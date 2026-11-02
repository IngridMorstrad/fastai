"""Tests for DataLoaders.from_registry (fastai.data.core).

These tests are NETWORK-FREE: they monkeypatch `fastai.data.external.untar_data`,
`fastai.vision.data.ImageDataLoaders.from_folder`, `.from_name_re`, and
`fastai.data.transforms.get_image_files` so that no dataset is ever downloaded and
no real filesystem layout is touched. `DataLoaders.from_registry` performs DEFERRED
imports of these symbols inside its method body, so patching the ORIGINAL definitions
(where they live) is what the deferred imports pick up at call time.

Covers:
  * case-insensitive / whitespace-tolerant name resolution to the correct URL,
  * dispatch to the correct builder with the resolved download path,
  * folder-based datasets dispatch to `from_folder`; `pets` dispatches to the
    filename-regex builder `from_name_re` (NOT `from_folder`, which would mislabel
    the flat oxford-iiit-pet `images/` folder as a single class),
  * caller kwargs forwarding (and overriding the spec defaults),
  * unsupported names raising a helpful ValueError (name + supported list),
  * the text-family name 'glue' raising ValueError rather than being handled.
"""
import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fastai.data.external as external
import fastai.vision.data as vision_data
import fastai.data.transforms as data_transforms
from fastai.data.core import DataLoaders


@pytest.fixture
def recorder(monkeypatch):
    """Patch untar_data + the image builders and record their calls.

    Returns a dict that captures the URL passed to untar_data, which builder was
    invoked ('from_folder' or 'from_name_re'), and the (path, kwargs) passed to
    that builder. Each builder stub returns a sentinel so we can assert
    from_registry returns exactly what the builder produced. `get_image_files` is
    stubbed to avoid touching a real filesystem for the name-regex (`pets`) path.
    """
    calls = {'url': None, 'path': None, 'builder_kwargs': None, 'builder': None}
    fake_path = Path('/fake/cached/dataset')
    sentinel = object()
    fake_fnames = ['Abyssinian_1.jpg', 'keeshond_34.jpg']

    def fake_untar_data(url, *args, **kwargs):
        calls['url'] = url
        return fake_path

    def fake_from_folder(cls, path, **kwargs):
        calls['builder'] = 'from_folder'
        calls['path'] = path
        calls['builder_kwargs'] = kwargs
        return sentinel

    def fake_from_name_re(cls, path, fnames, pat, **kwargs):
        calls['builder'] = 'from_name_re'
        calls['path'] = path
        calls['fnames'] = fnames
        calls['pat'] = pat
        calls['builder_kwargs'] = kwargs
        return sentinel

    def fake_get_image_files(path, *args, **kwargs):
        calls['get_image_files_path'] = path
        return fake_fnames

    monkeypatch.setattr(external, 'untar_data', fake_untar_data)
    monkeypatch.setattr(vision_data.ImageDataLoaders, 'from_folder',
                        classmethod(fake_from_folder))
    monkeypatch.setattr(vision_data.ImageDataLoaders, 'from_name_re',
                        classmethod(fake_from_name_re))
    monkeypatch.setattr(data_transforms, 'get_image_files', fake_get_image_files)
    calls['fake_path'] = fake_path
    calls['fake_fnames'] = fake_fnames
    calls['sentinel'] = sentinel
    return calls


def test_known_name_dispatches_to_correct_url_and_builder(recorder):
    result = DataLoaders.from_registry('cifar')
    assert recorder['url'] == external.URLs.CIFAR
    assert recorder['builder'] == 'from_folder'
    assert recorder['path'] == recorder['fake_path']
    assert result is recorder['sentinel']


@pytest.mark.parametrize('name', ['CIFAR', '  cifar  ', 'CiFaR', '\tcifar\n'])
def test_name_resolution_is_case_insensitive_and_trimmed(name, recorder):
    DataLoaders.from_registry(name)
    assert recorder['url'] == external.URLs.CIFAR


def test_each_registered_name_resolves_to_its_url(recorder):
    expected = {
        'cifar': external.URLs.CIFAR,
        'cifar100': external.URLs.CIFAR_100,
        'mnist': external.URLs.MNIST,
        'mnist_sample': external.URLs.MNIST_SAMPLE,
        'mnist_tiny': external.URLs.MNIST_TINY,
        'imagenette': external.URLs.IMAGENETTE,
        'imagenette_160': external.URLs.IMAGENETTE_160,
        'imagenette_320': external.URLs.IMAGENETTE_320,
        'imagewoof': external.URLs.IMAGEWOOF,
        'pets': external.URLs.PETS,
    }
    # pets uses the filename-regex builder; every other dataset uses from_folder.
    expected_builder = {name: ('from_name_re' if name == 'pets' else 'from_folder')
                        for name in expected}
    for name, url in expected.items():
        DataLoaders.from_registry(name)
        assert recorder['url'] == url, f"{name} -> wrong url"
        assert recorder['builder'] == expected_builder[name], f"{name} -> wrong builder"


def test_pets_dispatches_to_name_regex_builder(recorder):
    # oxford-iiit-pet ships a FLAT images/ folder with the class in the filename, so pets must
    # use from_name_re over path/'images', NOT from_folder (which would mislabel every image).
    result = DataLoaders.from_registry('pets')
    assert recorder['url'] == external.URLs.PETS
    assert recorder['builder'] == 'from_name_re'
    assert result is recorder['sentinel']
    # builder runs over the images/ subfolder, and get_image_files is called on that same path
    assert recorder['path'] == recorder['fake_path']/'images'
    assert recorder['get_image_files_path'] == recorder['fake_path']/'images'
    assert recorder['fnames'] is recorder['fake_fnames']
    # a regex that captures the class prefix before the trailing _<digits>.jpg
    assert recorder['pat'] == r'(.+)_\d+.jpg$'
    # spec defaults are applied (random 20% validation split with a fixed seed)
    assert recorder['builder_kwargs']['valid_pct'] == 0.2
    assert recorder['builder_kwargs']['seed'] == 42


def test_pets_caller_kwargs_override_spec_defaults(recorder):
    DataLoaders.from_registry('pets', valid_pct=0.3, seed=7, bs=8)
    bk = recorder['builder_kwargs']
    assert bk['valid_pct'] == 0.3
    assert bk['seed'] == 7
    assert bk['bs'] == 8


def test_caller_kwargs_forwarded_to_builder(recorder):
    item_tfms = ['dummy_tfm']
    DataLoaders.from_registry('mnist_sample', bs=16, item_tfms=item_tfms)
    bk = recorder['builder_kwargs']
    assert bk['bs'] == 16
    assert bk['item_tfms'] is item_tfms


def test_caller_kwargs_override_spec_defaults(recorder):
    # 'imagenette' default spec uses valid='val'; caller override must win.
    DataLoaders.from_registry('imagenette', valid='validation')
    assert recorder['builder_kwargs']['valid'] == 'validation'


def test_spec_default_kwargs_applied_when_not_overridden(recorder):
    DataLoaders.from_registry('imagenette')
    bk = recorder['builder_kwargs']
    assert bk['train'] == 'train'
    assert bk['valid'] == 'val'


def test_unsupported_name_raises_valueerror_with_name_and_supported_list():
    with pytest.raises(ValueError) as exc:
        DataLoaders.from_registry('not_a_real_dataset')
    msg = str(exc.value)
    assert 'not_a_real_dataset' in msg
    # every supported name must be listed
    for name in ['cifar', 'cifar100', 'mnist', 'mnist_sample', 'mnist_tiny',
                 'imagenette', 'imagenette_160', 'imagenette_320', 'imagewoof',
                 'pets']:
        assert name in msg, f"supported name {name!r} missing from error"


def test_glue_raises_valueerror_not_silently_handled():
    with pytest.raises(ValueError) as exc:
        DataLoaders.from_registry('glue')
    msg = str(exc.value)
    assert 'glue' in msg
    # it must flag glue as an unsupported text family, not return empty data
    assert 'text' in msg.lower()


def test_glue_is_case_insensitive_too():
    with pytest.raises(ValueError) as exc:
        DataLoaders.from_registry('  GLUE ')
    assert 'GLUE' in str(exc.value)
