"""Tests for DataLoaders.from_registry (fastai.data.core).

These tests are NETWORK-FREE: they monkeypatch `fastai.data.external.untar_data`
and `fastai.vision.data.ImageDataLoaders.from_folder` so that no dataset is ever
downloaded. `DataLoaders.from_registry` performs DEFERRED imports of both symbols
inside its method body, so patching the ORIGINAL definitions (where they live) is
what the deferred imports pick up at call time.

Covers:
  * case-insensitive / whitespace-tolerant name resolution to the correct URL,
  * dispatch to the correct builder with the resolved download path,
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
from fastai.data.core import DataLoaders


@pytest.fixture
def recorder(monkeypatch):
    """Patch untar_data + ImageDataLoaders.from_folder and record their calls.

    Returns a dict that captures the URL passed to untar_data and the
    (path, kwargs) passed to the builder. The builder stub returns a sentinel
    so we can assert from_registry returns exactly what the builder produced.
    """
    calls = {'url': None, 'path': None, 'builder_kwargs': None}
    fake_path = Path('/fake/cached/dataset')
    sentinel = object()

    def fake_untar_data(url, *args, **kwargs):
        calls['url'] = url
        return fake_path

    def fake_from_folder(cls, path, **kwargs):
        calls['path'] = path
        calls['builder_kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(external, 'untar_data', fake_untar_data)
    monkeypatch.setattr(vision_data.ImageDataLoaders, 'from_folder',
                        classmethod(fake_from_folder))
    calls['fake_path'] = fake_path
    calls['sentinel'] = sentinel
    return calls


def test_known_name_dispatches_to_correct_url_and_builder(recorder):
    result = DataLoaders.from_registry('cifar')
    assert recorder['url'] == external.URLs.CIFAR
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
    for name, url in expected.items():
        DataLoaders.from_registry(name)
        assert recorder['url'] == url, f"{name} -> wrong url"


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
