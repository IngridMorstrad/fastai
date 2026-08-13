"""Tests for DataLoaders.from_registry classmethod."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_from_registry_exists():
    """DataLoaders.from_registry should be a callable classmethod."""
    from fastai.data.core import DataLoaders
    assert hasattr(DataLoaders, 'from_registry')
    assert callable(DataLoaders.from_registry)


def test_from_registry_unknown_name_raises():
    """Passing an unknown dataset name should raise ValueError."""
    from fastai.data.core import DataLoaders
    with pytest.raises(ValueError, match="Unknown dataset 'nonexistent'"):
        DataLoaders.from_registry('nonexistent')


def test_from_registry_unknown_name_lists_available():
    """The ValueError should list all available dataset names."""
    from fastai.data.core import DataLoaders
    with pytest.raises(ValueError, match="Available:") as exc_info:
        DataLoaders.from_registry('bad_name')
    msg = str(exc_info.value)
    for name in ['mnist', 'cifar10', 'imagenette', 'imagewoof', 'imagewang']:
        assert name in msg


def test_registry_contains_expected_keys():
    """The _REGISTRY should contain at minimum the specified datasets."""
    from fastai.data.external import _REGISTRY
    expected = [
        'mnist', 'mnist_sample', 'mnist_tiny',
        'cifar10', 'cifar100',
        'imagenette', 'imagenette_160', 'imagenette_320',
        'imagewoof', 'imagewoof_160', 'imagewoof_320',
        'imagewang', 'imagewang_160', 'imagewang_320',
    ]
    for name in expected:
        assert name in _REGISTRY, f"'{name}' not in _REGISTRY"


def test_from_registry_case_insensitive():
    """from_registry should accept names case-insensitively."""
    from fastai.data.core import DataLoaders
    # These should not raise ValueError (they will fail at download, but we mock that)
    with patch('fastai.data.external.untar_data', return_value=Path('/tmp/fake')):
        with patch('fastai.vision.data.ImageDataLoaders.from_folder', return_value=MagicMock()):
            DataLoaders.from_registry('MNIST')
            DataLoaders.from_registry('Cifar10')
            DataLoaders.from_registry('ImageNette')


def test_from_registry_calls_untar_data_and_from_folder():
    """from_registry should call untar_data with the correct URL and ImageDataLoaders.from_folder."""
    from fastai.data.core import DataLoaders
    from fastai.data.external import URLs

    fake_path = Path('/tmp/fake_dataset')
    mock_dls = MagicMock()

    with patch('fastai.data.external.untar_data', return_value=fake_path) as mock_untar:
        with patch('fastai.vision.data.ImageDataLoaders.from_folder', return_value=mock_dls) as mock_from_folder:
            result = DataLoaders.from_registry('mnist', bs=32, item_tfms='fake_tfm')

    mock_untar.assert_called_once_with(URLs.MNIST)
    mock_from_folder.assert_called_once_with(
        fake_path, bs=32, item_tfms='fake_tfm', batch_tfms=None
    )
    assert result is mock_dls


def test_from_registry_passes_kwargs():
    """from_registry should forward **kwargs to ImageDataLoaders.from_folder."""
    from fastai.data.core import DataLoaders

    fake_path = Path('/tmp/fake')
    mock_dls = MagicMock()

    with patch('fastai.data.external.untar_data', return_value=fake_path):
        with patch('fastai.vision.data.ImageDataLoaders.from_folder', return_value=mock_dls) as mock_ff:
            DataLoaders.from_registry('cifar10', bs=128, valid_pct=0.3, seed=42)

    mock_ff.assert_called_once_with(
        fake_path, bs=128, item_tfms=None, batch_tfms=None, valid_pct=0.3, seed=42
    )
