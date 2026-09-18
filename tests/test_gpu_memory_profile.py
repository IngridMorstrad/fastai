"""Tests for GPUMemoryProfileCallback."""

import pytest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from fastai.callback.training import GPUMemoryProfileCallback


class _MockLearn:
    """Minimal mock of a Learner for testing callbacks."""
    def __init__(self, model):
        self.model = model


class TestGPUMemoryProfileCallbackInit:
    """Tests for constructor and default parameter values."""

    def test_default_parameters(self):
        cb = GPUMemoryProfileCallback()
        assert cb.log_every == 1
        assert cb.peak_only is False

    def test_custom_parameters(self):
        cb = GPUMemoryProfileCallback(log_every=5, peak_only=True)
        assert cb.log_every == 5
        assert cb.peak_only is True

    def test_order_is_set(self):
        cb = GPUMemoryProfileCallback()
        assert cb.order == 70

    def test_run_valid_is_false(self):
        cb = GPUMemoryProfileCallback()
        assert cb.run_valid is False


class TestGPUMemoryProfileCallbackNoCUDA:
    """Tests for behaviour when CUDA is not available."""

    def test_before_fit_disables_when_no_cuda(self):
        cb = GPUMemoryProfileCallback()
        cb.learn = _MockLearn(nn.Linear(2, 1))
        with patch('torch.cuda.is_available', return_value=False):
            cb.before_fit()
        assert cb.run is False

    def test_before_fit_disables_when_model_on_cpu(self):
        cb = GPUMemoryProfileCallback()
        model = nn.Linear(2, 1)  # CPU by default
        cb.learn = _MockLearn(model)
        with patch('torch.cuda.is_available', return_value=True):
            cb.before_fit()
        assert cb.run is False


class TestGPUMemoryProfileCallbackWithMockedCUDA:
    """Tests with CUDA functions mocked to avoid requiring a GPU."""

    def _setup_cb_with_cuda_model(self, log_every=1, peak_only=False):
        """Create a callback with a mock CUDA model."""
        cb = GPUMemoryProfileCallback(log_every=log_every, peak_only=peak_only)
        model = nn.Linear(2, 1)

        # Create a fake CUDA device for the parameters
        fake_device = torch.device('cuda', 0)
        fake_param = MagicMock()
        fake_param.device = fake_device

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([fake_param])

        cb.learn = _MockLearn(mock_model)
        return cb

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_before_fit_initializes_tracking(self, mock_reset, mock_avail):
        cb = self._setup_cb_with_cuda_model()
        cb.before_fit()
        assert cb.peak_log == []
        assert cb._epoch_peak == 0.
        assert cb.device == torch.device('cuda', 0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.max_memory_allocated', return_value=50 * 1024**2)
    @patch('torch.cuda.memory_allocated', return_value=30 * 1024**2)
    def test_after_batch_tracks_peak(self, mock_alloc, mock_peak, mock_reset, mock_avail):
        cb = self._setup_cb_with_cuda_model()
        cb.before_fit()
        cb.before_train()
        # Simulate batch context
        cb.iter = 0
        cb.n_iter = 10
        cb.after_batch()
        assert cb._epoch_peak == pytest.approx(50.0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.max_memory_allocated', return_value=50 * 1024**2)
    @patch('torch.cuda.memory_allocated', return_value=30 * 1024**2)
    def test_after_batch_peak_only_suppresses_step_print(self, mock_alloc, mock_peak, mock_reset, mock_avail, capsys):
        cb = self._setup_cb_with_cuda_model(peak_only=True)
        cb.before_fit()
        cb.before_train()
        cb.iter = 0
        cb.n_iter = 10
        cb.after_batch()
        captured = capsys.readouterr()
        # peak_only=True should suppress per-step output
        assert '[GPU] step' not in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.max_memory_allocated', return_value=50 * 1024**2)
    @patch('torch.cuda.memory_allocated', return_value=30 * 1024**2)
    def test_after_batch_prints_at_log_interval(self, mock_alloc, mock_peak, mock_reset, mock_avail, capsys):
        cb = self._setup_cb_with_cuda_model(log_every=2)
        cb.before_fit()
        cb.before_train()
        # iter=0 → step 1, not a multiple of 2 → no print
        cb.iter = 0
        cb.n_iter = 10
        cb.after_batch()
        captured = capsys.readouterr()
        assert '[GPU] step' not in captured.out
        # iter=1 → step 2, multiple of 2 → print
        cb.iter = 1
        cb.after_batch()
        captured = capsys.readouterr()
        assert '[GPU] step 2/10' in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.max_memory_allocated', return_value=100 * 1024**2)
    @patch('torch.cuda.memory_allocated', return_value=60 * 1024**2)
    def test_after_train_logs_epoch_peak(self, mock_alloc, mock_peak, mock_reset, mock_avail, capsys):
        cb = self._setup_cb_with_cuda_model(peak_only=True)
        cb.before_fit()
        cb.before_train()
        cb.epoch = 0
        cb.iter = 0
        cb.n_iter = 1
        cb.after_batch()
        cb.after_train()
        captured = capsys.readouterr()
        assert '[GPU] epoch 0: peak memory = 100.0MB' in captured.out
        assert len(cb.peak_log) == 1
        assert cb.peak_log[0] == pytest.approx(100.0)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_after_fit_prints_overall_peak(self, mock_reset, mock_avail, capsys):
        cb = self._setup_cb_with_cuda_model()
        cb.before_fit()
        cb.peak_log = [50.0, 120.0, 80.0]
        cb.after_fit()
        captured = capsys.readouterr()
        assert '[GPU] training peak across all epochs: 120.0MB' in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_after_fit_no_output_when_empty(self, mock_reset, mock_avail, capsys):
        cb = self._setup_cb_with_cuda_model()
        cb.before_fit()
        # peak_log is empty
        cb.after_fit()
        captured = capsys.readouterr()
        assert 'training peak' not in captured.out

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_multiple_epochs_accumulate_peak_log(self, mock_reset, mock_avail):
        cb = self._setup_cb_with_cuda_model()
        cb.before_fit()

        # Epoch 0 with peak of 200MB
        with patch('torch.cuda.max_memory_allocated', return_value=200 * 1024**2), \
             patch('torch.cuda.memory_allocated', return_value=100 * 1024**2):
            cb.before_train()
            cb.epoch = 0
            cb.iter = 0
            cb.n_iter = 1
            cb.after_batch()
            cb.after_train()

        # Epoch 1 with peak of 150MB
        with patch('torch.cuda.max_memory_allocated', return_value=150 * 1024**2), \
             patch('torch.cuda.memory_allocated', return_value=80 * 1024**2):
            cb.before_train()
            cb.epoch = 1
            cb.iter = 0
            cb.n_iter = 1
            cb.after_batch()
            cb.after_train()

        assert len(cb.peak_log) == 2
        assert cb.peak_log[0] == pytest.approx(200.0)
        assert cb.peak_log[1] == pytest.approx(150.0)

    def test_plot_peaks_callable(self):
        """plot_peaks method exists and is callable."""
        cb = GPUMemoryProfileCallback()
        assert callable(cb.plot_peaks)
