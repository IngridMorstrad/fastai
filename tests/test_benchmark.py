"""Tests for Learner.benchmark() added via fastai.benchmark."""

import pytest
import torch
import torch.nn as nn

# Importing the module patches Learner.benchmark onto the real Learner class.
from fastai.benchmark import BenchmarkResult, _benchmark_impl, _format_table


# ---------------------------------------------------------------------------
# Minimal mock learner (same pattern as tests/test_gradient_noise.py)
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    """Tiny model: Linear(4, 2)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _MockDL:
    """Minimal mock DataLoader that yields one batch of the given shape."""

    def __init__(self, input_shape=(3, 4), n=8):
        self._input_shape = input_shape
        self._n = n

    def __iter__(self):
        xb = torch.randn(self._n, *self._input_shape[1:])
        yb = torch.randn(self._n, 1)
        yield (xb, yb)

    def __len__(self):
        return 1


class _MockDLs:
    """Minimal mock DataLoaders exposing a ``.valid`` attribute."""

    def __init__(self, input_shape=(3, 4), n=8):
        self.valid = _MockDL(input_shape=input_shape, n=n)
        self.train = _MockDL(input_shape=input_shape, n=n)


class _MockLearn:
    """Minimal mock of a Learner for testing benchmark without full fastai stack."""

    def __init__(self, model=None, dls=None):
        self.model = model or _SimpleModel()
        self.dls = dls or _MockDLs()
        self.training = False
        self._logged = []
        self.logger = self._logged.append


# ---------------------------------------------------------------------------
# Tests using _MockLearn (fast, no fastai data pipeline dependency)
# ---------------------------------------------------------------------------

class TestBenchmarkMock:
    """Unit tests using the lightweight _MockLearn fixture."""

    def test_default_parameters(self):
        """benchmark() with defaults returns 5 results (one per default batch size)."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(1, 8, 16, 32, 64))
        assert len(results) == 5

    def test_custom_batch_sizes(self):
        """Results count matches the supplied batch_sizes."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(2, 4))
        assert len(results) == 2
        assert results[0].batch_size == 2
        assert results[1].batch_size == 4

    def test_single_batch_size(self):
        """Edge case: a single batch size works."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(8,))
        assert len(results) == 1
        assert results[0].batch_size == 8

    def test_result_fields(self):
        """Each result contains all expected fields with correct types."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=5, warmup_batches=1)
        r = results[0]
        assert isinstance(r, BenchmarkResult)
        assert isinstance(r.batch_size, int)
        assert isinstance(r.n_batches, int)
        assert isinstance(r.total_samples, int)
        assert isinstance(r.avg_latency_ms, float)
        assert isinstance(r.std_latency_ms, float)
        assert isinstance(r.throughput_samples_per_sec, float)
        assert isinstance(r.peak_memory_bytes, int)

    def test_total_samples_calculation(self):
        """total_samples == batch_size * n_batches."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(7,), n_batches=13)
        assert results[0].total_samples == 7 * 13

    def test_latency_positive(self):
        """Latency values are positive (forward passes take non-zero time)."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=5)
        assert results[0].avg_latency_ms > 0

    def test_std_latency_non_negative(self):
        """Standard deviation of latency is non-negative."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=5)
        assert results[0].std_latency_ms >= 0

    def test_throughput_positive(self):
        """Throughput is positive for any real workload."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=5)
        assert results[0].throughput_samples_per_sec > 0

    def test_memory_non_negative(self):
        """Peak memory bytes is non-negative."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=3)
        assert results[0].peak_memory_bytes >= 0

    def test_n_batches_one(self):
        """Edge case: n_batches=1 works and std_latency is 0 (single sample)."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(2,), n_batches=1, warmup_batches=0)
        assert results[0].n_batches == 1
        assert results[0].std_latency_ms == 0.0  # single measurement

    def test_restores_training_state_when_training(self):
        """Model training state is restored after benchmarking (was training)."""
        learn = _MockLearn()
        learn.model.train()
        assert learn.model.training is True
        _benchmark_impl(learn, batch_sizes=(2,), n_batches=2, warmup_batches=1)
        assert learn.model.training is True

    def test_restores_training_state_when_eval(self):
        """Model eval state is preserved after benchmarking."""
        learn = _MockLearn()
        learn.model.eval()
        assert learn.model.training is False
        _benchmark_impl(learn, batch_sizes=(2,), n_batches=2, warmup_batches=1)
        assert learn.model.training is False

    def test_logger_receives_summary(self):
        """The learner's logger is called with a formatted summary string."""
        learn = _MockLearn()
        _benchmark_impl(learn, batch_sizes=(1,), n_batches=2)
        assert len(learn._logged) == 1
        table = learn._logged[0]
        assert "Batch" in table
        assert "Avg(ms)" in table
        assert "Throughput" in table

    def test_format_table_header_fields(self):
        """_format_table produces expected column headers."""
        r = BenchmarkResult(
            batch_size=4, n_batches=10, total_samples=40,
            avg_latency_ms=1.5, std_latency_ms=0.2,
            throughput_samples_per_sec=2666.0, peak_memory_bytes=1024,
        )
        table = _format_table([r])
        assert "Batch" in table
        assert "Samples" in table
        assert "PeakMem(B)" in table

    def test_multiple_batch_sizes_order(self):
        """Results are returned in the same order as batch_sizes."""
        learn = _MockLearn()
        sizes = (32, 1, 16)
        results = _benchmark_impl(learn, batch_sizes=sizes, n_batches=2, warmup_batches=1)
        for res, expected in zip(results, sizes):
            assert res.batch_size == expected

    def test_warmup_batches_zero(self):
        """warmup_batches=0 is valid (no warmup)."""
        learn = _MockLearn()
        results = _benchmark_impl(learn, batch_sizes=(4,), n_batches=3, warmup_batches=0)
        assert len(results) == 1
        assert results[0].avg_latency_ms > 0


# ---------------------------------------------------------------------------
# Integration test with synth_learner (requires full fastai data pipeline)
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    """Integration tests using synth_learner from fastai.test_utils."""

    @pytest.fixture(autouse=True)
    def _import_synth(self):
        """Import synth_learner; skip the whole class if the data pipeline is unavailable."""
        try:
            from fastai.test_utils import synth_learner as _sl
            # Importing benchmark patches the method onto Learner
            import fastai.benchmark  # noqa: F401
            self._synth_learner = _sl
        except Exception:
            pytest.skip("Full fastai data pipeline not available")

    def _make_learner(self, **kwargs):
        return self._synth_learner(**kwargs)

    def test_synth_learner_default(self):
        """benchmark() works out of the box on a synth_learner."""
        learn = self._make_learner()
        results = learn.benchmark(batch_sizes=(1, 2), n_batches=3, warmup_batches=1)
        assert len(results) == 2
        for r in results:
            assert r.avg_latency_ms > 0
            assert r.throughput_samples_per_sec > 0
            assert r.peak_memory_bytes >= 0

    def test_synth_learner_restores_state(self):
        """Model training flag restored after benchmark on real Learner."""
        learn = self._make_learner()
        learn.model.train()
        assert learn.model.training is True
        learn.benchmark(batch_sizes=(2,), n_batches=2, warmup_batches=1)
        assert learn.model.training is True

    def test_synth_learner_returns_benchmark_result(self):
        """Return type is list of BenchmarkResult on real Learner."""
        learn = self._make_learner()
        results = learn.benchmark(batch_sizes=(1,), n_batches=2)
        assert isinstance(results, list)
        assert isinstance(results[0], BenchmarkResult)
