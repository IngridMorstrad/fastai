"""Tests for GANDiscriminativeLR learning-rate guarding."""

from fastai.vision.gan import GANDiscriminativeLR


class _FakeOpt:
    """Minimal optimizer exposing just `hypers` and `set_hyper`."""

    def __init__(self, lr):
        self.hypers = [{'lr': lr}]

    def set_hyper(self, k, v):
        self.hypers[0][k] = v


class _FakeGANTrainer:
    def __init__(self, gen_mode):
        self.gen_mode = gen_mode


class _FakeLearn:
    def __init__(self, lr, gen_mode):
        self.opt = _FakeOpt(lr)
        self.gan_trainer = _FakeGANTrainer(gen_mode)


def _make_cb(lr, gen_mode, training, mult_lr=5.):
    cb = GANDiscriminativeLR(mult_lr=mult_lr)
    cb.learn = _FakeLearn(lr, gen_mode)
    cb.training = training
    return cb


class TestGANDiscriminativeLR:
    """Unit tests covering the self.training guard on before/after batch."""

    def test_after_batch_no_divide_during_validation(self):
        """During validation (training=False) the LR must not be divided."""
        cb = _make_cb(lr=0.01, gen_mode=False, training=False)
        cb.after_batch()
        assert cb.learn.opt.hypers[0]['lr'] == 0.01

    def test_before_batch_no_multiply_during_validation(self):
        """During validation (training=False) the LR must not be multiplied."""
        cb = _make_cb(lr=0.01, gen_mode=False, training=False)
        cb.before_batch()
        assert cb.learn.opt.hypers[0]['lr'] == 0.01

    def test_training_roundtrip_restores_lr(self):
        """During training the LR is multiplied then divided back to the original."""
        cb = _make_cb(lr=0.01, gen_mode=False, training=True)
        cb.before_batch()
        assert cb.learn.opt.hypers[0]['lr'] == 0.05
        cb.after_batch()
        assert cb.learn.opt.hypers[0]['lr'] == 0.01

    def test_after_batch_no_divide_in_gen_mode(self):
        """In generator mode the LR must not be divided even during training."""
        cb = _make_cb(lr=0.01, gen_mode=True, training=True)
        cb.after_batch()
        assert cb.learn.opt.hypers[0]['lr'] == 0.01
