"""
Contract tests: checkpoint format and backend interface.

These guard the interfaces that let work proceed in parallel. If a checkpoint
written today cannot be read tomorrow, or a backend silently changes its
signature, the benchmark and the parity gate both become meaningless.
"""

import numpy as np
import pytest

from pymvn import (AngularHead, BACKENDS, LiftParams, fit_lift, lift,
                   load_checkpoint, save_checkpoint)


def test_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    W = np.exp(1j * rng.uniform(0, 2 * np.pi, (10, 65)))
    Z = rng.normal(size=(200, 64))
    lp = fit_lift(Z, "minmax")

    p = save_checkpoint(tmp_path / "m.npz", W, lp,
                        backbone="test", dataset="synthetic", accuracy_fp32=0.9)
    ck = load_checkpoint(p)

    np.testing.assert_allclose(ck.W, W, atol=1e-12)
    assert ck.n_classes == 10 and ck.d1 == 65
    assert ck.meta["backbone"] == "test"
    assert ck.lift.kind == "minmax"


def test_metadata_cannot_shadow_save_checkpoint_parameters(tmp_path):
    """
    Metadata is forwarded as **kwargs, so a meta key named after one of
    save_checkpoint's own parameters collides with it. That is how the hybrid
    branch of train_zoo.py died at the final line, after paying for the whole
    training run: it passed lift=<LiftParams> positionally and lift=<str>
    through **extra.

    Guarding the contract rather than the caller: any meta key colliding with a
    parameter name must fail loudly here, not in a script an hour in.
    """
    import inspect
    reserved = set(inspect.signature(save_checkpoint).parameters) - {"meta"}
    assert {"path", "W", "lift"} <= reserved

    W = np.ones((2, 4), dtype=np.complex128)
    lp = fit_lift(np.zeros((5, 3)), "minmax")
    for name in ("lift", "W"):
        with pytest.raises(TypeError):
            save_checkpoint(tmp_path / "x.npz", W, lp, **{name: "collides"})


def test_hybrid_checkpoint_save_path(tmp_path):
    """The exact call shape train_zoo.py uses for backbone-based models."""
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(64, 16))
    lp = fit_lift(Z, "minmax")
    W = np.exp(1j * rng.uniform(0, 2 * np.pi, (10, 17)))
    p = save_checkpoint(tmp_path / "resnet18_mnist.npz", W, lp,
                        backbone="resnet18", dataset="mnist",
                        accuracy_fp32=0.97, epochs=5,
                        feature_dim=16, fully_phase_native=False)
    ck = load_checkpoint(p)
    assert ck.lift.kind == "minmax"          # the lift kind survives without
    assert ck.meta["backbone"] == "resnet18"  # a redundant meta key
    assert ck.meta["fully_phase_native"] is False


def test_checkpoint_has_no_pickle(tmp_path):
    """allow_pickle=False must suffice -- checkpoints carry no Python objects."""
    rng = np.random.default_rng(1)
    W = np.exp(1j * rng.uniform(0, 2 * np.pi, (3, 9)))
    p = save_checkpoint(tmp_path / "m.npz", W, None, note="plain")
    with np.load(p, allow_pickle=False) as z:
        assert "W_real" in z.files and "W_imag" in z.files


@pytest.mark.parametrize("kind", ["minmax", "tanh", "cdf", "learned"])
def test_all_lifts_land_on_the_torus(kind):
    """Every lift must produce unit-modulus output -- that is the whole point."""
    rng = np.random.default_rng(2)
    Ztr = rng.normal(size=(500, 32)) * 3.0
    Zte = rng.normal(size=(100, 32)) * 3.0
    C = lift(Zte, fit_lift(Ztr, kind))
    assert C.dtype == np.complex128
    np.testing.assert_allclose(np.abs(C), 1.0, atol=1e-12)


def test_lift_params_serialize(tmp_path):
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(300, 16))
    for kind in ("minmax", "tanh", "cdf", "learned"):
        lp = fit_lift(Z, kind)
        back = LiftParams.from_dict(lp.to_dict())
        assert back.kind == lp.kind
        np.testing.assert_allclose(lift(Z, back), lift(Z, lp), atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_interface_is_uniform(backend):
    """Every backend accepts the same inputs and returns the same shapes."""
    if backend == "onnx":
        pytest.importorskip("onnxruntime")
    if backend == "neon":
        from pymvn import neon_available
        if not neon_available():
            pytest.skip("kernel not built for this host")

    rng = np.random.default_rng(4)
    W = np.exp(1j * rng.uniform(0, 2 * np.pi, (5, 33)))
    X = np.exp(1j * rng.uniform(0, 2 * np.pi, (17, 32)))

    h = AngularHead(W, b=4, backend=backend)
    assert h.logits(X).shape == (17, 5)
    assert h.predict(X).shape == (17,)
    assert 0.0 <= h.accuracy(X, np.zeros(17, dtype=int)) <= 1.0
    assert set(h.mac_report()) >= {"head_macs", "multiplier_free", "lut_bytes"}


def test_rejects_unknown_backend():
    W = np.ones((2, 3), dtype=np.complex128)
    with pytest.raises(ValueError):
        AngularHead(W, backend="does_not_exist")
