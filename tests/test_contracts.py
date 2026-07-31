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
