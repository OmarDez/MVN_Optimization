"""
Cross-backend parity.

THE MERGE GATE. A fast kernel that computes the wrong thing is worth nothing, so
no branch merges while these fail.

Two distinct notions of agreement are checked, and conflating them is a common
way to fool yourself:

  * EXACT parity -- backends that implement the same arithmetic must produce
    numerically identical results up to floating-point tolerance
    (complex128 vs complex64 vs onnx).

  * QUANTIZED parity -- the angular backends compute a genuinely different,
    lower-precision quantity. They cannot match bit for bit. What must hold is
    that they agree with the reference ANGULAR computation exactly, and that
    they agree with the full-precision model on PREDICTIONS at a high rate.

The `angular_naive` backend is the reference for the second group: it is the
literal transcription of the algebra, so `angular_tiled` and `neon` are required
to match it essentially exactly. Any drift there is a kernel bug, not a
quantization effect.
"""

import numpy as np
import pytest

from pymvn import AngularHead, neon_available

BITS = [3, 4, 5, 6]
N, D, K = 128, 257, 10


@pytest.fixture(scope="module")
def fixture():
    """A random but reproducible head plus inputs already on S^1."""
    rng = np.random.default_rng(20260731)
    W = np.exp(1j * rng.uniform(0, 2 * np.pi, (K, D + 1))) / np.sqrt(D + 1)
    X = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, D)))
    return W, X


def _head(W, b, backend, **kw):
    return AngularHead(W, b=b, backend=backend, **kw)


# ---------------------------------------------------------------------------
# Group 1: backends implementing identical arithmetic
# ---------------------------------------------------------------------------

def test_complex64_matches_complex128(fixture):
    W, X = fixture
    z64 = _head(W, 4, "complex64").logits(X)
    z128 = _head(W, 4, "complex128").logits(X)
    np.testing.assert_allclose(z64, z128, rtol=1e-4, atol=1e-4)


def test_onnx_matches_complex128(fixture):
    """
    The four-GEMM decomposition must reproduce complex arithmetic exactly.
    If this fails, the baseline is wrong and every speedup number is invalid.
    """
    pytest.importorskip("onnxruntime")
    W, X = fixture
    z_onnx = _head(W, 4, "onnx").logits(X)
    z_ref = _head(W, 4, "complex128").logits(X)
    np.testing.assert_allclose(z_onnx, z_ref, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# Group 2: angular backends must match the angular reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", BITS)
def test_tiled_matches_naive(fixture, b):
    """
    Same algebra, different memory schedule. Tolerance accounts only for the
    int8 LUT rounding that `angular_tiled` uses and `angular_naive` does not.
    """
    W, X = fixture
    z_naive = _head(W, b, "angular_naive").logits(X)
    z_tiled = _head(W, b, "angular_tiled").logits(X)

    # Per-term int8 rounding error is <= 0.5/127; over D+1 terms it accumulates
    # at most linearly.
    tol = (D + 1) * 0.5 / 127.0
    np.testing.assert_allclose(z_tiled.real, z_naive.real, atol=tol)
    np.testing.assert_allclose(z_tiled.imag, z_naive.imag, atol=tol)


@pytest.mark.parametrize("tile", [(16, 64), (32, 128), (64, 256), (128, 512)])
def test_tiling_is_schedule_invariant(fixture, tile):
    """
    Tile size is a performance knob and must never change the result. This
    catches off-by-one errors in the blocking loops.
    """
    W, X = fixture
    ref = _head(W, 4, "angular_tiled", tile=(64, 256)).logits(X)
    got = _head(W, 4, "angular_tiled", tile=tile).logits(X)
    np.testing.assert_allclose(got, ref, atol=1e-9)


@pytest.mark.skipif(not neon_available(), reason="kernel not built for this host")
@pytest.mark.parametrize("b", BITS)
def test_neon_matches_tiled(fixture, b):
    """
    The C kernel and the NumPy tiled path perform the identical integer
    computation, so they must agree to the last bit.
    """
    W, X = fixture
    z_neon = _head(W, b, "neon").logits(X)
    z_tiled = _head(W, b, "angular_tiled").logits(X)
    np.testing.assert_allclose(z_neon, z_tiled, atol=1e-9,
                               err_msg="NEON kernel diverges -- kernel bug")


# ---------------------------------------------------------------------------
# Group 3: prediction agreement against full precision
# ---------------------------------------------------------------------------

REFERENCE_BITS = 8  # "full precision" for agreement purposes


@pytest.mark.parametrize("b,min_agreement", [(3, 0.65), (4, 0.78), (5, 0.85), (6, 0.93)])
def test_prediction_agreement_vs_full_precision(fixture, b, min_agreement):
    """
    How often does the b-bit angular head pick the same class as the
    full-precision complex head?

    IMPORTANT -- these floors are calibrated on RANDOM weights, which is the
    worst case: random class logits are nearly tied, so a fraction of a sector
    of phase error is enough to flip the argmax. A trained model concentrates
    the decision margin and does substantially better.

    That is also why the comparison is against a fixed full-precision reference
    rather than against the same-b complex head: the latter is not monotone in
    b on random weights, because both sides move at once.

    Tighten these thresholds once real checkpoints land in models/
    (see PLAN.md, Experiment E3).
    """
    W, X = fixture
    p_ref = _head(W, REFERENCE_BITS, "complex128").predict(X)
    p_ang = _head(W, b, "angular_tiled").predict(X)
    agreement = float((p_ref == p_ang).mean())
    assert agreement >= min_agreement, (
        f"b={b}: agreement {agreement:.3f} < {min_agreement}"
    )


def test_agreement_improves_with_bits(fixture):
    """
    Spending more phase bits must buy accuracy. Checked end to end rather than
    step by step, since adjacent widths can tie on a small sample.
    """
    W, X = fixture
    p_ref = _head(W, REFERENCE_BITS, "complex128").predict(X)
    scores = [float((p_ref == _head(W, b, "angular_tiled").predict(X)).mean())
              for b in (2, 3, 4, 5, 6)]
    assert scores[-1] > scores[0], f"more bits did not help: {scores}"
    assert scores[-1] >= 0.90, f"b=6 should be close to full precision: {scores}"


# ---------------------------------------------------------------------------
# Group 4: the pre-indexed (phase-native) input path
# ---------------------------------------------------------------------------

ANGULAR_BACKENDS = ["angular_naive", "angular_tiled", "neon"]


@pytest.mark.parametrize("backend", ANGULAR_BACKENDS)
@pytest.mark.parametrize("b", BITS)
def test_pre_indexed_matches_complex_input(fixture, backend, b):
    """
    Feeding indices directly must be BIT-IDENTICAL to letting the head convert
    them itself. The conversion is pure overhead, not part of the arithmetic --
    if these ever diverge, the fast path is computing something else.
    """
    if backend == "neon" and not neon_available():
        pytest.skip("kernel not built for this host")
    W, X = fixture
    h = _head(W, b, backend)
    kX = h.to_indices(X)
    np.testing.assert_array_equal(h.logits(kX, indices=True), h.logits(X))


def test_to_indices_shape_and_dtype(fixture):
    """Indices are uint8 (the kernel's operand type) and carry the bias column."""
    W, X = fixture
    h = _head(W, 4, "angular_tiled")
    kX = h.to_indices(X)
    assert kX.dtype == np.uint8
    assert kX.shape == (X.shape[0], D + 1)
    assert kX.max() < h.L
    assert np.array_equal(kX[:, 0], np.zeros(X.shape[0]))  # bias index is 0


def test_pre_indexed_accepts_missing_bias_column(fixture):
    """(n, d) indices are padded with the bias index exactly like (n, d) complex."""
    W, X = fixture
    h = _head(W, 4, "angular_tiled")
    kX = h.to_indices(X)
    np.testing.assert_array_equal(h.logits(kX[:, 1:], indices=True),
                                  h.logits(kX, indices=True))


def test_pre_indexed_rejects_wrong_width(fixture):
    W, X = fixture
    h = _head(W, 4, "angular_tiled")
    with pytest.raises(ValueError):
        h.logits(h.to_indices(X)[:, :5], indices=True)


# ---------------------------------------------------------------------------
# Group 5: the contract the MAC claim rests on
# ---------------------------------------------------------------------------

def test_mac_report_requires_phase_only(fixture):
    """
    Keeping the modulus reintroduces a real multiply. The accounting must say
    so, otherwise the headline claim would be overstated.
    """
    W, _ = fixture
    assert _head(W, 4, "neon", phase_only=True).mac_report()["multiplier_free"]
    assert not _head(W, 4, "neon", phase_only=False).mac_report()["multiplier_free"]


def test_bias_column_handled_consistently(fixture):
    """Callers may pass X with or without the leading bias column."""
    W, X = fixture
    h = _head(W, 4, "complex128")
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    np.testing.assert_allclose(h.logits(X), h.logits(np.hstack([ones, X])))
