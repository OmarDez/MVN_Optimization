"""
Tests for the phase algebra.

These are not incidental unit tests -- they are the machine-checked statement of
the project's central claim. If test_group_isomorphism fails, the multiplier-free
datapath is not valid and nothing downstream means anything.
"""

import numpy as np
import pytest

from pymvn import algebra as alg

BITS = [1, 2, 3, 4, 5, 6, 8]


@pytest.mark.parametrize("b", BITS)
def test_roundtrip_index_phase(b):
    """Index -> phase -> index is the identity on Z_L."""
    L = alg.n_roots(b)
    k = np.arange(L, dtype=np.int64)
    back = alg.phase_to_index(alg.index_to_phase(k, b), b)
    assert np.array_equal(back.astype(np.int64), k)


@pytest.mark.parametrize("b", BITS)
def test_group_isomorphism(b):
    """
    THE CENTRAL CLAIM.

    For every pair of roots of unity, complex multiplication equals modular
    integer addition of their indices. Exhaustive over the whole group.
    """
    L = alg.n_roots(b)
    ka, kb = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    ka, kb = ka.ravel(), kb.ravel()

    # Left side: honest complex multiplication.
    lhs = alg.index_to_unit(ka, b) * alg.index_to_unit(kb, b)

    # Right side: one integer add, then look up.
    rhs = alg.index_to_unit(alg.group_mul(ka.astype(np.int32),
                                          kb.astype(np.int32), b), b)

    np.testing.assert_allclose(lhs, rhs, atol=1e-12,
                               err_msg="mu_L -> Z_L homomorphism violated")


@pytest.mark.parametrize("b", BITS)
def test_group_axioms(b):
    """Identity, inverse and closure -- mu_L really is a group under this map."""
    L = alg.n_roots(b)
    k = np.arange(L, dtype=np.int32)
    zero = np.zeros_like(k)

    assert np.array_equal(alg.group_mul(k, zero, b), k)              # identity
    assert np.array_equal(alg.group_mul(k, alg.group_inv(k, b), b), zero)  # inverse
    assert np.all(alg.group_mul(k, k, b) < L)                        # closure


@pytest.mark.parametrize("b", BITS)
def test_division_is_subtraction(b):
    """The dual operation used by phase-coherence attention."""
    L = alg.n_roots(b)
    rng = np.random.default_rng(0)
    ka = rng.integers(0, L, 500).astype(np.int32)
    kb = rng.integers(0, L, 500).astype(np.int32)

    lhs = alg.index_to_unit(ka, b) * np.conj(alg.index_to_unit(kb, b))
    rhs = alg.index_to_unit(alg.group_div(ka, kb, b), b)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize("b", BITS)
def test_quantization_error_bounded(b):
    """
    Quantization error never exceeds half a sector. This is the bound that
    justifies choosing b from the ablation rather than by intuition.
    """
    L = alg.n_roots(b)
    rng = np.random.default_rng(1)
    theta = rng.uniform(-np.pi, np.pi, 20_000)
    k = alg.phase_to_index(theta, b)
    err = np.angle(np.exp(1j * theta) * np.conj(alg.index_to_unit(k, b)))
    assert np.abs(err).max() <= np.pi / L + 1e-9


@pytest.mark.parametrize("b", BITS)
def test_lut_fits_expected_width(b):
    """
    The hardware-relevant fact: for b <= 4 the int8 cosine table is at most
    16 bytes, i.e. exactly one AArch64 NEON register and one vqtbl1q_u8 operand.
    """
    c8 = alg.cos_lut_i8(b)
    assert c8.dtype == np.int8
    assert c8.nbytes == alg.n_roots(b)
    if b <= 4:
        assert c8.nbytes <= 16, "b<=4 must fit a single NEON table lookup"


@pytest.mark.parametrize("b", BITS)
def test_angular_distance_is_a_metric(b):
    """Geodesic distance on the circle: symmetric, zero iff equal, bounded."""
    L = alg.n_roots(b)
    rng = np.random.default_rng(2)
    ka = rng.integers(0, L, 400).astype(np.int32)
    kb = rng.integers(0, L, 400).astype(np.int32)

    d_ab = alg.angular_distance(ka, kb, b)
    d_ba = alg.angular_distance(kb, ka, b)
    assert np.array_equal(d_ab, d_ba)
    assert np.all(alg.angular_distance(ka, ka, b) == 0)
    assert d_ab.max() <= L // 2


def test_phase_only_preserves_unit_modulus():
    """phase_only quantization must land exactly on S^1 -- else no isomorphism."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=(10, 64)) + 1j * rng.normal(size=(10, 64))
    Wq = alg.quantize_unit(W, 4, phase_only=True)
    np.testing.assert_allclose(np.abs(Wq), 1.0, atol=1e-12)


def test_full_polar_preserves_modulus():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(10, 64)) + 1j * rng.normal(size=(10, 64))
    Wq = alg.quantize_unit(W, 4, phase_only=False)
    np.testing.assert_allclose(np.abs(Wq), np.abs(W), rtol=1e-12)
