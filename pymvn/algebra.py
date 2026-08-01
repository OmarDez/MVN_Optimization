"""
Phase algebra: the (S^1, x) -> (Z_2^b, +) isomorphism.

This module contains the primitives that the whole project rests on. Everything
else -- the kernels, the benchmarks, the tests -- is an implementation or a
measurement of what is defined here.

--------------------------------------------------------------------------
THE CORE IDENTITY
--------------------------------------------------------------------------
Unit-modulus complex numbers form a group under multiplication:

    S^1 = { e^{i*theta} : theta in [0, 2*pi) },     |w| = 1

Restricting the phase to b bits confines weights to the cyclic subgroup of
2^b-th roots of unity:

    mu_L = { e^{i*2*pi*j/L} : j = 0..L-1 },         L = 2^b

and mu_L is isomorphic to the additive group of integers modulo L:

    phi : mu_L  ->  Z_L,      phi(e^{i*2*pi*j/L}) = j

Because phi is a group isomorphism, multiplication becomes addition:

    e^{i*2*pi*k_w/L} * e^{i*2*pi*k_x/L} = e^{i*2*pi*((k_w + k_x) mod L)/L}

    ==>    k_out = (k_w + k_x) mod L

A complex multiply (4 real multiplies + 2 adds) collapses to ONE integer add.
When L is a power of two the modulo is a bitwise AND with (L - 1), so the
operation is a single-cycle integer instruction on any CPU.

--------------------------------------------------------------------------
WHERE THE ISOMORPHISM STOPS (state this honestly)
--------------------------------------------------------------------------
The homomorphism covers the PRODUCT only. Accumulation leaves the group:

    | sum_i w_i x_i |  !=  1   in general

so the weighted sum z = sum_i w_i x_i is NOT a point of mu_L, and its phase is
not an integer index. Reconstruction therefore requires mapping each product
index back to a Cartesian pair before summing:

    Re(z) = sum_i cos(2*pi*k_i/L)        Im(z) = sum_i sin(2*pi*k_i/L)

Those two cosine/sine evaluations are table lookups over an L-entry table. This
is the real cost of the scheme, and the reason the width b matters far beyond
model accuracy: for b <= 4 the table has at most 16 entries, which is exactly
the operand width of the AArch64 `TBL` table-lookup instruction (see
kernels/angular_neon.c).

--------------------------------------------------------------------------
THE STATE SPACE IS A TORUS
--------------------------------------------------------------------------
A layer with d unit-modulus inputs does not live in R^d or C^d. It lives on the
d-torus

    T^d = (S^1)^d

and, once quantized, on the finite lattice (Z_L)^d. This has measurable
consequences: the natural metric is geodesic (angular) distance, not Euclidean
distance, and wrap-around at 0 == 2*pi is a real topological feature rather
than a numerical edge case. It is also why FFT phase-only preprocessing works
so well as an input encoding -- it places the data on the correct manifold
before the model ever sees it.

--------------------------------------------------------------------------
THE DUAL OPERATION
--------------------------------------------------------------------------
Phase-coherence attention uses the mirror image of the product rule:

    Re<Q, K> = sum_i cos(phi^Q_i - phi^K_i)     ->   k = (k_Q - k_K) mod L

Product is an integer ADD; coherence is an integer SUB. Same datapath, same
lookup table, same hardware. This module exposes both.

--------------------------------------------------------------------------
WHY THE FFT IS NOT A COINCIDENCE
--------------------------------------------------------------------------
By Pontryagin duality the character group of S^1 is Z: the irreducible
representations of the circle are indexed by integers. Fourier analysis is
therefore the canonical change of basis for data living on S^1. The encoding
(FFT), the model (MVN), the quantization (roots of unity) and the hardware
(modular integer adder) are four views of one structure.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "n_roots",
    "phase_to_index",
    "index_to_phase",
    "index_to_unit",
    "cos_lut",
    "sin_lut",
    "cos_lut_i8",
    "sin_lut_i8",
    "LUT_SCALE_I8",
    "group_mul",
    "group_div",
    "group_inv",
    "angular_distance",
    "quantize_unit",
    "pack_indices",
    "unpack_indices",
]

# Fixed-point scale used for the int8 lookup tables consumed by the NEON
# kernel. cos/sin in [-1, 1] map to [-127, 127]; 127 (not 128) keeps the range
# symmetric and avoids the -128 saturation corner.
LUT_SCALE_I8 = 127


def n_roots(b: int) -> int:
    """Return L = 2**b, the number of representable phases."""
    if not (1 <= b <= 16):
        raise ValueError(f"b must be in [1, 16], got {b}")
    return 1 << b


def phase_to_index(theta: np.ndarray, b: int) -> np.ndarray:
    """
    Map angles to root indices: theta -> k in {0, ..., L-1}.

    Uses round-half-away-from-zero on the scaled angle, then wraps into [0, L).
    Angles need not be pre-wrapped; the modulo handles any real input.

    Returns uint8 when L <= 256 (the case the kernels care about), else uint16.
    """
    L = n_roots(b)
    scaled = np.asarray(theta, dtype=np.float64) * (L / (2.0 * np.pi))
    k = np.floor(scaled + 0.5).astype(np.int64) % L
    return k.astype(np.uint8 if L <= 256 else np.uint16)


def index_to_phase(k: np.ndarray, b: int) -> np.ndarray:
    """Map root indices back to angles in [0, 2*pi)."""
    L = n_roots(b)
    return np.asarray(k, dtype=np.float64) * (2.0 * np.pi / L)


def index_to_unit(k: np.ndarray, b: int) -> np.ndarray:
    """Map root indices to the corresponding points of mu_L in C."""
    return np.exp(1j * index_to_phase(k, b))


def cos_lut(b: int) -> np.ndarray:
    """Float64 cosine table of length L = 2**b."""
    L = n_roots(b)
    return np.cos(2.0 * np.pi * np.arange(L) / L)


def sin_lut(b: int) -> np.ndarray:
    """Float64 sine table of length L = 2**b."""
    L = n_roots(b)
    return np.sin(2.0 * np.pi * np.arange(L) / L)


def cos_lut_i8(b: int) -> np.ndarray:
    """
    Int8 cosine table. For b <= 4 this is at most 16 bytes, which fits a single
    128-bit NEON vector and can be indexed with one `vqtbl1q_u8` instruction.
    """
    return np.rint(cos_lut(b) * LUT_SCALE_I8).astype(np.int8)


def sin_lut_i8(b: int) -> np.ndarray:
    """Int8 sine table. See cos_lut_i8."""
    return np.rint(sin_lut(b) * LUT_SCALE_I8).astype(np.int8)


def group_mul(k_a: np.ndarray, k_b: np.ndarray, b: int) -> np.ndarray:
    """
    The isomorphism in action: multiplication in mu_L is addition in Z_L.

    Since L is a power of two, `% L` compiles to a bitwise AND with (L - 1).
    """
    L = n_roots(b)
    return ((k_a.astype(np.int32) + k_b.astype(np.int32)) & (L - 1)).astype(k_a.dtype)


def group_div(k_a: np.ndarray, k_b: np.ndarray, b: int) -> np.ndarray:
    """
    Division in mu_L is subtraction in Z_L. This is the dual operation used by
    phase-coherence attention.
    """
    L = n_roots(b)
    return ((k_a.astype(np.int32) - k_b.astype(np.int32)) & (L - 1)).astype(k_a.dtype)


def group_inv(k: np.ndarray, b: int) -> np.ndarray:
    """Inverse element: conjugation on S^1 is negation in Z_L."""
    L = n_roots(b)
    return ((-k.astype(np.int32)) & (L - 1)).astype(k.dtype)


def angular_distance(k_a: np.ndarray, k_b: np.ndarray, b: int) -> np.ndarray:
    """
    Geodesic distance on the discretized circle, in index units.

    This is the natural metric on T^d and the correct notion of "close" for
    every quantization decision in this project. Range: [0, L/2].
    """
    L = n_roots(b)
    d = np.abs(k_a.astype(np.int32) - k_b.astype(np.int32)) % L
    return np.minimum(d, L - d)


def pack_indices(k: np.ndarray, b: int) -> np.ndarray:
    """
    Pack two b-bit indices per byte along the last axis. Requires b <= 4.

    This is where the 32x weight compression lives: at b = 4 a phase index needs
    a nibble, not a byte, so the head's weights halve again against the uint8
    form. The last axis is zero-padded to an even length; index 0 is the group
    identity, so a padded column contributes cos(0) = 1 and must be trimmed by
    the caller rather than summed blindly.

    Layout: low nibble is the even position, high nibble the odd one. The C
    kernel's `angular_gemm_packed` assumes exactly this.
    """
    if b > 4:
        raise ValueError(f"packing needs b <= 4 (two indices per byte), got b={b}")
    k = np.ascontiguousarray(np.asarray(k, dtype=np.uint8))
    if k.shape[-1] % 2:
        pad = np.zeros(k.shape[:-1] + (1,), dtype=np.uint8)
        k = np.concatenate([k, pad], axis=-1)
    lo = k[..., 0::2] & np.uint8(0x0F)
    hi = k[..., 1::2] & np.uint8(0x0F)
    return np.ascontiguousarray((lo | (hi << np.uint8(4))).astype(np.uint8))


def unpack_indices(packed: np.ndarray, b: int, width: int | None = None) -> np.ndarray:
    """Inverse of pack_indices. `width` trims the padding column when odd."""
    if b > 4:
        raise ValueError(f"packing needs b <= 4, got b={b}")
    packed = np.asarray(packed, dtype=np.uint8)
    lo = packed & np.uint8(0x0F)
    hi = packed >> np.uint8(4)
    out = np.empty(packed.shape[:-1] + (2 * packed.shape[-1],), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out if width is None else out[..., :width]


def quantize_unit(w: np.ndarray, b: int, phase_only: bool = True) -> np.ndarray:
    """
    Project complex weights onto mu_L (phase_only=True) or onto r * mu_L
    (phase_only=False, "full-polar": keep the modulus, quantize only the angle).

    Only the phase_only case yields the group isomorphism -- with r != 1 the
    product is a scaled rotation and a real multiply survives. The full-polar
    mode exists so the accuracy cost of dropping r can be measured rather than
    assumed.
    """
    w = np.asarray(w)
    k = phase_to_index(np.angle(w), b)
    unit = index_to_unit(k, b)
    return unit if phase_only else np.abs(w) * unit
