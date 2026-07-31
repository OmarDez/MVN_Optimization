"""
PolarQuant: quantize the angle, not the magnitude.

Rotation-based schemes (QuaRot, SpinQuant) rotate to suppress outliers and then
quantize magnitudes on Cartesian axes. For a weight that already lives on S^1
the natural quantity is the ANGLE, so PolarQuant discretizes it directly to one
of 2^b roots of unity:

    w = r * e^{i*phi}

    phi_hat = (2*pi / 2^b) * floor(phi * 2^b / (2*pi) + 1/2)

    w_hat = r * e^{i*phi_hat}   (full-polar)
          =     e^{i*phi_hat}   (phase-only, r = 1)

Only the phase-only form yields the group isomorphism that makes the datapath
multiplier-free. The full-polar form is kept so the accuracy cost of discarding
r can be measured -- that trade is the central ablation of this project.

The scheme is data-oblivious: the range of phi is fixed at [-pi, pi), so no
calibration set is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import algebra as alg

__all__ = ["QuantStats", "polar_quantize", "compression_stats", "ablate_bits"]


@dataclass
class QuantStats:
    bits_phase: int
    bits_norm: int
    n_weights: int
    phase_only: bool

    @property
    def bits_quantized(self) -> int:
        per = self.bits_phase + (0 if self.phase_only else self.bits_norm)
        return per * self.n_weights

    @property
    def bits_original(self) -> int:
        return 128 * self.n_weights          # complex128

    @property
    def compression_ratio(self) -> float:
        return self.bits_original / max(self.bits_quantized, 1)


def polar_quantize(W: np.ndarray, b: int, phase_only: bool = True) -> np.ndarray:
    """Project complex weights onto mu_L (or r * mu_L). See algebra.quantize_unit."""
    return alg.quantize_unit(W, b, phase_only)


def compression_stats(W: np.ndarray, b: int, bits_norm: int = 8,
                      phase_only: bool = True) -> QuantStats:
    return QuantStats(bits_phase=b, bits_norm=bits_norm,
                      n_weights=int(np.asarray(W).size), phase_only=phase_only)


def ablate_bits(head_factory, X, y, bits=(1, 2, 3, 4, 5, 6, 8)) -> dict:
    """
    Sweep phase bit-width for both quantization modes.

    `head_factory(b, phase_only) -> object with .accuracy(X, y)`.

    Returns {"bits": [...], "full": [...], "phase_only": [...]}. This is the
    figure that answers "how many bits of phase does the network actually
    need?", and the answer determines whether the 16-byte NEON lookup table
    is sufficient.
    """
    out = {"bits": list(bits), "full": [], "phase_only": []}
    for b in bits:
        out["full"].append(head_factory(b, False).accuracy(X, y))
        out["phase_only"].append(head_factory(b, True).accuracy(X, y))
    return out
