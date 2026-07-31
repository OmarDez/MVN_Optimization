"""
Encodings: getting real-valued data onto the torus T^d.

Two entry points exist, corresponding to the two model families in the zoo.

1. `encode_fft` -- for raw images fed to a pure MLMVN. Takes the 2-D spectrum,
   discards magnitude, keeps phase. This is the "phase-only" preprocessing of
   Oppenheim & Lim (1981): the structural content of an image lives in the
   phase of its spectrum, not the magnitude.

2. `lift` -- for feature vectors produced by a real-valued backbone. Maps
   R^d -> T^d so an angular head can consume them.

The lift is the interface between the real world and the phase world, and its
form is an open empirical question. Four variants are implemented so the choice
can be measured instead of assumed. See PLAN.md, Experiment E2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = ["encode_fft", "encode_pixels", "LiftParams", "fit_lift", "lift", "LiftKind"]

LiftKind = Literal["minmax", "tanh", "cdf", "learned"]


# ----------------------------------------------------------------------------
# Image -> S^1 (pure MLMVN path)
# ----------------------------------------------------------------------------

def encode_fft(X: np.ndarray, freq_size: int = 12, image_shape=(28, 28)) -> np.ndarray:
    """
    Image -> 2-D FFT -> fftshift -> central low-frequency crop -> unit phase.

    Args:
        X: (n, H*W) array of pixel values. uint8 [0,255] or float [0,1].
        freq_size: side of the centred crop; output dimension is freq_size**2.
        image_shape: (H, W) of the input images.

    Returns:
        (n, freq_size**2) complex128 array with every entry on S^1.
    """
    H, W = image_shape
    n = X.shape[0]
    imgs = np.asarray(X, dtype=np.float64).reshape(n, H, W)
    if imgs.max() > 1.5:  # heuristic: still in [0, 255]
        imgs = imgs / 255.0

    spec = np.fft.fftshift(np.fft.fft2(imgs, axes=(1, 2)), axes=(1, 2))

    cy, cx = H // 2, W // 2
    half = freq_size // 2
    lo_y, lo_x = cy - half, cx - half
    crop = spec[:, lo_y:lo_y + freq_size, lo_x:lo_x + freq_size]

    phase = np.angle(crop).reshape(n, -1)
    return np.exp(1j * phase)


def encode_pixels(X: np.ndarray) -> np.ndarray:
    """
    Trivial baseline encoding: pixel value -> e^{i*pi*v}. Exists purely as the
    control condition that demonstrates the FFT encoding is doing real work.
    """
    v = np.asarray(X, dtype=np.float64)
    if v.max() > 1.5:
        v = v / 255.0
    return np.exp(1j * np.pi * v)


# ----------------------------------------------------------------------------
# R^d -> T^d (hybrid backbone path)
# ----------------------------------------------------------------------------

@dataclass
class LiftParams:
    """
    Fitted parameters of a lift map. Serialized with the checkpoint so that
    inference is fully reproducible without the training data.
    """
    kind: LiftKind = "minmax"
    lo: np.ndarray = field(default=None)        # (d,) minmax lower bound
    hi: np.ndarray = field(default=None)        # (d,) minmax upper bound
    tau: float = 1.0                            # tanh temperature
    quantiles: np.ndarray = field(default=None) # (d, Q) cdf knots
    a: np.ndarray = field(default=None)         # (d,) learned scale
    b: np.ndarray = field(default=None)         # (d,) learned offset

    def to_dict(self) -> dict:
        out = {"lift_kind": self.kind, "lift_tau": np.float64(self.tau)}
        for name in ("lo", "hi", "quantiles", "a", "b"):
            v = getattr(self, name)
            if v is not None:
                out[f"lift_{name}"] = np.asarray(v)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "LiftParams":
        get = lambda k: np.asarray(d[f"lift_{k}"]) if f"lift_{k}" in d else None
        return cls(
            kind=str(d.get("lift_kind", "minmax")),
            lo=get("lo"), hi=get("hi"),
            tau=float(d.get("lift_tau", 1.0)),
            quantiles=get("quantiles"), a=get("a"), b=get("b"),
        )


def fit_lift(Z: np.ndarray, kind: LiftKind = "minmax", tau: float = 1.0,
             n_quantiles: int = 256) -> LiftParams:
    """
    Fit a lift on TRAINING features only. Never refit on test data -- doing so
    leaks distribution information and inflates accuracy.

    Args:
        Z: (n, d) real feature matrix from the backbone.
    """
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]

    if kind == "minmax":
        return LiftParams(kind="minmax", lo=Z.min(0), hi=Z.max(0))

    if kind == "tanh":
        # Robust scale: median absolute deviation, immune to outliers.
        med = np.median(Z, axis=0)
        mad = np.median(np.abs(Z - med), axis=0) + 1e-9
        return LiftParams(kind="tanh", lo=med, hi=mad, tau=tau)

    if kind == "cdf":
        qs = np.linspace(0.0, 1.0, n_quantiles)
        knots = np.quantile(Z, qs, axis=0).T           # (d, Q)
        return LiftParams(kind="cdf", quantiles=knots)

    if kind == "learned":
        # Initialized from minmax; a and b are then refined by the training
        # loop (scripts/train_zoo.py). Stored so inference is exact.
        lo, hi = Z.min(0), Z.max(0)
        rng = np.maximum(hi - lo, 1e-9)
        return LiftParams(kind="learned", a=np.pi / rng, b=-np.pi * lo / rng,
                          lo=lo, hi=hi)

    raise ValueError(f"unknown lift kind: {kind}")


def lift(Z: np.ndarray, p: LiftParams) -> np.ndarray:
    """
    Apply a fitted lift: R^d -> T^d. Returns complex128 with |entry| == 1.

    minmax  : e^{i*pi*normalized(z)}          -- range [0, pi], half the circle
    tanh    : e^{i*pi*tanh((z-med)/(tau*mad))} -- outlier-robust
    cdf     : e^{i*2*pi*F(z)}                 -- uniform on the full circle
    learned : e^{i*(a*z + b)}                 -- per-channel affine, trainable
    """
    Z = np.asarray(Z, dtype=np.float64)

    if p.kind == "minmax":
        rng = np.maximum(p.hi - p.lo, 1e-9)
        theta = np.pi * np.clip((Z - p.lo) / rng, 0.0, 1.0)

    elif p.kind == "tanh":
        theta = np.pi * np.tanh((Z - p.lo) / (p.tau * p.hi))

    elif p.kind == "cdf":
        # Empirical CDF by interpolation against the stored quantile knots.
        d = Z.shape[1]
        Q = p.quantiles.shape[1]
        grid = np.linspace(0.0, 1.0, Q)
        theta = np.empty_like(Z)
        for j in range(d):
            theta[:, j] = 2.0 * np.pi * np.interp(Z[:, j], p.quantiles[j], grid)

    elif p.kind == "learned":
        theta = p.a * Z + p.b

    else:
        raise ValueError(f"unknown lift kind: {p.kind}")

    return np.exp(1j * theta)
