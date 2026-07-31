"""
Checkpoint contract.

A model checkpoint is a single .npz that carries everything inference needs:
weights, lift parameters and metadata. Nothing about training, nothing about
the dataset, no pickled Python objects. This is what makes benchmarks portable
-- train wherever, measure on Arm.

Layout (models/<name>.npz):

    W_real     float64 (k, d+1)   real part of the complex weights
    W_imag     float64 (k, d+1)   imaginary part
    lift_*     ...                fitted lift parameters (see encode.LiftParams)
    meta       str (json)         backbone, dataset, fp accuracy, timestamp

The bias term is column 0 of W, matching AngularHead's convention.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np

from .encode import LiftParams

__all__ = ["save_checkpoint", "load_checkpoint", "Checkpoint"]


class Checkpoint(dict):
    """Thin dict wrapper with attribute-style access to the common fields."""

    @property
    def W(self) -> np.ndarray:
        return self["W_real"] + 1j * self["W_imag"]

    @property
    def lift(self) -> LiftParams:
        return LiftParams.from_dict(self)

    @property
    def meta(self) -> dict:
        return json.loads(str(self["meta"]))

    @property
    def n_classes(self) -> int:
        return int(self["W_real"].shape[0])

    @property
    def d1(self) -> int:
        return int(self["W_real"].shape[1])


def save_checkpoint(path: str | pathlib.Path, W: np.ndarray,
                    lift: LiftParams | None = None, **meta: Any) -> pathlib.Path:
    """
    Freeze a trained head. `meta` should always record at minimum:
      backbone, dataset, accuracy_fp32, and how the model was trained.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W = np.asarray(W, dtype=np.complex128)

    payload: dict[str, Any] = {
        "W_real": np.ascontiguousarray(W.real),
        "W_imag": np.ascontiguousarray(W.imag),
    }
    if lift is not None:
        payload.update(lift.to_dict())

    meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("n_classes", int(W.shape[0]))
    meta.setdefault("d1", int(W.shape[1]))
    payload["meta"] = json.dumps(meta, default=str)

    np.savez_compressed(path, **payload)
    return path


def load_checkpoint(path: str | pathlib.Path) -> Checkpoint:
    """Load a frozen head. Raises if the required arrays are missing."""
    with np.load(pathlib.Path(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    for required in ("W_real", "W_imag"):
        if required not in data:
            raise KeyError(f"checkpoint missing required array: {required}")
    return Checkpoint(data)
