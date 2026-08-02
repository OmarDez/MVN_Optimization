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

    W_hidden_real  float64 (h, d0+1)   optional: the layer that PRODUCES the
    W_hidden_imag  float64 (h, d0+1)   head's input, for MLMVN checkpoints

The bias term is column 0 of W, matching AngularHead's convention.

Only W_real and W_imag are required, and `load_checkpoint` returns every array
the file happens to contain. That is what makes W_hidden_* an additive
extension rather than a contract change: files written before it existed still
load, readers that do not know about it still work, and readers that do can
now regenerate the head's input instead of feeding it random points on S^1.

Without it an MLMVN checkpoint stores its own output layer and nothing that can
produce the activations that layer expects -- so its recorded accuracy_fp32 is
not reproducible from the artifact, which defeats the purpose of freezing one.
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
    def W_hidden(self) -> np.ndarray | None:
        """The layer that produces this head's input, or None if not stored."""
        if "W_hidden_real" not in self:
            return None
        return self["W_hidden_real"] + 1j * self["W_hidden_imag"]

    @property
    def n_classes(self) -> int:
        return int(self["W_real"].shape[0])

    @property
    def d1(self) -> int:
        return int(self["W_real"].shape[1])


def save_checkpoint(path: str | pathlib.Path, W: np.ndarray,
                    lift: LiftParams | None = None,
                    W_hidden: np.ndarray | None = None,
                    **meta: Any) -> pathlib.Path:
    """
    Freeze a trained head. `meta` should always record at minimum:
      backbone, dataset, accuracy_fp32, and how the model was trained.

    `W_hidden` is the preceding layer, when there is one whose weights would
    otherwise be lost. Pass it for MLMVN, whose head is useless without it.

    Note that every `meta` key is a keyword argument, so a key named after one
    of these parameters collides with it and raises TypeError. That is
    deliberate -- silently dropping metadata would be worse -- and is why the
    hidden-unit COUNT travels as meta["hidden"] while the hidden WEIGHTS travel
    as this parameter.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W = np.asarray(W, dtype=np.complex128)

    payload: dict[str, Any] = {
        "W_real": np.ascontiguousarray(W.real),
        "W_imag": np.ascontiguousarray(W.imag),
    }
    if W_hidden is not None:
        Wh = np.asarray(W_hidden, dtype=np.complex128)
        payload["W_hidden_real"] = np.ascontiguousarray(Wh.real)
        payload["W_hidden_imag"] = np.ascontiguousarray(Wh.imag)
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
