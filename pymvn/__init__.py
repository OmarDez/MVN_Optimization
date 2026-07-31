"""
pymvn-arm -- phase-native inference for Multi-Valued Neuron models on Arm64.

Quick start
-----------
    from pymvn import AngularHead, load_checkpoint, lift

    ckpt = load_checkpoint("models/resnet18_mnist.npz")
    X    = lift(features, ckpt.lift)

    ref  = AngularHead(ckpt.W, b=4, backend="onnx")
    fast = AngularHead(ckpt.W, b=4, backend="neon")

    assert (ref.predict(X) == fast.predict(X)).mean() > 0.999
"""

from .algebra import (  # noqa: F401
    angular_distance, cos_lut, cos_lut_i8, group_div, group_inv, group_mul,
    index_to_phase, index_to_unit, n_roots, phase_to_index, quantize_unit,
    sin_lut, sin_lut_i8,
)
from .angular import AngularHead, BACKENDS, neon_available  # noqa: F401
from .core import MLMVN, MVNHead, make_targets  # noqa: F401
from .encode import LiftParams, encode_fft, encode_pixels, fit_lift, lift  # noqa: F401
from .io import Checkpoint, load_checkpoint, save_checkpoint  # noqa: F401

__version__ = "0.1.0"
