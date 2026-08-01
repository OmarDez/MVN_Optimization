"""
AngularHead: a single MVN classification layer with interchangeable backends.

Every backend computes the same mathematical object

    z_j = sum_{i=0}^{d} w_{j,i} * x_i          (index 0 is the bias term)
    prediction = argmax_j Re(z_j)

but differs in how it represents and executes the product. Because they share
one interface, the parity test and the benchmark harness are both trivial: swap
a string, compare the outputs.

    BACKEND          REPRESENTATION                          ROLE
    ---------------  --------------------------------------  --------------------
    complex128       full complex arithmetic                 ground truth
    complex64        full complex, single precision          cheap reference
    onnx             real/imag split, 4 real GEMMs           HONEST BASELINE
    angular_naive    integer indices, float LUT              proof of concept
    angular_tiled    uint8 indices, int8 LUT, cache tiling   portable optimum
    neon             uint8 + vqtbl1q_u8 via C                Arm-specific optimum

The number to report is angular_tiled and neon MEASURED AGAINST `onnx`, not
against complex128. `onnx` is what any competent engineer would deploy today;
beating a deliberately slow reference would prove nothing.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from typing import Optional

import numpy as np

from . import algebra as alg

__all__ = ["AngularHead", "BACKENDS", "neon_available"]

BACKENDS = (
    "complex128",
    "complex64",
    "onnx",
    "angular_naive",
    "angular_tiled",
    "neon",
)

_LIB_PATH = pathlib.Path(__file__).resolve().parent.parent / "kernels" / "libangular.so"
_lib = None


def _load_lib():
    """Load the compiled C kernel, or return None if it was never built."""
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        return None
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.angular_gemm.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # kx  (n, d)
        ctypes.POINTER(ctypes.c_uint8),   # kw  (k, d)
        ctypes.POINTER(ctypes.c_int8),    # lut_cos (L)
        ctypes.POINTER(ctypes.c_int8),    # lut_sin (L)
        ctypes.POINTER(ctypes.c_int32),   # acc_re (n, k)
        ctypes.POINTER(ctypes.c_int32),   # acc_im (n, k)
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, k, d, L
    ]
    lib.angular_gemm.restype = None
    lib.angular_has_neon.argtypes = []
    lib.angular_has_neon.restype = ctypes.c_int
    if hasattr(lib, "angular_gemm_packed"):
        lib.angular_gemm_packed.argtypes = lib.angular_gemm.argtypes
        lib.angular_gemm_packed.restype = None
    _lib = lib
    return _lib


def neon_available() -> bool:
    """True when the C kernel is built AND compiled with the NEON path active."""
    lib = _load_lib()
    return bool(lib) and bool(lib.angular_has_neon())


class AngularHead:
    """
    Args:
        W: (k, d+1) complex weights. Column 0 is the bias.
        b: phase bit-width. L = 2**b.
        backend: one of BACKENDS.
        phase_only: if True, weights are projected onto mu_L (|w| = 1), which
            is what makes the group isomorphism -- and therefore the
            multiplier-free datapath -- valid. If False, the modulus is kept
            and only the angle is quantized ("full-polar"): more accurate, but
            a real multiply survives per MAC.
        tile: (tile_n, tile_d) for the angular_tiled backend.
        packed: store weights as two 4-bit indices per byte (b <= 4 only) and
            unpack them inside the kernel. Halves the weight footprint; the
            throughput cost is what the benchmark exists to measure. `neon` only.
    """

    def __init__(self, W: np.ndarray, b: int = 4, backend: str = "complex128",
                 phase_only: bool = True, tile: tuple[int, int] = (64, 256),
                 packed: bool = False):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.W_fp = np.asarray(W, dtype=np.complex128)
        self.n_classes, self.d1 = self.W_fp.shape
        self.b = int(b)
        self.L = alg.n_roots(self.b)
        self.backend = backend
        self.phase_only = bool(phase_only)
        self.tile = tile

        # Quantized weights, in both representations.
        self.W_q = alg.quantize_unit(self.W_fp, self.b, self.phase_only)
        self.kW = np.ascontiguousarray(
            alg.phase_to_index(np.angle(self.W_fp), self.b), dtype=np.uint8
        )

        if packed:
            if backend != "neon":
                raise ValueError("packed=True is only implemented for backend='neon'")
            if self.b > 4:
                raise ValueError(f"packed=True needs b <= 4, got b={self.b}")
            if self.d1 % 2:
                raise ValueError(
                    f"packed=True needs an even feature count; this head has "
                    f"d1={self.d1}. Pad W with a zero column."
                )
        self.packed = bool(packed)
        self.kW_packed = alg.pack_indices(self.kW, self.b) if packed else None

        self._ort_sess = None  # built lazily

    # -- helpers ------------------------------------------------------------

    def _with_bias(self, X: np.ndarray) -> np.ndarray:
        """Prepend the constant-1 input that the bias weight multiplies."""
        X = np.asarray(X)
        if X.shape[1] == self.d1:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)

    def _idx_with_bias(self, kX: np.ndarray) -> np.ndarray:
        """
        Same for index inputs. The bias input is the constant 1 = e^{i*0}, whose
        phase index is 0 -- so the padding column is zeros, not ones.
        """
        kX = np.ascontiguousarray(np.asarray(kX, dtype=np.uint8))
        if kX.shape[1] == self.d1:
            return kX
        if kX.shape[1] != self.d1 - 1:
            raise ValueError(
                f"expected indices of shape (n, {self.d1}) or (n, {self.d1 - 1}), "
                f"got {kX.shape}"
            )
        zeros = np.zeros((kX.shape[0], 1), dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate([zeros, kX], axis=1))

    def to_indices(self, X: np.ndarray) -> np.ndarray:
        """
        Complex activations on S^1 -> uint8 phase indices, shape (n, d1),
        bias column included.

        A genuinely phase-native pipeline never calls this: activations would
        already BE indices, arriving from the previous angular layer or straight
        from the FFT. It is public so the benchmark can time the conversion
        separately instead of burying it inside the kernel measurement -- see
        `logits(..., indices=True)`.
        """
        Xb = self._with_bias(np.asarray(X, dtype=np.complex128))
        return np.ascontiguousarray(alg.phase_to_index(np.angle(Xb), self.b))

    # -- backends -----------------------------------------------------------

    def _z_complex(self, X, dtype) -> np.ndarray:
        Xb = self._with_bias(np.asarray(X, dtype=dtype))
        W = self.W_q.astype(dtype)
        return Xb @ W.T

    def _z_onnx(self, X) -> np.ndarray:
        """
        The honest baseline: complex arithmetic expressed as four real GEMMs,
        which is exactly how a complex-valued layer must be deployed today
        because ONNX Runtime, ExecuTorch and LiteRT have no complex64 support.

            (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        """
        import onnxruntime as ort  # local import: optional dependency

        Xb = self._with_bias(np.asarray(X, dtype=np.complex128))
        Xr = np.ascontiguousarray(Xb.real, dtype=np.float32)
        Xi = np.ascontiguousarray(Xb.imag, dtype=np.float32)

        if self._ort_sess is None:
            self._ort_sess = self._build_ort_session()

        out = self._ort_sess.run(None, {"Xr": Xr, "Xi": Xi})
        return out[0].astype(np.float64) + 1j * out[1].astype(np.float64)

    def _build_ort_session(self):
        import onnx
        from onnx import TensorProto, helper
        import onnxruntime as ort

        d1, k = self.d1, self.n_classes
        Wr = np.ascontiguousarray(self.W_q.real.T, dtype=np.float32)  # (d1, k)
        Wi = np.ascontiguousarray(self.W_q.imag.T, dtype=np.float32)

        init = [
            helper.make_tensor("Wr", TensorProto.FLOAT, [d1, k], Wr.ravel()),
            helper.make_tensor("Wi", TensorProto.FLOAT, [d1, k], Wi.ravel()),
        ]
        nodes = [
            helper.make_node("MatMul", ["Xr", "Wr"], ["ac"]),
            helper.make_node("MatMul", ["Xi", "Wi"], ["bd"]),
            helper.make_node("MatMul", ["Xr", "Wi"], ["ad"]),
            helper.make_node("MatMul", ["Xi", "Wr"], ["bc"]),
            helper.make_node("Sub", ["ac", "bd"], ["Zr"]),
            helper.make_node("Add", ["ad", "bc"], ["Zi"]),
        ]
        graph = helper.make_graph(
            nodes, "complex_gemm",
            [helper.make_tensor_value_info("Xr", TensorProto.FLOAT, [None, d1]),
             helper.make_tensor_value_info("Xi", TensorProto.FLOAT, [None, d1])],
            [helper.make_tensor_value_info("Zr", TensorProto.FLOAT, [None, k]),
             helper.make_tensor_value_info("Zi", TensorProto.FLOAT, [None, k])],
            initializer=init,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("PYMVN_THREADS", "1"))
        return ort.InferenceSession(model.SerializeToString(), so,
                                    providers=["CPUExecutionProvider"])

    def _z_angular_naive(self, kX: np.ndarray) -> np.ndarray:
        """
        Direct transcription of the isomorphism. Materializes the full
        (n, k, d1) index tensor, which is why it is slow and memory-hungry --
        it exists to prove correctness, not speed.
        """
        kX = kX.astype(np.int32)
        kW = self.kW.astype(np.int32)

        kSum = (kX[:, None, :] + kW[None, :, :]) & (self.L - 1)
        c, s = alg.cos_lut(self.b), alg.sin_lut(self.b)
        return c[kSum].sum(axis=2) + 1j * s[kSum].sum(axis=2)

    def _z_angular_tiled(self, kX: np.ndarray) -> np.ndarray:
        """
        Cache-blocked integer datapath.

        Three optimizations over the naive form, all of them measurable:
          1. uint8 indices instead of int64  -> 8x less index traffic
          2. int8 LUT + int32 accumulation   -> integer-only inner loop
          3. tiling over (batch, features)   -> working set held in L1D

        Accumulator range check: with an int8 LUT scaled by 127 and d1 terms,
        |acc| <= 127 * d1. For d1 = 4097 that is ~520k, comfortably inside
        int32 and far outside int16 -- hence int32.
        """
        kW = self.kW
        n, d1 = kX.shape
        k = self.n_classes
        mask = np.uint8(self.L - 1)

        c8 = alg.cos_lut_i8(self.b)
        s8 = alg.sin_lut_i8(self.b)

        acc_re = np.zeros((n, k), dtype=np.int32)
        acc_im = np.zeros((n, k), dtype=np.int32)
        tn, td = self.tile

        for i0 in range(0, n, tn):
            i1 = min(i0 + tn, n)
            for j0 in range(0, d1, td):
                j1 = min(j0 + td, d1)
                # (tn, k, td) index block, uint8 throughout.
                blk = (kX[i0:i1, None, j0:j1] + kW[None, :, j0:j1]) & mask
                acc_re[i0:i1] += c8[blk].sum(axis=2, dtype=np.int32)
                acc_im[i0:i1] += s8[blk].sum(axis=2, dtype=np.int32)

        scale = 1.0 / alg.LUT_SCALE_I8
        return acc_re * scale + 1j * (acc_im * scale)

    def _z_neon(self, kX: np.ndarray) -> np.ndarray:
        """Hand-written C kernel using the AArch64 TBL instruction."""
        lib = _load_lib()
        if lib is None:
            raise RuntimeError(
                "kernels/libangular.so not found. Run: bash kernels/build.sh"
            )

        n, d1 = kX.shape
        k = self.n_classes

        c8 = np.ascontiguousarray(alg.cos_lut_i8(self.b))
        s8 = np.ascontiguousarray(alg.sin_lut_i8(self.b))
        acc_re = np.zeros((n, k), dtype=np.int32)
        acc_im = np.zeros((n, k), dtype=np.int32)

        p8 = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        pi8 = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        p32 = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        if self.packed:
            if not hasattr(lib, "angular_gemm_packed"):
                raise RuntimeError(
                    "libangular.so predates packed weights. Run: bash kernels/build.sh"
                )
            fn, kw = lib.angular_gemm_packed, self.kW_packed
        else:
            fn, kw = lib.angular_gemm, self.kW
        fn(p8(kX), p8(np.ascontiguousarray(kw)),
           pi8(c8), pi8(s8), p32(acc_re), p32(acc_im),
           n, k, d1, self.L)

        scale = 1.0 / alg.LUT_SCALE_I8
        return acc_re * scale + 1j * (acc_im * scale)

    # -- public API ---------------------------------------------------------

    def logits(self, X: np.ndarray, indices: bool = False) -> np.ndarray:
        """
        Return the complex pre-activation z, shape (n, n_classes).

        Args:
            X: (n, d) or (n, d+1) complex activations on S^1. With
                `indices=True`, uint8 phase indices of the same shape instead --
                already on Z_L, so the angular backends skip the np.angle
                conversion entirely. That is the phase-native case: indices
                arrive from the previous angular layer or from the FFT and never
                leave the integer domain.

        The complex backends cannot consume indices natively -- they need
        Cartesian values -- so with `indices=True` they reconstruct via the
        lookup. That reconstruction feeds them b-bit activations, so they return
        a DIFFERENT (quantized) quantity than in complex mode, not merely a
        differently-timed one. The angular backends return bit-identical results
        in both modes, which is what makes the flag a clean measurement of the
        conversion cost.
        """
        angular = self.backend in ("angular_naive", "angular_tiled", "neon")

        if angular:
            kX = self._idx_with_bias(X) if indices else self.to_indices(X)
            if self.backend == "angular_naive":
                return self._z_angular_naive(kX)
            if self.backend == "angular_tiled":
                return self._z_angular_tiled(kX)
            return self._z_neon(kX)

        if indices:
            X = alg.index_to_unit(self._idx_with_bias(X), self.b)
        if self.backend == "complex128":
            return self._z_complex(X, np.complex128)
        if self.backend == "complex64":
            return self._z_complex(X, np.complex64).astype(np.complex128)
        if self.backend == "onnx":
            return self._z_onnx(X)
        raise AssertionError("unreachable")

    def predict(self, X: np.ndarray, indices: bool = False) -> np.ndarray:
        """Class prediction: argmax over Re(z), the one-vs-all MVN rule."""
        return np.real(self.logits(X, indices=indices)).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray, indices: bool = False) -> float:
        return float((self.predict(X, indices=indices) == np.asarray(y)).mean())

    def mac_report(self) -> dict:
        """
        Per-sample MAC accounting for the head. Used to substantiate the
        "multiplier-free" claim with a number rather than an adjective.
        """
        macs = self.n_classes * self.d1
        multiplier_free = self.backend.startswith("angular") or self.backend == "neon"
        if multiplier_free and not self.phase_only:
            multiplier_free = False  # |w| != 1 reintroduces a real multiply
        n_w = self.W_fp.size
        weight_bytes = self.kW_packed.nbytes if self.packed else self.kW.nbytes
        return {
            "head_macs": macs,
            "real_mults_per_mac": 0 if multiplier_free else 4,
            "int_adds_per_mac": 1 if multiplier_free else 0,
            "lut_bytes": self.L,
            "multiplier_free": multiplier_free,
            "packed": self.packed,
            # Byte accounting for the compression claim. complex128 is the
            # unquantized form; fp32 real/imag is what actually ships today.
            "weight_bytes": int(weight_bytes),
            "weight_bytes_complex128": int(n_w * 16),
            "weight_bytes_fp32_reim": int(n_w * 8),
            "compression_vs_fp32_reim": (n_w * 8) / max(weight_bytes, 1),
        }
