#!/usr/bin/env python3
"""
Benchmark harness.

Measures every backend on the same inputs and emits one JSON record per
(model, backend, b, batch) cell, following the results contract in PLAN.md.

Design decisions that matter for credibility:

  * MEDIAN AND p95, never a single run. CI runners are shared machines; a lone
    timing is noise. Default is 30 repetitions after 5 warm-up iterations.

  * SPEEDUP IS REPORTED AGAINST `onnx`, not against complex128. The ONNX
    four-GEMM path is what anyone would actually deploy today, because no
    mainstream edge runtime supports complex64. Beating a deliberately slow
    reference would prove nothing.

  * SINGLE THREAD BY DEFAULT. Thread-count differences between backends would
    otherwise dominate and obscure the arithmetic effect being measured.

  * ACCURACY IS RECORDED ALONGSIDE LATENCY. A fast kernel that changed the
    predictions is a regression, not an optimization.

Usage
-----
    # synthetic sweep, no checkpoint required
    python bench/run_bench.py --synthetic --out results/arm64.json

    # against frozen checkpoints
    python bench/run_bench.py --models models/*.npz --out results/arm64.json

    # quick smoke test
    python bench/run_bench.py --synthetic --repeat 3 --batches 64
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pymvn import AngularHead, load_checkpoint, neon_available  # noqa: E402

DEFAULT_BACKENDS = ["complex128", "complex64", "onnx", "angular_tiled"]


# ---------------------------------------------------------------------------
# Host description -- goes into every record so results are self-identifying
# ---------------------------------------------------------------------------

def host_info() -> dict:
    info = {
        "arch": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "neon_kernel": neon_available(),
        "threads": int(os.environ.get("PYMVN_THREADS", "1")),
    }
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu"] = line.split(":", 1)[1].strip()
            elif line.startswith("Flags:") or line.startswith("Features:"):
                feats = line.split(":", 1)[1].split()
                # Record only the features relevant to this kernel.
                info["features"] = sorted(
                    f for f in feats
                    if f in {"asimd", "neon", "sve", "sve2", "i8mm", "bf16", "dotprod"}
                )
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _stats(samples: list[float]) -> dict:
    s = np.asarray(samples)
    return {
        "median_ms": float(np.median(s)),
        "mean_ms": float(s.mean()),
        "p95_ms": float(np.percentile(s, 95)),
        "min_ms": float(s.min()),
        "std_ms": float(s.std()),
        "repeats": len(s),
    }


def _time(fn, repeat: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return _stats(samples)


def time_backend(head: AngularHead, X: np.ndarray, repeat: int, warmup: int,
                 indices: bool = False) -> dict:
    """Return latency statistics in milliseconds for one full forward pass."""
    return _time(lambda: head.logits(X, indices=indices), repeat, warmup)


def time_conversion(head: AngularHead, X: np.ndarray, repeat: int,
                    warmup: int) -> dict:
    """
    Cost of complex activations -> phase indices (np.angle plus rounding).

    This is float work that a genuinely phase-native pipeline never performs:
    activations would already be indices, arriving from the previous angular
    layer or straight from the FFT. Measuring it separately keeps it out of the
    kernel number instead of silently inflating it.
    """
    return _time(lambda: head.to_indices(X), repeat, warmup)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

def synthetic_cases(seed: int = 0):
    """
    Five shapes spanning the interesting regime.

    The real hybrid head (512 features x 10 classes) is far too small to measure
    anything -- Python and allocation overhead dominate completely. The larger
    layers exist to expose the crossover point where the angular datapath
    overtakes BLAS, which is itself the result worth reporting.

    The measured neon/onnx ratio grows monotonically with layer size and a
    log-log fit over the first three points gives ratio ~ MACs^0.277, which
    projects parity at ~7.3G MACs. The last two shapes bracket that prediction:

        shape             MACs @ batch 2048   neon/onnx
        head_512x10                 10.5M     0.163   measured
        layer_2048x64                269M     0.397   measured
        layer_4096x256              2.15G     0.712   measured
        layer_6144x384              4.83G     ~0.90   projected
        layer_8192x512              8.59G     ~1.05   projected

    If the ratio flattens below 1.0 instead, the kernel went memory-bound before
    catching BLAS -- an equally reportable result, since it bounds the ceiling.
    """
    rng = np.random.default_rng(seed)
    for name, d, k in [
        ("head_512x10", 512, 10),        # the real hybrid head
        ("layer_2048x64", 2048, 64),     # mid-size
        ("layer_4096x256", 4096, 256),   # large; where tiling should pay off
        ("layer_6144x384", 6144, 384),   # brackets the projected crossover
        ("layer_8192x512", 8192, 512),   # expected to cross 1.0
    ]:
        W = np.exp(1j * rng.uniform(0, 2 * np.pi, (k, d + 1))) / np.sqrt(d + 1)
        yield name, W, d


def checkpoint_cases(paths):
    for p in paths:
        ck = load_checkpoint(p)
        yield pathlib.Path(p).stem, ck.W, ck.d1 - 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], help="checkpoint .npz files")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic layers")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="restrict --synthetic to these case names. Without it "
                         "every shape runs, and the largest are ruinously slow "
                         "under angular_tiled -- pair them with --backends.")
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=[3, 4])
    ap.add_argument("--batches", nargs="*", type=int, default=[64, 512, 2048])
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tile", nargs=2, type=int, default=[64, 256])
    ap.add_argument("--pre-indexed", action="store_true",
                    help="feed the angular backends uint8 phase indices instead "
                         "of complex activations, and time the conversion "
                         "separately. This is the phase-native regime.")
    ap.add_argument("--out", default="results/bench.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    backends = list(args.backends) if args.backends else list(DEFAULT_BACKENDS)
    if neon_available() and "neon" not in backends:
        backends.append("neon")

    host = host_info()
    print(f"host: {host.get('cpu', host['arch'])}  |  arch={host['arch']}  "
          f"|  neon_kernel={host['neon_kernel']}", flush=True)
    print(f"backends: {', '.join(backends)}\n", flush=True)

    cases = list(synthetic_cases(args.seed)) if args.synthetic else []
    if args.cases is not None:
        known = {name for name, _, _ in cases}
        unknown = set(args.cases) - known
        if unknown:
            ap.error(f"unknown case(s) {sorted(unknown)}; known: {sorted(known)}")
        cases = [c for c in cases if c[0] in set(args.cases)]
    cases += list(checkpoint_cases(args.models))
    if not cases:
        ap.error("nothing to benchmark: pass --synthetic and/or --models")

    rng = np.random.default_rng(args.seed + 1)
    records = []

    for model_name, W, d in cases:
        for batch in args.batches:
            X = np.exp(1j * rng.uniform(0, 2 * np.pi, (batch, d)))
            ref_pred = AngularHead(W, b=8, backend="complex128").predict(X)

            for b in args.bits:
                row = {}
                convert_ms = None
                for backend in backends:
                    head = AngularHead(W, b=b, backend=backend,
                                       tile=tuple(args.tile))

                    # Only the angular backends can consume indices natively.
                    # The complex baselines keep their full-precision input --
                    # feeding them b-bit activations would change what the
                    # baseline IS, not merely how fast it runs.
                    angular = backend in ("angular_naive", "angular_tiled", "neon")
                    use_idx = args.pre_indexed and angular
                    try:
                        if use_idx:
                            if convert_ms is None:
                                convert_ms = time_conversion(
                                    head, X, args.repeat, args.warmup)["median_ms"]
                            Xin = head.to_indices(X)
                        else:
                            Xin = X
                        stats = time_backend(head, Xin, args.repeat, args.warmup,
                                             indices=use_idx)
                    except Exception as exc:
                        print(f"  [skip] {backend}: {exc}", flush=True)
                        continue

                    pred = head.predict(Xin, indices=use_idx)
                    rec = {
                        "model": model_name, "backend": backend, "b": b,
                        "batch": int(batch), "d": int(d),
                        "n_classes": int(W.shape[0]),
                        "macs": int(W.shape[0] * (d + 1) * batch),
                        "agreement_vs_fp": float((pred == ref_pred).mean()),
                        "input_mode": "indices" if use_idx else "complex",
                        "convert_ms": convert_ms if use_idx else None,
                        **head.mac_report(), **stats, **host,
                    }
                    records.append(rec)
                    row[backend] = stats["median_ms"]

                base = row.get("onnx")
                head_line = f"{model_name:16s} b={b} batch={batch:5d}"
                parts = []
                for backend in backends:
                    if backend not in row:
                        continue
                    ms = row[backend]
                    sp = f" ({base / ms:5.2f}x)" if base and backend != "onnx" else ""
                    parts.append(f"{backend}={ms:8.2f}ms{sp}")
                if convert_ms is not None:
                    parts.append(f"[convert={convert_ms:.2f}ms]")
                print(f"  {head_line}  " + "  ".join(parts), flush=True)

                if base:
                    for rec in records[-len(row):]:
                        rec["speedup_vs_onnx"] = base / rec["median_ms"]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
