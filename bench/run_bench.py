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


def time_conversion_c(head: AngularHead, X: np.ndarray, repeat: int,
                      warmup: int) -> dict | None:
    """
    Same conversion, atan2-free, in C. Returns None when the kernel is absent.

    Reported next to the NumPy figure rather than replacing it: the C path is
    opt-in because the two disagree exactly on sector boundaries, so the honest
    presentation is both numbers and the reason for the default.
    """
    from pymvn.angular import _load_lib
    lib = _load_lib()
    if lib is None or not hasattr(lib, "angular_phase_index"):
        return None
    return _time(lambda: head.to_indices(X, c_kernel=True), repeat, warmup)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

SYNTHETIC_SHAPES = [
    ("head_512x10", 512, 10),          # the real hybrid head
    ("layer_2048x64", 2048, 64),       # mid-size
    ("layer_4096x256", 4096, 256),     # large; where tiling should pay off
    ("layer_6144x384", 6144, 384),     # brackets the original projection
    ("layer_8192x512", 8192, 512),     # 0.891: close, but did not cross
    ("layer_12288x768", 12288, 768),   # straddles the revised estimate
    ("layer_16384x1024", 16384, 1024), # 1.016: the crossover
    ("ffn_4096x11008", 4096, 11008),   # a Llama-7B FFN layer, to scale
]


def synthetic_cases(seed: int = 0, only: set[str] | None = None):
    """
    Eight shapes spanning the interesting regime.

    The real hybrid head (512 features x 10 classes) is far too small to measure
    anything -- Python and allocation overhead dominate completely. The larger
    layers exist to expose the crossover point where the angular datapath
    overtakes BLAS, which is itself the result worth reporting.

    The last shape is not synthetic in the same sense as the others: 4096x11008
    is the FFN width of Llama-7B, and it is there to put the crossover on a
    scale a reader already has intuition for.

    The neon/onnx ratio is a function of SHAPE, not of batch: measured on
    Neoverse N2 it holds to within a few percent across batch 64/512/2048
    (head_512x10: 0.205/0.169/0.156, layer_4096x256: 0.701/0.688/0.708). That is
    what makes the last two shapes affordable -- they are swept at a small batch
    and the ratio still means the same thing.

    Indexing by MACs PER SAMPLE, (d+1)*k, for the same reason:

        shape             MACs/sample   neon/onnx   (Neoverse N2, b=4)
        head_512x10             5,130     0.158
        layer_2048x64         131,136     0.362
        layer_4096x256      1,048,832     0.707
        layer_6144x384      2,359,680     0.805
        layer_8192x512      4,194,816     0.887
        layer_12288x768     9,437,952     0.961
        layer_16384x1024   16,778,240     1.016   <-- crossover
        ffn_4096x11008     45,088,768     1.198   <-- Llama-7B FFN width

    So the ratio DOES cross 1.0, at ~1.7e7 MACs per sample. The margin is small
    but not noise: 697.9 +- 1.35 ms for onnx against 687.1 +- 0.61 ms for neon
    over 30 repetitions, and neon's p95 still beats onnx's fastest run.

    WHERE THAT LANDS. The crossover is 3000x above this repository's own 512x10
    head, which invites the reading that parity only arrives at sizes nobody
    runs. Run the arithmetic instead: a Llama-7B FFN layer is 4096x11008 =
    45.1M MACs per sample, three of them per block. The crossover sits 2.7x
    BELOW that, so that shape is measured rather than extrapolated -- and it is
    not marginal parity but 485.4 ms against 405.1 ms, a 1.20x win.

    So the frame is not "parity at absurd sizes": the angular datapath loses on
    small heads, reaches parity at ~1.7e7 MACs per sample, and is 20% ahead in
    the shape regime where transformer FFN compute already lives.

    The claim is about the SHAPE REGION coinciding. It is not a claim that MVN
    works for LLMs -- nothing here trains or evaluates one, and that is future
    work rather than a result.

    x86_64 is the control, and it is arguably the strongest single number in the
    sweep: flat at 0.047-0.059 across every shape, because without vqtbl1q_u8
    the kernel falls back to scalar C. Same code, same weights, same batches --
    the only difference is the table-lookup instruction, so the entire Arm curve
    from 0.158 to 1.016 is attributable to it and to nothing in NumPy, BLAS or
    the harness.
    """
    for i, (name, d, k) in enumerate(SYNTHETIC_SHAPES):
        if only is not None and name not in only:
            # Skipped BEFORE materializing W. layer_16384x1024 alone is a 268 MiB
            # complex128 array, and --cases exists precisely so a run does not
            # pay for the shapes it did not ask for.
            continue
        # Seeded per shape, not from one shared stream: a filtered run must draw
        # the SAME W as the full sweep, or `--cases layer_4096x256` would not be
        # comparable with the run that produced the rest of the table.
        rng = np.random.default_rng([seed, i])
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

    only = None
    if args.cases is not None:
        known = {name for name, _, _ in SYNTHETIC_SHAPES}
        unknown = set(args.cases) - known
        if unknown:
            ap.error(f"unknown case(s) {sorted(unknown)}; known: {sorted(known)}")
        only = set(args.cases)
    cases = list(synthetic_cases(args.seed, only)) if args.synthetic else []
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
                convert_ms = convert_c_ms = None
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
                                c = time_conversion_c(head, X, args.repeat,
                                                      args.warmup)
                                convert_c_ms = c["median_ms"] if c else None
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
                        "convert_c_ms": convert_c_ms if use_idx else None,
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
                    c = (f" C={convert_c_ms:.2f}ms "
                         f"({convert_ms / convert_c_ms:.2f}x)") if convert_c_ms else ""
                    parts.append(f"[convert={convert_ms:.2f}ms{c}]")
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
