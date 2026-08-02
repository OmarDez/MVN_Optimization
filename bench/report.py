#!/usr/bin/env python3
"""
Turn benchmark JSON into the tables and figures that go in the write-up.

    python bench/report.py results/*.json --out docs/
"""
from __future__ import annotations

import argparse, json, pathlib, sys
from collections import defaultdict


def load(paths):
    recs = []
    for p in paths:
        recs += json.loads(pathlib.Path(p).read_text())
    return recs


#: Backends that are candidate deployment targets. complex128/complex64 are
#: reference implementations, not things anyone ships -- ranking against them
#: would manufacture a speedup out of NumPy's convenience, which is exactly the
#: self-deception PLAN.md section 6 warns about.
ANGULAR = ("angular_naive", "angular_tiled", "neon")


def table_speedup(recs) -> str:
    """Median latency per backend, with the angular path ranked against ONNX."""
    rows = defaultdict(dict)
    for r in recs:
        key = (r["arch"], r["model"], r["b"], r["batch"])
        rows[key][r["backend"]] = r["median_ms"]

    backends = sorted({r["backend"] for r in recs})
    out = ["| arch | model | b | batch | " + " | ".join(backends)
           + " | best angular vs onnx |",
           "|---|---|---|---|" + "---|" * (len(backends) + 1)]
    for key in sorted(rows):
        row = rows[key]
        base = row.get("onnx")
        cells = [f"{row[b]:.2f}" if b in row else "-" for b in backends]
        best = min((v for k, v in row.items() if k in ANGULAR), default=None)
        sp = f"{base / best:.3f}x" if base and best else "-"
        out.append("| " + " | ".join(str(k) for k in key) + " | " +
                   " | ".join(cells) + f" | {sp} |")
    return "\n".join(out)


def table_crossover(recs) -> str:
    """
    The neon/onnx ratio against layer size. This is the curve that decides
    whether multiplier-free inference ever overtakes a tuned BLAS GEMM, so it
    gets its own table rather than being buried in the latency dump.
    """
    # Complex input only. Mixing in the pre-indexed rows would put two different
    # questions in one table -- and, since the mode is not a column, would look
    # like duplicate rows disagreeing with each other.
    rows = defaultdict(dict)
    for r in recs:
        if r.get("input_mode", "complex") != "complex":
            continue
        key = (r["arch"], r["model"], r["b"], r["batch"])
        rows[key][r["backend"]] = r
    def layer_macs(row):
        """Per-sample MACs. This -- not the batch -- is what moves the ratio."""
        r = row.get("neon") or row.get("onnx")
        return r["n_classes"] * (r["d"] + 1)

    # Weights travel in the same row as latency on purpose. At parity the
    # interesting statement is not "as fast as BLAS" -- that alone is easy to
    # wave away -- but "as fast as BLAS, on weights 16x smaller". Splitting the
    # two into separate tables lets a reader take either one in isolation, which
    # is exactly the reading to avoid.
    out = ["| arch | model | layer MACs | b | batch | onnx ms | neon ms "
           "| neon/onnx | fp32 re/im | uint8 idx | packed b-bit | smaller by |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(rows, key=lambda k: (k[0], layer_macs(rows[k]), k[3])):
        row = rows[key]
        if "onnx" not in row or "neon" not in row:
            continue
        ratio = row["onnx"]["median_ms"] / row["neon"]["median_ms"]
        arch, model, b, batch = key
        n = layer_macs(row)
        mib = lambda bits: n * bits / 8 / 1024 ** 2
        out.append(f"| {arch} | {model} | {n / 1e6:.3f}M "
                   f"| {b} | {batch} | "
                   f"{row['onnx']['median_ms']:.2f} | {row['neon']['median_ms']:.2f} | "
                   f"**{ratio:.3f}** | {mib(64):.1f} MiB | {mib(8):.1f} MiB | "
                   f"{mib(b):.1f} MiB | **{64 / b:.0f}x** |")
    return "\n".join(out)


def table_preindexed(recs) -> str:
    """
    What the complex->index conversion costs, and what removing it buys.

    A phase-native front end (an FFT encoder, or a preceding angular layer)
    hands the head indices already. Timing the conversion inside the kernel
    measurement charges the angular path for work that would never happen in
    the deployment it is being proposed for.

    The overhead scales as ~1/k -- conversion is O(n*d), the layer is O(n*k*d)
    -- so it hurts most on the small-k shapes, which is exactly the regime this
    repository actually deploys.
    """
    # Everything below comes from ONE process: the indexed timing, the
    # conversion timing and the onnx baseline are all measured in the same
    # invocation, and the complex-mode cost is their sum rather than a number
    # joined in from a different run. Joining across runs looked fine and was
    # not -- on a loaded box the two processes disagreed by more than the
    # quantity being measured.
    idx = [r for r in recs if r.get("input_mode") == "indices"
           and r["backend"] in ANGULAR and r.get("convert_ms")]
    onnx = {(r["arch"], r["model"], r["b"], r["batch"]): r for r in recs
            if r["backend"] == "onnx"}

    out = ["| arch | model | k | b | batch | backend | kernel ms | convert ms "
           "| complex-in ms | conversion was | ratio then | ratio now |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(idx, key=lambda r: (r["arch"], r["n_classes"] * (r["d"] + 1),
                                        r["batch"], r["backend"])):
        conv = r["convert_ms"]
        total = r["median_ms"] + conv
        base = onnx.get((r["arch"], r["model"], r["b"], r["batch"]))
        if base is None:
            continue
        then, now = base["median_ms"] / total, base["median_ms"] / r["median_ms"]
        out.append(f"| {r['arch']} | {r['model']} | {r['n_classes']} | {r['b']} "
                   f"| {r['batch']} | {r['backend']} | {r['median_ms']:.2f} "
                   f"| {conv:.2f} | {total:.2f} | {conv / total * 100:.0f}% "
                   f"| {then:.3f} | **{now:.3f}** |")
    return "\n".join(out)


def table_memory(recs) -> str:
    """Weight footprint per representation. The claim that does not depend on
    beating a tuned GEMM."""
    seen = {}
    for r in recs:
        seen[(r["model"], r["d"], r["n_classes"], r["b"])] = r
    out = ["| model | weights | complex128 | fp32 re/im | uint8 idx | packed b-bit | vs fp32 |",
           "|---|---|---|---|---|---|---|"]
    for (model, d, k, b), r in sorted(seen.items()):
        n = k * (d + 1)
        kb = lambda bits: n * bits / 8 / 1024
        out.append(f"| {model} (b={b}) | {n:,} | {kb(128):.0f} KiB | {kb(64):.0f} KiB | "
                   f"{kb(8):.0f} KiB | {kb(b):.0f} KiB | **{64 / b:.0f}x** |")
    return "\n".join(out)


def table_accuracy(recs) -> str:
    out = ["| model | b | backend | agreement vs fp | multiplier-free |", "|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: (x["model"], x["b"], x["backend"])):
        if r["batch"] != max(x["batch"] for x in recs):
            continue
        out.append(f"| {r['model']} | {r['b']} | {r['backend']} | "
                   f"{r['agreement_vs_fp']:.4f} | {r['multiplier_free']} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()

    recs = load(a.inputs)
    if not recs:
        print("no records", file=sys.stderr)
        return 1

    md = ["# Benchmark results\n",
          f"_{len(recs)} records; hosts: "
          + ", ".join(sorted({r.get('cpu', r['arch']) for r in recs})) + "_\n",
          "## Crossover: neon vs ONNX against layer size\n", table_crossover(recs), "",
          "## Phase-native input: what the conversion was costing\n",
          table_preindexed(recs), "",
          "## Latency\n", table_speedup(recs), "",
          "## Weight memory\n", table_memory(recs), "",
          "## Accuracy\n", table_accuracy(recs), ""]

    out = pathlib.Path(a.out) / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
