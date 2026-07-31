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


def table_speedup(recs) -> str:
    """Median latency per backend, with speedup relative to the ONNX baseline."""
    rows = defaultdict(dict)
    for r in recs:
        key = (r["arch"], r["model"], r["b"], r["batch"])
        rows[key][r["backend"]] = r["median_ms"]

    backends = sorted({r["backend"] for r in recs})
    out = ["| arch | model | b | batch | " + " | ".join(backends) + " | best vs onnx |",
           "|---|---|---|---|" + "---|" * (len(backends) + 1)]
    for key in sorted(rows):
        row = rows[key]
        base = row.get("onnx")
        cells = [f"{row[b]:.2f}" if b in row else "-" for b in backends]
        best = min((v for k, v in row.items() if k != "onnx"), default=None)
        sp = f"{base / best:.2f}x" if base and best else "-"
        out.append("| " + " | ".join(str(k) for k in key) + " | " +
                   " | ".join(cells) + f" | {sp} |")
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
