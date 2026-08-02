#!/usr/bin/env python3
"""
Draw the E3 bit-width ablation (PLAN.md 5, "the most important figure").

Reads the JSON that scripts/ablate_bits.py writes and produces
docs/fig_bitwidth_ablation.png.

The figure has to carry two messages at once, so it has two panels:

  left   accuracy against b, for both phase-only and full-polar. The reader
         should be able to see where the curve flattens (b = 4) and, separately,
         the constant vertical gap between the two modes -- that gap is the
         price of a multiplier-free datapath, and it does not shrink with b.

  right  the LUT size against b on a log axis, with the 16-byte NEON register
         marked. This is the hardware half of the claim: the b where accuracy
         saturates is the largest b whose table still fits one vqtbl1q_u8
         operand. Putting it beside the accuracy panel is the whole argument.

    python bench/plot_ablation.py results/e3-mlmvn-mnist.json --out docs/
"""
from __future__ import annotations

import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")          # headless: this runs in CI
import matplotlib.pyplot as plt

NEON_REGISTER_BYTES = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--name", default="fig_bitwidth_ablation.png")
    a = ap.parse_args()

    rows = []
    for p in a.json:
        rows += json.loads(pathlib.Path(p).read_text())
    rows.sort(key=lambda r: r["b"])
    if not rows:
        raise SystemExit("no records")

    bs = [r["b"] for r in rows]
    po = [r["accuracy"] for r in rows]
    fp = [r["accuracy_full_polar"] for r in rows]
    lut = [r["lut_bytes"] for r in rows]
    model = pathlib.Path(rows[0]["model"]).stem
    n = rows[0]["n_test"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.plot(bs, fp, "o-", color="#1f77b4", label="full-polar (keeps |w|)")
    ax.plot(bs, po, "s-", color="#d62728", label="phase-only (multiplier-free)")
    ax.fill_between(bs, po, fp, color="#d62728", alpha=0.10)

    # Annotate the gap where it is flat, not at its widest -- the honest
    # summary is the asymptote, not the b=2 outlier.
    i4 = bs.index(4) if 4 in bs else len(bs) // 2
    ax.annotate(f"cost of discarding |w|\n{fp[i4] - po[i4]:+.4f}",
                xy=(bs[i4], (fp[i4] + po[i4]) / 2),
                xytext=(bs[i4] + 1.4, (fp[i4] + po[i4]) / 2 - 0.055),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    ax.axvline(4, color="gray", ls=":", lw=1)
    ax.text(4.06, min(po) + 0.01, "b = 4", fontsize=8, color="gray")
    ax.set_xlabel("phase bits b")
    ax.set_ylabel(f"accuracy ({n:,} test samples)")
    ax.set_title(f"E3: phase resolution vs accuracy\n{model}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    bx.semilogy(bs, lut, "o-", color="#2ca02c")
    bx.axhline(NEON_REGISTER_BYTES, color="#ff7f0e", ls="--", lw=1.2)
    bx.text(bs[0] + 0.1, NEON_REGISTER_BYTES * 1.25,
            "16 B = one NEON register = one vqtbl1q_u8",
            fontsize=8, color="#ff7f0e")
    bx.axvspan(bs[0] - 0.3, 4, color="#2ca02c", alpha=0.08)
    bx.text(2.0, max(lut) * 0.4, "fits the\nlookup datapath",
            fontsize=8, color="#2ca02c", ha="center")
    bx.set_xlabel("phase bits b")
    bx.set_ylabel("cos/sin LUT size (bytes)")
    bx.set_title("The hardware coincidence\naccuracy saturates exactly where the "
                 "table stops fitting", fontsize=10)
    bx.grid(alpha=0.25, which="both")

    fig.tight_layout()
    out = pathlib.Path(a.out) / a.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
