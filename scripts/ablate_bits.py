#!/usr/bin/env python3
"""
Experiment E3: how many phase bits does the head actually need?

On REAL activations. That distinction is the whole point of this script -- the
parity tests sweep b on random points of S^1, which is the worst case, because
random class logits are nearly tied and a fraction of a sector of phase error
flips the argmax. A trained model concentrates the decision margin, so the
thresholds calibrated on random weights say almost nothing about the deployed
one. PLAN.md E3 asks for the curve on a checkpoint, and this produces it.

Requires a checkpoint carrying the layer that PRODUCES the head's input
(W_hidden_*, see io.py). Without it the head can only be fed noise.

    python scripts/ablate_bits.py --model models/mlmvn_fft_mnist.npz
    python scripts/ablate_bits.py --model models/mlmvn_fft_mnist.npz --out results/e3.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pymvn import AngularHead, encode_fft, load_checkpoint  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REFERENCE_BITS = 8  # "full precision" for agreement purposes


def hidden_activations(ck, X, shape) -> np.ndarray:
    """Reproduce the head's input: FFT phase encoding, then the hidden layer."""
    W1 = ck.W_hidden
    if W1 is None:
        raise SystemExit(
            f"checkpoint has no W_hidden_*; its accuracy is not reproducible "
            f"from the artifact and the head cannot be fed real activations")
    C = encode_fft(X, int(ck.meta["freq_size"]), shape)
    ones = np.ones((C.shape[0], 1), dtype=np.complex128)
    Zh = np.concatenate([ones, C], axis=1) @ W1.T
    return Zh / np.maximum(np.abs(Zh), 1e-12)   # phase-only: back onto S^1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mlmvn_fft_mnist.npz")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--activations", default=None,
                    help="precomputed activation slice (tests/data/*.npz). Skips "
                         "the dataset entirely, so this runs on a CI box with no "
                         "torchvision -- which is how E3 gets measured on Arm.")
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--backend", default="angular_tiled")
    ap.add_argument("--prune-taus", type=float, nargs="+", default=None,
                    help="also sweep magnitude-pruning thresholds at --prune-b")
    ap.add_argument("--prune-b", type=int, default=4)
    ap.add_argument("--prune-out", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ck = load_checkpoint(a.model)
    if a.activations:
        with np.load(a.activations, allow_pickle=False) as z:
            H = np.exp(1j * z["phase"].astype(np.float64))
            yte = z["y"].astype(int)
    else:
        from train_zoo import load_dataset
        _, _, Xte, yte, shape = load_dataset(a.dataset)
        H = hidden_activations(ck, Xte, shape)

    # The reference is the phase-only head at high b, so the sweep below isolates
    # phase resolution. It is NOT the number in the checkpoint: that was measured
    # by MLMVN itself, which keeps |W|.
    ref = AngularHead(ck.W, b=REFERENCE_BITS, backend="complex128")
    p_ref = ref.predict(H)
    acc_ref = float((p_ref == yte).mean())
    mod = np.abs(ck.W)
    print(f"{a.model}  recorded accuracy_fp32={ck.meta.get('accuracy_fp32')}")
    print(f"phase-only reference (b={REFERENCE_BITS}): {acc_ref:.4f}  "
          f"({H.shape[0]} test samples, {H.shape[1]} hidden units)")
    print(f"|W| in [{mod.min():.4f}, {mod.max():.4f}], std {mod.std():.4f} "
          f"-- unit modulus would be exactly 1\n")

    # Both columns, because they are two different questions. Phase quantization
    # is the knob E3 is about; discarding the MODULUS is the separate, larger
    # decision that makes the datapath multiplier-free at all. Reporting only
    # the phase-only column would hide the cost of the headline claim inside the
    # quantization curve, where it does not belong.
    print(f"{'b':>2}  {'L':>4}  {'phase-only':>11}  {'full-polar':>11}  "
          f"{'modulus':>8}  {'agreement':>10}  {'LUT':>6}")
    rows = []
    for b in a.bits:
        h = AngularHead(ck.W, b=b, backend=a.backend)
        pred = h.predict(H)
        acc = float((pred == yte).mean())
        agree = float((pred == p_ref).mean())
        # Same head, modulus kept: costs a real multiply per MAC.
        acc_fp = float((AngularHead(ck.W, b=b, backend="complex128",
                                    phase_only=False).predict(H) == yte).mean())
        rep = h.mac_report()
        rows.append({"b": b, "L": 2 ** b, "accuracy": acc,
                     "accuracy_full_polar": acc_fp,
                     "modulus_cost": acc_fp - acc,
                     "delta_vs_fp": acc - acc_ref, "agreement_vs_fp": agree,
                     "lut_bytes": rep["lut_bytes"],
                     "weight_bytes": rep["weight_bytes"],
                     "compression_vs_fp32_reim": rep["compression_vs_fp32_reim"],
                     "backend": a.backend, "model": a.model,
                     "n_test": int(H.shape[0])})
        print(f"{b:>2}  {2**b:>4}  {acc:>11.4f}  {acc_fp:>11.4f}  "
              f"{acc_fp - acc:>+8.4f}  {agree:>10.4f}  {rep['lut_bytes']:>4}B")

    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")

    if a.prune_taus:
        prune_rows = sweep_pruning(ck, H, yte, a.prune_b, a.prune_taus, a.backend)
        if a.prune_out:
            p = pathlib.Path(a.prune_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(prune_rows, indent=2))
            print(f"wrote {p}")
    return 0


def sweep_pruning(ck, H, y, b, taus, backend) -> list[dict]:
    """
    Magnitude pruning: the second compression axis, and a partial answer to the
    modulus cost.

    Phase-only rescales every weight to |w| = 1, so a weight training left at
    |w| = 0.0015 gets its contribution multiplied by ~666x. Zeroing the smallest
    weights should therefore HELP, not merely cost less than it saves -- and it
    does, which is evidence that the 2.8-point modulus gap is amplified noise
    rather than lost magnitude information.
    """
    po = float((AngularHead(ck.W, b=b, backend=backend).predict(H) == y).mean())
    fp = float((AngularHead(ck.W, b=b, backend="complex128",
                            phase_only=False).predict(H) == y).mean())
    gap = fp - po
    print(f"\nmagnitude pruning at b={b} "
          f"(phase-only {po:.4f}, full-polar {fp:.4f}, gap {gap:.4f})")
    print(f"{'tau':>6}  {'sparsity':>8}  {'accuracy':>9}  {'gap closed':>10}  "
          f"{'active MACs':>12}")

    rows = []
    for tau in [0.0] + list(taus):
        h = AngularHead(ck.W, b=b, backend=backend, prune_tau=tau)
        acc = float((h.predict(H) == y).mean())
        rep = h.mac_report()
        recovered = (acc - po) / gap if gap else 0.0
        rows.append({"tau": tau, "sparsity": h.sparsity, "accuracy": acc,
                     "gap_closed": recovered, "phase_only": po,
                     "full_polar": fp, "b": b,
                     "active_macs": rep["active_macs"],
                     "head_macs": rep["head_macs"],
                     "mask_bytes": rep["mask_bytes"],
                     "weight_bytes": rep["weight_bytes"],
                     "weight_bytes_sparse": rep["weight_bytes_sparse"],
                     "compression_vs_fp32_reim": rep["compression_vs_fp32_reim"],
                     "n_test": int(H.shape[0])})
        print(f"{tau:>6.3f}  {h.sparsity:>8.3f}  {acc:>9.4f}  "
              f"{recovered * 100:>9.1f}%  {rep['active_macs']:>12,}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
