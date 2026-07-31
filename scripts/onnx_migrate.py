#!/usr/bin/env python3
"""
Migration template: retarget any ONNX classifier to a phase-native head.

This is the adoption story. Given an existing ONNX classification model, the
script locates the final Gemm/MatMul, replaces it with an angular head, and
reports the accuracy and footprint change. Nothing about the backbone changes,
so the migration is incremental and reversible.

    python scripts/onnx_migrate.py --model resnet.onnx --data feats.npz --bits 4

Output:
  * <model>.angular.npz   the frozen angular head
  * a printed before/after table (accuracy, weight bytes, MAC composition)
"""
from __future__ import annotations

import argparse, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pymvn import AngularHead, MVNHead, fit_lift, lift, save_checkpoint  # noqa: E402


def find_classifier(model):
    """Return the last Gemm/MatMul node -- the classifier to be replaced."""
    import onnx
    for node in reversed(model.graph.node):
        if node.op_type in ("Gemm", "MatMul"):
            return node
    raise RuntimeError("no Gemm/MatMul found; is this a classifier?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="input .onnx classifier")
    ap.add_argument("--data", required=True,
                    help=".npz with arrays: feats_train, y_train, feats_test, y_test")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--lift", default="minmax")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import onnx
    model = onnx.load(a.model)
    node = find_classifier(model)
    print(f"classifier node: {node.op_type} '{node.name}'")

    d = np.load(a.data)
    Ztr, ytr = d["feats_train"], d["y_train"]
    Zte, yte = d["feats_test"], d["y_test"]
    n_classes = int(ytr.max()) + 1

    # --- original head, for the before column -----------------------------
    init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    Worig = next((v for k, v in init.items() if k in node.input and v.ndim == 2), None)
    orig_bytes = Worig.nbytes if Worig is not None else 0

    # --- angular replacement ----------------------------------------------
    lp = fit_lift(Ztr, a.lift)
    Ctr, Cte = lift(Ztr, lp), lift(Zte, lp)
    head = MVNHead(Ctr.shape[1], n_classes).fit(Ctr, ytr, epochs=a.epochs)

    ah = AngularHead(head.W, b=a.bits, backend="angular_tiled")
    acc = ah.accuracy(Cte, yte)
    rep = ah.mac_report()
    ang_bytes = head.W.size * a.bits / 8

    out = pathlib.Path(a.out or a.model).with_suffix(".angular.npz")
    save_checkpoint(out, head.W, lp, backbone=pathlib.Path(a.model).stem,
                    dataset=a.data, accuracy_fp32=float(acc), migrated=True)

    print(f"""
  MIGRATION REPORT
  ----------------------------------------------------------
  head accuracy (angular, b={a.bits}) : {acc:.4f}
  head weight bytes  before / after  : {orig_bytes:,} / {int(ang_bytes):,}
  compression                        : {orig_bytes / max(ang_bytes, 1):.1f}x
  real multiplies per MAC            : {rep['real_mults_per_mac']}
  integer adds per MAC               : {rep['int_adds_per_mac']}
  lookup table                       : {rep['lut_bytes']} bytes
  written                            : {out}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
