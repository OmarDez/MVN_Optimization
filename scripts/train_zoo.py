#!/usr/bin/env python3
"""
Train the model zoo and freeze checkpoints into models/.

Every entry produces the same artifact -- an (k, d+1) complex weight matrix plus
lift parameters -- so a single AngularHead can serve all of them. That
uniformity is what makes "works with several models" a property of the design
rather than a pile of special cases.

    python scripts/train_zoo.py --model mlmvn_fft --dataset mnist
    python scripts/train_zoo.py --model resnet18   --dataset mnist
    python scripts/train_zoo.py --model mobilenetv3 --dataset cifar10
    python scripts/train_zoo.py --all

Training runs wherever is convenient (a GPU box, Colab, a laptop). Only
inference is ever benchmarked on Arm.
"""
from __future__ import annotations

import argparse, pathlib, sys, time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pymvn import (MLMVN, MVNHead, encode_fft, fit_lift, lift,  # noqa: E402
                   save_checkpoint)

ZOO = ["mlmvn_fft", "resnet18", "mobilenetv3", "vit_tiny"]


def load_dataset(name: str):
    """MNIST/CIFAR-10 via torchvision; falls back to sklearn digits offline."""
    if name in ("mnist", "cifar10"):
        try:
            import torchvision
            root = "./data"
            if name == "mnist":
                tr = torchvision.datasets.MNIST(root, train=True, download=True)
                te = torchvision.datasets.MNIST(root, train=False, download=True)
                Xtr = tr.data.numpy().reshape(len(tr), -1)
                Xte = te.data.numpy().reshape(len(te), -1)
                return Xtr, tr.targets.numpy(), Xte, te.targets.numpy(), (28, 28)
            tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
            te = torchvision.datasets.CIFAR10(root, train=False, download=True)
            g = lambda d: np.asarray(d.data).mean(axis=3).reshape(len(d.data), -1)
            return (g(tr), np.asarray(tr.targets), g(te), np.asarray(te.targets), (32, 32))
        except Exception as e:
            print(f"[warn] torchvision unavailable ({e}); using sklearn digits")

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = np.repeat(np.repeat(d.images, 4, axis=1), 4, axis=2)[:, :28, :28]
    X = (X / X.max() * 255).reshape(len(d.images), -1).astype(np.uint8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.2, random_state=0)
    return Xtr, ytr, Xte, yte, (28, 28)


def backbone_features(model: str, Xtr, Xte, shape, seed: int = 0):
    """
    Return real-valued features from a torch backbone.

    The backbone is constructed with weights=None and never trained: these are
    RANDOM convolutional projections, and the MVN head is the only thing that
    learns. That is a deliberate lower bound on the head -- whatever accuracy it
    reaches here, it reaches without help from the feature extractor -- but it
    is not the setting PLAN.md 3.2 quotes, so do not compare the two numbers.

    Seeded because of it. An unseeded random backbone makes the checkpoint
    irreproducible: the head weights would refer to features nobody can
    regenerate. The seed goes into the checkpoint metadata.
    """
    import torch, torch.nn as nn
    from torchvision import models

    torch.manual_seed(seed)

    H, W = shape
    to_t = lambda X: torch.tensor(X, dtype=torch.float32).reshape(-1, 1, H, W) / 255.0
    Itr, Ite = to_t(Xtr), to_t(Xte)
    if model in ("mobilenetv3", "vit_tiny"):
        Itr, Ite = Itr.repeat(1, 3, 1, 1), Ite.repeat(1, 3, 1, 1)

    if model == "resnet18":
        net = models.resnet18(weights=None, num_classes=10)
        net.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        feat = nn.Sequential(*list(net.children())[:-1])
    elif model == "mobilenetv3":
        net = models.mobilenet_v3_small(weights=None, num_classes=10)
        feat = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
    elif model == "vit_tiny":
        import torch.nn.functional as F
        net = models.vit_b_16(weights=None, num_classes=10)
        feat = None  # placeholder: see PLAN.md, stretch goal S1
        raise NotImplementedError("vit_tiny is a stretch goal; see PLAN.md")
    else:
        raise ValueError(model)

    feat.eval()
    with torch.no_grad():
        f = lambda I: torch.cat([feat(I[i:i + 256]).flatten(1)
                                 for i in range(0, len(I), 256)]).numpy()
        return f(Itr), f(Ite)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ZOO, default="mlmvn_fft")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--lift", default="minmax",
                    choices=["minmax", "tanh", "cdf", "learned"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--freq-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0,
                    help="backbone init seed; recorded in the checkpoint")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()

    targets = ZOO[:3] if a.all else [a.model]
    Xtr, ytr, Xte, yte, shape = load_dataset(a.dataset)
    print(f"dataset={a.dataset}  train={Xtr.shape}  test={Xte.shape}")

    for model in targets:
        t0 = time.time()
        print(f"\n=== {model} ===")

        if model == "mlmvn_fft":
            # Fully phase-native: every inference MAC can be multiplier-free.
            Ctr = encode_fft(Xtr, a.freq_size, shape)
            Cte = encode_fft(Xte, a.freq_size, shape)
            net = MLMVN(Ctr.shape[1], a.hidden, 10)
            net.fit(Ctr, ytr, epochs=a.epochs)
            acc = net.accuracy(Cte, yte)
            # Freeze the OUTPUT layer as the benchmarked head -- and W1 with it.
            # The head alone cannot reproduce the accuracy recorded below,
            # because the activations it expects come out of the hidden layer.
            W, lp, W_hidden = net.W2, None, net.W1
            extra = {"hidden": a.hidden, "freq_size": a.freq_size,
                     "encoding": "fft_phase_only", "fully_phase_native": True}
        else:
            W_hidden = None
            Ztr, Zte = backbone_features(model, Xtr, Xte, shape, seed=a.seed)
            lp = fit_lift(Ztr, a.lift)
            Ctr, Cte = lift(Ztr, lp), lift(Zte, lp)
            head = MVNHead(Ctr.shape[1], 10).fit(Ctr, ytr, epochs=a.epochs)
            acc = head.accuracy(Cte, yte)
            W = head.W
            # No "lift" key here: it would collide with save_checkpoint's own
            # `lift` parameter through **extra. The kind is already serialized
            # as part of LiftParams, so recording it again would be redundant
            # as well as fatal.
            extra = {"feature_dim": int(Ztr.shape[1]),
                     "backbone_seed": int(a.seed),
                     "backbone_trained": False,
                     "fully_phase_native": False}

        out = pathlib.Path(a.outdir) / f"{model}_{a.dataset}.npz"
        save_checkpoint(out, W, lp, W_hidden=W_hidden,
                        backbone=model, dataset=a.dataset,
                        accuracy_fp32=float(acc), epochs=a.epochs, **extra)
        print(f"  test accuracy {acc:.4f}  [{time.time() - t0:.1f}s]  -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
