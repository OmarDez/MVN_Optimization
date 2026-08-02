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


def load_dataset(name: str, color: bool = False):
    """
    MNIST/CIFAR-10 via torchvision; falls back to sklearn digits offline.

    `color` decides how CIFAR-10 comes back, and it matters. The FFT encoder
    consumes a 2-D image, so the pure MLMVN path needs greyscale. A backbone
    does not: averaging the channels first throws away two thirds of the input
    before a network that expects three. Shapes are returned as (C, H, W) so the
    caller never has to guess which convention it got.
    """
    if name in ("mnist", "cifar10"):
        try:
            import torchvision
            root = "./data"
            if name == "mnist":
                tr = torchvision.datasets.MNIST(root, train=True, download=True)
                te = torchvision.datasets.MNIST(root, train=False, download=True)
                Xtr = tr.data.numpy().reshape(len(tr), -1)
                Xte = te.data.numpy().reshape(len(te), -1)
                return Xtr, tr.targets.numpy(), Xte, te.targets.numpy(), (1, 28, 28)
            tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
            te = torchvision.datasets.CIFAR10(root, train=False, download=True)
            if color:
                # (N, H, W, 3) -> (N, 3, H, W), flattened.
                f = lambda d: np.asarray(d.data).transpose(0, 3, 1, 2).reshape(len(d.data), -1)
                return (f(tr), np.asarray(tr.targets), f(te),
                        np.asarray(te.targets), (3, 32, 32))
            g = lambda d: np.asarray(d.data).mean(axis=3).reshape(len(d.data), -1)
            return (g(tr), np.asarray(tr.targets), g(te),
                    np.asarray(te.targets), (1, 32, 32))
        except Exception as e:
            print(f"[warn] torchvision unavailable ({e}); using sklearn digits")

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = np.repeat(np.repeat(d.images, 4, axis=1), 4, axis=2)[:, :28, :28]
    X = (X / X.max() * 255).reshape(len(d.images), -1).astype(np.uint8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.2, random_state=0)
    return Xtr, ytr, Xte, yte, (1, 28, 28)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def backbone_features(model: str, Xtr, Xte, shape, seed: int = 0,
                      pretrained: bool = True, size: int = 112):
    """
    Return real-valued features from a FROZEN PRETRAINED torch backbone.

    This is ordinary transfer learning: ImageNet features, no fine-tuning, and
    the MVN head is the only thing that learns. That is the honest setting for
    what the zoo is meant to demonstrate -- that `AngularHead` is agnostic to
    the backbone -- since the quality of the backbone is not what is under test.

    `pretrained=False` keeps the old behaviour (random projections) as a lower
    bound on the head. It is seeded so the checkpoint stays reproducible; with
    pretrained weights the seed is irrelevant but still recorded.

    Two details that are easy to get wrong and expensive to miss:
      * Greyscale input is REPEATED across three channels, never fed to a
        3-channel stem as one. Colour input keeps its channels.
      * Inputs are resized and normalized with the ImageNet statistics the
        weights were trained under. Skipping this quietly costs a lot of
        accuracy, and it is the single easiest thing to leave out.

    `size=112` rather than the canonical 224: measured on this box, 224 costs
    ~31 minutes to extract 70k features against ~14 at 112, and the features
    only feed a linear-ish head. Raise it if the accuracy matters more than the
    wall clock.
    """
    import torch, torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    torch.manual_seed(seed)

    C, H, W = shape
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def to_t(X):
        I = torch.tensor(np.asarray(X), dtype=torch.float32).reshape(-1, C, H, W) / 255.0
        if C == 1:
            I = I.repeat(1, 3, 1, 1)          # grey -> 3 channels, not averaged
        I = F.interpolate(I, size=(size, size), mode="bilinear", align_corners=False)
        return (I - mean) / std

    if model == "resnet18":
        net = models.resnet18(weights="DEFAULT" if pretrained else None)
        feat = nn.Sequential(*list(net.children())[:-1])
    elif model == "mobilenetv3":
        net = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        feat = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
    elif model == "vit_tiny":
        raise NotImplementedError("vit_tiny is a stretch goal; see PLAN.md")
    else:
        raise ValueError(model)

    feat.eval()
    with torch.no_grad():
        def f(X):
            out = []
            for i in range(0, len(X), 128):
                out.append(feat(to_t(X[i:i + 128])).flatten(1).numpy())
            return np.concatenate(out)
        return f(Xtr), f(Xte)


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
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random backbone instead of ImageNet weights: a lower "
                         "bound on the head rather than a usable model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()

    targets = ZOO[:3] if a.all else [a.model]
    pretrained = not a.no_pretrained

    for model in targets:
        t0 = time.time()
        print(f"\n=== {model} ===")
        # Colour only for the backbones: the FFT encoder takes a 2-D image.
        Xtr, ytr, Xte, yte, shape = load_dataset(a.dataset,
                                                 color=(model != "mlmvn_fft"))
        print(f"dataset={a.dataset}  train={Xtr.shape}  shape={shape}")

        if model == "mlmvn_fft":
            # Fully phase-native: every inference MAC can be multiplier-free.
            Ctr = encode_fft(Xtr, a.freq_size, shape[1:])
            Cte = encode_fft(Xte, a.freq_size, shape[1:])
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
            Ztr, Zte = backbone_features(model, Xtr, Xte, shape, seed=a.seed,
                                         pretrained=pretrained)
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
                     "backbone_pretrained": bool(pretrained),
                     "backbone_finetuned": False,
                     "fully_phase_native": False}

        out = pathlib.Path(a.outdir) / f"{model}_{a.dataset}.npz"
        save_checkpoint(out, W, lp, W_hidden=W_hidden,
                        backbone=model, dataset=a.dataset,
                        accuracy_fp32=float(acc), epochs=a.epochs, **extra)
        print(f"  test accuracy {acc:.4f}  [{time.time() - t0:.1f}s]  -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
