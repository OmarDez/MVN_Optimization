"""
MVN learning by error correction.

The MVN activation P(z) = z/|z| is not holomorphic, so classical backpropagation
does not apply. Aizenberg's error-correction rule instead ROTATES weights toward
the desired root:

    w <- w + (C / (n+1)) * (eps_desired - eps_actual) * conj(x)

Two trainers live here:

  * `MVNHead`   -- a single one-vs-all MVN layer. This is what the hybrid
                   models use, and what gets frozen into a checkpoint and
                   handed to AngularHead for inference.

  * `MLMVN`     -- one hidden layer plus an output layer, with hidden-layer
                   error distributed through a truncated SVD pseudo-inverse of
                   the output weights. This is the pure-phase model where every
                   inference MAC can be made multiplier-free.

Training runs in float64 on whatever machine is convenient. Only inference is
benchmarked on Arm.
"""

from __future__ import annotations

import math
import time

import numpy as np

__all__ = ["MVNHead", "MLMVN", "make_targets"]


def make_targets(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    One-vs-all targets on S^1: +1 for the true class, -1 otherwise.
    The two targets sit at opposite poles, so the decision rule reduces to the
    sign of Re(z) -- which is what argmax over Re(z) implements.
    """
    y = np.asarray(y).astype(int)
    T = -np.ones((len(y), n_classes), dtype=np.complex128)
    T[np.arange(len(y)), y] = 1.0
    return T


class MVNHead:
    """Single one-vs-all MVN layer trained by error correction."""

    def __init__(self, n_inputs: int, n_classes: int = 10, lr: float = 1.0,
                 seed: int = 7, geometry: str = "ambient"):
        if geometry not in ("ambient", "tangent"):
            raise ValueError(f"geometry must be 'ambient' or 'tangent', "
                             f"got {geometry!r}")
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, (n_classes, n_inputs + 1))
        self.geometry = geometry
        if geometry == "tangent":
            # State IS the phase. There is no modulus to drift, so nothing has
            # to be projected back afterwards.
            self.phi = phases
            self.W = np.exp(1j * phases)
        else:
            self.W = np.exp(1j * phases) / np.sqrt(n_inputs + 1)
        self.n_classes = n_classes
        self.lr = lr

    @staticmethod
    def _with_bias(X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=np.complex128)
        return np.concatenate([ones, np.asarray(X, dtype=np.complex128)], axis=1)

    def forward(self, X: np.ndarray):
        Xb = self._with_bias(X)
        Z = Xb @ self.W.T
        return Xb, Z / np.maximum(np.abs(Z), 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, O = self.forward(X)
        return np.real(O).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 3,
            tol: float = 1e-3, seed: int = 0, verbose: bool = False):
        X = np.asarray(X, dtype=np.complex128)
        T = make_targets(y, self.n_classes)
        rng = np.random.default_rng(seed)
        scale = self.lr / self.W.shape[1]

        for ep in range(epochs):
            for i in rng.permutation(len(y)):
                Xb, O = self.forward(X[i:i + 1])
                eps = T[i] - O[0]
                if np.max(np.abs(eps)) < tol:
                    continue
                if self.geometry == "tangent":
                    # Riemannian SGD on the torus T^d, which PLAN.md 2.5 already
                    # names as the state space. The ambient rule updates in C^d
                    # and leaves the manifold; projecting back afterwards is
                    # EXTRINSIC and destroys, every step, the magnitude the
                    # correction rule uses as an implicit learning rate.
                    #
                    # The tangent space at w = e^{i.phi} is i.w.R, so the
                    # component of the ambient step along it is
                    #     a = Im(dW . conj(w)),   dW = scale . eps . conj(x)
                    # and moving along the tangent by `a` is, to first order,
                    # rotating phi by `a`. The weight never leaves S^1, so there
                    # is no modulus to destroy.
                    dphi = scale * np.imag(
                        eps[:, None] * np.conj(Xb[0])[None, :] * np.conj(self.W))
                    self.phi += dphi
                    self.W = np.exp(1j * self.phi)
                else:
                    self.W += scale * eps[:, None] * np.conj(Xb[0])[None, :]
            if verbose:
                print(f"  epoch {ep + 1}: train acc {self.accuracy(X, y):.4f}")
        return self


class MLMVN:
    """
    Two-layer MLMVN. Hidden-layer error is obtained by pushing the output error
    through a truncated pseudo-inverse of the output weights, computed once via
    SVD and refreshed periodically. That replaces the per-neuron loop of the
    naive formulation and is the source of the large training speedup.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_classes: int,
                 lr: float = 1.0, beta: float = 0.5, dead_zone: float = 0.0,
                 seed: int = 7, geometry: str = "ambient"):
        if geometry not in ("ambient", "tangent"):
            raise ValueError(f"geometry must be 'ambient' or 'tangent', "
                             f"got {geometry!r}")
        self.geometry = geometry
        rng = np.random.default_rng(seed)
        self.W1 = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_hidden, n_inputs + 1)))
        self.W1 /= np.sqrt(n_inputs + 1)
        self.W2 = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_classes, n_hidden + 1)))
        self.W2 /= np.sqrt(n_hidden + 1)
        if geometry == "tangent":
            # Unit modulus from the start, and phase is the state that moves.
            self.W1, self.W2 = self.W1 / np.abs(self.W1), self.W2 / np.abs(self.W2)
            self.phi1, self.phi2 = np.angle(self.W1), np.angle(self.W2)

        self.n_inputs, self.n_hidden, self.n_classes = n_inputs, n_hidden, n_classes
        self.lr, self.beta, self.dead_zone = lr, beta, dead_zone
        self._pinv = None
        self._svd_rank = 0

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _bias(v: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones(1, dtype=np.complex128), v])

    def forward(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, dtype=np.complex128))
        ones = np.ones((X.shape[0], 1), dtype=np.complex128)
        Xb = np.concatenate([ones, X], axis=1)

        Zh = Xb @ self.W1.T
        Yh = Zh / np.maximum(np.abs(Zh), 1e-12)
        Hb = np.concatenate([ones, Yh], axis=1)
        Zo = Hb @ self.W2.T
        return Yh, Zh, Zo

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, Zo = self.forward(X)
        return np.real(Zo).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    # -- training -----------------------------------------------------------

    def _precompute_pinv(self, energy: float = 0.99) -> None:
        """Truncated pseudo-inverse of W2 (bias column excluded)."""
        W2h = self.W2[:, 1:]
        U, s, Vh = np.linalg.svd(W2h, full_matrices=False)
        e = s ** 2
        total = e.sum()
        if total < 1e-20:
            self._pinv = np.zeros((self.n_hidden, self.n_classes), dtype=np.complex128)
            self._svd_rank = 0
            return
        cum = np.cumsum(e) / total
        rank = min(int((cum < energy).sum()) + 1, len(s))
        self._pinv = (Vh[:rank].conj().T * (1.0 / s[:rank])[None, :]) @ U[:, :rank].conj().T
        self._svd_rank = rank

    def _train_one(self, xi, yi, ti, lr) -> tuple[bool, float]:
        Xb = self._bias(xi)
        Zh = Xb @ self.W1.T
        Zh_mag = np.maximum(np.abs(Zh), 1e-12)
        Yh = Zh / Zh_mag
        Hb = self._bias(Yh)
        Zo = Hb @ self.W2.T

        correct = int(np.real(Zo).argmax()) == yi
        Yo = Zo / np.maximum(np.abs(Zo), 1e-12)
        eps_full = ti - Yo
        err_sq = float((np.abs(eps_full) ** 2).sum())

        wrong = (np.real(Zo) < 0).astype(int) != (np.real(ti) < 0).astype(int)
        pe = np.abs(np.angle(ti) - np.angle(Zo))
        pe = np.where(pe > math.pi, 2 * math.pi - pe, pe)
        soft = (~wrong) & (pe > self.dead_zone)

        eps = np.zeros_like(eps_full)
        eps[wrong] = eps_full[wrong]
        eps[soft] = self.beta * eps_full[soft]
        if not np.any(np.abs(eps) > 0):
            return correct, err_sq

        mask = np.abs(eps) > 0
        eps_h = (self._pinv @ eps) / Zh_mag
        act = np.abs(eps_h) > 1e-10

        if self.geometry == "tangent":
            # Riemannian SGD on T^d: project the ambient step onto the tangent
            # i.w.R and integrate the phase directly, so the weights never leave
            # the manifold. See MVNHead.fit for the derivation.
            self.phi2[mask] += (lr / (self.n_hidden + 1)) * np.imag(
                eps[mask][:, None] * np.conj(Hb)[None, :] * np.conj(self.W2[mask]))
            self.W2[mask] = np.exp(1j * self.phi2[mask])
            if np.any(act):
                self.phi1[act] += (lr / (self.n_inputs + 1)) * np.imag(
                    eps_h[act][:, None] * np.conj(Xb)[None, :] * np.conj(self.W1[act]))
                self.W1[act] = np.exp(1j * self.phi1[act])
            return correct, err_sq

        self.W2[mask] += (lr / (self.n_hidden + 1)) * eps[mask][:, None] * np.conj(Hb)[None, :]
        if np.any(act):
            self.W1[act] += (lr / (self.n_inputs + 1)) * eps_h[act][:, None] * np.conj(Xb)[None, :]

        return correct, err_sq

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
            lr: float = None, svd_every: int = 500, svd_energy: float = 0.99,
            seed: int = 0, verbose: bool = True) -> dict:
        X = np.asarray(X, dtype=np.complex128)
        y = np.asarray(y).astype(int)
        T = make_targets(y, self.n_classes)
        lr = self.lr if lr is None else lr
        rng = np.random.default_rng(seed)

        self._precompute_pinv(svd_energy)
        history = {"train_acc": [], "rmse": [], "time": []}

        for ep in range(epochs):
            t0 = time.time()
            n_ok, err_tot = 0, 0.0
            for c, i in enumerate(rng.permutation(len(y))):
                if c and c % svd_every == 0:
                    self._precompute_pinv(svd_energy)
                ok, e = self._train_one(X[i], int(y[i]), T[i], lr)
                n_ok += int(ok)
                err_tot += e

            acc = n_ok / len(y)
            rmse = math.sqrt(err_tot / (len(y) * self.n_classes))
            history["train_acc"].append(acc)
            history["rmse"].append(rmse)
            history["time"].append(time.time() - t0)
            if verbose:
                print(f"  epoch {ep + 1:3d}  acc={acc:.4f}  rmse={rmse:.4f}  "
                      f"rank={self._svd_rank}  [{history['time'][-1]:.1f}s]")

        return history
