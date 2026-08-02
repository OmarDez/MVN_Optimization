"""
Riemannian training on the torus.

PLAN.md 2.5 says the state space is T^d = (S^1)^d. The default trainer does not
act like it: the error-correction rule updates in the ambient C^d and lets the
modulus wander, which is what makes `phase_only=True` cost accuracy at inference
(E3). Projecting back onto the circle after each ambient step is EXTRINSIC and
measurably worse -- it destroys, every step, the magnitude the correction rule
uses as an implicit learning rate.

`geometry="tangent"` instead projects the update onto the tangent space i.w.R
and integrates the phase, so the weights never leave the manifold and there is
no magnitude to destroy.

These tests check the property that makes the whole thing worth doing -- the
weights stay on S^1 exactly -- and that the default path is untouched.
"""

import numpy as np
import pytest

from pymvn import MLMVN, MVNHead


def _data(seed=0, n=120, d=16, k=3):
    rng = np.random.default_rng(seed)
    C = np.exp(1j * rng.uniform(0, 2 * np.pi, (n, d)))
    y = rng.integers(0, k, n)
    return C, y


@pytest.mark.parametrize("cls,args", [(MVNHead, (16, 3)), (MLMVN, (16, 12, 3))])
def test_tangent_updates_never_leave_the_circle(cls, args):
    """
    The defining property. Not "close to 1" -- the update is a phase increment,
    so the modulus is 1 by construction and any drift means the ambient path
    ran by mistake.
    """
    C, y = _data()
    net = cls(*args, geometry="tangent")
    net.fit(C, y, epochs=2)
    for W in ([net.W] if cls is MVNHead else [net.W1, net.W2]):
        np.testing.assert_allclose(np.abs(W), 1.0, atol=1e-12)


@pytest.mark.parametrize("cls,args", [(MVNHead, (16, 3)), (MLMVN, (16, 12, 3))])
def test_ambient_is_the_default_and_does_leave_the_circle(cls, args):
    """
    The contrast, and a guard that the default path was not quietly changed:
    ambient training is still what happens unless asked otherwise, and it does
    move the modulus away from 1.
    """
    C, y = _data()
    net = cls(*args)
    assert net.geometry == "ambient"
    net.fit(C, y, epochs=2)
    W = net.W if cls is MVNHead else net.W2
    assert not np.allclose(np.abs(W), np.abs(W).flat[0], atol=1e-9), \
        "ambient training should spread the modulus"


@pytest.mark.parametrize("cls,args", [(MVNHead, (16, 3)), (MLMVN, (16, 12, 3))])
def test_tangent_makes_phase_only_free(cls, args):
    """
    The point of the exercise. If every weight has |w| = 1, then discarding the
    modulus at inference discards nothing, so the phase-only head must agree
    with the full-polar one exactly -- not approximately.
    """
    from pymvn import AngularHead
    C, y = _data()
    net = cls(*args, geometry="tangent")
    net.fit(C, y, epochs=2)
    W = net.W if cls is MVNHead else net.W2
    X = C if cls is MVNHead else net.forward(C)[0]

    po = AngularHead(W, b=8, backend="complex128", phase_only=True).predict(X)
    fp = AngularHead(W, b=8, backend="complex128", phase_only=False).predict(X)
    np.testing.assert_array_equal(po, fp)


def test_rejects_unknown_geometry():
    with pytest.raises(ValueError):
        MVNHead(8, 3, geometry="riemannian")
    with pytest.raises(ValueError):
        MLMVN(8, 6, 3, geometry="riemannian")


def test_tangent_still_learns():
    """
    Staying on the manifold is worthless if the model stops training. A
    separable problem must still be learned above chance.
    """
    rng = np.random.default_rng(3)
    k, d, n = 3, 24, 300
    centers = np.exp(1j * rng.uniform(0, 2 * np.pi, (k, d)))
    y = rng.integers(0, k, n)
    C = centers[y] * np.exp(1j * rng.normal(0, 0.25, (n, d)))

    net = MVNHead(d, k, geometry="tangent").fit(C, y, epochs=6)
    assert net.accuracy(C, y) > 0.6, "tangent trainer failed to learn"
