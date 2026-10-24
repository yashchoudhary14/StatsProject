"""
Lomax (Pareto type II) distribution.

Parameterisation used throughout the project
    f(x; alpha, lam) = alpha * lam * (1 + lam*x)^(-(alpha+1)),   x > 0
    F(x; alpha, lam) = 1 - (1 + lam*x)^(-alpha)
    R(x; alpha, lam) = (1 + lam*x)^(-alpha)              (reliability / survival)
    h(x; alpha, lam) = alpha*lam / (1 + lam*x)           (hazard rate, decreasing)
    MdTF            = (2^(1/alpha) - 1) / lam            (median time to failure)

alpha > 0 is the shape parameter (it controls the tail) and lam > 0 is the
scale-type parameter (1/lam is the usual Lomax scale).
"""
import numpy as np


def pdf(x, alpha, lam):
    x = np.asarray(x, dtype=float)
    return alpha * lam * (1.0 + lam * x) ** (-(alpha + 1.0))


def cdf(x, alpha, lam):
    x = np.asarray(x, dtype=float)
    return 1.0 - (1.0 + lam * x) ** (-alpha)


def reliability(t, alpha, lam):
    return (1.0 + lam * t) ** (-alpha)


def hazard(t, alpha, lam):
    return alpha * lam / (1.0 + lam * t)


def mdtf(alpha, lam):
    return (2.0 ** (1.0 / alpha) - 1.0) / lam


def quantile(p, alpha, lam):
    p = np.asarray(p, dtype=float)
    return ((1.0 - p) ** (-1.0 / alpha) - 1.0) / lam


def rvs(size, alpha, lam, rng):
    return quantile(rng.uniform(size=size), alpha, lam)


def survival_characteristics(alpha, lam, t0):
    """Reliability, hazard and median time to failure (vectorised)."""
    alpha = np.asarray(alpha, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return {
        "R": reliability(t0, alpha, lam),
        "h": hazard(t0, alpha, lam),
        "MdTF": mdtf(alpha, lam),
    }
