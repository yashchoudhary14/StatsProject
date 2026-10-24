"""
Likelihood of Lomax(alpha_i, lam) data under block GPHC.

    l(alpha, lam) = sum_i D_i log(alpha_i) + D log(lam) - sum_ij log(1 + lam x_ij)
                    - sum_i alpha_i * Delta_i(lam)

    Delta_i(lam) = sum_j (1 + R_ij) log(1 + lam x_ij) + RT_i log(1 + lam T_i)

Delta_i(lam) is the "total time on test" of facility i after the transformation
u = log(1 + lam x), which maps Lomax(alpha_i, lam) lifetimes to exponential
lifetimes with rate alpha_i.
"""
import numpy as np

from .censoring import BlockData


class Prepared:
    """Pre-computed arrays for fast (vectorised over lam) likelihood work."""

    def __init__(self, data: BlockData):
        self.data = data
        self.k = data.k
        self.Di = data.D.astype(float)
        self.Dtot = float(self.Di.sum())
        self.x, self.w, self.starts, self.T, self.RT = data.concat()
        self.scale = float(np.median(self.x))

    # -- Delta_i(lam) and its derivatives; lam may be a scalar or 1-d array --
    def Delta(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        L = np.log1p(lam[:, None] * self.x[None, :]) * self.w[None, :]
        S = np.add.reduceat(L, self.starts, axis=1)
        S += self.RT[None, :] * np.log1p(lam[:, None] * self.T[None, :])
        return S                                            # (N, k)

    def dDelta(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        L = self.w[None, :] * self.x[None, :] / (1.0 + lam[:, None] * self.x[None, :])
        S = np.add.reduceat(L, self.starts, axis=1)
        S += self.RT[None, :] * self.T[None, :] / (1.0 + lam[:, None] * self.T[None, :])
        return S

    def d2Delta(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        L = -self.w[None, :] * self.x[None, :] ** 2 / (1.0 + lam[:, None] * self.x[None, :]) ** 2
        S = np.add.reduceat(L, self.starts, axis=1)
        S -= self.RT[None, :] * self.T[None, :] ** 2 / (1.0 + lam[:, None] * self.T[None, :]) ** 2
        return S

    def sumlog1p(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        return np.log1p(lam[:, None] * self.x[None, :]).sum(axis=1)

    # -- likelihood pieces --------------------------------------------------
    def loglik(self, alpha, lam):
        alpha = np.asarray(alpha, dtype=float)
        De = self.Delta(lam)[0]
        return (np.sum(self.Di * np.log(alpha)) + self.Dtot * np.log(lam)
                - self.sumlog1p(lam)[0] - np.sum(alpha * De))

    def profile(self, lam):
        """Profile log-likelihood of lam (alpha_i replaced by D_i/Delta_i)."""
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        De = self.Delta(lam)
        return (np.sum(self.Di * np.log(self.Di / De), axis=1) - self.Dtot
                + self.Dtot * np.log(lam) - self.sumlog1p(lam))

    def existence_condition(self):
        """Sufficient condition for an interior MLE of lam: the slope of the
        profile log-likelihood at lam -> 0+ is positive, i.e.
            sum_i D_i B_i / (2 A_i) > sum_ij x_ij,
        with A_i = sum (1+R)x + RT*T and B_i = sum (1+R)x^2 + RT*T^2."""
        A = np.add.reduceat(self.w * self.x, self.starts) + self.RT * self.T
        B = np.add.reduceat(self.w * self.x ** 2, self.starts) + self.RT * self.T ** 2
        slope0 = np.sum(self.Di * B / (2 * A)) - self.x.sum()
        return slope0 > 0, slope0

    def observed_information(self, alpha, lam):
        """Observed Fisher information for theta = (alpha_1..alpha_k, lam)."""
        alpha = np.asarray(alpha, dtype=float)
        k = self.k
        I = np.zeros((k + 1, k + 1))
        I[np.arange(k), np.arange(k)] = self.Di / alpha ** 2
        dD = self.dDelta(lam)[0]
        I[:k, k] = I[k, :k] = dD
        d2 = self.d2Delta(lam)[0]
        I[k, k] = (self.Dtot / lam ** 2
                   - np.sum(self.x ** 2 / (1.0 + lam * self.x) ** 2)
                   + np.sum(alpha * d2))
        return I
