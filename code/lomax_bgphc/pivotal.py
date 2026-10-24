"""
Pivotal-quantity (generalised pivotal / generalised confidence interval)
inference, adapting Lodhi, Gangopadhyay & Bhattacharya (2024) to the Lomax
distribution.

Key facts (facility i, D_i observed failures, x_i1 < ... < x_iD):
  U_ij = alpha_i log(1 + lam x_ij) is a progressively censored Exp(1) sample.
  Put L_ij = log(1 + lam x_ij), c_ij = n_i - sum_{l<j}(R_il + 1) and
      W_ij(lam) = sum_{l<j} (R_il + 1) L_il + c_ij L_ij    (j = 1..D_i).
  Then
      P_i(lam) = 2 sum_{j=1}^{D_i-1} log( W_iD / W_ij )  ~  chi2( 2(D_i - 1) ),
  free of alpha_i, and P(lam) = sum_i P_i(lam) ~ chi2( 2 sum_i (D_i - 1) ).
  For the Lomax distribution P(lam) is strictly DECREASING in lam, falls to 0
  as lam -> inf and tends to the finite value P(0+) (the exponential limit)
  as lam -> 0.  Hence P(lam) = c has a solution only when c < P(0+); we draw
  c from the chi-square distribution truncated to (0, P(0+)).
  Given lam, 2 alpha_i Delta_i(lam) ~ chi2(2 D_i)  gives a pivot for alpha_i.

Under GPHC the stopping point is random (Cases I-III), so these pivots are
exact for progressive Type-II data and approximate under GPHC; their
accuracy is checked in the simulation study through coverage probabilities.
"""
import numpy as np
from scipy import stats

from .likelihood import Prepared


class PivotalEngine:
    def __init__(self, prep: Prepared):
        self.prep = prep
        self.fac = []
        for f in prep.data.facilities:
            Rp1 = f.R + 1.0
            excl_counts = np.cumsum(Rp1) - Rp1
            c = f.n - excl_counts                      # c_ij
            self.fac.append((f.x, Rp1, c, f.D))
        self.df = 2.0 * np.sum([max(f.D - 1, 0) for f in prep.data.facilities])

    def _P_from_L(self, Ls):
        total = 0.0
        for (x, Rp1, c, D), L in zip(self.fac, Ls):
            if D < 2:
                continue
            a = Rp1[None, :] * L
            W = (np.cumsum(a, axis=1) - a) + c[None, :] * L
            total = total + 2.0 * np.sum(np.log(W[:, -1:]) - np.log(W[:, :-1]), axis=1)
        return total

    def P(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        Ls = [np.log1p(lam[:, None] * x[None, :]) for (x, _, _, _) in self.fac]
        return self._P_from_L(Ls)

    def P0(self):
        """Limit of P(lam) as lam -> 0+ (log(1+lam x) ~ lam x)."""
        Ls = [x[None, :] for (x, _, _, _) in self.fac]
        return float(self._P_from_L(Ls)[0])

    def solve_lambda(self, targets, n_iter=60):
        """Solve P(lam) = target (vectorised bisection on log lam)."""
        targets = np.asarray(targets, dtype=float)
        s = np.log(1.0 / self.prep.scale)
        lo = np.full_like(targets, s - 25.0)   # P(lo) close to P(0+)  (> target)
        hi = np.full_like(targets, s + 30.0)   # P(hi) small            (< target)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            pm = self.P(np.exp(mid))
            go_right = pm > targets             # P decreasing => root is right of mid
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        return np.exp(0.5 * (lo + hi))

    def sample(self, N, rng):
        """N generalised pivotal draws of (lam, alpha_1..alpha_k)."""
        P0 = self.P0()
        F0 = stats.chi2.cdf(P0, self.df)
        u = rng.uniform(size=N) * F0
        u = np.clip(u, 1e-300, None)
        targets = stats.chi2.ppf(u, self.df)
        lam = self.solve_lambda(targets)
        De = self.prep.Delta(lam)                                  # (N, k)
        chi = rng.chisquare(2.0 * self.prep.Di[None, :], size=(N, self.prep.k))
        alpha = chi / (2.0 * De)
        return {"lam": lam, "alpha": alpha, "trunc_prob": F0, "P0": P0}
