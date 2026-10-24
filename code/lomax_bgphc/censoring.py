"""
Block generalized progressive hybrid censoring (BGPHC).

A total of n units is split into k blocks (test facilities); block i holds n_i
units.  Every block runs its own generalized progressive hybrid censoring
scheme (GPHCS, Cho, Sun & Lee 2015) with plan (n_i, m_i, kappa_i, T_i, R_i):

  * kappa_i < m_i <= n_i, T_i > 0 and R_i = (R_i1, ..., R_im_i) with
    sum(R_i) = n_i - m_i.
  * At the j-th failure, R_ij surviving units are withdrawn at random.
  * The block stops at T_i* = max( X_{kappa_i}, min( X_{m_i}, T_i ) ):
        Case I   (T_i < X_kappa)         : stop at X_kappa, D_i = kappa_i
        Case II  (X_kappa <= T_i < X_m)  : stop at T_i,     D_i = J_i
        Case III (X_m <= T_i)            : stop at X_m,     D_i = m_i
    where J_i is the number of failures observed before T_i.  At the stopping
    point every surviving unit is withdrawn.

For every block we store the observed failure times, the removals that were
actually made at each failure and the number RT_i removed at T_i (Case II only).
With this convention the likelihood of block i is always

    prod_j f(x_ij) [1 - F(x_ij)]^{R_ij}  *  [1 - F(T_i)]^{RT_i}.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import model


@dataclass
class GPHCPlan:
    n: int
    m: int
    kappa: int
    T: float
    R: np.ndarray  # planned removals, length m, sum = n - m

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=int)
        if len(self.R) != self.m:
            raise ValueError("R must have length m")
        if self.R.sum() != self.n - self.m:
            raise ValueError("sum(R) must equal n - m")
        if not (1 <= self.kappa < self.m <= self.n):
            raise ValueError("need 1 <= kappa < m <= n")


@dataclass
class FacilityData:
    x: np.ndarray   # observed (ordered) failure times, length D
    R: np.ndarray   # removals actually made at each observed failure, length D
    RT: int         # units withdrawn at time T (only Case II, otherwise 0)
    T: float
    n: int
    case: int       # 1, 2 or 3

    @property
    def D(self):
        return len(self.x)

    def check(self):
        assert self.D + self.R.sum() + self.RT == self.n, "units do not add up"


@dataclass
class BlockData:
    facilities: List[FacilityData] = field(default_factory=list)

    @property
    def k(self):
        return len(self.facilities)

    @property
    def D(self):
        return np.array([f.D for f in self.facilities])

    @property
    def cases(self):
        return [f.case for f in self.facilities]

    def concat(self):
        """Concatenated arrays used by the vectorised code."""
        x = np.concatenate([f.x for f in self.facilities])
        w = np.concatenate([1.0 + f.R for f in self.facilities])
        starts = np.cumsum([0] + [f.D for f in self.facilities])[:-1]
        T = np.array([f.T for f in self.facilities], dtype=float)
        RT = np.array([f.RT for f in self.facilities], dtype=float)
        return x, w, starts, T, RT


# ---------------------------------------------------------------------------
# generation of progressive Type-II samples (Balakrishnan & Sandhu, 1995)
# ---------------------------------------------------------------------------
def progressive_type2_uniform(n, m, R, rng):
    """Progressively Type-II censored sample of size m from U(0,1)."""
    R = np.asarray(R)
    W = rng.uniform(size=m)
    # E_i = i + R_m + R_{m-1} + ... + R_{m-i+1},  i = 1..m
    E = np.arange(1, m + 1) + np.cumsum(R[::-1])
    V = W ** (1.0 / E)
    # U_i = 1 - V_m V_{m-1} ... V_{m-i+1}
    U = 1.0 - np.cumprod(V[::-1])
    return U


def apply_gphc_rule(x_prog, plan: GPHCPlan) -> FacilityData:
    """Turn a full progressive Type-II sample (length m) into GPHCS data."""
    x_prog = np.asarray(x_prog, dtype=float)
    n, m, kap, T, R = plan.n, plan.m, plan.kappa, plan.T, plan.R
    if x_prog[kap - 1] > T:                       # Case I
        D = kap
        Robs = R[:D].copy()
        Robs[-1] = n - D - R[:D - 1].sum()
        return FacilityData(x_prog[:D].copy(), Robs, 0, T, n, 1)
    if x_prog[m - 1] <= T:                        # Case III
        return FacilityData(x_prog.copy(), R.copy(), 0, T, n, 3)
    J = int(np.sum(x_prog <= T))                  # Case II, kappa <= J < m
    Robs = R[:J].copy()
    RT = n - J - Robs.sum()
    return FacilityData(x_prog[:J].copy(), Robs, int(RT), T, n, 2)


def generate_facility(plan: GPHCPlan, alpha, lam, rng) -> FacilityData:
    U = progressive_type2_uniform(plan.n, plan.m, plan.R, rng)
    x = model.quantile(U, alpha, lam)
    return apply_gphc_rule(x, plan)


def generate_block(plans, alphas, lam, rng) -> BlockData:
    return BlockData([generate_facility(p, a, lam, rng) for p, a in zip(plans, alphas)])


# ---------------------------------------------------------------------------
# applying the scheme to an observed complete sample (used for real data)
# ---------------------------------------------------------------------------
def censor_complete_sample(x_complete, plan: GPHCPlan, rng) -> FacilityData:
    """Run the GPHCS on a complete sample: at the j-th failure R_j of the
    surviving units are withdrawn completely at random."""
    surv = np.sort(np.asarray(x_complete, dtype=float))
    n, m, kap, T, R = plan.n, plan.m, plan.kappa, plan.T, plan.R
    assert len(surv) == n
    xs, Rs = [], []
    for j in range(1, m + 1):
        t = surv[0]
        if t > T and (j - 1) >= kap:              # Case II: stopped at T
            return FacilityData(np.array(xs), np.array(Rs, dtype=int),
                                int(len(surv)), T, n, 2)
        xs.append(t)
        surv = surv[1:]
        if j == m:                                # Case III
            Rs.append(len(surv))
            return FacilityData(np.array(xs), np.array(Rs, dtype=int), 0, T, n, 3)
        if j == kap and t > T:                    # Case I
            Rs.append(len(surv))
            return FacilityData(np.array(xs), np.array(Rs, dtype=int), 0, T, n, 1)
        r = int(R[j - 1])
        if r > 0:
            drop = rng.choice(len(surv), size=r, replace=False)
            surv = np.delete(surv, drop)
        Rs.append(r)
    raise RuntimeError("unreachable")
