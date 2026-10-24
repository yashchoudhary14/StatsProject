"""
Bayesian estimation with a joint (dependent) Gamma-Dirichlet prior,
following the idea of Sultana, Cetinkaya, Raqab & Kundu (2026) (Pena & Gupta
1990 prior), for both the unrestricted and the order-restricted case.

Prior
  * alpha = (alpha_1..alpha_k) ~ GD(a0, b0, c_1..c_k):
        alpha_+ = sum alpha_i ~ Gamma(a0, b0)  and
        (alpha_1/alpha_+, ..., alpha_k/alpha_+) | alpha_+ ~ Dirichlet(c_1..c_k)
    density  prop. to  prod alpha_i^(c_i-1) e^(-b0 alpha_i) * alpha_+^(a0 - c_+)
    (a0 = c_+ gives independent Gamma(c_i, b0) priors, otherwise dependent).
  * lam ~ Gamma(a, b), independent of alpha.
  * order-restricted case: alpha_1 <= ... <= alpha_k with the ordered
    Gamma-Dirichlet (OGD) prior  prop. to
        e^(-b0 alpha_+) alpha_+^(a0 - c_+) sum_{perm} prod_j alpha_{p(j)}^(c_j - 1).

Posterior (unrestricted)
  pi(alpha, lam | data) prop. to
      lam^(a-1+D) e^(-b lam) prod_ij (1+lam x_ij)^(-1)
      * prod_i alpha_i^(D_i + c_i - 1) e^(-alpha_i (b0 + Delta_i(lam)))
      * alpha_+^(a0 - c_+)

Computation: importance sampling.
  Proposal  q(lam, alpha) = g(lam) * prod_i Gamma(alpha_i | D_i + c_i, b0 + Delta_i(lam)),
  where g is the marginal posterior of lam when a0 = c_+ (independent priors),
      g(lam) prop. to lam^(a-1+D) e^(-b lam) prod(1+lam x)^(-1)
                     prod_i (b0 + Delta_i(lam))^(-(D_i + c_i)),
  sampled by numerical inverse-cdf on a fine grid of log(lam).
  Importance weight (unrestricted):  w = alpha_+^(a0 - c_+) * g(lam)/g_grid(lam).
  For the order-restricted posterior the proposal draws are sorted, and the
  weight is target / (density of the sorted proposal vector), which requires a
  sum over the k! permutations (exactly as in the OGD prior).
"""
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy import special

from .likelihood import Prepared


@dataclass
class GDPrior:
    a0: float
    b0: float
    c: np.ndarray
    a: float   # lam ~ Gamma(a, b)
    b: float

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)

    @property
    def c_sum(self):
        return float(self.c.sum())


def gd_prior_from_guess(alpha_guess, lam_guess, sd_frac=0.5, conc=None, lam_sd_frac=0.5):
    """Elicit GD hyper-parameters from prior guesses:
    E(alpha_+) = sum(alpha_guess), sd(alpha_+) = sd_frac * E(alpha_+),
    Dirichlet means proportional to alpha_guess with concentration `conc`
    (default conc = k, which makes the prior dependent because a0 != c_+),
    and lam ~ Gamma with mean lam_guess and sd lam_sd_frac * lam_guess."""
    alpha_guess = np.asarray(alpha_guess, dtype=float)
    k = len(alpha_guess)
    s = alpha_guess.sum()
    a0 = 1.0 / sd_frac ** 2
    b0 = a0 / s
    conc = float(k) if conc is None else conc
    c = conc * alpha_guess / s
    a = 1.0 / lam_sd_frac ** 2
    b = a / lam_guess
    return GDPrior(a0, b0, c, a, b)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _log_g(prep: Prepared, prior: GDPrior, lam):
    """log of the (unnormalised) lam-marginal under independent priors."""
    lam = np.atleast_1d(lam)
    De = prep.Delta(lam)
    return ((prior.a - 1.0 + prep.Dtot) * np.log(lam) - prior.b * lam
            - prep.sumlog1p(lam)
            - np.sum((prep.Di + prior.c)[None, :] * np.log(prior.b0 + De), axis=1))


class LambdaGrid:
    """Inverse-cdf sampler for lam on a fine grid of u = log(lam)."""

    def __init__(self, prep, prior, n_coarse=801, n_fine=3000, width=40.0):
        s = np.log(1.0 / prep.scale)
        u = s + np.linspace(-width / 2, width / 2, n_coarse)
        lu = _log_g(prep, prior, np.exp(u)) + u          # density of u
        keep = np.where(lu > lu.max() - 40.0)[0]
        lo = u[max(keep[0] - 1, 0)]
        hi = u[min(keep[-1] + 1, n_coarse - 1)]
        edges = np.linspace(lo, hi, n_fine + 1)
        mids = 0.5 * (edges[1:] + edges[:-1])
        lmid = _log_g(prep, prior, np.exp(mids)) + mids
        p = np.exp(lmid - lmid.max())
        self.du = edges[1] - edges[0]
        self.p = p / p.sum()
        self.edges = edges
        self.cdf = np.cumsum(self.p)

    def sample(self, M, rng):
        cells = np.searchsorted(self.cdf, rng.uniform(size=M) * self.cdf[-1])
        cells = np.minimum(cells, len(self.p) - 1)
        u = self.edges[cells] + rng.uniform(size=M) * self.du
        lam = np.exp(u)
        log_q_lam = np.log(self.p[cells] / self.du) - u      # density wrt lam
        return lam, log_q_lam


def _log_gamma_pdf(x, shape, rate):
    return (shape * np.log(rate) - special.gammaln(shape)
            + (shape - 1.0) * np.log(x) - rate * x)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
def posterior_unrestricted(prep: Prepared, prior: GDPrior, M, rng, grid=None):
    grid = grid or LambdaGrid(prep, prior)
    lam, log_q_lam = grid.sample(M, rng)
    De = prep.Delta(lam)
    shape = prep.Di + prior.c
    rate = prior.b0 + De
    alpha = rng.gamma(shape[None, :], 1.0 / rate)
    # log g(lam) incl. the Gamma normalising constants that the proposal uses
    log_g = _log_g(prep, prior, lam) + np.sum(special.gammaln(shape))
    logw = (prior.a0 - prior.c_sum) * np.log(alpha.sum(axis=1)) + log_g - log_q_lam
    return {"lam": lam, "alpha": alpha, "logw": logw}


def posterior_order_restricted(prep: Prepared, prior: GDPrior, M, rng, grid=None):
    grid = grid or LambdaGrid(prep, prior)
    lam, log_q_lam = grid.sample(M, rng)
    De = prep.Delta(lam)
    shape = prep.Di + prior.c
    rate = prior.b0 + De
    raw = rng.gamma(shape[None, :], 1.0 / rate)
    a = np.sort(raw, axis=1)                         # ordered draws
    k = prep.k
    perms = list(permutations(range(k)))
    # density of the sorted proposal vector: sum over assignments
    terms = np.empty((M, len(perms)))
    for p_i, perm in enumerate(perms):
        # facility i's gamma evaluated at the sorted value a[:, perm[i]]
        terms[:, p_i] = np.sum(_log_gamma_pdf(a[:, list(perm)], shape[None, :], rate), axis=1)
    log_q_alpha = special.logsumexp(terms, axis=1)
    # OGD prior: sum over permutations of prod_j a_{perm(j)}^(c_j - 1)
    pterms = np.empty((M, len(perms)))
    for p_i, perm in enumerate(perms):
        pterms[:, p_i] = np.sum((prior.c[None, :] - 1.0) * np.log(a[:, list(perm)]), axis=1)
    log_prior_alpha = (special.logsumexp(pterms, axis=1) - prior.b0 * a.sum(axis=1)
                       + (prior.a0 - prior.c_sum) * np.log(a.sum(axis=1)))
    log_target = ((prior.a - 1.0 + prep.Dtot) * np.log(lam) - prior.b * lam
                  - prep.sumlog1p(lam)
                  + np.sum(prep.Di[None, :] * np.log(a) - a * De, axis=1)
                  + log_prior_alpha)
    logw = log_target - log_q_lam - log_q_alpha
    return {"lam": lam, "alpha": a, "logw": logw}


# ---------------------------------------------------------------------------
# summaries of weighted samples
# ---------------------------------------------------------------------------
def normalise(logw):
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def ess(w):
    return 1.0 / np.sum(w ** 2)


def wmean(x, w):
    return float(np.sum(w * x))


def linex(x, w, h):
    """Bayes estimate under LINEX loss L(d, theta) = e^{h(d-theta)} - h(d-theta) - 1."""
    m = -h * x
    mx = np.max(m)
    return float(-(mx + np.log(np.sum(w * np.exp(m - mx)))) / h)


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(np.interp(q, cw - 0.5 * w[o], x[o]))


def whpd(x, w, level=0.95):
    """Shortest interval with posterior (importance) weight >= level,
    weighted version of Chen & Shao (1999)."""
    o = np.argsort(x)
    xs, cw = x[o], np.cumsum(w[o])
    cw0 = np.r_[0.0, cw[:-1]]                       # weight before index i
    j = np.searchsorted(cw, cw0 + level - 1e-12)    # first j with mass >= level
    valid = j < len(xs)
    if not np.any(valid):
        return float(xs[0]), float(xs[-1])
    i_idx = np.where(valid)[0]
    widths = xs[j[valid]] - xs[i_idx]
    b = int(np.argmin(widths))
    return float(xs[i_idx[b]]), float(xs[j[valid][b]])
