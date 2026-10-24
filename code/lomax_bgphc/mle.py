"""
Maximum likelihood estimation (unrestricted and order restricted) and
asymptotic confidence intervals (observed information + delta method).
"""
import numpy as np
from scipy import optimize, stats

from .likelihood import Prepared


def _maximise_1d(fun, u_center, half_width=14.0, n_grid=561):
    """Maximise a scalar function of u = log(lam): coarse grid, then Brent."""
    grid = u_center + np.linspace(-half_width, half_width, n_grid)
    vals = fun(grid)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    i = int(np.argmax(vals))
    on_boundary = i in (0, n_grid - 1)
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    res = optimize.minimize_scalar(lambda u: -fun(np.array([u]))[0],
                                   bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-10})
    return res.x, -res.fun, on_boundary


# ---------------------------------------------------------------------------
# unrestricted MLE
# ---------------------------------------------------------------------------
def mle(prep: Prepared):
    u_hat, ll, boundary = _maximise_1d(lambda u: prep.profile(np.exp(u)),
                                       np.log(1.0 / prep.scale))
    lam = float(np.exp(u_hat))
    alpha = prep.Di / prep.Delta(lam)[0]
    cond, _ = prep.existence_condition()
    return {"alpha": alpha, "lam": lam, "loglik": ll,
            "exists": (not boundary), "existence_condition": cond}


# ---------------------------------------------------------------------------
# order-restricted MLE  (alpha_1 <= alpha_2 <= ... <= alpha_k)
# ---------------------------------------------------------------------------
def pava_increasing(num, den):
    """Maximiser of sum_i [num_i log a_i - a_i den_i] subject to
    a_1 <= ... <= a_k.  Pool-adjacent-violators with pooled value
    sum(num)/sum(den); identical to the min-max formula of Sultana et al."""
    blocks = []                                   # [sum_num, sum_den, size]
    for a, b in zip(num, den):
        blocks.append([a, b, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a2, b2, s2 = blocks.pop()
            blocks[-1][0] += a2
            blocks[-1][1] += b2
            blocks[-1][2] += s2
    out = []
    for a, b, s in blocks:
        out += [a / b] * s
    return np.array(out)


def minmax_formula(num, den):
    """Direct min-max (isotonic) formula, used only to cross-check PAVA:
    1/alpha_u = min_{s<=u} max_{t>=u} sum_{s..t} den / sum_{s..t} num."""
    k = len(num)
    out = np.empty(k)
    for u in range(k):
        best = np.inf
        for s in range(u + 1):
            worst = -np.inf
            for t in range(u, k):
                worst = max(worst, den[s:t + 1].sum() / num[s:t + 1].sum())
            best = min(best, worst)
        out[u] = 1.0 / best
    return out


def or_alpha_given_lam(prep: Prepared, lam):
    return pava_increasing(prep.Di, prep.Delta(lam)[0])


def or_profile(prep: Prepared, lam_arr):
    lam_arr = np.atleast_1d(lam_arr)
    De = prep.Delta(lam_arr)
    out = np.empty(len(lam_arr))
    for r, lam in enumerate(lam_arr):
        a = pava_increasing(prep.Di, De[r])
        out[r] = (np.sum(prep.Di * np.log(a)) + prep.Dtot * np.log(lam)
                  - np.sum(np.log1p(lam * prep.x)) - np.sum(a * De[r]))
    return out


def or_mle(prep: Prepared):
    u_hat, ll, boundary = _maximise_1d(lambda u: or_profile(prep, np.exp(u)),
                                       np.log(1.0 / prep.scale), n_grid=281)
    lam = float(np.exp(u_hat))
    return {"alpha": or_alpha_given_lam(prep, lam), "lam": lam,
            "loglik": ll, "exists": (not boundary)}


# ---------------------------------------------------------------------------
# asymptotic confidence intervals
# ---------------------------------------------------------------------------
def numerical_gradient(g, theta, eps=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.empty_like(theta)
    for j in range(len(theta)):
        h = eps * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        grad[j] = (g(tp) - g(tm)) / (2 * h)
    return grad


def aci(est, se, level=0.95, kind="normal"):
    """Asymptotic CI.
    kind='normal' : est -/+ z*se, lower limit truncated at 0 (as in Lodhi et al.)
    kind='prob'   : est -/+ z*se truncated to [0, 1] (for reliabilities)
    kind='log'    : delta method on the log scale (as in Sultana et al.)
    kind='logit'  : delta method on the logit scale."""
    z = stats.norm.ppf(0.5 + level / 2)
    if kind == "normal":
        return max(est - z * se, 0.0), est + z * se
    if kind == "prob":
        return max(est - z * se, 0.0), min(est + z * se, 1.0)
    if kind == "log":
        f = np.exp(z * se / est)
        return est / f, est * f
    if kind == "logit":
        lo_ = np.log(est / (1 - est))
        s = se / (est * (1 - est))
        return 1 / (1 + np.exp(-(lo_ - z * s))), 1 / (1 + np.exp(-(lo_ + z * s)))
    return est - z * se, est + z * se


def mle_with_intervals(prep: Prepared, functionals, level=0.95):
    """functionals: dict name -> (g(theta) where theta=(alpha..., lam), kind).
    Returns MLE fit plus estimate / se / ACI of every functional.
    (Convenience helper; estimators.analyse does the same for all methods.)"""
    fit = mle(prep)
    theta = np.r_[fit["alpha"], fit["lam"]]
    info = prep.observed_information(fit["alpha"], fit["lam"])
    try:
        cov = np.linalg.inv(info)
        ok = np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0)
    except np.linalg.LinAlgError:
        cov, ok = np.full_like(info, np.nan), False
    out = {}
    for name, (g, kind) in functionals.items():
        est = float(g(theta))
        if ok:
            grad = numerical_gradient(g, theta)
            var = float(grad @ cov @ grad)
            se = np.sqrt(max(var, 0.0))
            lo, hi = aci(est, se, level, kind) if se > 0 else (np.nan, np.nan)
        else:
            se, lo, hi = np.nan, np.nan, np.nan
        out[name] = {"est": est, "se": se, "lo": lo, "hi": hi}
    fit["cov"], fit["info_ok"] = cov, ok
    return fit, out
