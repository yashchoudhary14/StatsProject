"""
One call that runs every estimation method on a block-GPHC data set.

Method keys in the returned dictionary
  MLE           maximum likelihood, asymptotic CI (observed information + delta method)
  PIV           pivotal generalised estimate (median of the GPQ draws) + generalised CI
  PIV_MEAN      pivotal generalised estimate defined as the mean of the draws
  OR_MLE        order-restricted MLE (alpha_1 <= ... <= alpha_k), point estimate only
  BU_SE_<p>     Bayes, unrestricted Gamma-Dirichlet joint prior <p>, squared error loss, HPD
  BU_LX_<p>     Bayes, unrestricted, LINEX loss (same HPD interval)
  BOR_SE_<p>    Bayes, order-restricted (ordered GD prior), squared error loss, HPD
  BOR_LX_<p>    Bayes, order-restricted, LINEX loss

Quantities: alpha_1..alpha_k, lam, alpha_pool (inverse-variance weighted
alpha, as in Lodhi et al.) and, for each facility, R_i(t0), h_i(t0), MdTF_i.
"""
import numpy as np

from . import bayes, mle as mlem, model, pivotal
from .likelihood import Prepared


def quantity_names(k):
    names = [f"alpha{i+1}" for i in range(k)] + ["lam", "alpha_pool"]
    for i in range(k):
        names += [f"R{i+1}", f"h{i+1}", f"MdTF{i+1}"]
    return names


def _derived(alpha, lam, t0):
    """alpha: (N,k), lam: (N,) -> dict name -> (N,) survival characteristics."""
    out = {}
    for i in range(alpha.shape[1]):
        out[f"R{i+1}"] = model.reliability(t0, alpha[:, i], lam)
        out[f"h{i+1}"] = model.hazard(t0, alpha[:, i], lam)
        out[f"MdTF{i+1}"] = model.mdtf(alpha[:, i], lam)
    return out


def _draws_dict(alpha, lam, t0, w):
    k = alpha.shape[1]
    d = {f"alpha{i+1}": alpha[:, i] for i in range(k)}
    d["lam"] = lam
    mean = alpha.T @ w
    var = ((alpha - mean) ** 2).T @ w
    pw = (1.0 / var) / np.sum(1.0 / var)
    d["alpha_pool"] = alpha @ pw
    d.update(_derived(alpha, lam, t0))
    return d


def summarise_draws(alpha, lam, t0, w=None, level=0.95, linex_h=None, interval="hpd"):
    """Point estimates and intervals from (weighted) draws.
    Returns dicts for: mean (SE loss), median, LINEX."""
    N = len(lam)
    w = np.full(N, 1.0 / N) if w is None else w
    out_mean, out_med, out_lx = {}, {}, {}
    for name, x in _draws_dict(alpha, lam, t0, w).items():
        if interval == "hpd":
            lo, hi = bayes.whpd(x, w, level)
        else:  # equal-tailed percentile interval
            lo = bayes.wquantile(x, w, (1 - level) / 2)
            hi = bayes.wquantile(x, w, (1 + level) / 2)
        out_mean[name] = {"est": bayes.wmean(x, w), "lo": lo, "hi": hi}
        out_med[name] = {"est": bayes.wquantile(x, w, 0.5), "lo": lo, "hi": hi}
        if linex_h is not None:
            out_lx[name] = {"est": bayes.linex(x, w, linex_h), "lo": lo, "hi": hi}
    return out_mean, out_med, out_lx


def analyse(data, priors, t0, rng, N_piv=1000, M_bayes=2000, level=0.95,
            linex_h=1.0, do_pivotal=True, keep_draws=False):
    """priors: dict label -> GDPrior (may be empty)."""
    prep = Prepared(data)
    k = prep.k
    res, diag = {}, {}

    # ---------------- maximum likelihood + ACI -----------------------------
    fit = mlem.mle(prep)
    diag["mle_exists"] = fit["exists"]
    diag["mle_condition"] = bool(fit["existence_condition"])
    theta = np.r_[fit["alpha"], fit["lam"]]
    info = prep.observed_information(fit["alpha"], fit["lam"])
    try:
        cov = np.linalg.inv(info)
        ok = bool(np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0))
    except np.linalg.LinAlgError:
        cov, ok = None, False
    diag["info_ok"] = ok
    if ok:
        pw = (1.0 / np.diag(cov)[:k]) / np.sum(1.0 / np.diag(cov)[:k])
    else:
        pw = prep.Di / prep.Dtot
    funcs = {f"alpha{i+1}": (lambda th, i=i: th[i], "normal") for i in range(k)}
    funcs["lam"] = (lambda th: th[k], "normal")
    funcs["alpha_pool"] = (lambda th: float(th[:k] @ pw), "normal")
    for i in range(k):
        funcs[f"R{i+1}"] = (lambda th, i=i: model.reliability(t0, th[i], th[k]), "prob")
        funcs[f"h{i+1}"] = (lambda th, i=i: model.hazard(t0, th[i], th[k]), "normal")
        funcs[f"MdTF{i+1}"] = (lambda th, i=i: model.mdtf(th[i], th[k]), "normal")
    out = {}
    for name, (g, kind) in funcs.items():
        est = float(g(theta))
        lo = hi = np.nan
        if ok:
            grad = mlem.numerical_gradient(g, theta)
            v = float(grad @ cov @ grad)
            if v > 0:
                lo, hi = mlem.aci(est, np.sqrt(v), level, kind)
        out[name] = {"est": est, "lo": lo, "hi": hi}
    res["MLE"] = out
    res["_mle"] = {"alpha": fit["alpha"], "lam": fit["lam"], "cov": cov, "loglik": fit["loglik"]}

    # ---------------- order-restricted MLE ---------------------------------
    orf = mlem.or_mle(prep)
    out = {f"alpha{i+1}": {"est": float(orf["alpha"][i])} for i in range(k)}
    out["lam"] = {"est": orf["lam"]}
    out["alpha_pool"] = {"est": float(orf["alpha"] @ pw)}
    for name, v in _derived(orf["alpha"][None, :], np.array([orf["lam"]]), t0).items():
        out[name] = {"est": float(v[0])}
    for v in out.values():
        v.setdefault("lo", np.nan)
        v.setdefault("hi", np.nan)
    res["OR_MLE"] = out
    res["_or_mle"] = orf

    # ---------------- pivotal ----------------------------------------------
    if do_pivotal:
        eng = pivotal.PivotalEngine(prep)
        s = eng.sample(N_piv, rng)
        mean_, med_, _ = summarise_draws(s["alpha"], s["lam"], t0, None, level,
                                         interval="equal")
        res["PIV"], res["PIV_MEAN"] = med_, mean_
        diag["piv_trunc_prob"] = float(s["trunc_prob"])
        if keep_draws:
            res["_piv_draws"] = s

    # ---------------- Bayes -------------------------------------------------
    for label, prior in priors.items():
        sfx = f"_{label}" if label else ""
        grid = bayes.LambdaGrid(prep, prior)
        p = bayes.posterior_unrestricted(prep, prior, M_bayes, rng, grid)
        w = bayes.normalise(p["logw"])
        mean_, _, lx = summarise_draws(p["alpha"], p["lam"], t0, w, level, linex_h)
        res["BU_SE" + sfx], res["BU_LX" + sfx] = mean_, lx
        diag["bu_ess" + sfx] = float(bayes.ess(w))
        if keep_draws:
            res["_bu_draws" + sfx] = (p, w)
        p = bayes.posterior_order_restricted(prep, prior, M_bayes, rng, grid)
        w = bayes.normalise(p["logw"])
        mean_, _, lx = summarise_draws(p["alpha"], p["lam"], t0, w, level, linex_h)
        res["BOR_SE" + sfx], res["BOR_LX" + sfx] = mean_, lx
        diag["bor_ess" + sfx] = float(bayes.ess(w))
        if keep_draws:
            res["_bor_draws" + sfx] = (p, w)

    res["_diag"] = diag
    return res
