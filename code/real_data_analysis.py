"""
Real data analysis: fatalities in British coal-mining disasters
(Table 14 of Lodhi, Gangopadhyay & Bhattacharya, 2024; original source
Hand et al., 1993 / Jarrett, 1979).  Three 20-year periods are treated as
three "test facilities" (blocks).

The tabulated values (scaled fatalities) are recorded in steps of 0.02 and
start at 0.20, the recording threshold (only disasters with 10 or more deaths
are listed).  Following the peaks-over-threshold idea, we model the EXCESS over
the threshold with a half-step continuity correction:

        y = x - 0.19        (so the smallest excess is 0.01 > 0).

Model: y_ij ~ Lomax(alpha_i, lam), i = 1, 2, 3 (common lam, period-specific
shape).  Order restriction: alpha_1 <= alpha_2 <= alpha_3, i.e. the tail of the
excess-fatality distribution became lighter over the century (mine safety
improved), which is the natural prior knowledge for this application.

Usage:  python real_data_analysis.py
Output: ../results/real_data/*.csv and ../results/figures/real_*.png
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lomax_bgphc import bayes, censoring, estimators, model, plotstyle  # noqa: E402
from lomax_bgphc import mle as mlem  # noqa: E402
from lomax_bgphc.likelihood import Prepared  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "real_data")
FIG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
plotstyle.apply()

PERIODS = ["1900-1919", "1920-1939", "1940-1959"]
RAW = [
    [0.20, 0.22, 0.24, 0.26, 0.28, 0.32, 0.50, 0.54, 0.54, 0.66, 0.70, 0.78, 1.04, 1.50,
     1.62, 1.76, 2.38, 2.72, 3.36, 6.88, 8.78],
    [0.20, 0.20, 0.22, 0.22, 0.24, 0.26, 0.26, 0.28, 0.28, 0.38, 0.54, 0.54, 0.54, 0.60,
     0.78, 0.90, 1.04, 1.16, 1.58, 5.30],
    [0.20, 0.22, 0.24, 0.24, 0.24, 0.26, 0.30, 0.30, 0.32, 0.34, 0.38, 0.42, 0.44, 0.90,
     1.14, 1.66, 2.08],
]
SHIFT = 0.19
Y = [np.round(np.array(r) - SHIFT, 2) for r in RAW]
N = [len(y) for y in Y]
T0 = 0.5            # "mission time": an excess of 0.5 units over the threshold
N_PIV, M_BAYES = 10000, 20000
SEED = 2025

# censoring designs applied to the complete data
DESIGNS = {
    "Complete": dict(m=N, kappa=[n - 1 for n in N], T=1e9, R="none"),
    "Scheme 1": dict(m=[17, 16, 14], kappa=[12, 11, 10], T=0.5, R="end"),
    "Scheme 2": dict(m=[17, 16, 14], kappa=[12, 11, 10], T=1.0, R="first1"),
}


def removal(n, m, kind):
    R = np.zeros(m, dtype=int)
    if kind == "end":
        R[-1] = n - m
    elif kind == "first1":
        R[0] = 1
        R[-1] = n - m - 1
    return R


def build(design_name, rng):
    d = DESIGNS[design_name]
    fac = []
    for i in range(3):
        plan = censoring.GPHCPlan(N[i], d["m"][i], d["kappa"][i], d["T"],
                                  removal(N[i], d["m"][i], d["R"]))
        f = censoring.censor_complete_sample(Y[i], plan, rng)
        f.check()
        fac.append(f)
    return censoring.BlockData(fac)


# ---------------------------------------------------------------------------
# helpers for model comparison on complete data
# ---------------------------------------------------------------------------
def single_block(y):
    n = len(y)
    plan = censoring.GPHCPlan(n, n, n - 1, 1e9, np.zeros(n, int))
    return censoring.BlockData([censoring.censor_complete_sample(y, plan, np.random.default_rng(0))])


def fit_equal_alpha(prep):
    """Common alpha AND common lam under block censoring."""
    def prof(lam):
        lam = np.atleast_1d(lam)
        De = prep.Delta(lam).sum(axis=1)
        return (prep.Dtot * np.log(prep.Dtot / De) - prep.Dtot + prep.Dtot * np.log(lam)
                - prep.sumlog1p(lam))
    u, ll, _ = mlem._maximise_1d(lambda u: prof(np.exp(u)), np.log(1 / prep.scale))
    lam = np.exp(u)
    return {"alpha": prep.Dtot / prep.Delta(lam).sum(), "lam": lam, "loglik": ll}


def exp_loglik(y):
    rate = len(y) / np.sum(y)
    return len(y) * np.log(rate) - rate * np.sum(y), rate


def fmt(v, d=4):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------
def descriptive():
    rows = []
    for p, y, r in zip(PERIODS, Y, RAW):
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        rows.append(dict(Period=p, n=len(y), Min=y.min(), Q1=q1, Median=med, Mean=y.mean(),
                         Q3=q3, Max=y.max(), SD=y.std(ddof=1), CV=y.std(ddof=1) / y.mean(),
                         Skewness=stats.skew(y, bias=False)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "descriptive_statistics.csv"), index=False)
    pd.DataFrame({"Period": sum([[p] * len(r) for p, r in zip(PERIODS, RAW)], []),
                  "x_original": sum([list(r) for r in RAW], []),
                  "y_excess": sum([list(y) for y in Y], [])}).to_csv(
        os.path.join(OUT, "coal_mining_data.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    bp = ax.boxplot(Y, vert=False, widths=0.5, patch_artist=True, tick_labels=PERIODS,
                    showfliers=False,
                    medianprops=dict(color=plotstyle.INK, linewidth=1.5))
    for patch, c in zip(bp["boxes"], plotstyle.FACILITY_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for i, (y, c) in enumerate(zip(Y, plotstyle.FACILITY_COLORS)):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(y))
        ax.scatter(y, np.full(len(y), i + 1) + jitter, s=14, color=c, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("excess fatalities y (log scale)")
    ax.set_title("Box plots by period")
    ax = axes[1]
    grid = np.logspace(-2, 1, 200)
    for y, p, c, ls in zip(Y, PERIODS, plotstyle.FACILITY_COLORS, plotstyle.FACILITY_STYLES):
        ys = np.sort(y)
        ax.step(ys, 1 - np.arange(1, len(ys) + 1) / len(ys) + 1e-9, where="post", color=c,
                linestyle=ls, label=p)
    ax.set_xscale("log")
    ax.set_xlabel("y (log scale)")
    ax.set_ylabel("empirical survival  P(Y > y)")
    ax.set_title("Empirical survival functions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_descriptive.png"))
    plt.close(fig)
    return df


def goodness_of_fit():
    rows = []
    sep_ll = 0.0
    for p, y in zip(PERIODS, Y):
        prep = Prepared(single_block(y))
        f = mlem.mle(prep)
        a, lam = f["alpha"][0], f["lam"]
        ks = stats.kstest(y, lambda t: model.cdf(t, a, lam))
        ll_exp, rate = exp_loglik(y)
        sep_ll += f["loglik"]
        rows.append(dict(Period=p, Model="Lomax (separate lam)", alpha=a, lam=lam,
                         loglik=f["loglik"], AIC=2 * 2 - 2 * f["loglik"], KS_D=ks.statistic,
                         KS_p=ks.pvalue))
        ks_e = stats.kstest(y, lambda t: 1 - np.exp(-rate * t))
        rows.append(dict(Period=p, Model="Exponential", alpha=np.nan, lam=rate, loglik=ll_exp,
                         AIC=2 - 2 * ll_exp, KS_D=ks_e.statistic, KS_p=ks_e.pvalue))
    full = censoring.BlockData([single_block(y).facilities[0] for y in Y])
    prep = Prepared(full)
    f = mlem.mle(prep)
    for i, (p, y) in enumerate(zip(PERIODS, Y)):
        ks = stats.kstest(y, lambda t: model.cdf(t, f["alpha"][i], f["lam"]))
        rows.append(dict(Period=p, Model="Lomax (common lam)", alpha=f["alpha"][i], lam=f["lam"],
                         loglik=np.nan, AIC=np.nan, KS_D=ks.statistic, KS_p=ks.pvalue))
    gof = pd.DataFrame(rows)
    gof.to_csv(os.path.join(OUT, "goodness_of_fit.csv"), index=False)

    eq = fit_equal_alpha(prep)
    lrt = []
    s1 = 2 * (sep_ll - f["loglik"])
    lrt.append(dict(Test="H0: lam_1 = lam_2 = lam_3 (common scale)", Data="Complete",
                    Statistic=s1, df=2, p_value=stats.chi2.sf(s1, 2)))
    s2 = 2 * (f["loglik"] - eq["loglik"])
    lrt.append(dict(Test="H0: alpha_1 = alpha_2 = alpha_3 (no period effect)", Data="Complete",
                    Statistic=s2, df=2, p_value=stats.chi2.sf(s2, 2)))
    return gof, lrt, f


def gof_figure(fit_common):
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for i, (p, y, c) in enumerate(zip(PERIODS, Y, plotstyle.FACILITY_COLORS)):
        a, lam = fit_common["alpha"][i], fit_common["lam"]
        ys = np.sort(y)
        n = len(ys)
        ecdf = np.arange(1, n + 1) / n
        pp = (np.arange(1, n + 1) - 0.5) / n
        tgrid = np.linspace(0, ys.max() * 1.05, 400)
        ax = axes[i, 0]
        ax.step(ys, ecdf, where="post", color=plotstyle.INK2, label="empirical")
        ax.plot(tgrid, model.cdf(tgrid, a, lam), color=c, label="fitted Lomax")
        ax.set_title(f"CDF  ({p})")
        ax.set_xlabel("y")
        ax.legend(loc="lower right")
        ax = axes[i, 1]
        ax.plot(model.cdf(ys, a, lam), pp, "o", color=c, markersize=5)
        ax.plot([0, 1], [0, 1], color=plotstyle.INK2, linewidth=1)
        ax.set_title(f"P-P plot  ({p})")
        ax.set_xlabel("fitted CDF")
        ax.set_ylabel("empirical CDF")
        ax = axes[i, 2]
        qth = model.quantile(pp, a, lam)
        ax.plot(qth, ys, "o", color=c, markersize=5)
        lim = [0, max(qth.max(), ys.max()) * 1.05]
        ax.plot(lim, lim, color=plotstyle.INK2, linewidth=1)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"Q-Q plot  ({p})")
        ax.set_xlabel("fitted quantile")
        ax.set_ylabel("sample quantile")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_gof.png"))
    plt.close(fig)


def censored_listing(datasets):
    rows = []
    for name, d in datasets.items():
        if name == "Complete":
            continue
        for p, f in zip(PERIODS, d.facilities):
            rows.append(dict(Design=name, Period=p, Case=["I", "II", "III"][f.case - 1], D=f.D,
                             T=f.T, RT=f.RT,
                             failures=" ".join(f"{v:.2f}" for v in f.x),
                             removals=" ".join(str(int(r)) for r in f.R)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "censored_datasets.csv"), index=False)
    return df


def profile_figure(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 3.4))
    for ax, (name, d) in zip(axes, datasets.items()):
        prep = Prepared(d)
        f = mlem.mle(prep)
        lam = np.linspace(0.2, 4 * f["lam"], 400)
        ax.plot(lam, prep.profile(lam), color=plotstyle.METHOD_COLORS["MLE"])
        ax.axvline(f["lam"], color=plotstyle.INK2, linestyle=":", linewidth=1.2)
        ax.annotate(f"MLE = {f['lam']:.3f}", (f["lam"], prep.profile(f["lam"])[0]),
                    textcoords="offset points", xytext=(8, -14), color=plotstyle.INK, fontsize=9)
        ax.set_title(name)
        ax.set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"profile log-likelihood  $\ell_p(\lambda)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_profile_loglik.png"))
    plt.close(fig)


METHOD_ORDER = [("MLE", "MLE", "ACI"), ("PIV", "Pivotal", "GCI"),
                ("OR_MLE", "OR-MLE", None),
                ("BU_SE", "Bayes-U (SELF)", "HPD"), ("BU_LX", "Bayes-U (LINEX)", None),
                ("BOR_SE", "Bayes-OR (SELF)", "HPD"), ("BOR_LX", "Bayes-OR (LINEX)", None)]
QUANT_ORDER = ["alpha1", "alpha2", "alpha3", "alpha_pool", "lam",
               "R1", "R2", "R3", "h1", "h2", "h3", "MdTF1", "MdTF2", "MdTF3"]


def estimates_table(name, res):
    rows = []
    for q in QUANT_ORDER:
        for key, label, ik in METHOD_ORDER:
            v = res[key][q]
            rows.append(dict(Design=name, Quantity=q, Method=label, Estimate=v["est"],
                             Lower=v["lo"] if ik else np.nan, Upper=v["hi"] if ik else np.nan,
                             Length=(v["hi"] - v["lo"]) if ik else np.nan,
                             Interval=ik or ""))
    return pd.DataFrame(rows)


def curves_figure(datasets_res, name):
    res, data = datasets_res
    t = np.linspace(0, 3, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (p, c, ls) in enumerate(zip(PERIODS, plotstyle.FACILITY_COLORS, plotstyle.FACILITY_STYLES)):
        a_m, l_m = res["_mle"]["alpha"][i], res["_mle"]["lam"]
        pb, wb = res["_bor_draws"]
        R_b = np.array([np.sum(wb * model.reliability(tt, pb["alpha"][:, i], pb["lam"])) for tt in t])
        h_b = np.array([np.sum(wb * model.hazard(tt, pb["alpha"][:, i], pb["lam"])) for tt in t])
        axes[0].plot(t, model.reliability(t, a_m, l_m), color=c, linestyle=ls, linewidth=1.3, alpha=0.6)
        axes[0].plot(t, R_b, color=c, linestyle=ls, label=p)
        axes[1].plot(t, model.hazard(t, a_m, l_m), color=c, linestyle=ls, linewidth=1.3, alpha=0.6)
        axes[1].plot(t, h_b, color=c, linestyle=ls, label=p)
    axes[0].set_title(f"Reliability R(t) - {name}")
    axes[0].set_xlabel("t (excess fatalities)")
    axes[0].set_ylabel("R(t) = P(Y > t)")
    axes[1].set_title(f"Hazard rate h(t) - {name}")
    axes[1].set_xlabel("t (excess fatalities)")
    axes[1].set_ylabel("h(t)")
    axes[0].legend(title="thick: Bayes-OR, thin: MLE", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"real_curves_{name.replace(' ', '').lower()}.png"))
    plt.close(fig)


def posterior_figure(res, name):
    pu, wu = res["_bu_draws"]
    po, wo = res["_bor_draws"]
    piv = res["_piv_draws"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    labels = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\lambda$"]
    for j, ax in enumerate(axes):
        get = (lambda d: d["alpha"][:, j]) if j < 3 else (lambda d: d["lam"])
        xs = [get(pu), get(po), get(piv)]
        hi = np.quantile(np.concatenate(xs), 0.99)
        grid = np.linspace(0, hi, 300)
        grid = grid[1:]
        for x, w, key, lab in [(xs[0], wu, "BU", "Bayes-U"), (xs[1], wo, "BOR", "Bayes-OR"),
                               (xs[2], None, "PIV", "Pivotal GPQ")]:
            # density estimated on the log scale (robust to heavy right tails),
            # then transformed back: f_X(x) = f_logX(log x) / x
            kde = stats.gaussian_kde(np.log(x), weights=w)
            ax.plot(grid, kde(np.log(grid)) / grid, color=plotstyle.METHOD_COLORS[key],
                    linestyle=plotstyle.METHOD_STYLES[key], label=lab)
        ax.axvline(res["MLE"][["alpha1", "alpha2", "alpha3", "lam"][j]]["est"],
                   color=plotstyle.INK2, linestyle=":", linewidth=1.2, label="MLE")
        ax.set_title(labels[j])
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Posterior densities and pivotal (GPQ) distributions - {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"real_posteriors_{name.replace(' ', '').lower()}.png"))
    plt.close(fig)


def censored_lrt(datasets):
    out = []
    for name, d in datasets.items():
        prep = Prepared(d)
        f = mlem.mle(prep)
        eq = fit_equal_alpha(prep)
        s = 2 * (f["loglik"] - eq["loglik"])
        if name != "Complete":
            out.append(dict(Test="H0: alpha_1 = alpha_2 = alpha_3 (no period effect)", Data=name,
                            Statistic=s, df=2, p_value=stats.chi2.sf(s, 2)))
    return out


def prior_sensitivity(datasets):
    rows = []
    rng = np.random.default_rng(SEED + 7)
    for name in ["Scheme 1", "Scheme 2"]:
        d = datasets[name]
        f = mlem.mle(Prepared(d))
        for sd in [0.5, 1.0, 2.0]:
            pri = bayes.gd_prior_from_guess(f["alpha"], f["lam"], sd_frac=sd, lam_sd_frac=sd)
            res = estimators.analyse(d, {"": pri}, T0, rng, M_bayes=M_BAYES, do_pivotal=False)
            for q in ["alpha1", "alpha2", "alpha3", "lam", "R1", "R2", "R3"]:
                for key in ["BU_SE", "BOR_SE"]:
                    v = res[key][q]
                    rows.append(dict(Design=name, prior_sd_fraction=sd, Method=key, Quantity=q,
                                     Estimate=v["est"], Lower=v["lo"], Upper=v["hi"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "prior_sensitivity.csv"), index=False)
    return df


def main():
    rng = np.random.default_rng(SEED)
    desc = descriptive()
    gof, lrt, fit_common = goodness_of_fit()
    gof_figure(fit_common)
    datasets = {name: build(name, rng) for name in DESIGNS}
    listing = censored_listing(datasets)
    profile_figure(datasets)
    lrt += censored_lrt(datasets)
    pd.DataFrame(lrt).to_csv(os.path.join(OUT, "likelihood_ratio_tests.csv"), index=False)

    all_tabs, diags = [], []
    for name, d in datasets.items():
        prep = Prepared(d)
        f = mlem.mle(prep)
        prior = bayes.gd_prior_from_guess(f["alpha"], f["lam"], sd_frac=1.0, lam_sd_frac=1.0)
        res = estimators.analyse(d, {"": prior}, T0, np.random.default_rng(SEED + len(all_tabs)),
                                 N_piv=N_PIV, M_bayes=M_BAYES, keep_draws=True)
        all_tabs.append(estimates_table(name, res))
        diags.append(dict(Design=name, cases=" ".join(["I", "II", "III"][c - 1] for c in d.cases),
                          D=" ".join(map(str, d.D)), **{k: v for k, v in res["_diag"].items()},
                          prior=f"a0={prior.a0:.3f}, b0={prior.b0:.3f}, c={np.round(prior.c, 3).tolist()}, "
                                f"lam~Gamma({prior.a:.3f},{prior.b:.3f})"))
        curves_figure((res, d), name)
        posterior_figure(res, name)
    est = pd.concat(all_tabs)
    est.to_csv(os.path.join(OUT, "estimates_all_methods.csv"), index=False)
    pd.DataFrame(diags).to_csv(os.path.join(OUT, "diagnostics.csv"), index=False)
    sens = prior_sensitivity(datasets)

    pd.set_option("display.width", 200)
    print(desc.round(4).to_string(index=False))
    print(gof.round(4).to_string(index=False))
    print(pd.DataFrame(lrt).round(4).to_string(index=False))
    print(listing.to_string(index=False))
    print(pd.DataFrame(diags).to_string(index=False))
    for name in DESIGNS:
        sub = est[(est.Design == name) & est.Quantity.isin(["alpha1", "alpha2", "alpha3", "lam", "R1"])]
        print("\n==", name)
        print(sub.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
