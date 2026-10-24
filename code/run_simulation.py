"""
Monte Carlo simulation study.

Model      : k = 3 test facilities, X_ij ~ Lomax(alpha_i, lam)
True values: alpha = (0.50, 0.75, 1.00)  (ordered: alpha_1 < alpha_2 < alpha_3), lam = 1
Mission time for reliability: t0 = 1
Designs    : 3 sample-size settings x 2 time limits T x 3 removal schemes = 18
Each design: `--reps` replications (default 2000).  Samples in which the MLE
does not exist (profile likelihood maximised at the boundary lam -> 0) are
discarded and regenerated; the number discarded is reported.

Usage:  python run_simulation.py [--reps 2000] [--workers 22]
Output: ../results/simulation/sim_summary.csv, sim_diagnostics.csv
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lomax_bgphc import bayes, censoring, estimators, model  # noqa: E402

TRUE_ALPHA = np.array([0.50, 0.75, 1.00])
TRUE_LAM = 1.0
T0 = 1.0
K = 3
SIZES = {"S1": (20, 15, 10), "S2": (30, 24, 15), "S3": (40, 32, 20)}   # (n_i, m_i, kappa_i)
T_VALUES = (3.0, 8.0)
SCHEMES = ("A", "B", "C")
QUANTITIES = ["alpha1", "alpha2", "alpha3", "lam", "R1", "R2", "R3"]
METHODS = ["MLE", "PIV", "PIV_MEAN", "OR_MLE",
           "BU_SE_P1", "BU_LX_P1", "BOR_SE_P1", "BOR_LX_P1",
           "BU_SE_P2", "BU_LX_P2", "BOR_SE_P2", "BOR_LX_P2"]
N_PIV = 1000
M_BAYES = 2000
SEED = 20251005


def removal_plan(n, m, scheme):
    R = np.zeros(m, dtype=int)
    if scheme == "A":          # all removals at the m-th failure (Type-II like)
        R[-1] = n - m
    elif scheme == "B":        # all removals at the first failure
        R[0] = n - m
    else:                      # half at the first and half at the m-th failure
        R[0] = (n - m) // 2
        R[-1] = n - m - R[0]
    return R


def designs():
    out = []
    for s_name, (n, m, kap) in SIZES.items():
        for T in T_VALUES:
            for sch in SCHEMES:
                out.append(dict(design=f"{s_name}-T{T:g}-{sch}", size=s_name, n=n, m=m,
                                kappa=kap, T=T, scheme=sch))
    return out


def priors():
    return {
        # P1: informative joint GD prior (prior sd = 50% of the prior mean)
        "P1": bayes.gd_prior_from_guess(TRUE_ALPHA, TRUE_LAM, sd_frac=0.5, lam_sd_frac=0.5),
        # P2: weakly informative (prior sd = 100% of the prior mean)
        "P2": bayes.gd_prior_from_guess(TRUE_ALPHA, TRUE_LAM, sd_frac=1.0, lam_sd_frac=1.0),
    }


def truth():
    t = {f"alpha{i+1}": TRUE_ALPHA[i] for i in range(K)}
    t["lam"] = TRUE_LAM
    for i in range(K):
        t[f"R{i+1}"] = model.reliability(T0, TRUE_ALPHA[i], TRUE_LAM)
    return t


def worker(task):
    d_idx, chunk, n_reps = task
    d = designs()[d_idx]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, d_idx, chunk]))
    plans = [censoring.GPHCPlan(d["n"], d["m"], d["kappa"], d["T"],
                                removal_plan(d["n"], d["m"], d["scheme"])) for _ in range(K)]
    pri = priors()
    out = np.full((n_reps, len(METHODS), len(QUANTITIES), 3), np.nan)
    diag = dict(discarded=0, case_counts=np.zeros(3, int), piv_trunc=[], ess_bu_p1=[],
                ess_bor_p1=[], ess_bu_p2=[], ess_bor_p2=[], order_violation=0, info_fail=0,
                D_total=[])
    r = 0
    while r < n_reps:
        data = censoring.generate_block(plans, TRUE_ALPHA, TRUE_LAM, rng)
        res = estimators.analyse(data, pri, T0, rng, N_piv=N_PIV, M_bayes=M_BAYES)
        if not res["_diag"]["mle_exists"]:
            diag["discarded"] += 1
            continue
        for c in data.cases:
            diag["case_counts"][c - 1] += 1
        diag["piv_trunc"].append(res["_diag"]["piv_trunc_prob"])
        diag["ess_bu_p1"].append(res["_diag"]["bu_ess_P1"])
        diag["ess_bor_p1"].append(res["_diag"]["bor_ess_P1"])
        diag["ess_bu_p2"].append(res["_diag"]["bu_ess_P2"])
        diag["ess_bor_p2"].append(res["_diag"]["bor_ess_P2"])
        diag["info_fail"] += int(not res["_diag"]["info_ok"])
        diag["D_total"].append(int(data.D.sum()))
        a = res["_mle"]["alpha"]
        diag["order_violation"] += int(np.any(np.diff(a) < 0))
        for mi, meth in enumerate(METHODS):
            for qi, q in enumerate(QUANTITIES):
                v = res[meth][q]
                out[r, mi, qi] = (v["est"], v["lo"], v["hi"])
        r += 1
    return d_idx, out, diag


def summarise(all_out, all_diag):
    tr = truth()
    rows, drows = [], []
    for d_idx, d in enumerate(designs()):
        arr = np.concatenate(all_out[d_idx], axis=0)
        for mi, meth in enumerate(METHODS):
            for qi, q in enumerate(QUANTITIES):
                est, lo, hi = arr[:, mi, qi, 0], arr[:, mi, qi, 1], arr[:, mi, qi, 2]
                err = est - tr[q]
                has_int = np.isfinite(lo) & np.isfinite(hi)
                rows.append(dict(
                    design=d["design"], size=d["size"], n=d["n"], m=d["m"], kappa=d["kappa"],
                    T=d["T"], scheme=d["scheme"], method=meth, quantity=q, true=tr[q],
                    mean_est=np.mean(est), median_est=np.median(est), bias=np.mean(err),
                    abs_bias=abs(np.mean(err)), mse=np.mean(err ** 2),
                    medae=np.median(np.abs(err)),
                    al=np.mean(hi[has_int] - lo[has_int]) if has_int.any() else np.nan,
                    cp=np.mean((lo[has_int] <= tr[q]) & (tr[q] <= hi[has_int])) if has_int.any() else np.nan,
                    n_int=int(has_int.sum()), reps=len(est)))
        dg = all_diag[d_idx]
        cases = sum(x["case_counts"] for x in dg)
        cat = lambda key: np.concatenate([np.asarray(x[key], float) for x in dg])
        reps = arr.shape[0]
        drows.append(dict(
            design=d["design"], reps=reps,
            discarded=sum(x["discarded"] for x in dg),
            caseI=cases[0] / cases.sum(), caseII=cases[1] / cases.sum(), caseIII=cases[2] / cases.sum(),
            mean_failures=cat("D_total").mean(),
            mle_order_violation=sum(x["order_violation"] for x in dg) / reps,
            info_fail=sum(x["info_fail"] for x in dg),
            piv_trunc_mean=cat("piv_trunc").mean(),
            ess_bu_p1_median=np.median(cat("ess_bu_p1")), ess_bor_p1_median=np.median(cat("ess_bor_p1")),
            ess_bu_p2_median=np.median(cat("ess_bu_p2")), ess_bor_p2_median=np.median(cat("ess_bor_p2")),
            ess_bor_p1_min=np.min(cat("ess_bor_p1")), ess_bor_p2_min=np.min(cat("ess_bor_p2"))))
    return pd.DataFrame(rows), pd.DataFrame(drows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "simulation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds = designs()
    tasks = []
    for d_idx in range(len(ds)):
        left, chunk = args.reps, 0
        while left > 0:
            nr = min(args.chunk, left)
            tasks.append((d_idx, chunk, nr))
            left -= nr
            chunk += 1
    t0 = time.time()
    all_out = {i: [] for i in range(len(ds))}
    all_diag = {i: [] for i in range(len(ds))}
    with Pool(args.workers) as pool:
        for done, (d_idx, out, diag) in enumerate(pool.imap_unordered(worker, tasks), 1):
            all_out[d_idx].append(out)
            all_diag[d_idx].append(diag)
            if done % 20 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} chunks  ({time.time() - t0:.0f}s)", flush=True)
    summ, diag = summarise(all_out, all_diag)
    summ.to_csv(os.path.join(args.out, "sim_summary.csv"), index=False)
    # raw point estimates (float32) for box plots: shape (designs, reps, methods, quantities)
    raw = np.stack([np.concatenate(all_out[i], axis=0)[:, :, :, 0] for i in range(len(ds))])
    np.savez_compressed(os.path.join(args.out, "sim_raw_estimates.npz"),
                        est=raw.astype(np.float32), designs=[d["design"] for d in ds],
                        methods=METHODS, quantities=QUANTITIES)
    diag.to_csv(os.path.join(args.out, "sim_diagnostics.csv"), index=False)
    with open(os.path.join(args.out, "sim_settings.txt"), "w") as f:
        f.write(f"true alpha = {TRUE_ALPHA.tolist()}, true lam = {TRUE_LAM}, t0 = {T0}\n")
        f.write(f"replications per design = {args.reps}, pivotal draws = {N_PIV}, "
                f"importance-sampling draws = {M_BAYES}, seed = {SEED}\n")
        for lab, p in priors().items():
            f.write(f"prior {lab}: a0={p.a0:.4f}, b0={p.b0:.4f}, c={np.round(p.c, 4).tolist()}, "
                    f"lam ~ Gamma(a={p.a:.4f}, b={p.b:.4f})\n")
        f.write(f"total run time = {time.time() - t0:.0f} s\n")
    print("saved to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
