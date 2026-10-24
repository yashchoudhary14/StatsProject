"""
Turn the raw simulation output into report-ready tables (CSV, Excel, Markdown
snippets) and figures.

Usage:  python make_sim_outputs.py      (after run_simulation.py)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lomax_bgphc import plotstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SIM = os.path.join(HERE, "..", "results", "simulation")
FIG = os.path.join(HERE, "..", "results", "figures")
MD = os.path.join(HERE, "..", "results", "report_tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MD, exist_ok=True)
plotstyle.apply()

S = pd.read_csv(os.path.join(SIM, "sim_summary.csv"))
DG = pd.read_csv(os.path.join(SIM, "sim_diagnostics.csv"))
DESIGNS = list(dict.fromkeys(S["design"]))
QLAB = {"alpha1": "α₁", "alpha2": "α₂", "alpha3": "α₃", "lam": "λ",
        "R1": "R₁(1)", "R2": "R₂(1)", "R3": "R₃(1)"}


def design_label(d):
    s, t, sch = d.split("-")
    return f"{s}, T={t[1:]}, {sch}"


def md_table(df, floatfmt="{:.4f}", index_name="Design"):
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join([":---"] + ["---:"] * len(df.columns)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("–" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save(name, df, floatfmt="{:.4f}"):
    df.to_csv(os.path.join(SIM, "tables", f"{name}.csv"))
    with open(os.path.join(MD, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md_table(df, floatfmt))
    return df


os.makedirs(os.path.join(SIM, "tables"), exist_ok=True)


def pivot(quantity, metric, methods, names):
    t = S[S.quantity == quantity].pivot(index="design", columns="method", values=metric)
    t = t.loc[DESIGNS, methods]
    t.columns = names
    t.index = [design_label(d) for d in t.index]
    return t


POINT = ["MLE", "PIV", "OR_MLE", "BU_SE_P1", "BOR_SE_P1", "BU_SE_P2", "BOR_SE_P2"]
POINT_N = ["MLE", "PIV", "OR-MLE", "BU-P1", "BOR-P1", "BU-P2", "BOR-P2"]
tables = {}

# --- design / diagnostics table ------------------------------------------
dg = DG.set_index("design").loc[DESIGNS]
diag = pd.DataFrame({
    "Case I %": 100 * dg.caseI, "Case II %": 100 * dg.caseII, "Case III %": 100 * dg.caseIII,
    "Avg failures D": dg.mean_failures,
    "MLE order violated %": 100 * dg.mle_order_violation,
    "Discarded": dg.discarded,
    "Median ESS (BOR-P2)": dg.ess_bor_p2_median,
})
diag.index = [design_label(d) for d in diag.index]
tables["sim_design_diagnostics"] = save("sim_design_diagnostics", diag, "{:.1f}")

# --- MSE / bias / MedAE tables ---------------------------------------------
for q in ["alpha1", "alpha2", "alpha3", "lam", "R1", "R2", "R3"]:
    fmt = "{:.4f}" if q.startswith("R") or q == "lam" else "{:.4f}"
    tables[f"sim_mse_{q}"] = save(f"sim_mse_{q}", pivot(q, "mse", POINT, POINT_N), fmt)
    tables[f"sim_bias_{q}"] = save(f"sim_bias_{q}", pivot(q, "bias", POINT, POINT_N), fmt)

med = pd.concat([pivot(q, "medae", ["MLE", "PIV", "BOR_SE_P2"], ["MLE", "PIV", "BOR"])
                 for q in ["alpha1", "alpha3", "lam"]], axis=1)
med.columns = [f"{QLAB[q]} {m}" for q in ["alpha1", "alpha3", "lam"] for m in ["MLE", "PIV", "BOR"]]
tables["sim_medae"] = save("sim_medae", med)

# --- interval tables ---------------------------------------------------------
INT = ["MLE", "PIV", "BU_SE_P2", "BOR_SE_P2"]
INT_N = ["ACI", "GCI", "HPD-U", "HPD-OR"]
for q in ["alpha1", "alpha2", "alpha3", "lam", "R1", "R3"]:
    al = pivot(q, "al", INT, [f"AL {n}" for n in INT_N])
    cp = pivot(q, "cp", INT, [f"CP {n}" for n in INT_N])
    tables[f"sim_int_{q}"] = save(f"sim_int_{q}", pd.concat([al, cp], axis=1), "{:.3f}")
INT1 = ["BU_SE_P1", "BOR_SE_P1"]
p1 = pd.concat([pd.concat([pivot(q, "al", INT1, [f"{QLAB[q]} AL U", f"{QLAB[q]} AL OR"]),
                           pivot(q, "cp", INT1, [f"{QLAB[q]} CP U", f"{QLAB[q]} CP OR"])], axis=1)
                for q in ["alpha1", "lam"]], axis=1)
tables["sim_int_prior1"] = save("sim_int_prior1", p1, "{:.3f}")

# --- SELF vs LINEX ------------------------------------------------------------
lx = pd.concat([pivot(q, "mse", ["BU_SE_P2", "BU_LX_P2", "BOR_SE_P2", "BOR_LX_P2"],
                      [f"{QLAB[q]} U-SE", f"{QLAB[q]} U-LX", f"{QLAB[q]} OR-SE", f"{QLAB[q]} OR-LX"])
                for q in ["alpha3", "lam"]], axis=1)
tables["sim_linex"] = save("sim_linex", lx)

# --- pivotal: median vs mean --------------------------------------------------
pm = pd.concat([pivot(q, "mse", ["PIV", "PIV_MEAN"], [f"{QLAB[q]} median", f"{QLAB[q]} mean"])
                for q in ["alpha1", "lam"]], axis=1)
tables["sim_piv_median_vs_mean"] = save("sim_piv_median_vs_mean", pm, "{:.3f}")

with pd.ExcelWriter(os.path.join(SIM, "simulation_tables.xlsx")) as xw:
    readme = pd.DataFrame({"Sheet": list(tables), "Note": ""})
    readme.loc[len(readme)] = ["sim_summary (full)", "every design x method x quantity: bias, MSE, MedAE, AL, CP"]
    readme.to_excel(xw, sheet_name="README", index=False)
    for name, t in tables.items():
        t.to_excel(xw, sheet_name=name[:31])
    S.to_excel(xw, sheet_name="sim_summary (full)", index=False)

# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
SIZES = {"S1": 20, "S2": 30, "S3": 40}
FIGMETH = [("MLE", "MLE"), ("PIV", "PIV"), ("OR_MLE", "OR_MLE"), ("BU_SE_P2", "BU"), ("BOR_SE_P2", "BOR")]


def line_panel(ax, q, metric, T, scheme, methods, logy=False, ref=None):
    for meth, key in methods:
        ys = [S[(S.design == f"{s}-T{T:g}-{scheme}") & (S.method == meth) & (S.quantity == q)][metric].iloc[0]
              for s in SIZES]
        ax.plot(list(SIZES.values()), ys, color=plotstyle.METHOD_COLORS[key],
                linestyle=plotstyle.METHOD_STYLES[key], marker=plotstyle.METHOD_MARKERS[key],
                label=plotstyle.METHOD_LABELS[key])
    if logy:
        ax.set_yscale("log")
    if ref is not None:
        ax.axhline(ref, color=plotstyle.INK2, linewidth=1, linestyle=":")
    ax.set_xticks(list(SIZES.values()))
    ax.set_xlabel(r"units per facility $n_i$")


for T in (3.0, 8.0):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for j, q in enumerate(["alpha1", "alpha3", "lam", "R1"]):
        line_panel(axes[0, j], q, "mse", T, "A", FIGMETH, logy=True)
        line_panel(axes[1, j], q, "medae", T, "A", FIGMETH, logy=True)
        axes[0, j].set_title(QLAB[q] if q != "R1" else "R₁(t₀=1)")
        axes[0, j].set_xlabel("")
    axes[0, 0].set_ylabel("MSE (log scale)")
    axes[1, 0].set_ylabel("median absolute error (log)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Accuracy vs sample size (T = {T:g}, scheme A, prior P2 for Bayes): "
                 "MSE (top) and median absolute error (bottom)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"sim_mse_T{T:g}.png"))
    plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
INTM = [("MLE", "MLE"), ("PIV", "PIV"), ("BU_SE_P2", "BU"), ("BOR_SE_P2", "BOR")]
for ax, q in zip(axes, ["alpha1", "alpha3", "lam", "R1"]):
    line_panel(ax, q, "cp", 8.0, "A", INTM, ref=0.95)
    ax.set_ylim(0.78, 1.01)
    ax.set_title(QLAB[q] if q != "R1" else "R₁(t₀=1)")
axes[0].set_ylabel("coverage probability")
axes[0].legend(fontsize=8, loc="lower right",
               labels=["ACI (MLE)", "GCI (pivotal)", "HPD (Bayes-U)", "HPD (Bayes-OR)"])
fig.suptitle("Coverage of 95% intervals vs sample size (T = 8, scheme A); dotted line = 0.95", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_coverage_T8.png"))
plt.close(fig)

# box plots of the raw estimates of alpha_1 (shows the heavy MLE tail)
raw = np.load(os.path.join(SIM, "sim_raw_estimates.npz"), allow_pickle=True)
est, dnames, mnames, qnames = raw["est"], list(raw["designs"]), list(raw["methods"]), list(raw["quantities"])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, d in zip(axes, ["S1-T3-A", "S2-T3-A", "S3-T8-A"]):
    di = dnames.index(d)
    data, labels, cols = [], [], []
    for meth, key in FIGMETH:
        x = est[di, :, mnames.index(meth), qnames.index("alpha1")]
        data.append(x)
        labels.append({"MLE": "MLE", "PIV": "PIV", "OR_MLE": "OR-MLE", "BU": "BU", "BOR": "BOR"}[key])
        cols.append(plotstyle.METHOD_COLORS[key])
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=plotstyle.INK, linewidth=1.4))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for j, x in enumerate(data):
        n_out = int(np.sum(x > 3))
        if n_out:
            ax.annotate(f"{n_out} > 3", (j + 1, 2.88), ha="center", fontsize=8, color=plotstyle.INK2)
    ax.axhline(0.5, color=plotstyle.INK2, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(design_label(d))
axes[0].set_ylabel(r"estimate of $\alpha_1$ (true 0.5, dotted)")
fig.suptitle("Sampling distribution of the estimators of α₁ (whiskers 1.5 IQR; number of estimates above 3 shown on top)\n"
             "MLE, PIV = pivotal, OR-MLE = order-restricted MLE, BU / BOR = Bayes unrestricted / order-restricted (P2)",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_boxplots_alpha1.png"))
plt.close(fig)

# effect of the censoring scheme and T (pivotal and Bayes-OR, S2)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, q in zip(axes, ["alpha1", "lam"]):
    xs = np.arange(3)
    width = 0.2
    for j, (T, meth, key) in enumerate([(3.0, "PIV", "PIV"), (8.0, "PIV", "PIV"),
                                        (3.0, "BOR_SE_P2", "BOR"), (8.0, "BOR_SE_P2", "BOR")]):
        ys = [S[(S.design == f"S2-T{T:g}-{sch}") & (S.method == meth) & (S.quantity == q)]["mse"].iloc[0]
              for sch in "ABC"]
        ax.bar(xs + (j - 1.5) * width, ys, width * 0.92, color=plotstyle.METHOD_COLORS[key],
               alpha=0.45 if T == 3.0 else 1.0,
               label=f"{plotstyle.METHOD_LABELS[key]}, T={T:g}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["A: removals at end", "B: at first failure", "C: half / half"], fontsize=8.5)
    ax.set_title(f"MSE of {QLAB[q]}  (S2: n_i = 30)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_scheme_effect.png"))
plt.close(fig)
print("tables:", len(tables), "-> ", os.path.abspath(MD))
