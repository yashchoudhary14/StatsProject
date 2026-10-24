"""Schematic of block generalized progressive hybrid censoring (three
facilities, one per GPHC case).  Output: ../results/figures/schematic_bgphc.png"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lomax_bgphc import plotstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

plotstyle.apply()
FIG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)

T = 6.0
rows = [
    # (label, failure times, removals at each failure, kappa, m, case text)
    ("Facility 1  (n₁ = 10, κ₁ = 4, m₁ = 6)", [1.2, 2.5, 4.4, 7.0], [1, 0, 0, 0], 4, 6,
     "Case I: T < X(κ) → stop at X(κ) = 4th failure"),
    ("Facility 2  (n₂ = 10, κ₂ = 4, m₂ = 6)", [0.8, 1.9, 3.1, 4.0, 5.2], [1, 0, 0, 0, 0], 4, 6,
     "Case II: X(κ) ≤ T < X(m) → stop at T"),
    ("Facility 3  (n₃ = 10, κ₃ = 4, m₃ = 6)", [0.6, 1.4, 2.2, 3.0, 4.1, 5.0], [1, 0, 0, 0, 0, 3], 4, 6,
     "Case III: X(m) ≤ T → stop at X(m) = 6th failure"),
]
colors = plotstyle.FACILITY_COLORS
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.grid(False)
for r, (lab, xs, Rs, kap, m, case) in enumerate(rows):
    y = 2 - r
    stop = xs[-1] if "Case I:" in case or "Case III" in case else T
    ax.plot([0, stop], [y, y], color=colors[r], linewidth=2.2)
    ax.plot([stop, 8.6], [y, y], color=colors[r], linewidth=1, linestyle=":", alpha=0.6)
    for j, (x, R) in enumerate(zip(xs, Rs)):
        ax.plot(x, y, marker="x", markersize=10, mew=2.2, color=colors[r])
        ax.annotate(f"X({j+1})", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=plotstyle.INK2)
        if R > 0 and x != stop:
            ax.annotate(f"remove R={R}", (x, y), textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=7.5, color=plotstyle.INK2)
    ax.plot(stop, y, marker="|", markersize=22, mew=2.5, color=plotstyle.INK)
    ax.annotate("STOP: all survivors removed", (stop, y), textcoords="offset points",
                xytext=(6, -17), fontsize=7.5, color=plotstyle.INK)
    ax.text(-0.15, y + 0.30, lab, fontsize=9, color=plotstyle.INK, fontweight="bold")
    ax.text(8.7, y, case, fontsize=8.5, va="center", color=plotstyle.INK)
ax.axvline(T, color=plotstyle.INK2, linestyle="--", linewidth=1.2)
ax.text(T, 2.62, "pre-fixed time T", ha="center", fontsize=9, color=plotstyle.INK2)
ax.set_xlim(-0.2, 14.5)
ax.set_ylim(-0.7, 2.8)
ax.set_yticks([])
ax.set_xticks(range(0, 9))
ax.set_xlabel("test time")
for s in ["left", "top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("Block generalized progressive hybrid censoring: each facility runs its own GPHC test "
             "and stops at T* = max{X(κ), min(X(m), T)}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "schematic_bgphc.png"))
print("saved")
