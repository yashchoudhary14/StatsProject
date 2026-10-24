"""Equation images (matplotlib mathtext) and an illustrative Lomax figure for the
slides and the guide.  Output: ../results/figures/eq_*.png, lomax_shapes.png"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lomax_bgphc import model, plotstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)
plotstyle.apply()
plt.rcParams["mathtext.fontset"] = "cm"

EQS = {
    "lomax": r"$f(x)=\alpha\lambda(1+\lambda x)^{-(\alpha+1)},\quad R(x)=(1+\lambda x)^{-\alpha},\quad h(x)=\dfrac{\alpha\lambda}{1+\lambda x}$",
    "transform": r"$U=\alpha\,\log(1+\lambda X)\ \sim\ \mathrm{Exp}(1)$",
    "stop": r"$T_i^{*}=\max\{X_{\kappa_i:m_i:n_i},\ \min(X_{m_i:m_i:n_i},\,T_i)\}$",
    "likelihood": r"$L(\alpha,\lambda)\ \propto\ \prod_{i=1}^{k}\alpha_i^{D_i}\ \lambda^{D}\prod_{i,j}(1+\lambda x_{ij})^{-1}\ \exp\{-\sum_{i}\alpha_i\Delta_i(\lambda)\}$",
    "delta": r"$\Delta_i(\lambda)=\sum_{j=1}^{D_i}(1+R_{ij})\log(1+\lambda x_{ij})+R_i^{T}\log(1+\lambda T_i)$",
    "mle": r"$\hat\alpha_i(\lambda)=\dfrac{D_i}{\Delta_i(\lambda)},\qquad \hat\lambda=\arg\max_\lambda\ \ell_p(\lambda)$",
    "exist": r"$S_0=\sum_i \dfrac{D_iB_i}{2A_i}-\sum_{i,j}x_{ij}>0\ \Rightarrow\ \hat\lambda\in(0,\infty)$ exists",
    "pivots": r"$\mathcal{P}(\lambda)=2\sum_{i}\sum_{j<D_i}\log\dfrac{W_{iD_i}}{W_{ij}}\sim\chi^2_{2\sum(D_i-1)},\qquad 2\alpha_i\Delta_i(\lambda)\sim\chi^2_{2D_i}$",
    "gd": r"$\pi(\alpha)\ \propto\ \prod_{i}\alpha_i^{c_i-1}\,e^{-b_0\alpha_+}\,\alpha_+^{\,a_0-c_+},\qquad \alpha_+=\sum_i\alpha_i$",
    "posterior": r"$\pi(\alpha,\lambda\,|\,\mathrm{data})\propto\lambda^{a+D-1}e^{-b\lambda}\prod(1+\lambda x_{ij})^{-1}\prod_i\alpha_i^{D_i+c_i-1}e^{-\alpha_i(b_0+\Delta_i)}\,\alpha_+^{a_0-c_+}$",
    "linex": r"$\hat\theta_{SE}=E(\theta\,|\,\mathrm{data}),\qquad \hat\theta_{LX}=-\dfrac{1}{h}\log E\left(e^{-h\theta}\,|\,\mathrm{data}\right)$",
    "ormle": r"$\tilde\alpha_u(\lambda)=\left[\min_{s\leq u}\ \max_{t\geq u}\ \dfrac{\sum_{r=s}^{t}\Delta_r(\lambda)}{\sum_{r=s}^{t}D_r}\right]^{-1}$",
}


def render(name, tex, size=22, color="#1b1d22"):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=color)
    fig.savefig(os.path.join(FIG, f"eq_{name}.png"), dpi=220, transparent=True,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


for k, v in EQS.items():
    render(k, v)

# Lomax density and hazard for several shapes
x = np.linspace(0, 4, 400)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for a, c, ls in zip([0.5, 1.0, 2.0], plotstyle.FACILITY_COLORS, plotstyle.FACILITY_STYLES):
    axes[0].plot(x, model.reliability(x, a, 1.0), color=c, linestyle=ls, label=f"α = {a:g}")
    axes[1].plot(x, model.hazard(x, a, 1.0), color=c, linestyle=ls, label=f"α = {a:g}")
axes[0].plot(x, np.exp(-x), color=plotstyle.INK2, linewidth=1.2, linestyle=":", label="Exp(1)")
axes[0].set_title("Reliability R(x) = P(X > x)")
axes[1].set_title("Hazard rate h(x) — always decreasing")
for ax in axes:
    ax.set_xlabel("x  (λ = 1)")
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "lomax_shapes.png"))
print("assets written")
