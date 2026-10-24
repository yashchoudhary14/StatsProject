---
title: "Pivotal-Based Classical and Bayesian Estimation for the Lomax Distribution under Block Generalized Progressive Hybrid Censoring"
subtitle: "DA 2141 Statistics for Data Science — Course Project Report"
author: "[Student name(s) and roll number(s)]"
date: "October 2026"
---

# Abstract {.unnumbered}

Life tests are often run in several test facilities at the same time, and they are usually stopped before every unit fails. This project studies **block generalized progressive hybrid censoring (BGPHC)**. A total of $n$ units is split into $k$ blocks. Each block is tested in its own facility under a generalized progressive hybrid censoring scheme, which guarantees a minimum number of failures while keeping the test time under control. We assume that the lifetime of a unit tested in facility $i$ follows a **Lomax** (Pareto type II) distribution with a facility-specific shape $\alpha_i$ and a common scale-type parameter $\lambda$, so the $\alpha_i$'s measure the *difference between test facilities*.

We derive the likelihood and four families of estimators:

* maximum likelihood estimators (MLEs), with a sufficient condition for existence and asymptotic confidence intervals from the observed Fisher information and the delta method;
* pivotal-quantity (generalized pivotal) estimators and generalized confidence intervals. We prove that the Lomax pivot for $\lambda$ is strictly *decreasing* and bounded, and handle this by a truncation step;
* Bayes estimators under squared-error and LINEX loss with a **joint (dependent) Gamma–Dirichlet prior**, computed by importance sampling, with HPD credible intervals;
* **order-restricted** MLEs (isotonic regression) and order-restricted Bayes estimators (ordered Gamma–Dirichlet prior) for $\alpha_1\le\alpha_2\le\dots\le\alpha_k$.

A Monte Carlo study covers 18 designs with 2,000 replications each. The MLE of the shape parameters is unstable in small or short tests. The pivotal method gives the most reliable frequentist intervals: its coverage is close to 95%, while the MLE intervals for $\lambda$ under-cover at 81–90%. The order-restricted Bayes estimator has the smallest errors overall. The methods are applied to excess fatalities in British coal-mining disasters in three 20-year periods. Lomax fits every period, a common $\lambda$ is supported (LRT $p = 0.65$), and the shape increases over time. This is consistent with the order restriction: catastrophic disasters became rarer.

**Keywords:** Lomax distribution; block censoring; generalized progressive hybrid censoring; maximum likelihood; generalized pivotal quantity; Gamma–Dirichlet prior; order restriction; importance sampling; HPD interval.

# 1. Introduction

## 1.1 Motivation

In reliability and survival studies we observe *lifetimes*: time to failure of a component, time to death, time until a customer leaves. Waiting for every unit to fail is usually too slow and too expensive, so the experiment is stopped early. The lifetimes that are not seen completely are **censored**: we only know that they are larger than some value. Ignoring censored units throws information away. Treating them as failures biases the results. The likelihood handles them correctly.

Two practical issues motivate this project:

1. **Several test facilities.** When there is not enough equipment to test all $n$ units at once, the units are split into $k$ groups and each group is tested on a different machine, in a different laboratory, or by a different operator. This is *block censoring* (Ahmadi et al., 2018, discussed in references [2] and [3]). Even nominally identical facilities give slightly different results, so the model should allow a **difference in different test facilities (DDTF)** [2].
2. **Flexible stopping rules.** A fixed test time (Type-I censoring) may produce very few failures. A fixed number of failures (Type-II censoring) may take very long. **Generalized progressive hybrid censoring** [4] guarantees at least $\kappa$ failures but otherwise stops at a pre-fixed time $T$ or at the $m$-th failure. It also allows units to be withdrawn during the test.

For the lifetime model we use the **Lomax distribution**. It has a decreasing hazard rate and a heavy (polynomial) right tail, and it arises as a gamma mixture of exponentials. It is widely used for heavy-tailed data such as insurance claims, income, internet traffic and extreme events.

## 1.2 Objectives

The project requirements and where they are addressed:

| Requirement | Where |
|:---|:---|
| Background and formulation of the likelihood | Sections 2–3 |
| Maximum likelihood estimates (and confidence intervals) | Section 4 |
| Pivotal-quantity based estimation | Section 5 |
| Bayesian estimation with joint priors | Section 6 |
| Order-restricted (and unrestricted) estimation | Section 7 |
| Complete simulation study | Section 8 |
| Validation on a real data set | Section 9 |

## 1.3 What is new in this project

References [1] and [2] develop likelihood and pivotal inference under *block progressive Type-II censoring* for the Burr-XII and inverse exponentiated Rayleigh distributions. Reference [3] develops classical and Bayesian inference with a joint Gamma–Dirichlet prior and order restrictions under *block Type-I hybrid censoring* for the Weibull distribution. This project combines the three ideas and applies them to a new distribution and a more general censoring scheme:

* a new censoring scheme, *block generalized progressive hybrid censoring*. Each block uses the generalized progressive hybrid scheme of [4], which contains block progressive Type-II censoring ($T=\infty$) and block Type-I hybrid censoring as special cases;
* a new lifetime model, the Lomax distribution, with a facility effect in the shape parameter;
* new theory:
  * a sufficient condition for existence of the MLE;
  * a proof that the Lomax pivot for $\lambda$ is strictly decreasing with a *finite* limit at $\lambda\to 0$. This differs from the IER case in [2] and requires a truncated pivotal algorithm;
  * a demonstration that vague priors give improper posteriors;
* an efficient importance-sampling algorithm for the dependent and the ordered Gamma–Dirichlet priors.

The project uses only one reference beyond the three papers that were provided: Cho, Sun & Lee (2015) [4], which defines the generalized progressive hybrid censoring scheme. Classical results quoted from the course textbooks [5] are standard material.

# 2. Background

## 2.1 The Lomax distribution

A random variable $X$ has a Lomax distribution with shape $\alpha>0$ and scale-type parameter $\lambda>0$, written $X\sim \mathrm{Lomax}(\alpha,\lambda)$, if

$$f(x;\alpha,\lambda)=\alpha\lambda\,(1+\lambda x)^{-(\alpha+1)},\qquad x>0, \qquad (2.1)$$

$$F(x;\alpha,\lambda)=1-(1+\lambda x)^{-\alpha},\qquad R(x;\alpha,\lambda)=1-F(x)=(1+\lambda x)^{-\alpha}. \qquad (2.2)$$

Important characteristics used later:

$$h(t;\alpha,\lambda)=\frac{f(t)}{R(t)}=\frac{\alpha\lambda}{1+\lambda t}\ (\text{decreasing}),\qquad \mathrm{MdTF}=\frac{2^{1/\alpha}-1}{\lambda},\qquad x_p=\frac{(1-p)^{-1/\alpha}-1}{\lambda}. \qquad (2.3)$$

The mean is $1/\{\lambda(\alpha-1)\}$ when $\alpha>1$ and is infinite otherwise, which is why we use the *median* time to failure (MdTF). Useful facts:

* **Mixture of exponentials.** If $X\mid\theta\sim\mathrm{Exp}(\theta)$ and $\theta\sim\mathrm{Gamma}(\alpha,\text{rate }1/\lambda)$, then $X\sim\mathrm{Lomax}(\alpha,\lambda)$. Heterogeneous exponential units produce Lomax data.
* **Peaks over threshold.** The Lomax is the generalized Pareto distribution with positive shape, the limiting distribution of exceedances over a high threshold (used in Section 9).
* **Exponential limit.** If $\lambda\to0$ and $\alpha\to\infty$ with $\alpha\lambda=\theta$ fixed, then $R(x)\to e^{-\theta x}$. This limit drives several results below.
* **Key transformation.** If $X\sim\mathrm{Lomax}(\alpha,\lambda)$ then

$$U=\alpha\log(1+\lambda X)\sim\mathrm{Exp}(1), \qquad (2.4)$$

  because $P(U>u)=P\big(X>(e^{u/\alpha}-1)/\lambda\big)=e^{-u}$. So $\log(1+\lambda X)$ is exponential with *rate* $\alpha$.

## 2.2 Censoring schemes

* **Type-I / Type-II censoring.** Stop at a fixed time $T$ or at the $m$-th failure.
* **Progressive Type-II censoring.** Fix $m$ and removal numbers $R_1,\dots,R_m$ with $m+\sum R_j=n$. At the $j$-th failure, $R_j$ surviving units are withdrawn at random. Remaining units are removed at the $m$-th failure. Withdrawing units early frees equipment and models drop-outs [1, 2].
* **Hybrid censoring.** Combines a time limit with a failure-count limit.
* **Generalized progressive hybrid censoring (GPHCS)** [4]. Fix integers $1\le\kappa<m\le n$, a time $T$ and removals $R_1,\dots,R_m$. The experiment stops at

$$T^*=\max\{X_{\kappa:m:n},\ \min(X_{m:m:n},T)\}. \qquad (2.5)$$

  So at least $\kappa$ failures are always observed (unlike ordinary progressive hybrid censoring), and the test never runs past $X_{m:m:n}$. Three cases are possible:

  | Case | Condition | Stop at | Failures observed $D$ | Units withdrawn at the stop |
  |:---|:---|:---|:---|:---|
  | I | $T<X_{\kappa:m:n}$ | $X_{\kappa:m:n}$ | $\kappa$ | $R^*_\kappa=n-\kappa-\sum_{j<\kappa}R_j$ |
  | II | $X_{\kappa:m:n}\le T<X_{m:m:n}$ | $T$ | $J$ (failures before $T$) | $R^{T}=n-J-\sum_{j\le J}R_j$ |
  | III | $X_{m:m:n}\le T$ | $X_{m:m:n}$ | $m$ | $R_m$ |

  Table: Table 1. The three cases of the generalized progressive hybrid censoring scheme.

## 2.3 Block generalized progressive hybrid censoring (BGPHC)

$n$ identical units are randomly divided into $k$ blocks of sizes $n_1,\dots,n_k$ ($\sum n_i=n$). Block $i$ is tested in facility $i$ under its own GPHCS with plan $(n_i,m_i,\kappa_i,T_i,\boldsymbol R_i)$, and all facilities run simultaneously. The observed data are

$$\text{facility } i:\quad (x_{i1},R_{i1}),\ (x_{i2},R_{i2}),\ \dots,\ (x_{iD_i},R_{iD_i});\ \ R_i^{T}\text{ units withdrawn at }T_i, \qquad i=1,\dots,k, \qquad (2.6)$$

where $D_i$ is the number of observed failures and $R_{ij}$ are the removals *actually made*. In Case I the last one is $R^*_{i\kappa_i}$, and $R^T_i=0$ except in Case II. Block progressive Type-II censoring [1, 2] is the special case $T_i=\infty$, and block Type-I hybrid censoring [3] is the special case $R_{ij}=0$ for $j<m_i$.

![Figure 1: Schematic of BGPHC with three facilities, one in each GPHCS case. Crosses are failures, "remove" marks progressive withdrawals and the black bar is the stopping time $T^*_i$.](../results/figures/schematic_bgphc.png){width=100%}

# 3. Model and likelihood

## 3.1 Assumptions

* **A1.** Lifetimes in facility $i$ are i.i.d. $\mathrm{Lomax}(\alpha_i,\lambda)$, $i=1,\dots,k$, and facilities are independent.
* **A2.** The scale-type parameter $\lambda$ is common to all facilities, because the facilities share the same failure mechanism. The shapes $\alpha_i$ are facility specific and measure the DDTF, as in [2] where one parameter carries the facility effect.
* **A3.** Withdrawals are random (non-informative).

Under A1–A3 the parameter vector is $\boldsymbol\theta=(\alpha_1,\dots,\alpha_k,\lambda)$. An LRT of A2 is in Section 9.

## 3.2 Likelihood of one block

Every GPHCS case gives the same algebraic form. Each observed failure contributes its density. Each unit withdrawn at a failure time $x_{ij}$ contributes the survival probability $R(x_{ij})$, and each unit withdrawn at $T_i$ contributes $R(T_i)$:

$$L_i(\alpha_i,\lambda)=C_i\prod_{j=1}^{D_i} f(x_{ij})\,\big[R(x_{ij})\big]^{R_{ij}}\ \big[R(T_i)\big]^{R_i^{T}}, \qquad (3.1)$$

with $C_i$ a constant free of the parameters. For the Lomax model, $f(x)R(x)^{R}=\alpha\lambda(1+\lambda x)^{-1}(1+\lambda x)^{-\alpha(1+R)}$, so

$$L_i(\alpha_i,\lambda)\propto \alpha_i^{D_i}\lambda^{D_i}\prod_{j=1}^{D_i}(1+\lambda x_{ij})^{-1}\exp\{-\alpha_i\,\Delta_i(\lambda)\}, \qquad (3.2)$$

$$\Delta_i(\lambda)=\sum_{j=1}^{D_i}(1+R_{ij})\log(1+\lambda x_{ij})+R_i^{T}\log(1+\lambda T_i). \qquad (3.3)$$

By (2.4), $\Delta_i(\lambda)$ is the **total time on test** of facility $i$ measured on the exponential scale $\log(1+\lambda x)$.

## 3.3 Full likelihood

With $D=\sum_i D_i$, independence across facilities gives

$$L(\boldsymbol\theta)\propto\Big(\prod_{i=1}^k\alpha_i^{D_i}\Big)\lambda^{D}\prod_{i=1}^{k}\prod_{j=1}^{D_i}(1+\lambda x_{ij})^{-1}\exp\Big\{-\sum_{i=1}^{k}\alpha_i\Delta_i(\lambda)\Big\}, \qquad (3.4)$$

$$\ell(\boldsymbol\theta)=\sum_{i=1}^{k}D_i\log\alpha_i+D\log\lambda-\sum_{i,j}\log(1+\lambda x_{ij})-\sum_{i=1}^{k}\alpha_i\Delta_i(\lambda). \qquad (3.5)$$

For fixed $\lambda$, the likelihood in $\alpha_i$ is a gamma kernel $\alpha_i^{D_i}e^{-\alpha_i\Delta_i}$. This single fact drives the closed-form MLE of $\alpha_i$, the pivots, the conjugate-like Bayesian updates and the isotonic order-restricted solution.

# 4. Maximum likelihood estimation

## 4.1 MLE of the facility effects for fixed $\lambda$

**Theorem 1.** For fixed $\lambda>0$ and $D_i\ge1$, $\ell$ is uniquely maximised in $\alpha_i$ at

$$\hat\alpha_i(\lambda)=\frac{D_i}{\Delta_i(\lambda)},\qquad i=1,\dots,k. \qquad (4.1)$$

*Proof.* $\partial\ell/\partial\alpha_i=D_i/\alpha_i-\Delta_i(\lambda)=0$ gives (4.1). Since $\partial^2\ell/\partial\alpha_i^2=-D_i/\alpha_i^2<0$ and $\ell$ is separable in the $\alpha_i$'s, the stationary point is the unique maximum. $\square$

## 4.2 Profile likelihood of $\lambda$: existence

Substituting (4.1) into (3.5) gives the profile log-likelihood

$$\ell_p(\lambda)=\sum_{i=1}^k D_i\log\frac{D_i}{\Delta_i(\lambda)}-D+D\log\lambda-\sum_{i,j}\log(1+\lambda x_{ij}). \qquad (4.2)$$

The MLE $\hat\lambda$ maximises (4.2), a one-dimensional problem, and then $\hat\alpha_i=\hat\alpha_i(\hat\lambda)$. The score equation is

$$\frac{D}{\lambda}-\sum_{i,j}\frac{x_{ij}}{1+\lambda x_{ij}}-\sum_{i=1}^k\frac{D_i\,\Delta_i'(\lambda)}{\Delta_i(\lambda)}=0,\qquad \Delta_i'(\lambda)=\sum_j\frac{(1+R_{ij})x_{ij}}{1+\lambda x_{ij}}+\frac{R_i^TT_i}{1+\lambda T_i}. \qquad (4.3)$$

It has no closed-form solution and is solved numerically: a log-scale grid search followed by Brent's method.

**Theorem 2 (existence).** Let $A_i=\sum_j(1+R_{ij})x_{ij}+R_i^TT_i$ and $B_i=\sum_j(1+R_{ij})x_{ij}^2+R_i^TT_i^2$.

1. $\ell_p(\lambda)\to-\infty$ as $\lambda\to\infty$.
2. As $\lambda\to0^{+}$, $\ell_p(\lambda)=\ell_E+\lambda S_0+O(\lambda^2)$, where $\ell_E=\sum_iD_i\log(D_i/A_i)-D$ is the maximised log-likelihood of the *exponential* model and
$$S_0=\sum_{i=1}^k\frac{D_iB_i}{2A_i}-\sum_{i,j}x_{ij}. \qquad (4.4)$$
3. Hence, if $S_0>0$, the MLE $\hat\lambda\in(0,\infty)$ exists.

*Proof.*

1. Since $\sum_j(1+R_{ij})+R_i^T=n_i$, we have $\Delta_i(\lambda)\sim n_i\log\lambda$ as $\lambda\to\infty$, so $-\sum D_i\log\Delta_i\approx-D\log\log\lambda\to-\infty$, while $D\log\lambda-\sum\log(1+\lambda x_{ij})\to-\sum\log x_{ij}$ is bounded.
2. Expand $\log(1+\lambda x)=\lambda x-\lambda^2x^2/2+O(\lambda^3)$, so $\Delta_i=\lambda A_i-\lambda^2B_i/2+O(\lambda^3)$ and $\log\Delta_i=\log\lambda+\log A_i-\lambda B_i/(2A_i)+O(\lambda^2)$. Substituting into (4.2), the $\log\lambda$ terms cancel and the result follows.
3. If $S_0>0$, then $\ell_p$ exceeds its limit $\ell_E$ at small $\lambda$ and tends to $-\infty$ at $\infty$. By continuity it attains its supremum at an interior point. $\square$

*Remark.* For one complete sample, $S_0>0$ reduces to $\overline{x^2}>2\bar x^2$, i.e. **coefficient of variation $>1$**. Lomax data must be "more variable than exponential" for $\lambda$ to be estimable. When the data look exponential, $\hat\lambda\to0$ and $\hat\alpha\to\infty$: the MLE does not exist and the model collapses to its exponential limit. Uniqueness is not guaranteed in general, but in all our data sets the profile likelihood is unimodal (Figure 8).

## 4.3 Observed Fisher information

The second derivatives of (3.5) give the $(k+1)\times(k+1)$ observed information $\mathbf I(\boldsymbol\theta)=-\partial^2\ell/\partial\boldsymbol\theta\partial\boldsymbol\theta^\top$ with elements

$$I_{\alpha_i\alpha_i}=\frac{D_i}{\alpha_i^2},\qquad I_{\alpha_i\alpha_l}=0\ (i\neq l),\qquad I_{\alpha_i\lambda}=\Delta_i'(\lambda),$$

$$I_{\lambda\lambda}=\frac{D}{\lambda^2}-\sum_{i,j}\frac{x_{ij}^2}{(1+\lambda x_{ij})^2}-\sum_{i=1}^k\alpha_i\Big[\sum_j\frac{(1+R_{ij})x_{ij}^2}{(1+\lambda x_{ij})^2}+\frac{R_i^TT_i^2}{(1+\lambda T_i)^2}\Big]. \qquad (4.5)$$

The implementation was checked against a numerical Hessian: the two agree to 4 decimals.

## 4.4 Asymptotic confidence intervals and the delta method

Under regularity conditions, $\hat{\boldsymbol\theta}$ is approximately $N_{k+1}(\boldsymbol\theta,\mathbf I^{-1}(\hat{\boldsymbol\theta}))$. For any smooth function $g(\boldsymbol\theta)$, such as $R_i(t_0)=(1+\lambda t_0)^{-\alpha_i}$, $h_i(t_0)$ or $\mathrm{MdTF}_i$, the delta method gives

$$\widehat{\mathrm{Var}}(\hat g)=\nabla g(\hat{\boldsymbol\theta})^\top\,\mathbf I^{-1}(\hat{\boldsymbol\theta})\,\nabla g(\hat{\boldsymbol\theta}),\qquad \text{ACI: }\ \hat g\pm z_{\tau/2}\sqrt{\widehat{\mathrm{Var}}(\hat g)}. \qquad (4.6)$$

Lower limits of positive quantities are truncated at 0 and reliabilities are truncated to $[0,1]$, as in [2]. Following [2], an overall (pooled) shape is also reported:

$$\bar\alpha=\sum_i w_i\alpha_i\Big/\sum_i w_i,\qquad w_i=1/\widehat{\mathrm{Var}}(\hat\alpha_i).$$

# 5. Pivotal-quantity inference

Pivotal inference [1, 2] (Weerahandi's generalized pivotal quantities, as cited in [2]) builds random quantities whose distribution is known and free of the unknown parameters. "Inverting" them gives a *distribution* for each parameter, from which point estimates and generalized confidence intervals (GCIs) follow. No large-sample approximation or numerical optimisation is needed.

## 5.1 Exponential spacings

Fix facility $i$, write $L_{ij}(\lambda)=\log(1+\lambda x_{ij})$, and let $c_{ij}=n_i-\sum_{l<j}(R_{il}+1)$ be the number of units at risk just before the $j$-th failure. By (2.4), $U_{ij}=\alpha_iL_{ij}$ is a progressively censored sample from $\mathrm{Exp}(1)$. The normalised spacings $V_{ij}=c_{ij}(U_{ij}-U_{i,j-1})$ are i.i.d. $\mathrm{Exp}(1)$ (Balakrishnan & Aggarwala, cited in [2]). Summation by parts gives

$$W_{ij}:=\sum_{l\le j}V_{il}=\alpha_i\Big[\sum_{l<j}(R_{il}+1)L_{il}+c_{ij}L_{ij}\Big],\qquad j=1,\dots,D_i. \qquad (5.1)$$

Then $Q_{ij}=W_{ij}/W_{iD_i}$, $j<D_i$, behave like ordered $U(0,1)$ variables independent of $W_{iD_i}\sim\mathrm{Gamma}(D_i,1)$. Two pivots follow:

$$\mathcal P_i(\lambda)=2\sum_{j=1}^{D_i-1}\log\frac{W_{iD_i}}{W_{ij}}\sim\chi^2_{2(D_i-1)},\qquad \mathcal R_i=2\alpha_i\,\Delta_i(\lambda)\sim\chi^2_{2D_i}. \qquad (5.2)$$

$\mathcal P_i$ does **not** depend on $\alpha_i$ because $\alpha_i$ cancels in the ratio. Summing over independent facilities gives

$$\mathcal P(\lambda)=\sum_{i=1}^k\mathcal P_i(\lambda)\sim\chi^2_{\nu},\qquad \nu=2\sum_{i=1}^k(D_i-1). \qquad (5.3)$$

The distribution in (5.3) was verified by simulation: with $T=\infty$, the Kolmogorov–Smirnov $p$-value of $\mathcal P(\lambda_{\text{true}})$ against $\chi^2_\nu$ was 0.15.

## 5.2 Monotonicity — a new result for the Lomax case

**Theorem 3.** For Lomax data, each $\mathcal P_i(\lambda)$ is strictly **decreasing** in $\lambda$. Moreover

$$\lim_{\lambda\to\infty}\mathcal P_i(\lambda)=0,\qquad \lim_{\lambda\to0^+}\mathcal P_i(\lambda)=\mathcal P_i(0^+)=2\sum_{j<D_i}\log\frac{W^0_{iD_i}}{W^0_{ij}}<\infty,$$

where $W^0$ is (5.1) with $L_{ij}$ replaced by $x_{ij}$.

*Proof.* (i) For $0<a<b$, $g(\lambda)=\log(1+\lambda a)/\log(1+\lambda b)$ is increasing. Indeed $g'(\lambda)>0\iff\psi(\lambda b)>\psi(\lambda a)$ with $\psi(z)=(1+z)\log(1+z)/z$, and $\psi'(z)=\{z-\log(1+z)\}/z^2>0$.

(ii) Using $c_{ij}=\sum_{l=j}^{D_i-1}(R_{il}+1)+c_{iD_i}$,

$$\frac{W_{iD_i}}{W_{ij}}=1+\frac{\sum_{l=j+1}^{D_i-1}(R_{il}+1)\big(\tfrac{L_{il}}{L_{ij}}-1\big)+c_{iD_i}\big(\tfrac{L_{iD_i}}{L_{ij}}-1\big)}{\sum_{l<j}(R_{il}+1)\tfrac{L_{il}}{L_{ij}}+c_{ij}}.$$

By (i), $L_{il}/L_{ij}$ decreases in $\lambda$ for $l>j$ (numerator) and increases for $l<j$ (denominator). So every ratio decreases and $\mathcal P_i$ decreases.

(iii) As $\lambda\to\infty$, $L_{il}/L_{ij}\to1$, so the ratio $\to1$ and $\mathcal P_i\to0$. As $\lambda\to0$, $L_{il}/L_{ij}\to x_{il}/x_{ij}$. $\square$

**Consequence.** In the IER model of [2] the pivot increases from 0 to $\infty$, so $\mathcal P(\lambda)=c$ always has a root. For the Lomax model $\mathcal P(\lambda)$ falls from the finite value $\mathcal P(0^+)$ to 0. The equation $\mathcal P(\lambda)=c$ has a unique root **only if $c<\mathcal P(0^+)$**. Values $c\ge\mathcal P(0^+)$ correspond to the exponential limit $\lambda\to0$. We therefore draw $c$ from $\chi^2_\nu$ **truncated to $(0,\mathcal P(0^+))$**, i.e. we condition on the Lomax model being non-degenerate. In the simulations the truncation removed only 0.2–10% of the $\chi^2$ mass.

## 5.3 Generalized pivotal quantities and the algorithm

For a root $\lambda$ of $\mathcal P(\lambda)=c$, (5.2) gives the generalized pivotal quantity $\alpha_i=\mathcal R_i/\{2\Delta_i(\lambda)\}$.

**Algorithm 1 (pivotal estimates).** For $s=1,\dots,N$ ($N=1000$ in the simulation, $10{,}000$ for the real data):

1. Draw $c^{(s)}\sim\chi^2_\nu$ truncated to $(0,\mathcal P(0^+))$ by inverse-cdf. Solve $\mathcal P(\lambda)=c^{(s)}$ for $\lambda^{(s)}$ by bisection on $\log\lambda$. Monotonicity guarantees a unique root.
2. For each facility draw $\mathcal R_i^{(s)}\sim\chi^2_{2D_i}$ and set $\alpha_i^{(s)}=\mathcal R_i^{(s)}/\{2\Delta_i(\lambda^{(s)})\}$.
3. Compute $g^{(s)}=g(\boldsymbol\alpha^{(s)},\lambda^{(s)})$ for every quantity of interest: $R_i(t_0)$, $h_i(t_0)$, $\mathrm{MdTF}_i$, $\bar\alpha$.

Return the generalized point estimate $\tilde g=\mathrm{median}\{g^{(s)}\}$ and the $100(1-2\delta)\%$ GCI $\big(g_{(N\delta)},g_{(N(1-\delta))}\big)$.

*Median versus mean.* Reference [2] uses the mean of the pivotal draws. For the Lomax model, Theorem 3 implies that small $c$ produce very large $\lambda^{(s)}$, and with them very large $\alpha_i^{(s)}$. The pivotal distributions are therefore extremely right-skewed and their mean is unstable. Table B7 (Appendix B) shows that, for the shape parameters, the mean-based estimator has MSEs that are often hundreds to thousands of times larger than those of the median-based one. We therefore use the **median**, a standard choice for generalized point estimates.

*Remark (GPHCS).* The pivots in (5.2) are exact for progressive Type-II data. Under GPHCS the stopping point is random (Cases I–III), so they are approximations. In Case II, $\Delta_i(\lambda)$ in $\mathcal R_i$ includes the units withdrawn at $T_i$, i.e. the full total time on test. Section 8 shows that the resulting GCIs nevertheless have coverage close to the nominal 95%.

# 6. Bayesian estimation with a joint prior

## 6.1 Point estimation as a decision problem

A Bayes estimator minimises the posterior expected loss and, equivalently, the Bayes risk $r(\pi,d)=E_\pi[E_\theta\{L(\theta,d(\boldsymbol X))\}]$. We use two loss functions:

* **Squared-error loss (SELF)** $L(d,\theta)=(d-\theta)^2$. The Bayes estimator is the posterior mean $\hat\theta_{S}=E(\theta\mid\text{data})$.
* **LINEX loss** $L(d,\theta)=e^{h(d-\theta)}-h(d-\theta)-1$. It is asymmetric: with $h>0$, over-estimation costs more than under-estimation, which suits reliability work where over-stating a lifetime is the dangerous error. The Bayes estimator is

$$\hat\theta_{LX}=-\frac1h\log E\big(e^{-h\theta}\mid\text{data}\big). \qquad (6.1)$$

We use $h=1$.

Interval estimates are $100(1-\gamma)\%$ highest posterior density (HPD) credible intervals: the shortest interval with posterior probability $1-\gamma$ (Chen & Shao, cited in [3]).

## 6.2 The joint Gamma–Dirichlet prior

Independent gamma priors ignore the fact that facility effects are related: they describe the *same* product. Following [3] (Peña & Gupta's construction), we use a **dependent** prior on $\boldsymbol\alpha$:

$$\alpha_+=\sum_i\alpha_i\sim\mathrm{Gamma}(a_0,b_0),\qquad \Big(\tfrac{\alpha_1}{\alpha_+},\dots,\tfrac{\alpha_k}{\alpha_+}\Big)\Big|\,\alpha_+\sim\mathrm{Dirichlet}(c_1,\dots,c_k).$$

Changing variables gives the Gamma–Dirichlet density $\mathrm{GD}(a_0,b_0,\boldsymbol c)$:

$$\pi(\boldsymbol\alpha)=\frac{\Gamma(c_+)\,b_0^{a_0}}{\Gamma(a_0)\prod_i\Gamma(c_i)}\prod_{i=1}^k\alpha_i^{c_i-1}\,e^{-b_0\alpha_+}\,\alpha_+^{a_0-c_+},\qquad c_+=\sum_ic_i. \qquad (6.2)$$

Its properties:

* $E(\alpha_i)=(a_0/b_0)(c_i/c_+)$.
* $\mathrm{Cov}(\alpha_i,\alpha_l)=\dfrac{c_ic_l}{b_0^2}\Big[\dfrac{a_0(a_0+1)}{c_+(c_+ +1)}-\dfrac{a_0^2}{c_+^2}\Big]$ for $i\ne l$. This is positive if $a_0<c_+$, zero if $a_0=c_+$ (independent $\mathrm{Gamma}(c_i,b_0)$ priors) and negative if $a_0>c_+$.

For $\lambda$ we use an independent $\mathrm{Gamma}(a,b)$ prior.

**Hyperparameter elicitation.** From prior guesses $\alpha_i^0$ and $\lambda^0$ with a relative prior uncertainty $f$, set:

* $E(\alpha_+)=s=\sum_i\alpha_i^0$ and $\mathrm{sd}(\alpha_+)=fs$, which gives $a_0=1/f^2$ and $b_0=a_0/s$;
* $c_i=k\alpha_i^0/s$ (Dirichlet concentration $k$);
* $a=1/f^2$ and $b=a/\lambda^0$.

The simulation uses two priors. **P1** (informative, $f=0.5$) has $a_0=4$ and negative dependence. **P2** (weakly informative, $f=1$) has $a_0=1$ and positive dependence. For the real data we follow [3] and centre the prior at the MLE with $f=1$, then check sensitivity to $f$.

## 6.3 Posterior

Combining (3.4) and (6.2),

$$\pi(\boldsymbol\alpha,\lambda\mid\text{data})\propto\lambda^{a+D-1}e^{-b\lambda}\prod_{i,j}(1+\lambda x_{ij})^{-1}\prod_{i=1}^k\alpha_i^{D_i+c_i-1}e^{-\alpha_i\{b_0+\Delta_i(\lambda)\}}\ \alpha_+^{a_0-c_+}. \qquad (6.3)$$

Neither the posterior means nor the HPD intervals are available in closed form.

**Proposition 4 (propriety).** With proper gamma priors ($a,b,b_0>0$), the posterior (6.3) is proper.

*Sketch.* Integrating out $\boldsymbol\alpha$ under $a_0=c_+$ gives

$$g(\lambda)\propto\lambda^{a+D-1}e^{-b\lambda}\prod(1+\lambda x_{ij})^{-1}\prod_i\{b_0+\Delta_i(\lambda)\}^{-(D_i+c_i)}\le b_0^{-\sum(D_i+c_i)}\lambda^{a+D-1}e^{-b\lambda},$$

which is integrable. The factor $\alpha_+^{a_0-c_+}$ has finite expectation under the gamma conditionals.

*Remark.* The likelihood does **not** vanish as $\lambda\to0$, because it tends to the exponential-model likelihood. With a "vague" prior such as $\pi(\lambda)\propto1/\lambda$ and $b_0\to0$, the marginal behaves like $\lambda^{-c_+-1}$ near 0, so the posterior is **improper**. A proper, at least weakly informative prior on $\lambda$ is therefore essential for Lomax models.

## 6.4 Importance-sampling algorithm

The posterior (6.3) factorises as $q(\lambda,\boldsymbol\alpha)\times\alpha_+^{a_0-c_+}$, where $q$ is the posterior under independent priors ($a_0=c_+$). Sampling from $q$ is easy and exact:

* the conditional of each $\alpha_i$ given $\lambda$ is $\mathrm{Gamma}\big(D_i+c_i,\ b_0+\Delta_i(\lambda)\big)$;
* the marginal $g(\lambda)$ above is one-dimensional, so it can be tabulated on a fine grid of $\log\lambda$ (3,000 cells) and sampled by inverse-cdf.

**Algorithm 2 (Bayes, unrestricted).** For $s=1,\dots,M$ ($M=2000$ in the simulation, $20{,}000$ for the real data):

1. Draw $\lambda^{(s)}$ from the grid approximation $\hat g$ of $g$.
2. Draw $\alpha_i^{(s)}\sim\mathrm{Gamma}\big(D_i+c_i,\ b_0+\Delta_i(\lambda^{(s)})\big)$, $i=1,\dots,k$.
3. Compute the importance weight $w_s=\big(\alpha_+^{(s)}\big)^{a_0-c_+}\,g(\lambda^{(s)})/\hat g(\lambda^{(s)})$.

With $\bar w_s=w_s/\sum w_s$, the estimates are $\hat\theta_S=\sum_s\bar w_s\theta^{(s)}$ and $\hat\theta_{LX}=-h^{-1}\log\sum_s\bar w_se^{-h\theta^{(s)}}$. The weighted HPD interval is the shortest interval $[\theta_{(L)},\theta_{(U)}]$ of the sorted draws with $\sum_{L\le s\le U}\bar w_{(s)}\ge1-\gamma$ (ref. [3], eq. 23). The effective sample size $\mathrm{ESS}=1/\sum\bar w_s^2$ monitors accuracy.

Compared with the Metropolis–Hastings sampler of [3], this algorithm has no burn-in, no autocorrelation and no tuning, and it is fully vectorised. It is exact when $a_0=c_+$.

# 7. Order-restricted inference

## 7.1 Why order restrictions?

Often we *know* how facilities compare: units are tested at increasing stress, by operators of increasing experience, or, in our data, in successive periods with improving safety. If facility $i$ produces stochastically longer lifetimes than facility $i+1$, then $R_i(t)\ge R_{i+1}(t)$ for all $t$, which for the Lomax model means

$$\alpha_1\le\alpha_2\le\dots\le\alpha_k. \qquad (7.1)$$

Unrestricted estimates can violate (7.1) by chance. In our simulation the unrestricted MLEs were out of order in 19–42% of the samples even though the true values are ordered. Imposing the restriction removes impossible estimates and shrinks variability.

## 7.2 Order-restricted MLE

For fixed $\lambda$, we maximise $\sum_i\{D_i\log\alpha_i-\alpha_i\Delta_i(\lambda)\}$ subject to (7.1). This is a generalized isotonic regression problem (Brunk; Pal, Mitra & Kundu, as cited in [3]). Its solution is

$$\tilde\alpha_u(\lambda)=\Big[\min_{s\le u}\max_{t\ge u}\frac{\sum_{r=s}^t\Delta_r(\lambda)}{\sum_{r=s}^tD_r}\Big]^{-1},\qquad u=1,\dots,k. \qquad (7.2)$$

We compute (7.2) with the **pool-adjacent-violators algorithm (PAVA)**: whenever two neighbouring blocks violate the order, they are pooled and receive the common value $\sum D/\sum\Delta$. In 500 random test cases PAVA reproduced the min–max formula (7.2) exactly. Then $\tilde\lambda$ maximises the restricted profile $\ell(\tilde{\boldsymbol\alpha}(\lambda),\lambda)$.

*Facts.* If the unrestricted MLEs already satisfy (7.1), then $\tilde{\boldsymbol\theta}=\hat{\boldsymbol\theta}$; this is the case in our real data. Standard asymptotic normality fails when the truth lies on the boundary of the cone, so we report point estimates only for the OR-MLE and use the Bayesian approach for intervals.

## 7.3 Order-restricted Bayes

Following [3], the prior on the ordered cone is the **ordered Gamma–Dirichlet (OGD)** density, i.e. the density of the order statistics of a GD vector:

$$\pi_{or}(\boldsymbol\alpha)\propto e^{-b_0\alpha_+}\alpha_+^{a_0-c_+}\sum_{\sigma\in\mathcal S_k}\prod_{j=1}^k\alpha_{\sigma(j)}^{c_j-1},\qquad 0<\alpha_1\le\dots\le\alpha_k, \qquad (7.3)$$

where $\mathcal S_k$ is the set of the $k!$ permutations. The posterior is (6.3) with the prior replaced by (7.3) and restricted to the cone.

**Algorithm 3 (Bayes, order restricted).** For $s=1,\dots,M$:

1. Draw $\lambda^{(s)}\sim\hat g$.
2. Draw $\alpha_i'\sim\mathrm{Gamma}(D_i+c_i,\ b_0+\Delta_i(\lambda^{(s)}))$ and **sort** them: $\boldsymbol\alpha^{(s)}=\mathrm{sort}(\boldsymbol\alpha')$.
3. The density of a sorted vector is a sum over permutations, $q(\boldsymbol\alpha\mid\lambda)=\sum_{\pi\in\mathcal S_k}\prod_i\mathrm{Ga}\big(\alpha_{\pi(i)};D_i+c_i,b_0+\Delta_i(\lambda)\big)$. The weight is $w_s=\pi_{or}(\boldsymbol\alpha^{(s)},\lambda^{(s)}\mid\text{data})\big/\{\hat g(\lambda^{(s)})\,q(\boldsymbol\alpha^{(s)}\mid\lambda^{(s)})\}$, computed in log scale with log-sum-exp.

Estimates, LINEX estimates and HPD intervals then follow exactly as in Section 6.4. Every draw lies in the cone, unlike rejection sampling.

# 8. Simulation study

## 8.1 Design

| Item | Setting |
|:---|:---|
| Facilities | $k=3$ |
| True parameters | $\boldsymbol\alpha=(0.50,0.75,1.00)$ (ordered), $\lambda=1$; mission time $t_0=1$, so $R_i(1)=(0.707,0.595,0.500)$ |
| Sample sizes $(n_i,m_i,\kappa_i)$, equal for all facilities | S1 $=(20,15,10)$, S2 $=(30,24,15)$, S3 $=(40,32,20)$ |
| Time limit | $T=3$ (short test) and $T=8$ (long test) |
| Removal schemes | A: $R=(0,\dots,0,n_i-m_i)$; B: $R=(n_i-m_i,0,\dots,0)$; C: half at the first and half at the $m$-th failure |
| Replications | 2,000 per design (18 designs); pivotal draws $N=1000$; importance draws $M=2000$ |
| Priors | P1 informative ($a_0=4,b_0=1.778,\boldsymbol c=(0.667,1,1.333)$, $\lambda\sim\mathrm{Ga}(4,4)$); P2 weakly informative ($a_0=1,b_0=0.444$, same $\boldsymbol c$, $\lambda\sim\mathrm{Ga}(1,1)$) |
| Criteria | Bias, mean squared error (MSE), median absolute error (MedAE), average length (AL) and coverage probability (CP) of 95% intervals |

Table: Table 2. Simulation design.

Samples are generated with the Balakrishnan–Sandhu algorithm (used in [2]) for progressive Type-II samples and then cut according to the GPHCS rule (2.5). Samples where the MLE does not exist (Theorem 2) are discarded and regenerated, as is standard. They were at most about 4% of the samples (S1, $T=3$) and under 1% for $n_i\ge30$. The heavy-tailed true values ($\alpha_i\le1$) make the Lomax shape clearly different from the exponential limit. With light tails and heavy censoring, $\alpha$ and $\lambda$ become nearly non-identifiable (Theorem 2). The pilot runs reported in Section 8.3 confirm this.

Method abbreviations used in the tables: **MLE**; **PIV** = pivotal (median of generalized pivotal draws); **OR-MLE** = order-restricted MLE; **BU** = Bayes, unrestricted GD prior; **BOR** = Bayes, order-restricted OGD prior; -P1/-P2 = prior; SE/LX = squared-error/LINEX loss; ACI/GCI/HPD-U/HPD-OR = the corresponding 95% intervals.

| Design | Case I % | Case II % | Case III % | Avg failures D | MLE order violated % | Discarded | Median ESS (BOR-P2) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 15.7 | 56.0 | 28.3 | 37.9 | 36.9 | 85.0 | 1016.4 |
| S1, T=3, B | 46.5 | 52.9 | 0.6 | 32.2 | 41.9 | 62.0 | 1045.1 |
| S1, T=3, C | 24.7 | 62.7 | 12.6 | 35.6 | 39.6 | 74.0 | 1026.0 |
| S1, T=8, A | 1.0 | 28.5 | 70.5 | 42.9 | 36.5 | 28.0 | 1148.9 |
| S1, T=8, B | 13.9 | 79.1 | 7.0 | 36.3 | 38.2 | 2.0 | 1221.7 |
| S1, T=8, C | 4.2 | 45.8 | 50.1 | 41.0 | 36.9 | 13.0 | 1202.0 |
| S2, T=3, A | 15.3 | 71.5 | 13.2 | 57.6 | 29.9 | 8.0 | 1169.7 |
| S2, T=3, B | 40.0 | 60.0 | 0.0 | 49.4 | 34.3 | 13.0 | 1168.9 |
| S2, T=3, C | 25.3 | 72.4 | 2.4 | 53.4 | 31.0 | 15.0 | 1174.0 |
| S2, T=8, A | 0.9 | 43.0 | 56.2 | 67.1 | 24.3 | 0.0 | 1387.8 |
| S2, T=8, B | 8.5 | 89.4 | 2.1 | 57.4 | 28.4 | 0.0 | 1416.9 |
| S2, T=8, C | 2.7 | 68.1 | 29.2 | 63.1 | 24.2 | 0.0 | 1432.0 |
| S3, T=3, A | 15.6 | 73.9 | 10.5 | 76.8 | 23.0 | 0.0 | 1288.1 |
| S3, T=3, B | 41.1 | 58.9 | 0.0 | 65.7 | 26.3 | 3.0 | 1287.0 |
| S3, T=3, C | 26.5 | 72.6 | 0.9 | 70.8 | 26.5 | 4.0 | 1282.4 |
| S3, T=8, A | 0.4 | 44.5 | 55.1 | 89.7 | 19.4 | 0.0 | 1515.3 |
| S3, T=8, B | 7.9 | 91.2 | 0.9 | 76.2 | 22.7 | 0.0 | 1525.5 |
| S3, T=8, C | 1.9 | 71.7 | 26.4 | 84.4 | 21.3 | 0.0 | 1549.5 |


Table: Table 3. Censoring behaviour in each design: percentage of facilities ending in Case I/II/III, average total number of failures, percentage of samples where the unrestricted MLEs violate the true order, number of discarded samples, and median effective sample size of the order-restricted importance sampler.

## 8.2 Results: point estimation

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 96.6433 | 0.1506 | 91.9218 | 0.0262 | 0.0108 | 0.0727 | 0.0369 |
| S1, T=3, B | 197.9653 | 0.2154 | 46.5465 | 0.0352 | 0.0118 | 0.0964 | 0.0409 |
| S1, T=3, C | 548.3284 | 0.1489 | 537.7264 | 0.0292 | 0.0103 | 0.0768 | 0.0346 |
| S1, T=8, A | 66.0948 | 0.1322 | 0.9314 | 0.0317 | 0.0142 | 0.0832 | 0.0449 |
| S1, T=8, B | 0.4782 | 0.2004 | 0.4274 | 0.0279 | 0.0117 | 0.0654 | 0.0334 |
| S1, T=8, C | 1.9904 | 0.1285 | 1.7173 | 0.0291 | 0.0129 | 0.0703 | 0.0365 |
| S2, T=3, A | 0.8571 | 0.1488 | 0.8321 | 0.0207 | 0.0109 | 0.0587 | 0.0384 |
| S2, T=3, B | 44.3214 | 0.1779 | 15.6856 | 0.0267 | 0.0118 | 0.0760 | 0.0436 |
| S2, T=3, C | 51.2546 | 0.1303 | 39.8860 | 0.0222 | 0.0111 | 0.0600 | 0.0375 |
| S2, T=8, A | 0.0984 | 0.0686 | 0.0873 | 0.0190 | 0.0119 | 0.0432 | 0.0310 |
| S2, T=8, B | 0.0712 | 0.0951 | 0.0647 | 0.0211 | 0.0116 | 0.0433 | 0.0280 |
| S2, T=8, C | 0.5836 | 0.0717 | 0.5771 | 0.0189 | 0.0114 | 0.0400 | 0.0276 |
| S3, T=3, A | 44.8992 | 0.1192 | 44.8899 | 0.0170 | 0.0108 | 0.0464 | 0.0348 |
| S3, T=3, B | 0.6616 | 0.1045 | 0.5957 | 0.0214 | 0.0118 | 0.0592 | 0.0402 |
| S3, T=3, C | 0.2129 | 0.0995 | 0.2022 | 0.0176 | 0.0103 | 0.0479 | 0.0342 |
| S3, T=8, A | 0.0470 | 0.0506 | 0.0442 | 0.0155 | 0.0109 | 0.0317 | 0.0251 |
| S3, T=8, B | 0.0377 | 0.0555 | 0.0355 | 0.0161 | 0.0106 | 0.0299 | 0.0220 |
| S3, T=8, C | 0.0414 | 0.0465 | 0.0390 | 0.0160 | 0.0109 | 0.0304 | 0.0232 |


Table: Table 4. MSE of the estimators of $\alpha_1$ (true 0.5).

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 379.2963 | 0.8118 | 374.9278 | 0.0923 | 0.0975 | 0.3304 | 0.3778 |
| S1, T=3, B | 401.0138 | 1.1148 | 186.3731 | 0.0943 | 0.1064 | 0.3348 | 0.4107 |
| S1, T=3, C | 8399.1593 | 0.9162 | 8355.0966 | 0.0857 | 0.0907 | 0.3113 | 0.3594 |
| S1, T=8, A | 94.4569 | 0.9698 | 10.7984 | 0.0889 | 0.1006 | 0.3455 | 0.4034 |
| S1, T=8, B | 4.0543 | 1.0808 | 4.2518 | 0.0976 | 0.1045 | 0.3090 | 0.3496 |
| S1, T=8, C | 52.7915 | 1.1480 | 52.1262 | 0.0984 | 0.1067 | 0.3537 | 0.3958 |
| S2, T=3, A | 7.0304 | 0.6733 | 7.0113 | 0.0686 | 0.0707 | 0.2577 | 0.2854 |
| S2, T=3, B | 81.7761 | 0.7528 | 61.0681 | 0.0694 | 0.0740 | 0.2592 | 0.2979 |
| S2, T=3, C | 627.8435 | 0.6715 | 563.9979 | 0.0707 | 0.0723 | 0.2533 | 0.2810 |
| S2, T=8, A | 0.8552 | 0.4890 | 0.8521 | 0.0763 | 0.0798 | 0.2467 | 0.2681 |
| S2, T=8, B | 0.3618 | 0.4904 | 0.3623 | 0.0668 | 0.0703 | 0.1873 | 0.2066 |
| S2, T=8, C | 3.5427 | 0.4819 | 3.5424 | 0.0763 | 0.0791 | 0.2273 | 0.2437 |
| S3, T=3, A | 531.5722 | 0.5920 | 531.5371 | 0.0622 | 0.0627 | 0.2206 | 0.2349 |
| S3, T=3, B | 3.2292 | 0.4989 | 3.1999 | 0.0666 | 0.0692 | 0.2361 | 0.2612 |
| S3, T=3, C | 6.4740 | 0.4315 | 6.4770 | 0.0549 | 0.0562 | 0.1905 | 0.2114 |
| S3, T=8, A | 0.3899 | 0.4665 | 0.3895 | 0.0638 | 0.0652 | 0.1860 | 0.1954 |
| S3, T=8, B | 0.2135 | 0.2967 | 0.2122 | 0.0586 | 0.0594 | 0.1426 | 0.1519 |
| S3, T=8, C | 0.2448 | 0.2867 | 0.2447 | 0.0623 | 0.0627 | 0.1588 | 0.1656 |


Table: Table 5. MSE of the estimators of $\alpha_3$ (true 1.0).

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.7315 | 1.9520 | 0.7410 | 0.0315 | 0.0317 | 0.1591 | 0.1598 |
| S1, T=3, B | 0.7686 | 2.0782 | 0.7774 | 0.0335 | 0.0336 | 0.1585 | 0.1590 |
| S1, T=3, C | 0.7326 | 1.8396 | 0.7430 | 0.0322 | 0.0325 | 0.1623 | 0.1610 |
| S1, T=8, A | 0.4673 | 0.9969 | 0.4731 | 0.0366 | 0.0374 | 0.1547 | 0.1578 |
| S1, T=8, B | 0.5201 | 1.0508 | 0.5263 | 0.0429 | 0.0430 | 0.1804 | 0.1815 |
| S1, T=8, C | 0.5359 | 1.1411 | 0.5407 | 0.0396 | 0.0400 | 0.1609 | 0.1618 |
| S2, T=3, A | 0.4454 | 0.8004 | 0.4476 | 0.0404 | 0.0402 | 0.1694 | 0.1695 |
| S2, T=3, B | 0.5247 | 1.0053 | 0.5266 | 0.0417 | 0.0416 | 0.1780 | 0.1755 |
| S2, T=3, C | 0.4570 | 0.8768 | 0.4616 | 0.0395 | 0.0397 | 0.1676 | 0.1683 |
| S2, T=8, A | 0.2586 | 0.4423 | 0.2580 | 0.0425 | 0.0424 | 0.1375 | 0.1369 |
| S2, T=8, B | 0.3039 | 0.5662 | 0.3060 | 0.0468 | 0.0473 | 0.1568 | 0.1572 |
| S2, T=8, C | 0.3029 | 0.4748 | 0.3045 | 0.0468 | 0.0470 | 0.1543 | 0.1549 |
| S3, T=3, A | 0.3093 | 0.4920 | 0.3100 | 0.0419 | 0.0418 | 0.1497 | 0.1487 |
| S3, T=3, B | 0.3313 | 0.6200 | 0.3326 | 0.0426 | 0.0426 | 0.1509 | 0.1503 |
| S3, T=3, C | 0.3136 | 0.5345 | 0.3146 | 0.0419 | 0.0418 | 0.1501 | 0.1496 |
| S3, T=8, A | 0.2128 | 0.3331 | 0.2132 | 0.0491 | 0.0490 | 0.1345 | 0.1344 |
| S3, T=8, B | 0.2010 | 0.3175 | 0.2017 | 0.0460 | 0.0459 | 0.1292 | 0.1295 |
| S3, T=8, C | 0.2121 | 0.3368 | 0.2121 | 0.0485 | 0.0484 | 0.1351 | 0.1341 |


Table: Table 6. MSE of the estimators of $\lambda$ (true 1.0).

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.0060 | 0.0053 | 0.0050 | 0.0043 | 0.0029 | 0.0046 | 0.0032 |
| S1, T=3, B | 0.0081 | 0.0075 | 0.0068 | 0.0054 | 0.0032 | 0.0058 | 0.0038 |
| S1, T=3, C | 0.0069 | 0.0061 | 0.0057 | 0.0048 | 0.0029 | 0.0051 | 0.0034 |
| S1, T=8, A | 0.0066 | 0.0067 | 0.0057 | 0.0048 | 0.0031 | 0.0051 | 0.0035 |
| S1, T=8, B | 0.0068 | 0.0063 | 0.0058 | 0.0046 | 0.0031 | 0.0051 | 0.0037 |
| S1, T=8, C | 0.0068 | 0.0066 | 0.0059 | 0.0048 | 0.0033 | 0.0052 | 0.0038 |
| S2, T=3, A | 0.0041 | 0.0038 | 0.0037 | 0.0032 | 0.0022 | 0.0034 | 0.0024 |
| S2, T=3, B | 0.0052 | 0.0049 | 0.0045 | 0.0038 | 0.0023 | 0.0040 | 0.0027 |
| S2, T=3, C | 0.0047 | 0.0044 | 0.0042 | 0.0036 | 0.0024 | 0.0038 | 0.0026 |
| S2, T=8, A | 0.0037 | 0.0038 | 0.0035 | 0.0029 | 0.0022 | 0.0031 | 0.0025 |
| S2, T=8, B | 0.0047 | 0.0048 | 0.0042 | 0.0036 | 0.0025 | 0.0039 | 0.0028 |
| S2, T=8, C | 0.0041 | 0.0042 | 0.0038 | 0.0032 | 0.0023 | 0.0034 | 0.0026 |
| S3, T=3, A | 0.0032 | 0.0031 | 0.0029 | 0.0026 | 0.0019 | 0.0028 | 0.0021 |
| S3, T=3, B | 0.0039 | 0.0038 | 0.0036 | 0.0030 | 0.0020 | 0.0032 | 0.0023 |
| S3, T=3, C | 0.0034 | 0.0033 | 0.0031 | 0.0027 | 0.0018 | 0.0029 | 0.0021 |
| S3, T=8, A | 0.0029 | 0.0030 | 0.0027 | 0.0023 | 0.0018 | 0.0025 | 0.0020 |
| S3, T=8, B | 0.0034 | 0.0035 | 0.0032 | 0.0027 | 0.0020 | 0.0030 | 0.0023 |
| S3, T=8, C | 0.0031 | 0.0032 | 0.0029 | 0.0025 | 0.0019 | 0.0027 | 0.0021 |


Table: Table 7. MSE of the estimators of the reliability $R_1(1)$ (true 0.7071).

| Design | α₁ MLE | α₁ PIV | α₁ BOR | α₃ MLE | α₃ PIV | α₃ BOR | λ MLE | λ PIV | λ BOR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.1792 | 0.1732 | 0.1103 | 0.4231 | 0.3630 | 0.3727 | 0.5129 | 0.4997 | 0.2493 |
| S1, T=3, B | 0.1930 | 0.1772 | 0.1194 | 0.4320 | 0.3811 | 0.4145 | 0.5134 | 0.5373 | 0.2538 |
| S1, T=3, C | 0.1742 | 0.1633 | 0.1106 | 0.3897 | 0.3565 | 0.3414 | 0.5039 | 0.4771 | 0.2488 |
| S1, T=8, A | 0.1692 | 0.1515 | 0.1150 | 0.3758 | 0.3382 | 0.3764 | 0.4687 | 0.4814 | 0.2675 |
| S1, T=8, B | 0.1504 | 0.1673 | 0.1062 | 0.3439 | 0.3561 | 0.3166 | 0.4317 | 0.4887 | 0.2667 |
| S1, T=8, C | 0.1523 | 0.1562 | 0.1064 | 0.3539 | 0.3264 | 0.3403 | 0.4432 | 0.4635 | 0.2572 |
| S2, T=3, A | 0.1501 | 0.1590 | 0.1153 | 0.3174 | 0.3275 | 0.2906 | 0.4246 | 0.4653 | 0.2672 |
| S2, T=3, B | 0.1516 | 0.1523 | 0.1187 | 0.3245 | 0.3451 | 0.3077 | 0.4223 | 0.4542 | 0.2577 |
| S2, T=3, C | 0.1458 | 0.1582 | 0.1120 | 0.3195 | 0.3281 | 0.3028 | 0.4030 | 0.4522 | 0.2530 |
| S2, T=8, A | 0.1178 | 0.1226 | 0.0964 | 0.2606 | 0.2642 | 0.2728 | 0.3399 | 0.3719 | 0.2459 |
| S2, T=8, B | 0.1199 | 0.1361 | 0.0968 | 0.2461 | 0.2686 | 0.2271 | 0.3374 | 0.3938 | 0.2526 |
| S2, T=8, C | 0.1178 | 0.1291 | 0.0948 | 0.2528 | 0.2612 | 0.2348 | 0.3399 | 0.3862 | 0.2445 |
| S3, T=3, A | 0.1200 | 0.1383 | 0.1009 | 0.2550 | 0.2793 | 0.2411 | 0.3435 | 0.4090 | 0.2448 |
| S3, T=3, B | 0.1296 | 0.1283 | 0.1105 | 0.2826 | 0.2953 | 0.2772 | 0.3557 | 0.4035 | 0.2450 |
| S3, T=3, C | 0.1227 | 0.1353 | 0.1049 | 0.2641 | 0.2788 | 0.2486 | 0.3543 | 0.3949 | 0.2442 |
| S3, T=8, A | 0.1036 | 0.1065 | 0.0922 | 0.2219 | 0.2305 | 0.2127 | 0.2950 | 0.3263 | 0.2315 |
| S3, T=8, B | 0.1039 | 0.1176 | 0.0881 | 0.2189 | 0.2416 | 0.1996 | 0.2923 | 0.3460 | 0.2389 |
| S3, T=8, C | 0.1046 | 0.1116 | 0.0895 | 0.2098 | 0.2213 | 0.1920 | 0.2915 | 0.3278 | 0.2321 |


Table: Table 8. Median absolute error (a robust accuracy measure) of MLE, pivotal and order-restricted Bayes (P2) estimators.

![Figure 2: MSE (top) and median absolute error (bottom), both on log scales, against the number of units per facility for $T=3$ (scheme A). The MSE of the MLE (and OR-MLE) of the shapes is erratic because a handful of samples give huge estimates, while its median error decreases normally.](../results/figures/sim_mse_T3.png){width=100%}

![Figure 3: Same as Figure 2 for the long test $T=8$.](../results/figures/sim_mse_T8.png){width=100%}

![Figure 4: Box plots of the 2,000 estimates of $\alpha_1$ in three designs. The MLE has a very long right tail when samples are small or tests are short; the pivotal and Bayes estimators do not.](../results/figures/sim_boxplots_alpha1.png){width=100%}

## 8.3 Results: interval estimation

| Design | AL ACI | AL GCI | AL HPD-U | AL HPD-OR | CP ACI | CP GCI | CP HPD-U | CP HPD-OR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 148.250 | 6.960 | 1.084 | 0.937 | 0.957 | 0.968 | 0.987 | 0.988 |
| S1, T=3, B | 198.894 | 7.504 | 1.111 | 0.930 | 0.965 | 0.970 | 0.979 | 0.981 |
| S1, T=3, C | 1036.344 | 7.024 | 1.100 | 0.938 | 0.958 | 0.968 | 0.985 | 0.985 |
| S1, T=8, A | 46.192 | 4.527 | 0.955 | 0.837 | 0.967 | 0.964 | 0.987 | 0.989 |
| S1, T=8, B | 1.669 | 5.337 | 0.937 | 0.800 | 0.965 | 0.948 | 0.983 | 0.979 |
| S1, T=8, C | 3.370 | 4.135 | 0.918 | 0.798 | 0.959 | 0.948 | 0.980 | 0.979 |
| S2, T=3, A | 2.603 | 5.395 | 0.935 | 0.848 | 0.953 | 0.956 | 0.984 | 0.985 |
| S2, T=3, B | 43.748 | 5.413 | 0.956 | 0.850 | 0.960 | 0.953 | 0.983 | 0.980 |
| S2, T=3, C | 92.821 | 5.136 | 0.933 | 0.840 | 0.950 | 0.959 | 0.977 | 0.977 |
| S2, T=8, A | 0.932 | 1.952 | 0.726 | 0.671 | 0.954 | 0.953 | 0.971 | 0.972 |
| S2, T=8, B | 0.889 | 2.419 | 0.749 | 0.674 | 0.959 | 0.948 | 0.970 | 0.972 |
| S2, T=8, C | 1.275 | 1.969 | 0.720 | 0.658 | 0.958 | 0.958 | 0.973 | 0.974 |
| S3, T=3, A | 67.424 | 3.903 | 0.815 | 0.763 | 0.954 | 0.949 | 0.975 | 0.977 |
| S3, T=3, B | 1.870 | 3.548 | 0.840 | 0.774 | 0.958 | 0.955 | 0.977 | 0.978 |
| S3, T=3, C | 1.321 | 3.646 | 0.825 | 0.765 | 0.958 | 0.956 | 0.980 | 0.976 |
| S3, T=8, A | 0.710 | 1.297 | 0.619 | 0.586 | 0.953 | 0.943 | 0.967 | 0.969 |
| S3, T=8, B | 0.705 | 1.353 | 0.639 | 0.592 | 0.959 | 0.952 | 0.971 | 0.972 |
| S3, T=8, C | 0.689 | 1.186 | 0.618 | 0.578 | 0.965 | 0.949 | 0.970 | 0.971 |


Table: Table 9. Average length (AL) and coverage probability (CP) of 95% intervals for $\alpha_1$.

| Design | AL ACI | AL GCI | AL HPD-U | AL HPD-OR | CP ACI | CP GCI | CP HPD-U | CP HPD-OR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 2.555 | 6.008 | 2.044 | 2.041 | 0.855 | 0.974 | 0.992 | 0.991 |
| S1, T=3, B | 2.390 | 6.173 | 1.979 | 1.969 | 0.815 | 0.970 | 0.988 | 0.988 |
| S1, T=3, C | 2.567 | 6.133 | 2.044 | 2.039 | 0.864 | 0.971 | 0.990 | 0.993 |
| S1, T=8, A | 2.118 | 4.594 | 1.830 | 1.831 | 0.812 | 0.967 | 0.973 | 0.971 |
| S1, T=8, B | 2.272 | 4.629 | 1.886 | 1.888 | 0.832 | 0.949 | 0.959 | 0.959 |
| S1, T=8, C | 2.158 | 4.485 | 1.837 | 1.835 | 0.831 | 0.956 | 0.958 | 0.956 |
| S2, T=3, A | 2.253 | 3.783 | 1.836 | 1.833 | 0.881 | 0.962 | 0.972 | 0.970 |
| S2, T=3, B | 2.226 | 4.030 | 1.832 | 1.827 | 0.856 | 0.957 | 0.965 | 0.967 |
| S2, T=3, C | 2.276 | 3.977 | 1.857 | 1.858 | 0.882 | 0.960 | 0.972 | 0.970 |
| S2, T=8, A | 1.862 | 2.912 | 1.601 | 1.599 | 0.873 | 0.961 | 0.952 | 0.952 |
| S2, T=8, B | 1.991 | 3.044 | 1.675 | 1.672 | 0.882 | 0.942 | 0.952 | 0.954 |
| S2, T=8, C | 1.923 | 2.913 | 1.630 | 1.631 | 0.876 | 0.950 | 0.947 | 0.951 |
| S3, T=3, A | 2.029 | 2.978 | 1.683 | 1.683 | 0.904 | 0.958 | 0.962 | 0.959 |
| S3, T=3, B | 2.008 | 3.179 | 1.684 | 1.672 | 0.869 | 0.953 | 0.958 | 0.948 |
| S3, T=3, C | 2.050 | 3.093 | 1.701 | 1.697 | 0.899 | 0.960 | 0.965 | 0.961 |
| S3, T=8, A | 1.673 | 2.353 | 1.465 | 1.461 | 0.883 | 0.944 | 0.935 | 0.933 |
| S3, T=8, B | 1.742 | 2.413 | 1.517 | 1.515 | 0.901 | 0.953 | 0.957 | 0.955 |
| S3, T=8, C | 1.698 | 2.337 | 1.486 | 1.483 | 0.897 | 0.953 | 0.950 | 0.950 |


Table: Table 10. AL and CP of 95% intervals for $\lambda$.

| Design | AL ACI | AL GCI | AL HPD-U | AL HPD-OR | CP ACI | CP GCI | CP HPD-U | CP HPD-OR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.312 | 0.306 | 0.281 | 0.238 | 0.943 | 0.964 | 0.950 | 0.946 |
| S1, T=3, B | 0.342 | 0.334 | 0.300 | 0.252 | 0.923 | 0.954 | 0.941 | 0.941 |
| S1, T=3, C | 0.324 | 0.317 | 0.290 | 0.243 | 0.930 | 0.956 | 0.941 | 0.942 |
| S1, T=8, A | 0.296 | 0.296 | 0.265 | 0.230 | 0.905 | 0.935 | 0.923 | 0.935 |
| S1, T=8, B | 0.323 | 0.321 | 0.290 | 0.248 | 0.923 | 0.958 | 0.947 | 0.929 |
| S1, T=8, C | 0.302 | 0.302 | 0.271 | 0.236 | 0.906 | 0.942 | 0.932 | 0.918 |
| S2, T=3, A | 0.256 | 0.253 | 0.237 | 0.207 | 0.943 | 0.958 | 0.944 | 0.950 |
| S2, T=3, B | 0.277 | 0.275 | 0.251 | 0.217 | 0.944 | 0.966 | 0.949 | 0.941 |
| S2, T=3, C | 0.265 | 0.263 | 0.244 | 0.212 | 0.934 | 0.953 | 0.940 | 0.941 |
| S2, T=8, A | 0.239 | 0.243 | 0.221 | 0.200 | 0.933 | 0.961 | 0.938 | 0.944 |
| S2, T=8, B | 0.260 | 0.262 | 0.241 | 0.214 | 0.928 | 0.950 | 0.947 | 0.942 |
| S2, T=8, C | 0.247 | 0.250 | 0.229 | 0.206 | 0.929 | 0.949 | 0.942 | 0.947 |
| S3, T=3, A | 0.221 | 0.221 | 0.207 | 0.187 | 0.939 | 0.952 | 0.940 | 0.949 |
| S3, T=3, B | 0.240 | 0.242 | 0.222 | 0.197 | 0.936 | 0.964 | 0.940 | 0.952 |
| S3, T=3, C | 0.230 | 0.230 | 0.215 | 0.191 | 0.944 | 0.961 | 0.948 | 0.957 |
| S3, T=8, A | 0.208 | 0.213 | 0.196 | 0.180 | 0.931 | 0.945 | 0.937 | 0.940 |
| S3, T=8, B | 0.226 | 0.229 | 0.213 | 0.193 | 0.934 | 0.954 | 0.938 | 0.941 |
| S3, T=8, C | 0.216 | 0.220 | 0.204 | 0.186 | 0.944 | 0.950 | 0.947 | 0.945 |


Table: Table 11. AL and CP of 95% intervals for $R_1(1)$.

![Figure 5: Coverage of the 95% intervals ($T=8$, scheme A). The dotted line marks the nominal 0.95.](../results/figures/sim_coverage_T8.png){width=100%}

## 8.4 Discussion of the simulation results

1. **Consistency.** For every method, bias, MSE, MedAE and interval length decrease as $n_i$ grows (S1 → S3) and as the test gets longer ($T=3\to8$). This matches the IER and Burr-XII studies [1, 2].
2. **The MLE of the shape parameters is fragile.** In small or short tests the MSE of $\hat\alpha_i$ is huge (Tables 4–5), sometimes above 100. The median absolute error (Table 8) and the box plots (Figure 4) show why: the *typical* MLE is reasonable, but a few samples look almost exponential. In those samples $\hat\lambda$ is pushed towards 0 and $\hat\alpha_i$ explodes (Theorem 2). The OR-MLE inherits this. The problem disappears for $n_i\ge30$ with $T=8$.
3. **Pivotal estimates are stable.** The median-based pivotal estimator of $\alpha_i$ has MSEs between about 0.05 and 1.1 in every design, often two to three orders of magnitude smaller than the MLE in small samples. For $\lambda$, however, the pivotal estimator has a *larger* MSE than the MLE, because its pivotal distribution is very right-skewed (Theorem 3).
4. **Pivotal intervals have the best frequentist coverage.** The GCIs have coverage between 93.5% and 97.5% in every design (Tables 9–11), even though the pivots are only approximate under GPHCS. The ACIs for $\lambda$ under-cover badly (81–90%) because $\hat\lambda$ is far from normal. Those for the reliabilities are slightly below 95%. GCIs are longer, the honest price for correct coverage.
5. **Bayes estimators have the smallest errors** for every quantity, under both priors. As expected, the informative prior P1 gains more than the weakly informative P2. The HPD intervals are the shortest of all. Their coverage is at or above 95% for $\alpha_i$ and $\lambda$, and close to nominal (92–96%) for the reliabilities. Coverage is slightly conservative for $\lambda$ because the priors are centred at the truth.
6. **The order restriction pays off.** Bayes-OR has the lowest MSE for $\alpha_1$, $\alpha_2$ and all three reliabilities, with shorter HPD intervals than Bayes-U and similar coverage. For the largest parameter $\alpha_3$ the restriction pushes the estimate up slightly, which adds a little bias. For $\lambda$ the restriction makes no difference, as expected since it concerns only $\boldsymbol\alpha$.
7. **Loss function.** LINEX ($h=1$) gives smaller estimates than SELF. Because the posteriors of $\alpha_3$ and $\lambda$ are right-skewed, this reduces their MSE (Table B6, Appendix B).
8. **Censoring scheme.** Removing units at the first failure (scheme B) gives the fewest failures and usually the largest errors. Scheme A (removals at the end) gives the most information. Increasing $T$ moves facilities from Cases I/II to Case III and increases $D$ (Table 3).

**Pilot runs.** A pilot with lighter tails ($\boldsymbol\alpha=(1,1.5,2)$, $\lambda=1$, $n_i=20$, $T=1$) produced non-existent MLEs in about 20% of samples and an extremely unstable MLE. This confirms that Lomax inference from censored data requires the tail to be visible within the test window, which is a design lesson for practitioners.

# 9. Real data analysis

## 9.1 Data and motivation

Reference [2] analyses the number of fatalities in British coal-mining disasters that claimed **10 or more lives** (Hand et al., 1993; Jarrett, 1979, cited in [2]), scaled as reported there. Three 20-year periods (1900–1919, 1920–1939, 1940–1959) play the role of three test facilities. The scaled values are recorded in steps of 0.02 (one death) starting at 0.20, the recording threshold. Following the peaks-over-threshold idea (Section 2.1), we model the **excess over the threshold**, with a half-step continuity correction:

$$y=x-0.19\qquad(\text{so }y\ge0.01>0).$$

The Lomax distribution is the natural model for exceedances. A common $\lambda$ with period-specific $\alpha_i$ means that the tail heaviness of disasters changed over time. Mine safety improved over the century (legislation, better ventilation and rescue methods), so catastrophic disasters should have become rarer. In other words the excess fatality should become stochastically smaller, which is exactly the order restriction $\alpha_1\le\alpha_2\le\alpha_3$.

| Period | n | Min | Q1 | Median | Mean | Q3 | Max | SD | CV | Skewness |
|:---------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1900-1919 | 21 | 0.010 | 0.130 | 0.510 | 1.490 | 1.570 | 8.590 | 2.251 | 1.511 | 2.325 |
| 1920-1939 | 20 | 0.010 | 0.065 | 0.270 | 0.586 | 0.620 | 5.110 | 1.131 | 1.930 | 3.708 |
| 1940-1959 | 17 | 0.010 | 0.050 | 0.130 | 0.379 | 0.250 | 1.890 | 0.555 | 1.462 | 1.934 |


Table: Table 12. Descriptive statistics of the excess fatalities $y$. All three periods are strongly right-skewed with coefficient of variation above 1, so the existence condition of Theorem 2 holds.

![Figure 6: Box plots (log scale) and empirical survival functions of the excess fatalities by period. Later periods lie lower, consistent with lighter tails.](../results/figures/real_descriptive.png){width=100%}

## 9.2 Goodness of fit and hypothesis tests

| Period | Model | α̂ | λ̂ | log L | AIC | KS D | KS p |
|:---------|-------------------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1900-1919 | Lomax (separate lam) | 1.328 | 1.244 | -26.29 | 56.57 | 0.107 | 0.950 |
| 1900-1919 | Exponential | – | 0.671 | -29.37 | 60.75 | 0.244 | 0.137 |
| 1920-1939 | Lomax (separate lam) | 1.294 | 3.626 | -4.54 | 13.08 | 0.154 | 0.677 |
| 1920-1939 | Exponential | – | 1.706 | -9.31 | 20.62 | 0.308 | 0.035 |
| 1940-1959 | Lomax (separate lam) | 1.532 | 3.787 | 1.80 | 0.40 | 0.125 | 0.924 |
| 1940-1959 | Exponential | – | 2.636 | -0.52 | 3.05 | 0.282 | 0.109 |
| 1900-1919 | Lomax (common lam) | 0.913 | 2.514 | – | – | 0.123 | 0.868 |
| 1920-1939 | Lomax (common lam) | 1.602 | 2.514 | – | – | 0.171 | 0.544 |
| 1940-1959 | Lomax (common lam) | 1.981 | 2.514 | – | – | 0.145 | 0.817 |


Table: Table 13. Goodness of fit on the complete data: separate Lomax fits, exponential fits, and the proposed Lomax model with common $\lambda$ (Kolmogorov–Smirnov distance and $p$-value).

| Test | Data | −2 log Λ | df | p-value |
|:----------------------------------|---------:|---------:|---------:|---------:|
| H₀: λ₁ = λ₂ = λ₃ (common scale) | Complete | 0.859 | 2 | 0.6509 |
| H₀: α₁ = α₂ = α₃ (no period effect) | Complete | 5.938 | 2 | 0.0513 |
| H₀: α₁ = α₂ = α₃ (no period effect) | Scheme 1 | 3.914 | 2 | 0.1413 |
| H₀: α₁ = α₂ = α₃ (no period effect) | Scheme 2 | 7.266 | 2 | 0.0264 |


Table: Table 14. Likelihood ratio tests (asymptotic $\chi^2_2$).

* Lomax has a **lower AIC than the exponential** in every period, and the exponential is rejected for 1920–1939 (KS $p=0.035$).
* Under the proposed model (common $\lambda$), all KS $p$-values exceed 0.5. The CDF, P–P and Q–Q plots (Figure 7) show a good fit.
* The LRT does **not reject a common $\lambda$** ($p=0.65$), which supports assumption A2.
* The LRT of *no period effect* ($\alpha_1=\alpha_2=\alpha_3$) is borderline on the complete data ($p=0.051$) and significant under censoring scheme 2 ($p=0.026$). So there is evidence of a difference between "facilities", and the estimates are ordered as the safety argument predicts.

![Figure 7: Empirical vs fitted CDF, P–P and Q–Q plots for the Lomax model with common $\lambda$.](../results/figures/real_gof.png){width=100%}

## 9.3 Block GPHC samples

To illustrate the methods, two BGPHC samples were generated from the complete data. Both use $(m_1,m_2,m_3)=(17,16,14)$ and $(\kappa_1,\kappa_2,\kappa_3)=(12,11,10)$:

* **Scheme 1:** $T=0.5$, removals at the end ($R=(0,\dots,0,n_i-m_i)$). The three periods end in Cases I, II and II.
* **Scheme 2:** $T=1.0$, one randomly chosen unit withdrawn at the first failure (seed 2025) and the rest at the end. The periods end in Cases II, III and III.

The complete data serve as a benchmark.

| Design | Period | Case | Dᵢ | Observed failures xᵢⱼ | Removals at failures | Rᵢᵀ |
|:---------|---------:|---------:|---------:|-----------------------------------:|-------------------:|---------:|
| Scheme 1 | 1900-1919 | I | 12 | 0.01 0.03 0.05 0.07 0.09 0.13 0.31 0.35 0.35 0.47 0.51 0.59 | R₁₂=9 | 0 |
| Scheme 1 | 1920-1939 | II | 14 | 0.01 0.01 0.03 0.03 0.05 0.07 0.07 0.09 0.09 0.19 0.35 0.35 0.35 0.41 | none | 6 |
| Scheme 1 | 1940-1959 | II | 13 | 0.01 0.03 0.05 0.05 0.05 0.07 0.11 0.11 0.13 0.15 0.19 0.23 0.25 | none | 4 |
| Scheme 2 | 1900-1919 | II | 12 | 0.01 0.03 0.05 0.07 0.09 0.13 0.31 0.35 0.35 0.51 0.59 0.85 | R₁=1 | 8 |
| Scheme 2 | 1920-1939 | III | 16 | 0.01 0.01 0.03 0.03 0.05 0.07 0.07 0.09 0.09 0.19 0.35 0.35 0.35 0.41 0.59 0.71 | R₁=1, R₁₆=3 | 0 |
| Scheme 2 | 1940-1959 | III | 14 | 0.01 0.03 0.05 0.05 0.05 0.07 0.11 0.11 0.13 0.15 0.19 0.23 0.25 0.71 | R₁=1, R₁₄=2 | 0 |


Table: Table 15. The censored samples (failures observed, removals at each failure, units withdrawn at $T$).

![Figure 8: Profile log-likelihood of $\lambda$ for the complete data and the two censored samples. Each curve has a single maximum, so the MLE exists and is unique (Theorem 2, condition $S_0>0$).](../results/figures/real_profile_loglik.png){width=100%}

## 9.4 Estimates

Mission time $t_0=0.5$, i.e. an excess of 25 deaths above the threshold. Then $R_i(0.5)$ is the probability that a disaster in period $i$ kills more than about 35 people. Pivotal results use $N=10{,}000$ draws and Bayes results use $M=20{,}000$ importance draws. The prior is centred at the MLE with $f=1$ (Section 6.2).

| Quantity | MLE | PIV | OR-MLE | BU-SE | BU-LX | BOR-SE | BOR-LX |
|:---------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| α₁ | 0.9129 | 0.7937 | 0.9129 | 0.9799 | 0.9292 | 0.9813 | 0.9305 |
| α₂ | 1.6024 | 1.3416 | 1.6024 | 1.7531 | 1.5702 | 1.6628 | 1.5114 |
| α₃ | 1.9811 | 1.6380 | 1.9811 | 2.1777 | 1.8766 | 2.3283 | 2.0363 |
| ᾱ (pooled) | 1.1414 | 0.9710 | 1.1414 | 1.2324 | 1.1625 | 1.2508 | 1.1790 |
| λ | 2.5140 | 3.3257 | 2.5140 | 2.6748 | 2.0979 | 2.6627 | 2.0902 |
| R₁(0.5) | 0.4756 | 0.4703 | 0.4756 | 0.4873 | 0.4840 | 0.4880 | 0.4850 |
| R₂(0.5) | 0.2713 | 0.2751 | 0.2713 | 0.2866 | 0.2835 | 0.3031 | 0.3007 |
| R₃(0.5) | 0.1993 | 0.2079 | 0.1993 | 0.2180 | 0.2150 | 0.1925 | 0.1906 |
| h₁(0.5) | 1.0169 | 0.9603 | 1.0169 | 1.0025 | 0.9799 | 1.0007 | 0.9793 |
| h₂(0.5) | 1.7849 | 1.6430 | 1.7849 | 1.7726 | 1.6952 | 1.6791 | 1.6262 |
| h₃(0.5) | 2.2067 | 2.0015 | 2.2067 | 2.1915 | 2.0522 | 2.3452 | 2.2332 |
| MdTF₁ | 0.4521 | 0.4396 | 0.4521 | 0.5092 | 0.4927 | 0.5093 | 0.4933 |
| MdTF₂ | 0.2153 | 0.2082 | 0.2153 | 0.2365 | 0.2337 | 0.2482 | 0.2459 |
| MdTF₃ | 0.1666 | 0.1630 | 0.1666 | 0.1846 | 0.1828 | 0.1659 | 0.1649 |


Table: Table 16. Point estimates for the complete data.

| Quantity | ACI (MLE) | GCI (pivotal) | HPD (Bayes-U) | HPD (Bayes-OR) |
|:---------|---------------------:|---------------------:|---------------------:|---------------------:|
| α₁ | [0.267, 1.558] (1.291) | [0.404, 1.751] (1.347) | [0.437, 1.651] (1.215) | [0.424, 1.662] (1.238) |
| α₂ | [0.297, 2.908] (2.611) | [0.645, 3.493] (2.848) | [0.672, 3.111] (2.439) | [0.680, 2.891] (2.212) |
| α₃ | [0.243, 3.720] (3.477) | [0.717, 4.429] (3.712) | [0.777, 4.017] (3.240) | [0.936, 4.084] (3.147) |
| ᾱ (pooled) | [0.360, 1.923] (1.564) | [0.527, 2.168] (1.641) | [0.613, 2.056] (1.443) | [0.618, 2.091] (1.473) |
| λ | [0.000, 5.373] (5.373) | [0.893, 9.855] (8.962) | [0.671, 5.306] (4.635) | [0.627, 5.246] (4.620) |
| R₁(0.5) | [0.306, 0.645] (0.338) | [0.308, 0.637] (0.329) | [0.324, 0.637] (0.313) | [0.336, 0.639] (0.304) |
| R₂(0.5) | [0.113, 0.430] (0.317) | [0.140, 0.454] (0.314) | [0.134, 0.437] (0.304) | [0.175, 0.443] (0.268) |
| R₃(0.5) | [0.046, 0.353] (0.307) | [0.086, 0.396] (0.310) | [0.079, 0.372] (0.293) | [0.081, 0.318] (0.237) |
| h₁(0.5) | [0.578, 1.456] (0.878) | [0.589, 1.448] (0.859) | [0.586, 1.415] (0.829) | [0.584, 1.397] (0.812) |
| h₂(0.5) | [0.937, 2.633] (1.695) | [0.964, 2.585] (1.621) | [1.023, 2.590] (1.567) | [1.074, 2.341] (1.267) |
| h₃(0.5) | [1.038, 3.376] (2.338) | [1.087, 3.335] (2.249) | [1.182, 3.283] (2.102) | [1.453, 3.340] (1.887) |
| MdTF₁ | [0.138, 0.767] (0.629) | [0.213, 0.967] (0.755) | [0.218, 0.883] (0.665) | [0.226, 0.878] (0.652) |
| MdTF₂ | [0.083, 0.347] (0.264) | [0.111, 0.414] (0.303) | [0.110, 0.384] (0.274) | [0.127, 0.383] (0.256) |
| MdTF₃ | [0.063, 0.270] (0.207) | [0.088, 0.328] (0.240) | [0.081, 0.300] (0.219) | [0.083, 0.253] (0.170) |


Table: Table 17. 95% interval estimates (length in brackets) for the complete data.

| Quantity | MLE | PIV | OR-MLE | BU-SE | BU-LX | BOR-SE | BOR-LX |
|:---------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| α₁ | 0.5725 | 0.7227 | 0.5725 | 0.6624 | 0.6158 | 0.6544 | 0.6106 |
| α₂ | 0.9701 | 1.2455 | 0.9701 | 1.1344 | 1.0081 | 1.0917 | 0.9847 |
| α₃ | 1.2418 | 1.6290 | 1.2418 | 1.4629 | 1.2570 | 1.5867 | 1.3685 |
| ᾱ (pooled) | 0.7356 | 0.8058 | 0.7356 | 0.8626 | 0.7994 | 0.8585 | 0.7960 |
| λ | 5.0852 | 3.4225 | 5.0852 | 5.3913 | 3.2574 | 5.3015 | 3.2207 |
| R₁(0.5) | 0.4848 | 0.4937 | 0.4848 | 0.4957 | 0.4910 | 0.5024 | 0.4984 |
| R₂(0.5) | 0.2932 | 0.2949 | 0.2932 | 0.3083 | 0.3039 | 0.3208 | 0.3178 |
| R₃(0.5) | 0.2079 | 0.2041 | 0.2079 | 0.2254 | 0.2215 | 0.1982 | 0.1957 |
| h₁(0.5) | 0.8218 | 0.8865 | 0.8218 | 0.8455 | 0.8102 | 0.8297 | 0.7993 |
| h₂(0.5) | 1.3925 | 1.5423 | 1.3925 | 1.4410 | 1.3461 | 1.3788 | 1.3104 |
| h₃(0.5) | 1.7825 | 2.0102 | 1.7825 | 1.8532 | 1.6907 | 1.9988 | 1.8446 |
| MdTF₁ | 0.4633 | 0.4865 | 0.4633 | 0.6383 | 0.5435 | 0.6287 | 0.5553 |
| MdTF₂ | 0.2051 | 0.2239 | 0.2051 | 0.2406 | 0.2350 | 0.2458 | 0.2422 |
| MdTF₃ | 0.1470 | 0.1596 | 0.1470 | 0.1701 | 0.1675 | 0.1488 | 0.1477 |


Table: Table 18. Point estimates for censored sample Scheme 1.

| Quantity | ACI (MLE) | GCI (pivotal) | HPD (Bayes-U) | HPD (Bayes-OR) |
|:---------|-----------------------:|-----------------------:|-----------------------:|-----------------------:|
| α₁ | [0.000, 1.169] (1.169) | [0.252, 9.142] (8.891) | [0.199, 1.312] (1.113) | [0.195, 1.276] (1.081) |
| α₂ | [0.000, 2.024] (2.024) | [0.409, 18.267] (17.858) | [0.348, 2.286] (1.938) | [0.340, 2.131] (1.791) |
| α₃ | [0.000, 2.674] (2.674) | [0.505, 24.238] (23.734) | [0.402, 2.980] (2.578) | [0.506, 3.151] (2.645) |
| ᾱ (pooled) | [0.009, 1.462] (1.454) | [0.298, 10.565] (10.267) | [0.316, 1.629] (1.314) | [0.329, 1.640] (1.310) |
| λ | [0.000, 12.753] (12.753) | [0.163, 16.125] (15.962) | [0.954, 11.631] (10.677) | [0.848, 11.490] (10.641) |
| R₁(0.5) | [0.286, 0.683] (0.397) | [0.304, 0.687] (0.383) | [0.303, 0.676] (0.373) | [0.331, 0.679] (0.348) |
| R₂(0.5) | [0.102, 0.485] (0.383) | [0.133, 0.510] (0.376) | [0.132, 0.494] (0.362) | [0.177, 0.475] (0.298) |
| R₃(0.5) | [0.023, 0.393] (0.370) | [0.068, 0.423] (0.354) | [0.069, 0.407] (0.337) | [0.063, 0.334] (0.271) |
| h₁(0.5) | [0.228, 1.416] (1.188) | [0.412, 1.774] (1.362) | [0.361, 1.387] (1.026) | [0.381, 1.338] (0.957) |
| h₂(0.5) | [0.358, 2.427] (2.068) | [0.683, 3.233] (2.551) | [0.646, 2.365] (1.719) | [0.695, 2.144] (1.449) |
| h₃(0.5) | [0.351, 3.214] (2.863) | [0.859, 4.353] (3.494) | [0.782, 3.078] (2.297) | [0.987, 3.207] (2.220) |
| MdTF₁ | [0.011, 0.915] (0.904) | [0.219, 1.676] (1.457) | [0.153, 1.374] (1.221) | [0.170, 1.339] (1.168) |
| MdTF₂ | [0.051, 0.360] (0.309) | [0.111, 0.525] (0.414) | [0.086, 0.451] (0.365) | [0.110, 0.410] (0.300) |
| MdTF₃ | [0.039, 0.255] (0.215) | [0.081, 0.354] (0.272) | [0.066, 0.312] (0.246) | [0.069, 0.243] (0.174) |


Table: Table 19. 95% interval estimates (length) for Scheme 1.

| Quantity | MLE | PIV | OR-MLE | BU-SE | BU-LX | BOR-SE | BOR-LX |
|:---------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| α₁ | 0.5148 | 0.5311 | 0.5148 | 0.5707 | 0.5421 | 0.5798 | 0.5509 |
| α₂ | 1.1184 | 1.1561 | 1.1184 | 1.2633 | 1.1355 | 1.1776 | 1.0709 |
| α₃ | 1.4028 | 1.4488 | 1.4028 | 1.5964 | 1.3891 | 1.7063 | 1.4932 |
| ᾱ (pooled) | 0.6763 | 0.6330 | 0.6763 | 0.7593 | 0.7192 | 0.7750 | 0.7333 |
| λ | 5.0357 | 4.6780 | 5.0357 | 5.3627 | 3.3873 | 5.3134 | 3.3821 |
| R₁(0.5) | 0.5234 | 0.5375 | 0.5234 | 0.5351 | 0.5307 | 0.5306 | 0.5264 |
| R₂(0.5) | 0.2449 | 0.2520 | 0.2449 | 0.2607 | 0.2572 | 0.2828 | 0.2799 |
| R₃(0.5) | 0.1713 | 0.1782 | 0.1713 | 0.1892 | 0.1860 | 0.1654 | 0.1635 |
| h₁(0.5) | 0.7369 | 0.7241 | 0.7369 | 0.7453 | 0.7203 | 0.7560 | 0.7315 |
| h₂(0.5) | 1.6010 | 1.6005 | 1.6010 | 1.6365 | 1.5335 | 1.5234 | 1.4459 |
| h₃(0.5) | 2.0081 | 2.0038 | 2.0081 | 2.0637 | 1.8844 | 2.2035 | 2.0388 |
| MdTF₁ | 0.5648 | 0.6045 | 0.5648 | 0.7940 | 0.6602 | 0.7763 | 0.6425 |
| MdTF₂ | 0.1705 | 0.1807 | 0.1705 | 0.1943 | 0.1915 | 0.2099 | 0.2073 |
| MdTF₃ | 0.1269 | 0.1340 | 0.1269 | 0.1445 | 0.1429 | 0.1282 | 0.1275 |


Table: Table 20. Point estimates for censored sample Scheme 2.

| Quantity | ACI (MLE) | GCI (pivotal) | HPD (Bayes-U) | HPD (Bayes-OR) |
|:---------|-----------------------:|-----------------------:|-----------------------:|----------------------:|
| α₁ | [0.059, 0.971] (0.912) | [0.221, 2.285] (2.064) | [0.191, 1.068] (0.877) | [0.186, 1.075] (0.889) |
| α₂ | [0.078, 2.158] (2.080) | [0.475, 5.751] (5.276) | [0.399, 2.381] (1.982) | [0.443, 2.219] (1.776) |
| α₃ | [0.000, 2.808] (2.808) | [0.541, 7.775] (7.234) | [0.536, 3.116] (2.579) | [0.583, 3.199] (2.615) |
| ᾱ (pooled) | [0.120, 1.232] (1.112) | [0.285, 2.825] (2.540) | [0.311, 1.358] (1.047) | [0.328, 1.405] (1.076) |
| λ | [0.000, 11.578] (11.578) | [0.584, 17.331] (16.747) | [0.824, 11.064] (10.240) | [0.998, 10.746] (9.748) |
| R₁(0.5) | [0.331, 0.716] (0.385) | [0.352, 0.721] (0.369) | [0.350, 0.713] (0.364) | [0.355, 0.706] (0.351) |
| R₂(0.5) | [0.075, 0.415] (0.339) | [0.114, 0.450] (0.337) | [0.107, 0.430] (0.323) | [0.136, 0.426] (0.291) |
| R₃(0.5) | [0.009, 0.333] (0.324) | [0.060, 0.387] (0.327) | [0.044, 0.342] (0.298) | [0.054, 0.292] (0.238) |
| h₁(0.5) | [0.261, 1.213] (0.953) | [0.360, 1.339] (0.979) | [0.332, 1.196] (0.864) | [0.356, 1.218] (0.862) |
| h₂(0.5) | [0.567, 2.635] (2.069) | [0.796, 3.003] (2.207) | [0.784, 2.603] (1.819) | [0.798, 2.352] (1.554) |
| h₃(0.5) | [0.575, 3.441] (2.865) | [0.917, 4.064] (3.147) | [0.954, 3.369] (2.415) | [1.103, 3.392] (2.288) |
| MdTF₁ | [0.000, 1.147] (1.147) | [0.248, 2.492] (2.244) | [0.176, 1.772] (1.596) | [0.177, 1.707] (1.531) |
| MdTF₂ | [0.051, 0.290] (0.238) | [0.090, 0.397] (0.307) | [0.078, 0.344] (0.266) | [0.094, 0.349] (0.255) |
| MdTF₃ | [0.038, 0.216] (0.177) | [0.068, 0.298] (0.229) | [0.056, 0.250] (0.194) | [0.064, 0.208] (0.145) |


Table: Table 21. 95% interval estimates (length) for Scheme 2.

![Figure 9: Posterior densities (Bayes unrestricted and order-restricted) and pivotal distributions for censored Scheme 1; dotted line = MLE.](../results/figures/real_posteriors_scheme1.png){width=100%}

![Figure 10: Estimated reliability and hazard functions for each period under Scheme 1 (thick: Bayes-OR, thin: MLE).](../results/figures/real_curves_scheme1.png){width=100%}

| Design: quantity | f=0.5 BU | f=0.5 BOR | f=1 BU | f=1 BOR | f=2 BU | f=2 BOR |
|:----------------|------------:|------------:|-------------:|-------------:|-------------:|-------------:|
| Scheme 1: α₁ | 0.603 (0.800) | 0.585 (0.734) | 0.667 (1.123) | 0.657 (1.097) | 0.814 (1.758) | 0.799 (1.695) |
| Scheme 1: α₂ | 1.023 (1.297) | 0.967 (1.093) | 1.140 (1.956) | 1.095 (1.781) | 1.426 (3.179) | 1.344 (2.996) |
| Scheme 1: α₃ | 1.306 (1.711) | 1.397 (1.610) | 1.477 (2.616) | 1.592 (2.631) | 1.857 (4.181) | 1.965 (4.448) |
| Scheme 1: λ | 5.229 (7.122) | 5.246 (7.101) | 5.335 (10.493) | 5.260 (10.483) | 5.118 (11.845) | 5.146 (11.549) |
| Scheme 1: R₁(0.5) | 0.496 (0.369) | 0.503 (0.339) | 0.495 (0.379) | 0.502 (0.349) | 0.497 (0.372) | 0.497 (0.346) |
| Scheme 1: R₂(0.5) | 0.309 (0.354) | 0.323 (0.290) | 0.308 (0.358) | 0.321 (0.295) | 0.304 (0.352) | 0.319 (0.306) |
| Scheme 1: R₃(0.5) | 0.228 (0.329) | 0.200 (0.256) | 0.224 (0.329) | 0.197 (0.261) | 0.222 (0.327) | 0.196 (0.264) |
| Scheme 2: α₁ | 0.538 (0.687) | 0.544 (0.678) | 0.577 (0.884) | 0.586 (0.898) | 0.630 (1.142) | 0.647 (1.165) |
| Scheme 2: α₂ | 1.170 (1.421) | 1.088 (1.215) | 1.269 (1.988) | 1.186 (1.810) | 1.419 (2.653) | 1.338 (2.468) |
| Scheme 2: α₃ | 1.467 (1.822) | 1.566 (1.761) | 1.607 (2.589) | 1.719 (2.661) | 1.802 (3.590) | 1.943 (3.718) |
| Scheme 2: λ | 5.206 (6.937) | 5.192 (6.900) | 5.273 (9.673) | 5.268 (9.989) | 5.378 (11.947) | 5.229 (11.385) |
| Scheme 2: R₁(0.5) | 0.532 (0.362) | 0.528 (0.350) | 0.533 (0.358) | 0.529 (0.352) | 0.536 (0.360) | 0.532 (0.352) |
| Scheme 2: R₂(0.5) | 0.260 (0.319) | 0.282 (0.284) | 0.261 (0.328) | 0.282 (0.293) | 0.260 (0.327) | 0.282 (0.291) |
| Scheme 2: R₃(0.5) | 0.190 (0.291) | 0.165 (0.229) | 0.188 (0.303) | 0.165 (0.237) | 0.191 (0.315) | 0.166 (0.244) |


Table: Table 22. Prior sensitivity: Bayes estimates (SELF) and 95% HPD intervals when the prior relative standard deviation is $f=0.5,1,2$.

## 9.5 Interpretation

1. **Facility (period) effect.** All methods give $\hat\alpha_1<\hat\alpha_2<\hat\alpha_3$. The excess-fatality distribution has the heaviest tail in 1900–1919 and the lightest in 1940–1959. On the complete data $\hat\alpha=(0.913,1.602,1.981)$ with $\hat\lambda=2.514$.
2. **Reliability.** On the complete data, $\hat R_1(0.5)\approx0.48$ for 1900–1919, $\hat R_2(0.5)\approx0.27$ for 1920–1939 and $\hat R_3(0.5)\approx0.20$ for 1940–1959 (Tables 16–21). The probability that a qualifying disaster kills more than about 35 people fell from roughly one-half to roughly one-fifth. The median excess falls in the same way, from $\widehat{\mathrm{MdTF}}_1\approx0.45$ to $\widehat{\mathrm{MdTF}}_3\approx0.17$, i.e. about 22 and 8 deaths above the threshold.
3. **Order restriction.** The unrestricted MLEs are already ordered, so OR-MLE = MLE. The order-restricted Bayes estimates differ slightly from the unrestricted ones and have shorter HPD intervals, particularly for the reliabilities. This matches the simulation findings.
4. **Effect of censoring.** The censored samples remove the largest disasters, so $\lambda$ becomes harder to estimate (wider intervals, larger $\hat\lambda$). The reliability estimates stay close to the complete-data values because $R(t_0)$ depends mainly on the observed part of the distribution.
5. **Methods.** In line with the simulation, the ACIs for $\lambda$ hit the lower bound 0, which signals that the normal approximation is unreliable here. The GCIs are wide but honest, and the HPD intervals are the shortest. Bayes estimates are stable across priors with $f=0.5$ to $2$ for the reliabilities (Table 22). The shapes and $\lambda$ are more prior-sensitive, as expected with about 40 failures.

# 10. Conclusions

* We formulated the likelihood of Lomax lifetimes under **block generalized progressive hybrid censoring**. This scheme unifies block progressive Type-II censoring [1, 2] and block Type-I hybrid censoring [3].
* The MLE of each facility effect has the closed form $D_i/\Delta_i(\lambda)$, and $\lambda$ solves a one-dimensional problem. The MLE exists when the data are "more variable than exponential" (Theorem 2).
* For the Lomax model the pivot for $\lambda$ is **decreasing and bounded** (Theorem 3), unlike the IER and Burr-XII cases [1, 2]. A truncated pivotal algorithm with **median-based** generalized estimates is needed, and it gives the most reliable frequentist intervals.
* The **joint Gamma–Dirichlet prior** [3] can be handled exactly by a simple importance sampler that exploits the gamma structure of the likelihood. Proper priors on $\lambda$ are necessary.
* **Order restrictions**, imposed by isotonic regression (MLE) or the ordered GD prior (Bayes), reduce estimation error whenever the ordering is genuine.
* On the coal-mining data the model fits well, a common $\lambda$ is supported, and the period effect is ordered. Large disasters became less likely over the 20th century.

**Recommendation.** Use the pivotal method when an objective, prior-free frequentist analysis is required and correct coverage matters. Use order-restricted Bayes when credible prior information about the ordering of the facilities is available. Avoid relying on the plain MLE and Wald intervals for Lomax shapes in small or short tests.

**Limitations and future work.**

* The pivots are approximate under GPHCS.
* Order-restricted MLE intervals could be obtained by bootstrap.
* Importance weights occasionally degenerate (minimum ESS in Appendix B) and could be stabilised with adaptive proposals.
* Optimal choice of $(m_i,\kappa_i,T_i,\boldsymbol R_i)$ (as in [3]) and facility-specific $\lambda_i$ are natural extensions.

# References {.unnumbered}

[1] Kumari, R., Tripathi, Y. M., Lodhi, C., & Wang, L. (2025). Pivotal based reliability inference for Burr-XII distribution under block progressive type-II censoring. *Communications in Statistics – Simulation and Computation*, 55(8), 3122–3143. https://doi.org/10.1080/03610918.2025.2483437

[2] Lodhi, C., Gangopadhyay, A. K., & Bhattacharya, P. (2025). Likelihood and pivotal-based reliability inference for inverse exponentiated Rayleigh distribution under block progressive censoring. *Quality Technology & Quantitative Management*, 22(4), 779–804. https://doi.org/10.1080/16843703.2024.2400435

[3] Sultana, F., Çetinkaya, Ç., Raqab, M. Z., & Kundu, D. (2026). Inference for block Type-I hybrid censored Weibull populations. *Communications in Statistics – Theory and Methods*, 55(11), 3436–3459. https://doi.org/10.1080/03610926.2025.2579720

[4] Cho, Y., Sun, H., & Lee, K. (2015). Exact likelihood inference for an exponential parameter under generalized progressive hybrid censoring scheme. *Statistical Methodology*, 23, 18–34. https://doi.org/10.1016/j.stamet.2014.09.002 — *the one additional reference used in this project.*

[5] Course texts: Casella, G., & Berger, R. L., *Statistical Inference*; Hogg, R. V., McKean, J., & Craig, A. T., *Introduction to Mathematical Statistics*, 7th ed. (likelihood, pivotal quantities, Bayes estimators, likelihood ratio tests).

Works such as Weerahandi (generalized pivots), Peña & Gupta (Gamma–Dirichlet prior), Chen & Shao (HPD intervals), Balakrishnan & Sandhu (progressive sample generation), Hand et al. and Jarrett (coal-mining data) are used only as cited inside references [2] and [3].

# Appendix A. Derivation details {.unnumbered}

**A.1 Lomax to exponential.** $P\{\alpha\log(1+\lambda X)>u\}=P\{X>(e^{u/\alpha}-1)/\lambda\}=(e^{u/\alpha})^{-\alpha}=e^{-u}$.

**A.2 Second derivatives.** From (3.5),

$$\partial\ell/\partial\lambda=D/\lambda-\sum x_{ij}/(1+\lambda x_{ij})-\sum_i\alpha_i\Delta_i'(\lambda),$$

$$\Delta_i''(\lambda)=-\sum_j(1+R_{ij})x_{ij}^2/(1+\lambda x_{ij})^2-R^T_iT_i^2/(1+\lambda T_i)^2.$$

Hence $\partial^2\ell/\partial\lambda^2=-D/\lambda^2+\sum x^2_{ij}/(1+\lambda x_{ij})^2-\sum_i\alpha_i\Delta_i''(\lambda)$, which gives (4.5).

**A.3 Summation by parts for (5.1).** With $U_{i0}=0$,

$$\sum_{l\le j}c_{il}(U_{il}-U_{i,l-1})=c_{ij}U_{ij}+\sum_{l<j}(c_{il}-c_{i,l+1})U_{il},$$

and $c_{il}-c_{i,l+1}=R_{il}+1$.

**A.4 Covariance of the GD prior.** $E(\alpha_i\alpha_l)=E(\alpha_+^2)E(p_ip_l)$, with $E(\alpha_+^2)=a_0(a_0+1)/b_0^2$ and $E(p_ip_l)=c_ic_l/\{c_+(c_++1)\}$. Subtracting $E\alpha_iE\alpha_l$ gives the expression in Section 6.2.

**A.5 Importance weight of Algorithm 2.** Target/proposal $=\big[g(\lambda)\prod_i\mathrm{Ga}(\alpha_i)\,\alpha_+^{a_0-c_+}\big]\big/\big[\hat g(\lambda)\prod_i\mathrm{Ga}(\alpha_i)\big]=\alpha_+^{a_0-c_+}\,g(\lambda)/\hat g(\lambda)$, up to a constant.

# Appendix B. Additional simulation tables {.unnumbered}

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 294.1276 | 0.3891 | 290.3015 | 0.0512 | 0.0211 | 0.1581 | 0.1003 |
| S1, T=3, B | 232.8374 | 0.5674 | 50.8437 | 0.0567 | 0.0213 | 0.1827 | 0.1101 |
| S1, T=3, C | 1841.5735 | 0.3827 | 1755.4976 | 0.0524 | 0.0199 | 0.1613 | 0.0971 |
| S1, T=8, A | 39.2422 | 0.4482 | 4.4624 | 0.0639 | 0.0252 | 0.2063 | 0.1199 |
| S1, T=8, B | 0.9110 | 0.4861 | 0.7964 | 0.0568 | 0.0232 | 0.1533 | 0.0921 |
| S1, T=8, C | 6.5803 | 0.3901 | 5.3038 | 0.0596 | 0.0240 | 0.1765 | 0.1032 |
| S2, T=3, A | 3.5453 | 0.3540 | 3.4001 | 0.0426 | 0.0219 | 0.1376 | 0.0985 |
| S2, T=3, B | 54.4608 | 0.4151 | 34.8410 | 0.0455 | 0.0213 | 0.1511 | 0.1028 |
| S2, T=3, C | 210.7803 | 0.3200 | 134.2023 | 0.0402 | 0.0196 | 0.1257 | 0.0894 |
| S2, T=8, A | 0.3331 | 0.2054 | 0.2722 | 0.0447 | 0.0230 | 0.1243 | 0.0827 |
| S2, T=8, B | 0.1782 | 0.2563 | 0.1566 | 0.0435 | 0.0221 | 0.1036 | 0.0697 |
| S2, T=8, C | 1.2166 | 0.1888 | 1.1800 | 0.0414 | 0.0229 | 0.1074 | 0.0755 |
| S3, T=3, A | 198.1706 | 0.2693 | 198.1440 | 0.0355 | 0.0204 | 0.1093 | 0.0833 |
| S3, T=3, B | 1.6919 | 0.2384 | 1.5590 | 0.0379 | 0.0208 | 0.1223 | 0.0920 |
| S3, T=3, C | 1.3673 | 0.2469 | 1.3200 | 0.0378 | 0.0201 | 0.1158 | 0.0835 |
| S3, T=8, A | 0.1688 | 0.1681 | 0.1555 | 0.0372 | 0.0222 | 0.0925 | 0.0682 |
| S3, T=8, B | 0.1041 | 0.1453 | 0.0946 | 0.0370 | 0.0211 | 0.0776 | 0.0555 |
| S3, T=8, C | 0.1127 | 0.1246 | 0.0970 | 0.0350 | 0.0208 | 0.0771 | 0.0553 |


Table: Table B1. MSE of the estimators of $\alpha_2$ (true 0.75).

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.9501 | 0.1039 | 0.8556 | 0.0577 | 0.0008 | 0.1460 | 0.0801 |
| S1, T=3, B | 1.1809 | 0.1549 | 0.6259 | 0.0752 | 0.0065 | 0.1718 | 0.0901 |
| S1, T=3, C | 1.3329 | 0.1131 | 1.1956 | 0.0668 | 0.0033 | 0.1547 | 0.0813 |
| S1, T=8, A | 0.5419 | 0.1052 | 0.2928 | 0.0626 | 0.0141 | 0.1450 | 0.0880 |
| S1, T=8, B | 0.1848 | 0.1571 | 0.1603 | 0.0567 | 0.0001 | 0.1226 | 0.0591 |
| S1, T=8, C | 0.2646 | 0.1050 | 0.2227 | 0.0509 | 0.0018 | 0.1223 | 0.0652 |
| S2, T=3, A | 0.2299 | 0.1258 | 0.2164 | 0.0552 | 0.0183 | 0.1285 | 0.0877 |
| S2, T=3, B | 0.5080 | 0.1382 | 0.3768 | 0.0675 | 0.0218 | 0.1470 | 0.0949 |
| S2, T=3, C | 0.4780 | 0.1129 | 0.4082 | 0.0552 | 0.0153 | 0.1271 | 0.0825 |
| S2, T=8, A | 0.1046 | 0.0648 | 0.0956 | 0.0416 | 0.0168 | 0.0905 | 0.0641 |
| S2, T=8, B | 0.0892 | 0.1016 | 0.0800 | 0.0474 | 0.0128 | 0.0892 | 0.0525 |
| S2, T=8, C | 0.1022 | 0.0709 | 0.0946 | 0.0378 | 0.0098 | 0.0800 | 0.0501 |
| S3, T=3, A | 0.2681 | 0.1021 | 0.2624 | 0.0444 | 0.0210 | 0.1038 | 0.0786 |
| S3, T=3, B | 0.1893 | 0.1060 | 0.1766 | 0.0614 | 0.0311 | 0.1278 | 0.0944 |
| S3, T=3, C | 0.1295 | 0.0967 | 0.1224 | 0.0491 | 0.0218 | 0.1097 | 0.0798 |
| S3, T=8, A | 0.0694 | 0.0513 | 0.0649 | 0.0376 | 0.0214 | 0.0725 | 0.0555 |
| S3, T=8, B | 0.0578 | 0.0695 | 0.0536 | 0.0371 | 0.0150 | 0.0668 | 0.0439 |
| S3, T=8, C | 0.0598 | 0.0566 | 0.0555 | 0.0373 | 0.0179 | 0.0666 | 0.0465 |


Table: Table B2. Bias of the estimators of $\alpha_1$.

| Design | MLE | PIV | OR-MLE | BU-P1 | BOR-P1 | BU-P2 | BOR-P2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.0085 | 0.3852 | 0.0229 | 0.0240 | 0.0251 | 0.0614 | 0.0630 |
| S1, T=3, B | -0.0664 | 0.3518 | -0.0465 | 0.0072 | 0.0076 | 0.0215 | 0.0218 |
| S1, T=3, C | 0.0147 | 0.3759 | 0.0289 | 0.0218 | 0.0222 | 0.0589 | 0.0593 |
| S1, T=8, A | -0.0946 | 0.2372 | -0.0737 | -0.0017 | 0.0003 | 0.0040 | 0.0091 |
| S1, T=8, B | 0.0008 | 0.1577 | 0.0117 | 0.0079 | 0.0091 | 0.0445 | 0.0475 |
| S1, T=8, C | -0.0544 | 0.2010 | -0.0376 | -0.0001 | 0.0018 | 0.0183 | 0.0227 |
| S2, T=3, A | 0.0243 | 0.1740 | 0.0286 | 0.0211 | 0.0210 | 0.0533 | 0.0535 |
| S2, T=3, B | -0.0096 | 0.1890 | -0.0022 | 0.0096 | 0.0091 | 0.0338 | 0.0329 |
| S2, T=3, C | 0.0254 | 0.2009 | 0.0312 | 0.0225 | 0.0228 | 0.0568 | 0.0577 |
| S2, T=8, A | -0.0498 | 0.1017 | -0.0437 | -0.0026 | -0.0023 | 0.0085 | 0.0090 |
| S2, T=8, B | 0.0092 | 0.0706 | 0.0136 | 0.0138 | 0.0139 | 0.0482 | 0.0492 |
| S2, T=8, C | -0.0059 | 0.0873 | -0.0018 | 0.0090 | 0.0089 | 0.0369 | 0.0377 |
| S3, T=3, A | 0.0145 | 0.1107 | 0.0167 | 0.0183 | 0.0183 | 0.0415 | 0.0415 |
| S3, T=3, B | -0.0266 | 0.1318 | -0.0221 | 0.0058 | 0.0046 | 0.0200 | 0.0164 |
| S3, T=3, C | 0.0086 | 0.1177 | 0.0114 | 0.0153 | 0.0150 | 0.0389 | 0.0383 |
| S3, T=8, A | -0.0175 | 0.0845 | -0.0139 | 0.0055 | 0.0047 | 0.0219 | 0.0217 |
| S3, T=8, B | 0.0032 | 0.0331 | 0.0056 | 0.0138 | 0.0137 | 0.0413 | 0.0412 |
| S3, T=8, C | 0.0094 | 0.0626 | 0.0116 | 0.0176 | 0.0171 | 0.0448 | 0.0442 |


Table: Table B3. Bias of the estimators of $\lambda$.

| Design | AL ACI | AL GCI | AL HPD-U | AL HPD-OR | CP ACI | CP GCI | CP HPD-U | CP HPD-OR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 312.170 | 16.867 | 2.257 | 2.295 | 0.931 | 0.973 | 0.989 | 0.997 |
| S1, T=3, B | 221.826 | 18.709 | 2.304 | 2.354 | 0.960 | 0.959 | 0.992 | 0.996 |
| S1, T=3, C | 4056.956 | 16.732 | 2.208 | 2.250 | 0.944 | 0.972 | 0.993 | 0.999 |
| S1, T=8, A | 57.751 | 14.256 | 2.164 | 2.175 | 0.953 | 0.961 | 0.984 | 0.994 |
| S1, T=8, B | 4.127 | 13.150 | 1.954 | 1.960 | 0.949 | 0.952 | 0.980 | 0.994 |
| S1, T=8, C | 13.526 | 13.359 | 2.059 | 2.054 | 0.950 | 0.959 | 0.985 | 0.991 |
| S2, T=3, A | 6.626 | 12.017 | 1.889 | 1.909 | 0.932 | 0.962 | 0.985 | 0.995 |
| S2, T=3, B | 65.986 | 12.295 | 1.950 | 1.980 | 0.951 | 0.957 | 0.987 | 0.990 |
| S2, T=3, C | 320.368 | 11.814 | 1.902 | 1.925 | 0.942 | 0.962 | 0.984 | 0.993 |
| S2, T=8, A | 2.367 | 5.867 | 1.690 | 1.679 | 0.960 | 0.957 | 0.977 | 0.982 |
| S2, T=8, B | 1.931 | 5.930 | 1.547 | 1.540 | 0.959 | 0.952 | 0.981 | 0.986 |
| S2, T=8, C | 3.065 | 5.349 | 1.578 | 1.565 | 0.951 | 0.952 | 0.977 | 0.982 |
| S3, T=3, A | 230.763 | 8.833 | 1.676 | 1.686 | 0.948 | 0.954 | 0.980 | 0.983 |
| S3, T=3, B | 4.130 | 8.182 | 1.772 | 1.782 | 0.951 | 0.956 | 0.983 | 0.989 |
| S3, T=3, C | 3.991 | 7.972 | 1.681 | 1.696 | 0.948 | 0.962 | 0.985 | 0.991 |
| S3, T=8, A | 1.734 | 3.871 | 1.422 | 1.407 | 0.953 | 0.946 | 0.972 | 0.976 |
| S3, T=8, B | 1.524 | 3.238 | 1.330 | 1.318 | 0.953 | 0.953 | 0.971 | 0.985 |
| S3, T=8, C | 1.545 | 3.044 | 1.328 | 1.314 | 0.953 | 0.950 | 0.972 | 0.980 |


Table: Table B4. AL and CP of 95% intervals for $\alpha_3$.

| Design | α₁ AL U | α₁ AL OR | α₁ CP U | α₁ CP OR | λ AL U | λ AL OR | λ CP U | λ CP OR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.740 | 0.619 | 0.982 | 0.984 | 1.381 | 1.381 | 1.000 | 1.000 |
| S1, T=3, B | 0.766 | 0.622 | 0.977 | 0.978 | 1.370 | 1.368 | 1.000 | 1.000 |
| S1, T=3, C | 0.757 | 0.624 | 0.980 | 0.982 | 1.384 | 1.383 | 1.000 | 1.000 |
| S1, T=8, A | 0.676 | 0.578 | 0.986 | 0.987 | 1.314 | 1.314 | 0.999 | 0.999 |
| S1, T=8, B | 0.708 | 0.587 | 0.977 | 0.979 | 1.335 | 1.334 | 0.997 | 0.997 |
| S1, T=8, C | 0.675 | 0.573 | 0.967 | 0.975 | 1.317 | 1.315 | 0.995 | 0.996 |
| S2, T=3, A | 0.652 | 0.576 | 0.983 | 0.986 | 1.305 | 1.303 | 1.000 | 1.000 |
| S2, T=3, B | 0.668 | 0.577 | 0.980 | 0.981 | 1.306 | 1.303 | 1.000 | 0.999 |
| S2, T=3, C | 0.657 | 0.574 | 0.979 | 0.981 | 1.315 | 1.313 | 1.000 | 0.999 |
| S2, T=8, A | 0.559 | 0.507 | 0.969 | 0.976 | 1.215 | 1.212 | 0.994 | 0.993 |
| S2, T=8, B | 0.597 | 0.525 | 0.973 | 0.976 | 1.246 | 1.245 | 0.991 | 0.991 |
| S2, T=8, C | 0.570 | 0.511 | 0.979 | 0.983 | 1.227 | 1.226 | 0.989 | 0.988 |
| S3, T=3, A | 0.586 | 0.537 | 0.977 | 0.981 | 1.243 | 1.241 | 0.996 | 0.995 |
| S3, T=3, B | 0.602 | 0.542 | 0.983 | 0.986 | 1.246 | 1.243 | 0.997 | 0.997 |
| S3, T=3, C | 0.594 | 0.539 | 0.982 | 0.984 | 1.250 | 1.250 | 0.995 | 0.995 |
| S3, T=8, A | 0.499 | 0.465 | 0.973 | 0.975 | 1.149 | 1.147 | 0.984 | 0.982 |
| S3, T=8, B | 0.529 | 0.482 | 0.974 | 0.974 | 1.176 | 1.175 | 0.986 | 0.986 |
| S3, T=8, C | 0.512 | 0.471 | 0.976 | 0.983 | 1.160 | 1.160 | 0.985 | 0.984 |


Table: Table B5. AL and CP of HPD intervals under the informative prior P1 (U = unrestricted, OR = order restricted).

| Design | α₃ U-SE | α₃ U-LX | α₃ OR-SE | α₃ OR-LX | λ U-SE | λ U-LX | λ OR-SE | λ OR-LX |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1, T=3, A | 0.3304 | 0.1455 | 0.3778 | 0.1655 | 0.1591 | 0.1023 | 0.1598 | 0.1026 |
| S1, T=3, B | 0.3348 | 0.1482 | 0.4107 | 0.1834 | 0.1585 | 0.1112 | 0.1590 | 0.1116 |
| S1, T=3, C | 0.3113 | 0.1406 | 0.3594 | 0.1595 | 0.1623 | 0.1050 | 0.1610 | 0.1044 |
| S1, T=8, A | 0.3455 | 0.1560 | 0.4034 | 0.1878 | 0.1547 | 0.1155 | 0.1578 | 0.1165 |
| S1, T=8, B | 0.3090 | 0.1587 | 0.3496 | 0.1794 | 0.1804 | 0.1237 | 0.1815 | 0.1241 |
| S1, T=8, C | 0.3537 | 0.1682 | 0.3958 | 0.1923 | 0.1609 | 0.1170 | 0.1618 | 0.1168 |
| S2, T=3, A | 0.2577 | 0.1323 | 0.2854 | 0.1448 | 0.1694 | 0.1172 | 0.1695 | 0.1172 |
| S2, T=3, B | 0.2592 | 0.1324 | 0.2979 | 0.1497 | 0.1780 | 0.1252 | 0.1755 | 0.1242 |
| S2, T=3, C | 0.2533 | 0.1307 | 0.2810 | 0.1423 | 0.1676 | 0.1148 | 0.1683 | 0.1153 |
| S2, T=8, A | 0.2467 | 0.1358 | 0.2681 | 0.1491 | 0.1375 | 0.1078 | 0.1369 | 0.1070 |
| S2, T=8, B | 0.1873 | 0.1129 | 0.2066 | 0.1234 | 0.1568 | 0.1133 | 0.1572 | 0.1137 |
| S2, T=8, C | 0.2273 | 0.1322 | 0.2437 | 0.1418 | 0.1543 | 0.1145 | 0.1549 | 0.1144 |
| S3, T=3, A | 0.2206 | 0.1228 | 0.2349 | 0.1296 | 0.1497 | 0.1101 | 0.1487 | 0.1094 |
| S3, T=3, B | 0.2361 | 0.1295 | 0.2612 | 0.1416 | 0.1509 | 0.1144 | 0.1503 | 0.1142 |
| S3, T=3, C | 0.1905 | 0.1067 | 0.2114 | 0.1163 | 0.1501 | 0.1107 | 0.1496 | 0.1102 |
| S3, T=8, A | 0.1860 | 0.1144 | 0.1954 | 0.1201 | 0.1345 | 0.1071 | 0.1344 | 0.1070 |
| S3, T=8, B | 0.1426 | 0.0950 | 0.1519 | 0.0999 | 0.1292 | 0.0987 | 0.1295 | 0.0991 |
| S3, T=8, C | 0.1588 | 0.1034 | 0.1656 | 0.1068 | 0.1351 | 0.1036 | 0.1341 | 0.1030 |


Table: Table B6. MSE of Bayes estimators under SELF (SE) and LINEX (LX, $h=1$), prior P2.

| Design | α₁ median | α₁ mean | λ median | λ mean |
|:---|---:|---:|---:|---:|
| S1, T=3, A | 0.151 | 68.769 | 1.952 | 3.893 |
| S1, T=3, B | 0.215 | 707.908 | 2.078 | 4.217 |
| S1, T=3, C | 0.149 | 829.153 | 1.840 | 3.715 |
| S1, T=8, A | 0.132 | 35.043 | 0.997 | 1.766 |
| S1, T=8, B | 0.200 | 137.758 | 1.051 | 1.969 |
| S1, T=8, C | 0.128 | 56.604 | 1.141 | 2.008 |
| S2, T=3, A | 0.149 | 398.187 | 0.800 | 1.213 |
| S2, T=3, B | 0.178 | 16.615 | 1.005 | 1.591 |
| S2, T=3, C | 0.130 | 23.330 | 0.877 | 1.380 |
| S2, T=8, A | 0.069 | 5.820 | 0.442 | 0.628 |
| S2, T=8, B | 0.095 | 15.959 | 0.566 | 0.834 |
| S2, T=8, C | 0.072 | 5.061 | 0.475 | 0.677 |
| S3, T=3, A | 0.119 | 9.438 | 0.492 | 0.669 |
| S3, T=3, B | 0.104 | 302.954 | 0.620 | 0.882 |
| S3, T=3, C | 0.099 | 11.373 | 0.534 | 0.747 |
| S3, T=8, A | 0.051 | 2.566 | 0.333 | 0.433 |
| S3, T=8, B | 0.055 | 0.346 | 0.317 | 0.422 |
| S3, T=8, C | 0.047 | 0.800 | 0.337 | 0.441 |


Table: Table B7. MSE of the pivotal estimator defined as the median vs the mean of the pivotal draws.

The complete results (every design × method × quantity: bias, MSE, MedAE, AL, CP) are in `results/simulation/simulation_tables.xlsx` and `sim_summary.csv`. Diagnostics, including the minimum importance-sampling ESS, are in `sim_diagnostics.csv`.

# Appendix C. Reproducibility {.unnumbered}

All computations use Python 3.12 (numpy, scipy, pandas, matplotlib). The code lives in the `code/` folder:

| File | Purpose |
|:---|:---|
| `lomax_bgphc/model.py` | Lomax functions ($f,F,R,h$, MdTF, quantiles) |
| `lomax_bgphc/censoring.py` | BGPHC data generation and application of the scheme to complete data |
| `lomax_bgphc/likelihood.py` | $\Delta_i(\lambda)$, log-likelihood, profile likelihood, observed information |
| `lomax_bgphc/mle.py` | MLE, order-restricted MLE (PAVA), asymptotic CIs |
| `lomax_bgphc/pivotal.py` | Pivots, Theorem 3, Algorithm 1 |
| `lomax_bgphc/bayes.py` | GD/OGD priors, Algorithms 2–3, LINEX, weighted HPD |
| `lomax_bgphc/estimators.py` | Runs every method on a data set |
| `run_simulation.py` | Simulation study (parallel; about 10 minutes on 22 cores) |
| `make_sim_outputs.py` | Simulation tables and figures |
| `real_data_analysis.py` | Section 9 |
| `build_report.py` | Assembles this report |

All random seeds are fixed (simulation seed 20251005, real-data seed 2025), so every number in this report can be reproduced exactly.
