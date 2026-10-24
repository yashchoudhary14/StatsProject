# Pivotal-based classical and Bayesian estimation for the Lomax distribution under block generalized progressive hybrid censoring

DA 2141 Statistics for Data Science — course project.

## Deliverables

| Deliverable | File |
|---|---|
| Background and likelihood, MLE, pivotal, Bayes, order-restricted (theory) | `report/Final_Report.pdf` §§2–7 |
| Simulation study | `report/Final_Report.pdf` §8, `results/simulation/` |
| Real-data validation | `report/Final_Report.pdf` §9, `results/real_data/` |
| Final report | `report/Final_Report.pdf` (editable: `report/Final_Report.docx`) |
| Presentation | `presentation/Presentation.pptx` (speaker notes on every slide), `presentation/Presentation.pdf` |

Before submitting, replace `[Student name(s) and roll number(s)]` on the report title page and on slide 1.

## References used

1. Kumari, Tripathi, Lodhi & Wang (2025), Burr-XII under block progressive Type-II censoring, *Comm. Stat. – Sim. & Comp.* (only the abstract was available; the PDF in this folder named *Variable selection for censored data…* is a different article).
2. Lodhi, Gangopadhyay & Bhattacharya (2025), IER distribution under block progressive censoring, *QTQM* 22(4) — block setup, pivotal method, coal-mining data.
3. Sultana, Çetinkaya, Raqab & Kundu (2026), block Type-I hybrid censored Weibull populations, *Comm. Stat. – Theory & Methods* 55(11), DOI 10.1080/03610926.2025.2579720 — joint Gamma–Dirichlet prior, order restrictions.
4. **One extra reference:** Cho, Sun & Lee (2015), *Statistical Methodology* 23, 18–34 — definition of generalized progressive hybrid censoring.

## Folder layout

```
code/
  lomax_bgphc/            Python package
    model.py              Lomax pdf/cdf/reliability/hazard/median/random numbers
    censoring.py          block GPHC data generation; applying the scheme to complete data
    likelihood.py         Delta_i(lambda), log-likelihood, profile likelihood, observed information
    mle.py                MLE, order-restricted MLE (PAVA), asymptotic CIs
    pivotal.py            pivots (Theorem 3) and the truncated GPQ algorithm
    bayes.py              Gamma-Dirichlet and ordered GD priors, importance sampling, LINEX, HPD
    estimators.py         runs every method on one data set
    plotstyle.py          shared figure style
  run_simulation.py       simulation study (parallel)
  make_sim_outputs.py     simulation tables (CSV/Excel/Markdown) and figures
  real_data_analysis.py   coal-mining data analysis
  make_schematic.py       censoring-scheme figure
  make_slide_assets.py    equation images and Lomax figure for the slides
  build_report.py         assembles report/Final_Report.md -> .docx -> .pdf
results/
  simulation/             sim_summary.csv, sim_diagnostics.csv, simulation_tables.xlsx, raw estimates
  real_data/              estimates, tests, censored samples, prior sensitivity
  figures/                all figures
  report_tables/          table snippets inserted into the report
report/                   report_template.md (source), Final_Report.md/.docx/.pdf
presentation/             build_deck.js (source), Presentation.pptx/.pdf
```

## Reproducing everything

Python 3.12:

```bash
pip install numpy scipy pandas matplotlib python-docx pypandoc_binary openpyxl pymupdf
```

```bash
python code/run_simulation.py
```

The full study takes about 10 minutes on 22 cores. `--reps 200` gives a quick version.

```bash
python code/make_sim_outputs.py
```

```bash
python code/real_data_analysis.py
```

```bash
python code/make_schematic.py
```

```bash
python code/make_slide_assets.py
```

```bash
python code/build_report.py
```

The PDF step uses Microsoft Word through COM (Windows); `--no-pdf` skips it. To rebuild the slides, install `pptxgenjs` with npm in any folder, set `NODE_PATH` to that folder's `node_modules`, and set `APPLY_THEME` to the pptx skill's `scripts/apply_theme.js`. Then run `node presentation/build_deck.js`.

Random seeds are fixed (simulation 20251005, real data 2025), so all numbers are reproducible.
