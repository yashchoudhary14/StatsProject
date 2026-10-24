"""
Assemble the final report:  report_template.md  ->  Final_Report.md / .docx / .pdf

  * {{TABLE:name}} placeholders are replaced by results/report_tables/name.md
    (written by make_sim_outputs.py);
  * {{RD:...}} placeholders are generated here from results/real_data/*.csv;
  * pandoc (pypandoc_binary) converts Markdown + LaTeX maths to Word with native
    equations, using a styled reference document;
  * Microsoft Word (COM automation, Windows) updates the table of contents and
    exports the PDF.

Usage:  python build_report.py [--no-pdf]
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pypandoc
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPORT = os.path.join(ROOT, "report")
TAB = os.path.join(ROOT, "results", "report_tables")
RD = os.path.join(ROOT, "results", "real_data")

QL = {"alpha1": "α₁", "alpha2": "α₂", "alpha3": "α₃", "alpha_pool": "ᾱ (pooled)", "lam": "λ",
      "R1": "R₁(0.5)", "R2": "R₂(0.5)", "R3": "R₃(0.5)", "h1": "h₁(0.5)", "h2": "h₂(0.5)", "h3": "h₃(0.5)",
      "MdTF1": "MdTF₁", "MdTF2": "MdTF₂", "MdTF3": "MdTF₃"}
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def pipe(df, index=True):
    """Pipe table whose separator dashes are proportional to the longest cell of
    each column (pandoc uses them as relative column widths)."""
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    body = [([str(idx)] if index else []) + [str(v) for v in row.values] for idx, row in df.iterrows()]
    widths = [max([len(str(c))] + [len(r[j]) for r in body]) for j, c in enumerate(cols)]
    widths = [min(max(w, 10), 36) for w in widths]
    seps = [":" + "-" * (widths[0] - 1)] + ["-" * (w - 1) + ":" for w in widths[1:]]
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(seps) + "|"]
    for cells in body:
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def f4(v, d=4):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def rd_tables():
    t = {}
    d = pd.read_csv(os.path.join(RD, "descriptive_statistics.csv")).set_index("Period")
    d.index.name = "Period"
    t["descriptive"] = pipe(d.apply(lambda c: c.map(lambda v: f4(v, 3) if isinstance(v, float) else v)))

    g = pd.read_csv(os.path.join(RD, "goodness_of_fit.csv"))
    g2 = pd.DataFrame({
        "Period": g.Period, "Model": g.Model,
        "α̂": g.alpha.map(lambda v: f4(v, 3)), "λ̂": g.lam.map(lambda v: f4(v, 3)),
        "log L": g.loglik.map(lambda v: f4(v, 2)), "AIC": g.AIC.map(lambda v: f4(v, 2)),
        "KS D": g.KS_D.map(lambda v: f4(v, 3)), "KS p": g.KS_p.map(lambda v: f4(v, 3))})
    t["gof"] = pipe(g2, index=False)

    l = pd.read_csv(os.path.join(RD, "likelihood_ratio_tests.csv"))
    l["Test"] = l["Test"].str.replace("H0: lam_1 = lam_2 = lam_3", "H₀: λ₁ = λ₂ = λ₃", regex=False)
    l["Test"] = l["Test"].str.replace("H0: alpha_1 = alpha_2 = alpha_3", "H₀: α₁ = α₂ = α₃", regex=False)
    l["Statistic"] = l.Statistic.map(lambda v: f4(v, 3))
    l["p_value"] = l.p_value.map(lambda v: f4(v, 4))
    l = l.rename(columns={"p_value": "p-value", "Statistic": "−2 log Λ"})
    t["lrt"] = pipe(l, index=False)

    c = pd.read_csv(os.path.join(RD, "censored_datasets.csv"))

    def nz(s):
        r = [int(v) for v in str(s).split()]
        nzs = [f"R{str(j + 1).translate(SUB)}={v}" for j, v in enumerate(r) if v > 0]
        return ", ".join(nzs) if nzs else "none"
    c2 = pd.DataFrame({"Design": c.Design, "Period": c.Period, "Case": c.Case, "Dᵢ": c.D,
                       "Observed failures xᵢⱼ": c.failures, "Removals at failures": c.removals.map(nz),
                       "Rᵢᵀ": c.RT})
    t["censored"] = pipe(c2, index=False)

    e = pd.read_csv(os.path.join(RD, "estimates_all_methods.csv"))
    methods = ["MLE", "Pivotal", "OR-MLE", "Bayes-U (SELF)", "Bayes-U (LINEX)", "Bayes-OR (SELF)", "Bayes-OR (LINEX)"]
    short = ["MLE", "PIV", "OR-MLE", "BU-SE", "BU-LX", "BOR-SE", "BOR-LX"]
    for des in e.Design.unique():
        sub = e[e.Design == des]
        pt = sub.pivot(index="Quantity", columns="Method", values="Estimate").loc[list(QL), methods]
        pt.columns = short
        pt.index = [QL[q] for q in pt.index]
        pt.index.name = "Quantity"
        t[f"point_{des}"] = pipe(pt.apply(lambda c: c.map(lambda v: f4(v, 4))))
        rows = {}
        for q in QL:
            r = {}
            for meth, lab in [("MLE", "ACI (MLE)"), ("Pivotal", "GCI (pivotal)"),
                              ("Bayes-U (SELF)", "HPD (Bayes-U)"), ("Bayes-OR (SELF)", "HPD (Bayes-OR)")]:
                v = sub[(sub.Quantity == q) & (sub.Method == meth)].iloc[0]
                r[lab] = f"[{v.Lower:.3f}, {v.Upper:.3f}] ({v.Length:.3f})"
            rows[QL[q]] = r
        it = pd.DataFrame(rows).T
        it.index.name = "Quantity"
        t[f"int_{des}"] = pipe(it)

    s = pd.read_csv(os.path.join(RD, "prior_sensitivity.csv"))
    s["cell"] = s.apply(lambda r: f"{r.Estimate:.3f} ({r.Upper - r.Lower:.3f})", axis=1)
    s["col"] = s.apply(lambda r: f"f={r.prior_sd_fraction:g} {'BU' if r.Method == 'BU_SE' else 'BOR'}", axis=1)
    s["row"] = s.apply(lambda r: f"{r.Design}: {QL[r.Quantity].replace('(0.5)', '(0.5)')}", axis=1)
    sp = s.pivot(index="row", columns="col", values="cell")
    order = [f"{d}: {QL[q]}" for d in ["Scheme 1", "Scheme 2"] for q in ["alpha1", "alpha2", "alpha3", "lam", "R1", "R2", "R3"]]
    sp = sp.loc[order, [f"f={f} {m}" for f in ["0.5", "1", "2"] for m in ["BU", "BOR"]]]
    sp.index.name = "Design: quantity"
    t["sensitivity"] = pipe(sp)
    return t


def assemble():
    s = open(os.path.join(REPORT, "report_template.md"), encoding="utf-8").read()

    def tab(m):
        return open(os.path.join(TAB, m.group(1) + ".md"), encoding="utf-8").read()
    s = re.sub(r"\{\{TABLE:([A-Za-z0-9_]+)\}\}", tab, s)
    rd = rd_tables()
    s = re.sub(r"\{\{RD:([^}]+)\}\}", lambda m: rd[m.group(1)], s)
    assert "{{" not in s, "unfilled placeholder"
    out = os.path.join(REPORT, "Final_Report.md")
    open(out, "w", encoding="utf-8").write(s)
    return out


# ---------------------------------------------------------------------------
# Word styling
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x1F, 0x38, 0x64)


def make_reference(path):
    data = subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"],
                          capture_output=True, check=True).stdout
    open(path, "wb").write(data)
    doc = Document(path)
    st = doc.styles
    for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
        if name in [x.name for x in st]:
            f = st[name].font
            f.name, f.size = "Cambria", Pt(11)
            st[name].element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")
    for name in ["Body Text", "First Paragraph"]:
        pf = st[name].paragraph_format
        pf.space_after = Pt(6)
        pf.line_spacing = 1.12
    for lvl, size in [(1, 16), (2, 13), (3, 11.5)]:
        h = st[f"Heading {lvl}"]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
        h.font.color.rgb = NAVY
        h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        h.paragraph_format.space_after = Pt(5)
        h.paragraph_format.keep_with_next = True
    for name, size, bold in [("Title", 20, True), ("Subtitle", 13, False), ("Author", 11, False), ("Date", 11, False)]:
        if name in [x.name for x in st]:
            s = st[name]
            s.font.name, s.font.size, s.font.bold = "Calibri", Pt(size), bold
            s.font.color.rgb = NAVY if name in ("Title", "Subtitle") else RGBColor(0x33, 0x33, 0x33)
    for name in ["Image Caption", "Table Caption", "Caption"]:
        if name in [x.name for x in st]:
            s = st[name]
            s.font.name, s.font.size, s.font.italic = "Calibri", Pt(9), True
            s.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    doc.save(path)


def _set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, val in kw.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val.get("val", "single"))
        el.set(qn("w:sz"), str(val.get("sz", 6)))
        el.set(qn("w:color"), val.get("color", "1F3864"))
        borders.append(el)


def polish_docx(path):
    doc = Document(path)
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        ncols = len(table.columns)
        size = Pt(9) if ncols <= 6 else Pt(8) if ncols <= 9 else Pt(7.5)
        for r_i, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    for run in p.runs:
                        run.font.size = size
                        run.font.name = "Calibri"
                        if r_i == 0:
                            run.font.bold = True
                if r_i == 0:
                    _set_cell_border(cell, top={"sz": 10}, bottom={"sz": 6})
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E8EDF5")
                    tcPr.append(shd)
                elif r_i == len(table.rows) - 1:
                    _set_cell_border(cell, bottom={"sz": 10})
    # keep figure captions with figures
    doc.save(path)


def to_pdf(docx_path, pdf_path):
    ps = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
  $doc = $word.Documents.Open('{docx_path}')
  foreach ($toc in $doc.TablesOfContents) {{ $toc.Update() }}
  $doc.Fields.Update() | Out-Null
  $doc.Save()
  $doc.SaveAs([ref]'{pdf_path}', [ref]17)
  $doc.Close([ref]0)
}} finally {{ $word.Quit() }}
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def build(md_path, stem, toc=True, pdf=True):
    ref = os.path.join(REPORT, "_reference.docx")
    make_reference(ref)
    docx_path = os.path.join(os.path.dirname(md_path), stem + ".docx")
    args = ["--reference-doc", ref, "--resource-path", os.path.dirname(md_path),
            "-f", "markdown+tex_math_dollars+pipe_tables+implicit_figures+link_attributes"]
    if toc:
        args += ["--toc", "--toc-depth=2"]
    pypandoc.convert_file(md_path, "docx", outputfile=docx_path, extra_args=args)
    polish_docx(docx_path)
    if pdf:
        to_pdf(docx_path, docx_path[:-5] + ".pdf")
    os.remove(ref)
    return docx_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    md = assemble()
    print("built", build(md, "Final_Report", pdf=not a.no_pdf))
