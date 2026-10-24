// Builds Presentation.pptx from the project results.
// Run:  NODE_PATH=<folder containing node_modules with pptxgenjs> node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "results", "figures", f);
const OUT = path.join(__dirname, "Presentation.pptx");

// ---------- helpers -------------------------------------------------------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// fit an image inside a box (x, y, w, h), keep aspect ratio, align top-centre
function fitImage(slide, file, x, y, w, h, opts = {}) {
  const s = pngSize(file);
  const r = s.w / s.h;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = opts.left ? x : x + (w - iw) / 2;
  const iy = opts.middle ? y + (h - ih) / 2 : y;
  slide.addImage({ path: file, x: ix, y: iy, w: iw, h: ih, objectName: opts.name || path.basename(file) });
  return { x: ix, y: iy, w: iw, h: ih };
}
function readCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const v = l.split(",");
    const o = {};
    head.forEach((h, i) => (o[h] = v[i]));
    return o;
  });
}

// ---------- theme ------------------------------------------------------------
const THEME = {
  name: "Coal and Ember",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1D22", lt1: "FFFFFF", dk2: "2B2F38", lt2: "F3F4F6",
    accent1: "E8692C", accent2: "2A78D6", accent3: "1BAF7A", accent4: "F2B134",
    accent5: "5B6573", accent6: "4A3AA7", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Lomax distribution under block generalized progressive hybrid censoring";
pres.subject = "DA 2141 Statistics for Data Science - course project";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.55, w: 11.7, h: 2.2,
      fontFace: "Cambria", fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4,
      fontFace: "Calibri", fontSize: 18, color: "D7DAE0", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3,
      fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0,
      fontFace: "Calibri", fontSize: 18, color: "D7DAE0" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.35, fontSize: 10, color: "AEB4BF" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontFace: "Cambria", fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Lomax under block generalized progressive hybrid censoring  ·  DA 2141",
      options: { x: 0.6, y: 6.98, w: 8, h: 0.3, fontFace: "Calibri", fontSize: 10, color: "8A909B", margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.6, h: 0.35, fontSize: 10, color: "8A909B" },
});

// ---------- reusable components --------------------------------------------
function numberCircle(slide, n, x, y, d = 0.55, fill = C.accent1) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `circle-${n}` });
  slide.addText(String(n), { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: "Cambria",
    fontSize: 18, bold: true, color: C.background1, margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: name || "card" });
}
function stat(slide, x, y, w, big, small, color = C.accent1) {
  slide.addText(big, { x, y, w, h: 0.85, fontFace: "Cambria", fontSize: 36, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(small, { x, y: y + 0.85, w, h: 0.75, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
}
function bullets(slide, items, x, y, w, h, size = 16) {
  const runs = items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } }));
  slide.addText(runs, { x, y, w, h, fontSize: size, color: C.text1, valign: "top", isTextBox: true });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ===========================================================================
// 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Pivotal-Based Classical and Bayesian Estimation for the Lomax Distribution under Block Generalized Progressive Hybrid Censoring", { placeholder: "title" });
s.addText([
  { text: "DA 2141 Statistics for Data Science  ·  Course project", options: { breakLine: true } },
  { text: "[Student name(s) and roll number(s)]  ·  October 2026" },
], { placeholder: "body" });
["MLE", "Pivotal", "Bayes (joint prior)", "Order-restricted"].forEach((t, i) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8 + i * 2.55, y: 5.7, w: 2.35, h: 0.55, rectRadius: 0.12,
    fill: { color: HEX.dk2 }, line: { color: HEX.accent1, width: 1.25 }, objectName: `tag-${i}` });
  s.addText(t, { x: 0.8 + i * 2.55, y: 5.7, w: 2.35, h: 0.55, align: "center", valign: "middle", fontSize: 14,
    color: C.background1, margin: 0, isTextBox: true });
});
s.addNotes("Introduce the title. We study how to estimate the parameters of a heavy-tailed lifetime model (Lomax) when units are tested in several facilities and the tests are stopped early under a flexible censoring rule. Four estimation approaches are compared: maximum likelihood, pivotal quantities, Bayes with a joint prior, and order-restricted versions.");

// 2. Project at a glance
s = content("The project at a glance", "Introduction");
const steps = [
  ["Background & likelihood", "Lomax model, block GPHC scheme, likelihood function"],
  ["Estimation", "MLE, pivotal, Bayes (joint prior), order-restricted MLE & Bayes"],
  ["Simulation", "18 designs × 2,000 replications, 12 estimators"],
  ["Real data", "British coal-mining disasters in 3 periods"],
  ["Report & code", "Reproducible Python package + 29-page report"],
];
steps.forEach(([h, t], i) => {
  const x = 0.6 + i * 2.48;
  card(s, x, 1.65, 2.3, 3.35, `step-card-${i}`);
  numberCircle(s, i + 1, x + 0.2, 1.9);
  s.addText(h, { x: x + 0.2, y: 2.65, w: 1.95, h: 0.9, fontFace: "Cambria", fontSize: 17, bold: true, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.2, y: 3.55, w: 1.95, h: 1.35, fontSize: 15, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
s.addText("New for Lomax: an MLE existence condition, a decreasing & bounded pivot (truncated algorithm), and an exact importance sampler for joint and ordered priors.",
  { x: 0.6, y: 5.35, w: 12.1, h: 0.7, fontSize: 16, italic: true, color: C.accent1, margin: 0, isTextBox: true });
s.addNotes("Five deliverables, matching the project brief. The bottom line lists the parts that are genuinely new compared with the reference papers.");

// 3. Motivation
s = content("Why this problem?", "Introduction");
card(s, 0.6, 1.45, 5.9, 3.55, "card-facilities");
numberCircle(s, 1, 0.85, 1.75);
s.addText("Several test facilities", { x: 1.55, y: 1.75, w: 4.8, h: 0.55, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
bullets(s, ["Not enough equipment to test all n units at once → split into k blocks (block censoring)",
  "Identical-looking facilities still give different results → model a facility effect αᵢ",
  "Common failure mechanism → shared parameter λ"], 0.85, 2.45, 5.4, 2.45, 15);
card(s, 6.8, 1.45, 5.9, 3.55, "card-censoring");
numberCircle(s, 2, 7.05, 1.75);
s.addText("Stop the test early (censoring)", { x: 7.75, y: 1.75, w: 4.8, h: 0.55, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
bullets(s, ["Type-I (fixed time) may give too few failures; Type-II (fixed count) may take too long",
  "Generalized progressive hybrid censoring: at least κ failures, stop by time T or the m-th failure",
  "Units may be withdrawn during the test (progressive removals)"], 7.05, 2.45, 5.4, 2.45, 15);
s.addText([
  { text: "Goal: ", options: { bold: true, color: C.accent1 } },
  { text: "estimate the facility effects αᵢ, the common λ, and the reliability R(t), hazard h(t) and median lifetime of each facility — with point and interval estimates, with and without the natural ordering α₁ ≤ α₂ ≤ α₃." },
], { x: 0.6, y: 5.3, w: 12.1, h: 1.0, fontSize: 17, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
s.addNotes("Two practical issues. First, the units are tested in several facilities. Second, the test is stopped early and units can be withdrawn. Censored units are not failures and not missing; the likelihood uses the fact that they survived.");

// 4. Lomax
pres.addSection({ title: "Model and likelihood" });
s = content("The Lomax (Pareto II) lifetime model", "Model and likelihood");
fitImage(s, FIG("eq_lomax.png"), 0.6, 1.45, 12.1, 0.75, { name: "eq-lomax" });
fitImage(s, FIG("lomax_shapes.png"), 0.6, 2.45, 7.2, 4.2, { name: "lomax-shapes" });
bullets(s, ["Heavy (polynomial) right tail; mean infinite if α ≤ 1",
  "Hazard always decreasing",
  "Gamma mixture of exponentials; exceedances over a high threshold",
  "Exponential limit: λ → 0, α → ∞",
  "Key trick:"], 8.1, 2.5, 4.6, 3.0, 15);
fitImage(s, FIG("eq_transform.png"), 8.3, 5.35, 4.3, 0.55, { name: "eq-transform", left: true });
s.addNotes("The Lomax density, reliability and hazard. Smaller alpha means a heavier tail. The transformation U = alpha log(1 + lambda X) is exponential with mean 1; it is the key to every result that follows.");

// 5. BGPHC
s = content("Block generalized progressive hybrid censoring (BGPHC)", "Model and likelihood");
fitImage(s, FIG("schematic_bgphc.png"), 0.6, 1.35, 12.1, 4.35, { name: "schematic" });
fitImage(s, FIG("eq_stop.png"), 0.6, 5.9, 5.6, 0.55, { name: "eq-stop", left: true });
s.addText("Case I: stop at the κ-th failure  ·  Case II: stop at T  ·  Case III: stop at the m-th failure. Each facility i records Dᵢ failures, removals Rᵢⱼ and Rᵢᵀ units withdrawn at T.",
  { x: 6.4, y: 5.8, w: 6.3, h: 0.85, fontSize: 14, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
s.addNotes("Each of k facilities runs its own GPHC test (Cho, Sun and Lee, 2015). At least kappa failures are always seen. Contains block progressive Type-II censoring (T infinite) and block Type-I hybrid censoring as special cases.");

// 6. Likelihood
s = content("One likelihood covers all three cases", "Model and likelihood");
fitImage(s, FIG("eq_likelihood.png"), 0.6, 1.6, 12.1, 0.95, { name: "eq-likelihood" });
fitImage(s, FIG("eq_delta.png"), 1.6, 2.8, 10.1, 0.7, { name: "eq-delta" });
card(s, 0.6, 4.0, 12.1, 2.5, "card-insight");
s.addText("Key insight", { x: 0.9, y: 4.2, w: 6, h: 0.5, fontFace: "Cambria", fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true });
bullets(s, ["Δᵢ(λ) is the total time on test on the exponential scale log(1 + λx)",
  "For fixed λ, αᵢ appears only as αᵢ^Dᵢ e^(−αᵢΔᵢ) — a gamma kernel",
  "→ closed-form MLE of αᵢ, chi-square pivots, gamma posterior conditionals, isotonic order-restricted solution"], 0.9, 4.75, 11.5, 1.7, 16);
s.addNotes("Failures contribute the density, withdrawn units contribute the survival function. The gamma-kernel structure in alpha is what makes all four methods tractable.");

// 7. MLE
pres.addSection({ title: "Estimation methods" });
s = content("Maximum likelihood estimation", "Estimation methods");
fitImage(s, FIG("eq_mle.png"), 0.6, 1.5, 6.4, 0.75, { name: "eq-mle", left: true });
fitImage(s, FIG("eq_exist.png"), 0.6, 2.45, 6.4, 0.65, { name: "eq-exist", left: true });
bullets(s, ["Profile likelihood ℓp(λ) → 1-D search (grid + Brent)",
  "Theorem 2: MLE exists if S₀ > 0 — for one complete sample this is CV > 1 (data more variable than exponential)",
  "Otherwise λ̂ → 0, α̂ → ∞ (exponential limit)",
  "Asymptotic CIs: observed Fisher information + delta method for R(t), h(t), MdTF"], 0.6, 3.35, 6.4, 3.3, 15);
const pr = fitImage(s, FIG("real_profile_loglik.png"), 7.2, 2.2, 5.5, 3.2, { name: "profile-plot" });
s.addText("Profile log-likelihood of λ, coal-mining data (complete, Scheme 1, Scheme 2): a single maximum",
  { x: 7.2, y: pr.y + pr.h + 0.1, w: 5.5, h: 0.4, fontSize: 11, color: C.text2, align: "center", margin: 0, isTextBox: true });
s.addNotes("For fixed lambda the MLE of alpha_i is D_i over Delta_i. Substituting gives a one-dimensional profile likelihood. Existence requires the data to look more variable than an exponential; this explains why the MLE can break down in small samples.");

// 8. Pivotal
s = content("Pivotal (generalized pivotal quantity) inference", "Estimation methods");
fitImage(s, FIG("eq_pivots.png"), 0.6, 1.45, 12.1, 0.8, { name: "eq-pivots" });
card(s, 0.6, 2.6, 5.8, 3.95, "card-theorem");
s.addText("Theorem 3 (new for Lomax)", { x: 0.85, y: 2.75, w: 5.4, h: 0.5, fontFace: "Cambria", fontSize: 19, bold: true, color: C.accent1, margin: 0, isTextBox: true });
bullets(s, ["𝒫(λ) is strictly decreasing in λ",
  "𝒫(λ) → 0 as λ → ∞, but 𝒫(0⁺) is finite (exponential limit)",
  "So 𝒫(λ) = c has a root only if c < 𝒫(0⁺) → draw c from a truncated χ²",
  "IER/Burr-XII pivots (refs 1–2) increase from 0 to ∞ — no truncation needed there"], 0.85, 3.3, 5.35, 3.2, 14);
const algo = ["Draw c ~ χ² truncated to (0, 𝒫(0⁺)); solve 𝒫(λ) = c (bisection)",
  "Draw χ²(2Dᵢ) and set αᵢ = χ² / (2Δᵢ(λ))",
  "Repeat N times → point estimate = median, 95% GCI = 2.5% and 97.5% quantiles"];
algo.forEach((t, i) => {
  numberCircle(s, i + 1, 6.8, 2.75 + i * 1.2, 0.5);
  s.addText(t, { x: 7.45, y: 2.7 + i * 1.2, w: 5.25, h: 1.0, fontSize: 15, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addText("Median, not mean: the pivotal draws are very right-skewed.", { x: 6.8, y: 6.2, w: 5.9, h: 0.4, fontSize: 13, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("The ratio pivot does not depend on alpha. For Lomax it decreases in lambda and is bounded, which is different from the reference papers, so we truncate the chi-square draws. Given lambda, a second chi-square pivot gives alpha_i. We use the median of the draws because the mean is unstable.");

// 9. Bayes
s = content("Bayesian estimation with a joint Gamma–Dirichlet prior", "Estimation methods");
fitImage(s, FIG("eq_gd.png"), 0.6, 1.45, 6.6, 0.7, { name: "eq-gd", left: true });
bullets(s, ["Sum α₊ ~ Gamma(a₀, b₀); shares αᵢ/α₊ ~ Dirichlet(c) (ref. 3)",
  "Dependent prior: correlation > 0 if a₀ < c₊, = 0 if a₀ = c₊, < 0 if a₀ > c₊",
  "λ ~ Gamma(a, b) — must be proper: vague priors give an improper posterior"], 0.6, 2.35, 6.6, 2.3, 15);
fitImage(s, FIG("eq_linex.png"), 0.6, 4.75, 6.6, 0.7, { name: "eq-linex", left: true });
s.addText("Squared-error and LINEX (h = 1) loss; 95% HPD credible intervals", { x: 0.6, y: 5.55, w: 6.6, h: 0.4, fontSize: 13, color: C.text2, margin: 0, isTextBox: true });
card(s, 7.5, 1.45, 5.2, 5.05, "card-is");
s.addText("Importance sampler (exact, no MCMC)", { x: 7.75, y: 1.6, w: 4.8, h: 0.5, fontFace: "Cambria", fontSize: 18, bold: true, color: C.accent1, margin: 0, isTextBox: true });
["Sample λ from its 1-D marginal on a fine grid", "Sample αᵢ | λ ~ Gamma(Dᵢ + cᵢ, b₀ + Δᵢ(λ))", "Weight w = α₊^(a₀ − c₊) → posterior means, LINEX, HPD"].forEach((t, i) => {
  numberCircle(s, i + 1, 7.75, 2.35 + i * 1.25, 0.5);
  s.addText(t, { x: 8.4, y: 2.3 + i * 1.25, w: 4.1, h: 1.05, fontSize: 15, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addText("No burn-in, no tuning, fully vectorised; exact when a₀ = c₊.", { x: 7.75, y: 6.0, w: 4.8, h: 0.4, fontSize: 12, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("The joint prior links the facility effects through their sum and shares. The posterior factorises into an easy part times a simple weight, so importance sampling is exact and fast. A proper prior on lambda is essential because the likelihood does not vanish at lambda to zero.");

// 10. Order restriction
s = content("Order-restricted inference: α₁ ≤ α₂ ≤ … ≤ αₖ", "Estimation methods");
stat(s, 0.6, 1.5, 3.6, "19–42%", "of simulated samples had unrestricted MLEs in the wrong order");
card(s, 4.6, 1.45, 8.1, 2.45, "card-ormle");
s.addText("Order-restricted MLE (isotonic regression)", { x: 4.85, y: 1.55, w: 7.6, h: 0.45, fontFace: "Cambria", fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
fitImage(s, FIG("eq_ormle.png"), 4.85, 2.0, 7.0, 1.25, { name: "eq-ormle", left: true });
s.addText("Computed by pool-adjacent-violators; then maximise over λ.", { x: 4.85, y: 3.35, w: 7.6, h: 0.45, fontSize: 13, color: C.text2, margin: 0, isTextBox: true });
card(s, 4.6, 4.15, 8.1, 2.35, "card-orbayes");
s.addText("Order-restricted Bayes (ordered GD prior, ref. 3)", { x: 4.85, y: 4.25, w: 7.6, h: 0.45, fontFace: "Cambria", fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
bullets(s, ["Prior = density of the order statistics of a GD vector (sum over k! permutations)",
  "Sampler: draw αᵢ' from the gamma conditionals, sort them, re-weight by target / density of the sorted vector"], 4.85, 4.75, 7.6, 1.7, 14);
bullets(s, ["Stress levels, operators or time periods often have a known ordering", "Coal data: mine safety improved, so tails got lighter"], 0.6, 3.4, 3.7, 3.0, 14);
s.addNotes("When we know the ordering, imposing it removes impossible estimates. The OR-MLE pools adjacent facilities that violate the order. The Bayesian version uses the ordered Gamma-Dirichlet prior; sorting the proposals keeps every draw inside the ordered region.");

// 11. Simulation design
pres.addSection({ title: "Simulation study" });
s = content("Simulation design", "Simulation study");
const rows = [
  ["Facilities / truth", "k = 3;  α = (0.50, 0.75, 1.00) (ordered),  λ = 1,  t₀ = 1"],
  ["Sizes (nᵢ, mᵢ, κᵢ)", "S1 = (20, 15, 10),  S2 = (30, 24, 15),  S3 = (40, 32, 20)"],
  ["Time limit", "T = 3 (short test),  T = 8 (long test)"],
  ["Removal schemes", "A: all at m-th failure;  B: all at 1st failure;  C: half / half"],
  ["Replications", "2,000 per design × 18 designs;  1,000 pivotal draws;  2,000 IS draws"],
  ["Priors", "P1 informative (a₀ = 4),  P2 weakly informative (a₀ = 1)"],
  ["Criteria", "Bias, MSE, median abs. error;  length and coverage of 95% intervals"],
];
s.addTable(rows.map((r, i) => [
  { text: r[0], options: { bold: true, color: C.text1, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } },
  { text: r[1], options: { color: C.text1, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }]),
  { x: 0.6, y: 1.5, w: 12.1, colW: [3.0, 9.1], fontSize: 16, rowH: 0.62, border: { type: "none" }, valign: "middle", margin: [0, 0.15, 0, 0.15] });
s.addText("Heavy tails (α ≤ 1) keep the Lomax clearly away from its exponential limit; a light-tailed pilot (α = 1–2) gave non-existent MLEs in about 20% of samples.",
  { x: 0.6, y: 6.05, w: 12.1, h: 0.6, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("Eighteen designs. Samples are generated with the Balakrishnan-Sandhu algorithm and cut by the GPHC rule. Each design produces estimates from twelve estimators.");

// 12. Accuracy
s = content("Accuracy: Bayes-OR is best; the MLE has rare but huge failures", "Simulation study");
fitImage(s, FIG("sim_mse_T3.png"), 0.6, 1.3, 8.3, 5.4, { name: "sim-mse" });
stat(s, 9.2, 1.45, 3.5, "96.6 vs 0.15", "MSE of α₁: MLE vs pivotal (n = 20, T = 3) — same typical error (0.18 vs 0.17)");
stat(s, 9.2, 3.25, 3.5, "18 / 18", "designs where order-restricted Bayes has the lowest MSE for α₂, R₂, R₃", HEX.accent3);
s.addText("Bottom row: median absolute error — robust to the few runaway MLEs.", { x: 9.2, y: 5.1, w: 3.5, h: 1.0, fontSize: 13, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("Top: MSE; bottom: median absolute error. The MLE's MSE explodes because a few near-exponential samples push alpha-hat to huge values, while its typical error is fine. Pivotal estimates are stable. Bayes, especially order-restricted, is the most accurate.");

// 13. Intervals
s = content("Intervals: pivotal GCIs keep their promise of 95%", "Simulation study");
fitImage(s, FIG("sim_coverage_T8.png"), 0.6, 1.35, 12.1, 3.3, { name: "sim-coverage" });
stat(s, 0.6, 4.85, 3.8, "93.5–97.5%", "coverage of pivotal GCIs in all 18 designs");
stat(s, 4.75, 4.85, 3.8, "81–90%", "coverage of MLE-based ACIs for λ (Wald fails)", "C0392B");
stat(s, 8.9, 4.85, 3.8, "Shortest", "Bayes HPD intervals, coverage ≥ 95% for α, λ; 92–96% for R(t)", HEX.accent3);
s.addNotes("The normal approximation is poor for lambda, so ACIs under-cover. Pivotal intervals are wider but honest. Bayes HPD intervals are the shortest; they over-cover slightly because the priors are centred at the truth.");

// 14. Boxplots
s = content("Why the MLE's MSE explodes", "Simulation study");
fitImage(s, FIG("sim_boxplots_alpha1.png"), 0.6, 1.35, 12.1, 3.9, { name: "sim-boxplots" });
bullets(s, ["Typical MLE is fine — the medians sit near the truth (0.5)",
  "But 90 of 2,000 MLEs exceed 3 when nᵢ = 20, T = 3: the sample looks exponential, λ̂ → 0, α̂ explodes (Theorem 2)",
  "Pivotal and Bayes estimators are robust; the problem fades for nᵢ ≥ 30 with a long test"], 0.6, 5.35, 12.1, 1.5, 15);
s.addNotes("Box plots of 2,000 estimates of alpha_1. The counts on top show how many MLEs fall above 3. This is the practical message: do not trust the plain MLE for Lomax shapes in small or short tests.");

// 15. Real data
pres.addSection({ title: "Real data" });
s = content("Real data: fatalities in British coal-mining disasters", "Real data");
fitImage(s, FIG("real_descriptive.png"), 0.6, 1.35, 7.9, 3.6, { name: "real-descriptive" });
bullets(s, ["Disasters with ≥ 10 deaths (data from ref. 2); 3 periods = 3 \"facilities\": 1900–19 (n = 21), 1920–39 (n = 20), 1940–59 (n = 17)",
  "Model the excess over the recording threshold: y = x − 0.19 (peaks over threshold)",
  "Prior knowledge: safety improved, so tails get lighter: α₁ ≤ α₂ ≤ α₃"], 8.8, 1.45, 3.9, 4.8, 14);
s.addText("All three periods are right-skewed with coefficient of variation > 1 → Lomax MLE exists.", { x: 0.6, y: 5.2, w: 7.9, h: 0.6, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("This is the coal-mining data analysed in the IER reference paper. The excess over the threshold is the natural Lomax variable. Later periods have lighter tails, consistent with improved safety.");

// 16. Model checking
s = content("Model checking: Lomax fits, common λ is supported", "Real data");
fitImage(s, FIG("real_gof.png"), 0.6, 1.3, 6.3, 5.5, { name: "real-gof" });
stat(s, 7.3, 1.4, 2.6, "p > 0.54", "KS tests, Lomax with common λ, all periods");
stat(s, 10.1, 1.4, 2.6, "p = 0.65", "LRT: H₀ λ₁ = λ₂ = λ₃ not rejected", HEX.accent3);
stat(s, 7.3, 3.35, 2.6, "p = 0.051", "LRT: H₀ α₁ = α₂ = α₃ (complete data)", HEX.accent2);
stat(s, 10.1, 3.35, 2.6, "p = 0.026", "same test, censored Scheme 2", HEX.accent2);
s.addText("Lomax beats the exponential on AIC in every period; the exponential is rejected for 1920–39 (KS p = 0.035).", { x: 7.3, y: 5.3, w: 5.4, h: 0.9, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("Goodness of fit with KS tests and probability plots, then likelihood ratio tests from the course: a common lambda is supported, and there is evidence that the shapes differ across periods.");

// 17. Estimates
s = content("Estimates: large disasters became much less likely", "Real data");
const est = readCsv(path.join(ROOT, "results", "real_data", "estimates_all_methods.csv")).filter((r) => r.Design === "Complete");
const get = (q, m) => Number(est.find((r) => r.Quantity === q && r.Method === m).Estimate).toFixed(3);
const qs = [["alpha1", "α₁ (1900–19)"], ["alpha2", "α₂ (1920–39)"], ["alpha3", "α₃ (1940–59)"], ["lam", "λ"], ["R1", "R₁(0.5)"], ["R2", "R₂(0.5)"], ["R3", "R₃(0.5)"]];
const ms = [["MLE", "MLE"], ["Pivotal", "Pivotal"], ["Bayes-U (SELF)", "Bayes-U"], ["Bayes-OR (SELF)", "Bayes-OR"]];
const head = [{ text: "Complete data", options: { bold: true, color: C.background1, fill: { color: HEX.dk2 } } }]
  .concat(ms.map(([, l]) => ({ text: l, options: { bold: true, color: C.background1, fill: { color: HEX.dk2 }, align: "right" } })));
const body = qs.map(([q, l], i) => [{ text: l, options: { color: C.text1, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }]
  .concat(ms.map(([m]) => ({ text: get(q, m), options: { color: C.text1, align: "right", fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }))));
s.addTable([head].concat(body), { x: 0.6, y: 1.45, w: 6.0, colW: [1.9, 1.0, 1.0, 1.05, 1.05], fontSize: 14, rowH: 0.5, border: { type: "none" }, valign: "middle", margin: [0, 0.1, 0, 0.1] });
fitImage(s, FIG("real_curves_scheme1.png"), 6.9, 1.4, 5.8, 3.0, { name: "real-curves" });
bullets(s, ["P(a disaster kills > ~35 people) fell from ≈ 0.48 (1900–19) to ≈ 0.20 (1940–59)",
  "Censoring (Schemes 1, 2) widens intervals for λ but barely moves R(t)",
  "Bayes-OR gives the shortest HPD intervals for the reliabilities"], 6.9, 4.55, 5.8, 2.2, 14);
s.addNotes("All methods agree on the ordering. R_i(0.5) is the probability that a qualifying disaster kills more than about 35 people; it dropped by more than half. The curves show Bayes-OR (thick) versus MLE (thin) for censored Scheme 1.");

// 18. Conclusions
pres.addSection({ title: "Conclusions" });
s = content("Conclusions and recommendations", "Conclusions");
const concl = [
  ["Pivotal → honest intervals", "Truncated, median-based GPQs give near-95% coverage without priors or large-sample theory.", HEX.accent1],
  ["Order-restricted Bayes → most accurate", "The joint GD prior plus a known ordering gives the smallest errors and the shortest intervals.", HEX.accent3],
  ["Be careful with the plain MLE", "Lomax shapes need visible tails: small or short tests can make the MLE (and Wald CIs) unreliable.", "C0392B"],
];
concl.forEach(([h, t, col], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.55, 3.9, 3.3, `concl-${i}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.8, w: 0.55, h: 0.55, fill: { color: col }, line: { color: col }, objectName: `concl-dot-${i}` });
  s.addText(h, { x: x + 0.25, y: 2.55, w: 3.4, h: 0.9, fontFace: "Cambria", fontSize: 19, bold: true, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.25, y: 3.45, w: 3.4, h: 1.7, fontSize: 15, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
s.addText("Future work: exact pivots under GPHC, bootstrap intervals for the OR-MLE, optimal censoring plans, facility-specific λᵢ.",
  { x: 0.6, y: 5.2, w: 12.1, h: 0.7, fontSize: 16, italic: true, color: C.text2, margin: 0, isTextBox: true });
s.addNotes("Three take-home messages, then possible extensions.");

// 19. References
s = content("References", "Conclusions");
s.addText([
  { text: "[1] Kumari, Tripathi, Lodhi & Wang (2025). Pivotal based reliability inference for Burr-XII distribution under block progressive type-II censoring. Comm. Stat. – Simulation & Computation 55(8).", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "[2] Lodhi, Gangopadhyay & Bhattacharya (2025). Likelihood and pivotal-based reliability inference for inverse exponentiated Rayleigh distribution under block progressive censoring. QTQM 22(4), 779–804.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "[3] Sultana, Çetinkaya, Raqab & Kundu (2026). Inference for block Type-I hybrid censored Weibull populations. Comm. Stat. – Theory & Methods 55(11), 3436–3459.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "[4] Cho, Sun & Lee (2015). Exact likelihood inference for an exponential parameter under generalized progressive hybrid censoring scheme. Statistical Methodology 23, 18–34. (the one additional reference)", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "Course texts: Casella & Berger, Statistical Inference; Hogg, McKean & Craig, Introduction to Mathematical Statistics." },
], { x: 0.6, y: 1.5, w: 12.1, h: 5.0, fontSize: 15, color: C.text1, valign: "top", isTextBox: true });
s.addNotes("Three reference papers from the brief plus one extra paper that defines the censoring scheme.");

// 20. Thank you
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusions" });
s.addText("Thank you — questions?", { placeholder: "title" });
s.addText("Code, results, report and this deck are fully reproducible (fixed random seeds).", { placeholder: "body" });
s.addNotes("Open the floor for questions.");

pres.writeFile({ fileName: OUT }).then(async () => {
  const { applyTheme } = require(process.env.APPLY_THEME);
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
});
