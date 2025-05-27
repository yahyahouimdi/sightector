/**
 * Builds docs/Sightector_Presentation.pptx (defense deck, 16 slides, with speaker notes).
 *
 *   cd scripts && npm install
 *   npm run slides
 *
 * Numbers come from artifacts/reports (benchmark.csv, ablation.json), so re-running
 * `python -m sightector evaluate` and this script keeps the slides in sync with the models.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "docs", "figures", f);
const SHOT = (f) => path.join(ROOT, "docs", "screenshots", f);
const OUT = path.join(ROOT, "docs", "Sightector_Presentation.pptx");

// ---------------------------------------------------------------- palette
const NAVY = "0C2547", BLUE = "0C6A9D", RED = "CC163F", INK2 = "52607A", LINE = "D6DDE8";
const TINT = "EEF3F9", WHITE = "FFFFFF", PINK = "FDE8EC";
const MODEL = { legacy: "9A9893", baseline: "2A78D6", bilstm: "EB6834", transformer: "1BAF7A" };
const HEAD = "Cambria", BODY = "Calibri";

// ------------------------------------------------------------------- data
function readBenchmark() {
  const lines = fs.readFileSync(path.join(ROOT, "artifacts/reports/benchmark.csv"), "utf8").trim().split(/\r?\n/);
  const cols = lines[0].split(",");
  const rows = lines.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));
  const get = (kind, set, metric = "macro_f1") => +rows.find((r) => r.kind === kind && r.set === set)[metric];
  return get;
}
const bench = readBenchmark();
const ablation = JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts/reports/ablation.json"), "utf8"));
const stats = JSON.parse(fs.readFileSync(path.join(ROOT, "data/processed/stats.json"), "utf8"));
const pct = (x) => `${(x * 100).toFixed(1)}`;
const fmt = (n) => n.toLocaleString("en-US");

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: `#${color}`, size }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// ------------------------------------------------------------------ build
(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.author = "IEEE ENIT SIGHT Group";
  pres.title = "Sightector v2 - Obfuscation-robust harassment detection";
  pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

  const footer = { text: "Sightector v2 · IEEE ENIT SIGHT Group", options: { x: 0.5, y: 5.22, w: 6, h: 0.25, fontSize: 9, color: INK2, fontFace: BODY } };
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: NAVY },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 6.2, h: 1.2, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.85, w: 6.2, h: 0.9, fontFace: BODY, fontSize: 18, color: "CADCFC", valign: "top", align: "left" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "DARK_LIST",
    background: { color: NAVY },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.5, w: 8.8, h: 0.9, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.6, w: 8.8, h: 2.6, fontFace: BODY, fontSize: 16, color: "CADCFC", valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: WHITE },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9, h: 0.65, fontFace: HEAD, fontSize: 28, bold: true, color: NAVY, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: footer },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontSize: 9, color: INK2, fontFace: BODY, align: "right" },
  });

  const content = (section, title) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const text = (s, t, o) => s.addText(t, { isTextBox: true, fontFace: BODY, fontSize: 14, color: NAVY, margin: 0, valign: "top", ...o });
  const card = (s, x, y, w, h, fill = TINT, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
  const chartText = { catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, dataLabelFontFace: BODY, legendFontFace: BODY,
                      catAxisLabelColor: INK2, valAxisLabelColor: INK2, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
                      dataLabelFontSize: 9, dataLabelColor: INK2, legendFontSize: 11, legendColor: INK2 };

  const icons = {
    bug: await icon(fa.FaBug, RED), leak: await icon(fa.FaCopy, RED), filter: await icon(fa.FaFilter, RED),
    db: await icon(fa.FaDatabase, BLUE), broom: await icon(fa.FaBroom, BLUE), shuffle: await icon(fa.FaRandom, BLUE),
    chart: await icon(fa.FaChartBar, MODEL.baseline), net: await icon(fa.FaProjectDiagram, MODEL.bilstm),
    brain: await icon(fa.FaBrain, MODEL.transformer), check: await icon(fa.FaCheckCircle, MODEL.transformer),
    warn: await icon(fa.FaExclamationTriangle, RED), tag: await icon(fa.FaTags, BLUE),
  };

  // 1 ── Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addImage({ path: path.join(ROOT, "server/static/logo.png"), x: 6.9, y: 0.25, w: 2.8, h: 2.8, transparency: 0, objectName: "logo" });
  s.addText("Sightector v2", { placeholder: "title" });
  s.addText("Obfuscation-robust online harassment detection with deep learning", { placeholder: "body" });
  text(s, "Final-year engineering project · IEEE ENIT SIGHT Group · École Nationale d'Ingénieurs de Tunis",
       { x: 0.6, y: 4.6, w: 8.8, h: 0.4, fontSize: 12, color: "CADCFC" });
  s.addNotes("Introduce the project: Sightector is a browser extension that tells a user whether a text they read is harassment. This presentation covers why the first version failed, how we rebuilt it end-to-end with deep learning, and the measured results.");

  // 2 ── Problem
  s = content("Introduction", "The problem: harassment hides behind spelling");
  text(s, "Harassers deliberately disguise words to slip past keyword filters. A word-level model sees only unknown tokens.",
       { x: 0.5, y: 1.05, w: 9, h: 0.6, fontSize: 15, color: INK2 });
  const tricks = [["Masking", "f**k · b*tch"], ["Inserted symbols", "fu*ck · f-u-c-k"], ["Leetspeak", "1d10t · sh!t"],
                  ["Elongation", "iiidiot · stuuupid"], ["Spacing", "f u c k you"], ["Look-alike letters", "ѕtupid · іdiot"]];
  tricks.forEach(([name, ex], i) => {
    const x = 0.5 + (i % 3) * 3.05, y = 1.85 + Math.floor(i / 3) * 1.5;
    card(s, x, y, 2.85, 1.25, i % 2 ? TINT : PINK, `trick-${i}`);
    text(s, name, { x: x + 0.2, y: y + 0.18, w: 2.5, h: 0.35, fontSize: 13, bold: true, color: INK2 });
    text(s, ex, { x: x + 0.2, y: y + 0.55, w: 2.5, h: 0.5, fontSize: 18, bold: true, color: RED, fontFace: "Consolas" });
  });
  s.addNotes("These are the real tricks our users see. Version 1 missed most of them: the client reported that typing fu*ck or iiidiot was not detected.");

  // 3 ── Audit
  s = content("Introduction", "Audit of v1: three root causes");
  const defects = [
    [icons.bug, "Inference bug", "The server called fit_transform() on every message: the learned IDF weights were thrown away."],
    [icons.leak, "Data leakage", "pd.concat([df, df]) before the split: ~80 % of test rows were also in training. The reported 97 % was memorisation."],
    [icons.filter, "Blind features", "Word-only TF-IDF + a 1 081-word stop-list (it deleted words like \"bad\"). Disguised words became unknown tokens."],
  ];
  defects.forEach(([img, h, b], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.15, 2.85, 2.55, TINT, `defect-${i}`);
    s.addImage({ data: img, x: x + 0.25, y: 1.38, w: 0.45, h: 0.45 });
    text(s, h, { x: x + 0.25, y: 1.95, w: 2.4, h: 0.4, fontSize: 16, bold: true });
    text(s, b, { x: x + 0.25, y: 2.38, w: 2.4, h: 1.25, fontSize: 12.5, color: INK2 });
  });
  text(s, [{ text: "Re-tested honestly, v1 scores ", options: {} },
           { text: `${bench("legacy", "challenge").toFixed(2)} macro-F1`, options: { bold: true, color: RED } },
           { text: " on our hand-written challenge set and recognises only ", options: {} },
           { text: "27 %", options: { bold: true, color: RED } },
           { text: " of obfuscated insults.", options: {} }],
       { x: 0.5, y: 4.0, w: 9, h: 0.6, fontSize: 16 });
  s.addNotes("We read the v1 code before changing anything. Three defects, each enough on its own to explain the bad behaviour. The 97 percent accuracy in the notebook was an artefact of data leakage.");

  // 4 ── Approach
  pres.addSection({ title: "Method" });
  s = content("Method", "Our approach: an end-to-end pipeline");
  s.addImage({ path: FIG("pipeline.png"), x: 0.5, y: 1.15, w: 9, h: 1.87 });
  const goals = [["Clean", "leakage-free data, enriched with real threats"], ["Robust", "de-obfuscation + adversarial augmentation"],
                 ["Deep", "BiLSTM and a fine-tuned Transformer vs a strong baseline"], ["Explainable", "severity + the words behind each decision"]];
  goals.forEach(([h, b], i) => {
    const x = 0.5 + i * 2.29;
    text(s, h, { x, y: 3.35, w: 2.1, h: 0.35, fontSize: 16, bold: true, color: i % 2 ? RED : BLUE });
    text(s, b, { x, y: 3.72, w: 2.1, h: 0.9, fontSize: 12.5, color: INK2 });
  });
  s.addNotes("The pipeline goes from raw data to evaluation. Every stage is a module of the sightector Python package and runs from one CLI.");

  // 5 ── Data
  s = content("Method", "Data: cleaned, de-duplicated and enriched");
  const kpis = [[fmt(stats.clean_rows), "messages after cleaning"], [fmt(stats.by_source.civil_comments.count), "added from Civil Comments"],
                [fmt(stats.duplicates_removed), "duplicates removed after normalisation"], ["80/10/10", "stratified train / val / test"]];
  kpis.forEach(([v, l], i) => {
    const y = 1.15 + i * 0.95;
    text(s, v, { x: 0.5, y, w: 2.2, h: 0.55, fontSize: 30, bold: true, color: i === 1 ? RED : BLUE, fontFace: HEAD });
    text(s, l, { x: 2.75, y: y + 0.12, w: 2.4, h: 0.5, fontSize: 13, color: INK2 });
  });
  s.addChart(pres.charts.BAR, [
    { name: "Clean", labels: ["Original dataset", "Civil Comments"], values: [stats.by_source.original.count - stats.by_source.original.harassment, stats.by_source.civil_comments.count - stats.by_source.civil_comments.harassment] },
    { name: "Harassment", labels: ["Original dataset", "Civil Comments"], values: [stats.by_source.original.harassment, stats.by_source.civil_comments.harassment] },
  ], { x: 5.3, y: 1.05, w: 4.3, h: 3.9, barDir: "bar", barGrouping: "stacked", chartColors: ["2A78D6", "E34948"],
       showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "ctr", dataLabelColor: WHITE, dataLabelFontSize: 10,
       valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "Messages by source and class",
       titleFontFace: BODY, titleFontSize: 13, titleColor: NAVY, ...chartText, dataLabelColor: WHITE });
  s.addNotes("Error analysis of the first models showed almost no threats in the original Wikipedia data. We enriched it with 9.4k real annotated comments: threats, identity attacks, and hard negatives, i.e. clean comments that use words like kill or die harmlessly. Duplicates are removed after normalisation so F*CK YOU and fuck you can never sit on both sides of the split.");

  // 6 ── Normalizer
  s = content("Method", "Obfuscation-aware normaliser");
  const ex = [["you are a fu*ck iiidiot", "you are a fuck idiot"], ["f u c k you", "fuck you"], ["u r a m0r0n", "u r a moron"],
              ["sh!t head", "shithead"], ["ѕtupid  (Cyrillic)", "stupid"], ["kill the process", "kill the process  (unchanged)"]];
  const rowsN = [[{ text: "Input", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "What the model reads", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
                 ...ex.map(([a, b], i) => [{ text: a, options: { fontFace: "Consolas", color: RED } }, { text: b, options: { fontFace: "Consolas", color: i === 5 ? INK2 : NAVY } }])];
  s.addTable(rowsN, { x: 0.5, y: 1.15, w: 5.3, colW: [2.65, 2.65], fontSize: 12, fontFace: BODY, border: { type: "solid", color: LINE, pt: 0.75 }, rowH: 0.42 });
  card(s, 6.1, 1.15, 3.4, 3.75, TINT, "normaliser-rules");
  text(s, "How it works", { x: 6.35, y: 1.3, w: 3, h: 0.4, fontSize: 16, bold: true });
  text(s, [
    { text: "Unicode folding and look-alike letters", options: { bullet: true, breakLine: true } },
    { text: "Join spaced letters", options: { bullet: true, breakLine: true } },
    { text: "Leetspeak, masks and wildcards", options: { bullet: true, breakLine: true } },
    { text: "Collapse letter elongation", options: { bullet: true, breakLine: true } },
    { text: "Re-join split words", options: { bullet: true, breakLine: true } },
    { text: "Every repair validated against a 145-term lexicon: clean text is never rewritten", options: { bullet: true, bold: true, color: BLUE } },
  ], { x: 6.35, y: 1.8, w: 3, h: 3, fontSize: 12.5, paraSpaceAfter: 5, color: INK2 });
  s.addNotes("The same function runs at training and at inference. The key design choice is safety: a repair is only applied if the result is a known offensive term, so 4you, class or kill the process are left alone. The decision itself is always made by the neural model.");

  // 7 ── Augmentation & evaluation sets
  s = content("Method", "Adversarial augmentation and honest evaluation");
  card(s, 0.5, 1.15, 4.3, 3.75, PINK, "augmentation");
  s.addImage({ data: icons.shuffle, x: 0.75, y: 1.35, w: 0.42, h: 0.42 });
  text(s, "Augmentation", { x: 1.3, y: 1.38, w: 3.2, h: 0.4, fontSize: 16, bold: true });
  text(s, [
    { text: "One obfuscated copy of each harassment message", options: { bullet: true, breakLine: true } },
    { text: "Noisy copies of half the clean messages: no \"weird spelling = toxic\" shortcut", options: { bullet: true, breakLine: true } },
    { text: `${fmt(stats.train)} → ${fmt(stats.train_augmented)} training messages`, options: { bullet: true, bold: true } },
  ], { x: 0.75, y: 1.95, w: 3.85, h: 1.9, fontSize: 13, paraSpaceAfter: 6, color: INK2 });
  text(s, "you are an idiot  →  you are an 1d!0t", { x: 0.75, y: 4.15, w: 3.85, h: 0.4, fontSize: 13, fontFace: "Consolas", color: RED });
  const sets = [["Test", `${fmt(stats.test)} held-out messages`], ["Obfuscated test", "every offensive word disguised (unseen seed)"],
                ["Challenge", "78 hand-written sentences: threats, sexism, tricky clean text"], ["Ablation", "normaliser × augmentation"]];
  sets.forEach(([h, b], i) => {
    const y = 1.15 + i * 0.95;
    card(s, 5.1, y, 4.4, 0.8, TINT, `set-${i}`);
    text(s, h, { x: 5.3, y: y + 0.12, w: 1.6, h: 0.55, fontSize: 14, bold: true, color: BLUE, valign: "middle" });
    text(s, b, { x: 6.95, y: y + 0.12, w: 2.45, h: 0.55, fontSize: 12, color: INK2, valign: "middle" });
  });
  s.addNotes("The challenge set was written by hand and is independent of every dataset. We deliberately did not tune on it, so it remains an honest test of real-world behaviour.");

  // 8 ── Models
  s = content("Method", "Three models, one interface");
  const models = [
    [icons.chart, "TF-IDF + LogReg", "Baseline", "word 1-2-grams + character 2-5-grams", "25 s training"],
    [icons.net, "BiLSTM + Attention", "Deep learning from scratch", "embeddings → BiLSTM → additive attention · 7.2 M params", "49 s training"],
    [icons.brain, "DistilRoBERTa", "Fine-tuned Transformer (deployed)", "6-layer self-attention, byte-level BPE · 82 M params", "7 min on RTX 4060"],
  ];
  models.forEach(([img, h, sub, b, t], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.15, 2.85, 3.0, i === 2 ? "E3F5EC" : TINT, `model-${i}`);
    s.addImage({ data: img, x: x + 0.25, y: 1.38, w: 0.5, h: 0.5 });
    text(s, h, { x: x + 0.25, y: 2.0, w: 2.45, h: 0.4, fontSize: 16, bold: true });
    text(s, sub, { x: x + 0.25, y: 2.4, w: 2.45, h: 0.35, fontSize: 12, italic: true, color: BLUE });
    text(s, b, { x: x + 0.25, y: 2.8, w: 2.45, h: 0.85, fontSize: 12, color: INK2 });
    text(s, t, { x: x + 0.25, y: 3.65, w: 2.45, h: 0.35, fontSize: 12, bold: true, color: INK2 });
  });
  text(s, "Shared training loop: BCE loss · AdamW · warm-up · fp16 · gradient clipping · early stopping · threshold tuned on validation",
       { x: 0.5, y: 4.4, w: 9, h: 0.5, fontSize: 13, color: NAVY });
  s.addNotes("All three models implement the same predict_proba interface, so the API and evaluation never depend on a specific model. Self-attention lets the Transformer read each word in context: kill the process versus I will kill you.");

  // 9 ── Training
  s = content("Method", "Training: early stopping against noisy labels");
  s.addImage({ path: FIG("learning_curves.png"), x: 1.05, y: 1.0, w: 7.9, h: 2.99 });
  [["Overfits early", "validation loss rises after epoch 1 (Transformer) and epoch 4 (BiLSTM)"],
   ["Why", "label noise in the source data (see error analysis)"],
   ["Fix", "early stopping restores the best weights automatically"]].forEach(([h, b], i) => {
    const x = 0.5 + i * 3.05;
    text(s, h, { x, y: 4.15, w: 2.85, h: 0.3, fontSize: 14, bold: true, color: i === 1 ? RED : BLUE });
    text(s, b, { x, y: 4.47, w: 2.85, h: 0.6, fontSize: 12, color: INK2 });
  });
  s.addNotes("Dashed lines are training loss, solid lines validation loss. The model quickly starts memorising mislabelled examples, so stopping early matters.");

  // 10 ── Results
  pres.addSection({ title: "Results" });
  s = content("Results", "Results: DistilRoBERTa is best on every set");
  const setNames = ["Test", "Obfuscated test", "Challenge"], setKeys = ["test", "test_obfuscated", "challenge"];
  s.addChart(pres.charts.BAR, ["legacy", "baseline", "bilstm", "transformer"].map((k) => ({
    name: { legacy: "Legacy v1", baseline: "TF-IDF + LogReg", bilstm: "BiLSTM + Attention", transformer: "DistilRoBERTa" }[k],
    labels: setNames, values: setKeys.map((sk) => +bench(k, sk).toFixed(3)),
  })), { x: 0.4, y: 1.0, w: 6.3, h: 4.05, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
         chartColors: [MODEL.legacy, MODEL.baseline, MODEL.bilstm, MODEL.transformer], showLegend: true, legendPos: "b",
         showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
         valAxisLabelFormatCode: "0.0", valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" },
         showTitle: true, title: "Macro-F1", titleFontFace: BODY, titleFontSize: 13, titleColor: NAVY, ...chartText });
  const tf = (sk, m = "macro_f1") => bench("transformer", sk, m);
  [[pct(tf("test", "accuracy")) + " %", "test accuracy"], [tf("test", "roc_auc").toFixed(3), "ROC-AUC"],
   [`${bench("legacy", "test_obfuscated").toFixed(2)} → ${tf("test_obfuscated").toFixed(2)}`, "macro-F1 on obfuscated text (v1 → v2)"]]
    .forEach(([v, l], i) => {
      const y = 1.15 + i * 1.25;
      text(s, v, { x: 6.95, y, w: 2.6, h: 0.55, fontSize: i === 2 ? 24 : 30, bold: true, color: i === 2 ? RED : MODEL.transformer, fontFace: HEAD });
      text(s, l, { x: 6.95, y: y + 0.58, w: 2.6, h: 0.45, fontSize: 12, color: INK2 });
    });
  s.addNotes("All numbers are measured on data never used for training or model selection. The axis starts at 0.5 to make the differences visible. The legacy model was re-run exactly as it was deployed, including the IDF bug.");

  // 11 ── Per-category
  s = content("Results", "Where each model succeeds: challenge set");
  s.addImage({ path: FIG("challenge_by_category.png"), x: 0.5, y: 1.05, w: 9, h: 3.0 });
  text(s, [
    { text: "Obfuscation: ", options: { bold: true, color: NAVY } }, { text: "v1 27 % → v2 91-100 %.   ", options: {} },
    { text: "Context: ", options: { bold: true, color: NAVY } }, { text: "only the Transformer is right on 94 % of tricky clean sentences.   ", options: {} },
    { text: "Still hard: ", options: { bold: true, color: RED } }, { text: "implicit threats and stereotype sexism.", options: {} },
  ], { x: 0.5, y: 4.2, w: 9, h: 0.75, fontSize: 13, color: INK2 });
  s.addNotes("Precision matters for a tool people trust: on the challenge set DistilRoBERTa reaches 0.977 precision. The keyword-driven models flag kill the process; the Transformer does not.");

  // 12 ── Ablation
  s = content("Results", "Ablation: the normaliser is the key component");
  s.addChart(pres.charts.BAR, [
    { name: "Obfuscated test", labels: ablation.map((r) => r.setting), values: ablation.map((r) => +r.test_obfuscated.toFixed(3)) },
    { name: "Challenge", labels: ablation.map((r) => r.setting), values: ablation.map((r) => +r.challenge.toFixed(3)) },
  ], { x: 0.4, y: 1.0, w: 5.9, h: 4.05, barDir: "bar", barGrouping: "clustered", chartColors: ["2A78D6", "EB6834"],
       showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
       valAxisMinVal: 0.7, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" },
       catAxisOrientation: "maxMin", showTitle: true, title: "Macro-F1 (DistilRoBERTa)", titleFontFace: BODY, titleFontSize: 13, titleColor: NAVY,
       ...chartText, catAxisLabelFontSize: 10 });
  text(s, [
    { text: "Normaliser: +5 to +10 points on obfuscated text, no cost on clean text", options: { bullet: true, breakLine: true } },
    { text: "Augmentation: +2.8 on the challenge set but slightly lower on test: within noise", options: { bullet: true, breakLine: true } },
    { text: "Next step: several seeds and a larger challenge set", options: { bullet: true } },
  ], { x: 6.55, y: 1.2, w: 3, h: 3.6, fontSize: 13, paraSpaceAfter: 8, color: INK2 });
  s.addNotes("We report the augmentation result honestly: it is not a clear win. The augmented copies are normalised before training, so most of the disguise is already removed. The deployed configuration was fixed before the experiment; choosing on the challenge set would turn it into a training signal.");

  // 13 ── Error analysis
  s = content("Results", "Error analysis: the model often beats the labels");
  card(s, 0.5, 1.15, 4.35, 3.0, PINK, "label-noise");
  s.addImage({ data: icons.tag, x: 0.75, y: 1.35, w: 0.42, h: 0.42 });
  text(s, "Label noise", { x: 1.3, y: 1.38, w: 3.3, h: 0.4, fontSize: 16, bold: true });
  text(s, "17 of the 31 most confident \"false positives\" contain explicit insults but are labelled clean in the source data:",
       { x: 0.75, y: 1.95, w: 3.9, h: 0.9, fontSize: 12.5, color: INK2 });
  text(s, "\"you fing rat you ratted me out you little prick\"  →  labelled clean",
       { x: 0.75, y: 2.95, w: 3.9, h: 0.8, fontSize: 12.5, italic: true, color: RED });
  card(s, 5.15, 1.15, 4.35, 3.0, TINT, "weaknesses");
  s.addImage({ data: icons.warn, x: 5.4, y: 1.35, w: 0.42, h: 0.42 });
  text(s, "Genuine weaknesses", { x: 5.95, y: 1.38, w: 3.3, h: 0.4, fontSize: 16, bold: true });
  text(s, [
    { text: "Implicit threats: \"i know where you live…\"", options: { bullet: true, breakLine: true } },
    { text: "Stereotype sexism: \"women belong in the kitchen\"", options: { bullet: true, breakLine: true } },
    { text: "Violent idioms: \"shoot me an email\"", options: { bullet: true, breakLine: true } },
    { text: "Fix: targeted data, not better spelling rules", options: { bullet: true, bold: true, color: BLUE } },
  ], { x: 5.4, y: 1.95, w: 3.9, h: 2.8, fontSize: 12.5, paraSpaceAfter: 6, color: INK2 });
  s.addNotes("Reading the errors one by one shows that many are labelling mistakes in the original dataset; the real error rate is lower than measured. The remaining genuine errors are semantic.");

  // 14 ── System
  pres.addSection({ title: "System" });
  s = content("System", "Clean architecture, from model to browser");
  s.addImage({ path: FIG("architecture.png"), x: 0.5, y: 1.05, w: 6.4, h: 2.53 });
  text(s, [
    { text: "Layers: text → data → models → training → inference → API", options: { bullet: true, breakLine: true } },
    { text: "One TextClassifier interface; model fallback", options: { bullet: true, breakLine: true } },
    { text: "Flask app factory with dependency injection", options: { bullet: true, breakLine: true } },
    { text: "31 pytest tests · 18 ms per message on CPU", options: { bullet: true } },
  ], { x: 7.1, y: 1.1, w: 2.45, h: 3.6, fontSize: 12.5, paraSpaceAfter: 6, color: INK2 });
  text(s, "POST /api/analyze  →  harassment · score · severity · de-obfuscated words · highlights",
       { x: 0.5, y: 3.85, w: 6.4, h: 0.4, fontSize: 12, fontFace: "Consolas", color: BLUE });
  s.addNotes("The browser extension and the web dashboard call the same REST API. The detector is injected into the Flask app, which makes the API testable with a fake model in under a second.");

  // 15 ── Demo
  s = content("System", "Demo: dashboard and browser extension");
  s.addImage({ path: SHOT("dashboard_obfuscated_insult.png"), x: 0.5, y: 1.05, w: 3.55, h: 2.77, objectName: "shot-insult" });
  s.addImage({ path: SHOT("dashboard_clean_message.png"), x: 4.2, y: 1.05, w: 3.55, h: 2.77, objectName: "shot-clean" });
  s.addImage({ path: SHOT("extension_harassment.png"), x: 7.95, y: 1.0, w: 1.55, h: 3.84, objectName: "shot-extension" });
  text(s, "fu*ck iiidiot: detected, words decoded", { x: 0.5, y: 3.95, w: 3.55, h: 0.35, fontSize: 12, color: INK2 });
  text(s, "\"kill the process\": harmless in context", { x: 4.2, y: 3.95, w: 3.55, h: 0.35, fontSize: 12, color: INK2 });
  s.addNotes("Live demo: run python -m server, open the dashboard, and try the example chips. Then load the unpacked extension, select text on any page and click Scan.");

  // 16 ── Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "DARK_LIST", sectionTitle: "Conclusion" });
  s.addText("Conclusion and future work", { placeholder: "title" });
  s.addText([
    { text: `Obfuscated text: ${bench("legacy", "test_obfuscated").toFixed(2)} → ${tf("test_obfuscated").toFixed(2)} macro-F1;  challenge set: ${bench("legacy", "challenge").toFixed(2)} → ${tf("challenge").toFixed(2)}`, options: { bullet: true, breakLine: true } },
    { text: "Next: relabel noisy data, collect implicit threats, add Arabic / French / Tunisian Derja", options: { bullet: true, breakLine: true } },
    { text: "Next: multi-label heads, larger encoders, in-browser ONNX inference", options: { bullet: true } },
  ], { placeholder: "body", fontSize: 16, paraSpaceAfter: 6 });
  text(s, "Thank you · Questions?", { x: 0.6, y: 4.6, w: 8.8, h: 0.5, fontSize: 20, bold: true, color: WHITE, fontFace: HEAD });
  s.addNotes("Summarise: concrete root causes found and fixed, honest evaluation, and a deployable explainable system. Then open questions.");

  await pres.writeFile({ fileName: OUT });
  console.log("saved", OUT);
})();
