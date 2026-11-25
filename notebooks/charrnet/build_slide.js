// One-slide PowerPoint version of charrnet_overview.png, built from native shapes.
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const { applyTheme } = require('./apply_theme.js');

const D = JSON.parse(fs.readFileSync('slide_data.json'));
const THEME = {
  name: 'ChaRRNet', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: '1F2933', lt1: 'FFFFFF', dk2: '3E4C59', lt2: 'E4E7EB',
    accent1: '2F6DB5', accent2: 'E07B28', accent3: '9AA5B1', accent4: 'D9E8FB',
    accent5: 'FDE5CF', accent6: '616E7C', hlink: '2F6DB5', folHlink: '616E7C' },
};
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'ChaRRNet overview';
const C = pres.SchemeColor;
const S = pres.shapes;
const BLUE = C.accent1, ORANGE = C.accent2, GREY = C.accent3, DARK = C.text2, MUTED = C.accent6;

pres.defineSlideMaster({
  title: 'FIGURE', background: { color: C.background1 },
  objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 0.5, y: 0.25, w: 12.33, h: 0.6,
    fontSize: 24, bold: true, color: C.text1, valign: 'middle', margin: 0 }, text: '' } }],
});
pres.addSection({ title: 'ChaRRNet' });
const s = pres.addSlide({ masterName: 'FIGURE', sectionTitle: 'ChaRRNet' });
s.addText('ChaRRNet: building multipath symmetry into an RF fingerprinting network', { placeholder: 'title', align: 'left' });

// ---- uniform scaling of everything below the title (SCALE env var, default 0.7)
const K = parseFloat(process.env.SCALE || '0.7');
const OX = (13.333 - 12.333 * K) / 2 - 0.5 * K, OY = 1.0 + (6.4 - 6.3 * K) / 2 - 1.0 * K;
function scaleOpts(o) {
  if (!o || o.placeholder) return o;
  o = Object.assign({}, o);
  if (o.x !== undefined) o.x = OX + o.x * K;
  if (o.y !== undefined) o.y = OY + o.y * K;
  if (o.w !== undefined) o.w *= K;
  if (o.h !== undefined) o.h *= K;
  if (o.fontSize) o.fontSize = Math.round(o.fontSize * K * 2) / 2;
  if (o.rectRadius) o.rectRadius *= K;
  if (o.margin) o.margin *= K;
  if (o.paraSpaceAfter) o.paraSpaceAfter *= K;
  if (o.line) o.line = Object.assign({}, o.line, o.line.width ? { width: Math.max(o.line.width * K, 0.25) } : {});
  if (o.points) o.points = o.points.map((p) => Object.assign({}, p, { x: p.x * K, y: p.y * K }));
  return o;
}
const scaleRuns = (t) => Array.isArray(t) ? t.map((r) => Object.assign({}, r, { options: r.options && r.options.fontSize
  ? Object.assign({}, r.options, { fontSize: Math.round(r.options.fontSize * K * 2) / 2 }) : r.options })) : t;
{
  const aS = s.addShape.bind(s), aT = s.addText.bind(s), aI = s.addImage.bind(s);
  s.addShape = (t, o) => aS(t, scaleOpts(o));
  s.addText = (t, o) => aT(o && o.placeholder ? t : scaleRuns(t), scaleOpts(o));
  s.addImage = (o) => aI(scaleOpts(o));
}

let nObj = 0;
const nm = (p) => `${p}-${++nObj}`;
const txt = (t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 11, color: C.text1,
  objectName: nm('text') }, o));
// line from (x1,y1) to (x2,y2) with correct flips so arrow heads point the right way
function line(x1, y1, x2, y2, o = {}) {
  s.addShape(S.LINE, Object.assign({
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.0001),
    h: Math.max(Math.abs(y2 - y1), 0.0001), flipH: x2 < x1, flipV: y2 < y1, objectName: nm('line'),
  }, o, { line: Object.assign({ color: DARK, width: 1 }, o.line || {}) }));
}
const panelTitle = (x, y, w, letter, t) => txt([
  { text: letter + '   ', options: { bold: true } }, { text: t, options: { bold: true } }],
  { x, y, w, h: 0.5, fontSize: 13, valign: 'top' });

// ================================================================ (a) pipeline
txt([{ text: 'a   Pipeline', options: { bold: true } }], { x: 0.5, y: 1.0, w: 3, h: 0.3, fontSize: 13 });
const boxes = [
  ['IQ burst + multipath', 'r = h ∗ x + n', C.background2],
  ['Kaiser STFT', 'windows share the same H(f)', C.background2],
  ['Log-polar map', 're^iθ → (ln r, θ): channel = shift', C.accent4],
  ['Equivariant layer', 'weighted Fréchet mean over windows', C.accent4],
  ['Invariant layer', 'distance of each window to the mean', C.accent5],
  ['Real 1D CNN', 'device ID (65 classes)', C.background2],
];
const bw = 1.8, gap = 0.306, by = 1.4, bh = 1.0;
boxes.forEach(([t, sub, fill], i) => {
  const x = 0.5 + i * (bw + gap);
  s.addText([
    { text: t, options: { bold: true, fontSize: 13, breakLine: true } },
    { text: sub, options: { fontSize: 11, color: DARK } }],
  { shape: S.ROUNDED_RECTANGLE, rectRadius: 0.12, x, y: by, w: bw, h: bh, fill: { color: fill },
    line: { color: DARK, width: 0.75 }, align: 'center', valign: 'middle', margin: 4, paraSpaceAfter: 3,
    objectName: nm('pipeline-box') });
  if (i < boxes.length - 1) line(x + bw + 0.04, by + bh / 2, x + bw + gap - 0.04, by + bh / 2,
    { line: { color: DARK, width: 1.25, endArrowType: 'triangle' } });
});
const bx0 = 0.5 + 2 * (bw + gap), bx1 = 0.5 + 4 * (bw + gap) + bw;
line(bx0, 2.52, bx1, 2.52, { line: { color: BLUE, width: 1.5 } });
txt('operations on the manifold ℝ⁺ × U(1), independently per frequency bin',
  { x: bx0, y: 2.57, w: bx1 - bx0, h: 0.28, align: 'center', color: BLUE, fontSize: 11 });

// shared plot geometry for the bottom row
const PT = 3.05, AY0 = 3.7, AY1 = 6.45;   // panel-title y, axes top/bottom

// ================================================================ (b) spectrogram + |H(f)|
panelTitle(0.5, PT, 4.0, 'b', 'A multipath channel multiplies every window by the same H(f)');
const sx = 1.1, sw = 2.35, hx = 3.55, hw = 0.85;
s.addImage({ path: 'slide_spectrogram.png', x: sx, y: AY0, w: sw, h: AY1 - AY0,
  altText: 'Spectrogram of a WiFi burst after a multipath channel', objectName: nm('spectrogram') });
const fy = (f) => AY0 + (10 - f) / 20 * (AY1 - AY0);         // MHz -> slide y
[-10, -5, 0, 5, 10].forEach((f) => {
  txt(String(f), { x: sx - 0.45, y: fy(f) - 0.1, w: 0.38, h: 0.2, align: 'right', fontSize: 10, color: DARK });
  line(sx - 0.05, fy(f), sx, fy(f), { line: { color: DARK, width: 0.75 } });
});
txt('Frequency (MHz)', { y: (AY0 + AY1) / 2 - 0.15, w: 1.4, h: 0.3, rotate: 270, align: 'center',
  fontSize: 11, color: DARK, x: sx - 1.05 });
txt('STFT window →', { x: sx, y: AY1 + 0.05, w: sw, h: 0.25, align: 'center', fontSize: 11, color: DARK });
line(sx, fy(D.f0), sx + sw, fy(D.f0), { line: { color: ORANGE, width: 2 } });
txt('bin f₀', { x: sx + 0.05, y: fy(D.f0) - 0.27, w: 0.5, h: 0.22, bold: true, color: ORANGE, fontSize: 10,
  fill: { color: C.background1 } });
// |H(f)| curve as a freeform
const hdbMin = -25, hdbMax = 5, hdx = (v) => (v - hdbMin) / (hdbMax - hdbMin) * hw;
const pts = D.fk.map((f, i) => ({ x: hdx(D.HdB[i]), y: fy(f) - AY0 }))
  .filter((p) => p.y >= 0 && p.y <= AY1 - AY0);
s.addShape(S.CUSTOM_GEOMETRY, { x: hx, y: AY0, w: hw, h: AY1 - AY0, points: pts,
  line: { color: DARK, width: 1.25 }, objectName: nm('H-curve') });
line(hx, AY1, hx + hw, AY1, { line: { color: DARK, width: 0.75 } });
[-20, 0].forEach((v) => txt(String(v), { x: hx + hdx(v) - 0.2, y: AY1 + 0.03, w: 0.4, h: 0.2, align: 'center',
  fontSize: 10, color: DARK }));
s.addShape(S.OVAL, { x: hx + hdx(D.HdB0) - 0.06, y: fy(D.f0) - 0.06, w: 0.12, h: 0.12,
  fill: { color: ORANGE }, line: { color: ORANGE }, objectName: nm('f0-marker') });
txt('|H(f)| (dB)', { x: hx - 0.1, y: AY1 + 0.25, w: hw + 0.2, h: 0.22, align: 'center', fontSize: 11, color: DARK });

// ================================================================ (c) tangent-space translation
panelTitle(4.85, PT, 3.8, 'c', 'In log-polar coordinates the channel is a shift: means move with it, distances do not');
const cx0 = 5.4, cx1 = 8.6;
const all = D.p0.concat(D.p1);
const thMin = Math.min(...all.map((p) => p[0])) - 0.35, thMax = Math.max(...D.p0.map((p) => p[0])) + 2.0;
const lrMin = Math.min(...all.map((p) => p[1])) - 0.25, lrMax = Math.max(...all.map((p) => p[1])) + 0.25;
const X = (t) => cx0 + (t - thMin) / (thMax - thMin) * (cx1 - cx0);
const Y = (l) => AY1 - (l - lrMin) / (lrMax - lrMin) * (AY1 - AY0);
line(cx0, AY1, cx1, AY1, { line: { color: DARK, width: 0.75 } });
line(cx0, AY0, cx0, AY1, { line: { color: DARK, width: 0.75 } });
[-2, -1, 0, 1].filter((t) => t >= thMin && t <= thMax).forEach((t) =>
  txt(String(t), { x: X(t) - 0.2, y: AY1 + 0.03, w: 0.4, h: 0.2, align: 'center', fontSize: 10, color: DARK }));
[0, 0.5, 1, 1.5, 2].filter((l) => l >= lrMin && l <= lrMax).forEach((l) =>
  txt(String(l), { x: cx0 - 0.42, y: Y(l) - 0.1, w: 0.36, h: 0.2, align: 'right', fontSize: 10, color: DARK }));
txt('phase θ (rad)', { x: cx0, y: AY1 + 0.25, w: cx1 - cx0, h: 0.22, align: 'center', fontSize: 11, color: DARK });
txt('log-magnitude ln r', { x: cx0 - 1.25, y: (AY0 + AY1) / 2 - 0.15, w: 1.7, h: 0.3, rotate: 270,
  align: 'center', fontSize: 11, color: DARK });
const mean = (P) => [P.reduce((a, p) => a + p[0], 0) / P.length, P.reduce((a, p) => a + p[1], 0) / P.length];
const m0 = mean(D.p0), m1 = mean(D.p1);
[[D.p0, m0, BLUE], [D.p1, m1, ORANGE]].forEach(([P, m, col]) => {
  P.forEach((p) => line(X(p[0]), Y(p[1]), X(m[0]), Y(m[1]), { line: { color: col, width: 1, dashType: 'sysDot' } }));
  P.forEach((p) => s.addShape(S.OVAL, { x: X(p[0]) - 0.065, y: Y(p[1]) - 0.065, w: 0.13, h: 0.13,
    fill: { color: col }, line: { color: col }, objectName: nm('window-point') }));
  s.addShape(S.STAR_5_POINT, { x: X(m[0]) - 0.15, y: Y(m[1]) - 0.15, w: 0.3, h: 0.3, fill: { color: col },
    line: { color: C.text1, width: 0.75 }, objectName: nm('frechet-mean') });
});
// arrow between means, shortened so it clears the stars
const ax0 = X(m0[0]), ay0 = Y(m0[1]), ax1 = X(m1[0]), ay1 = Y(m1[1]);
const L = Math.hypot(ax1 - ax0, ay1 - ay0), k = 0.2 / L;
line(ax0 + (ax1 - ax0) * k, ay0 + (ay1 - ay0) * k, ax1 - (ax1 - ax0) * k, ay1 - (ay1 - ay0) * k,
  { line: { color: DARK, width: 2.25, endArrowType: 'triangle' } });
txt('channel: shift by\n(∠H(f₀), ln|H(f₀)|)', { x: (ax0 + ax1) / 2 + 0.3, y: (ay0 + ay1) / 2 - 0.05, w: 1.5,
  h: 0.44, fontSize: 10, color: DARK });
// legend (shapes + labels)
const lgx = X(thMax) - 1.9, lgy = AY1 - 0.95;
const lg = [['oval', BLUE, 'clean windows'], ['oval', ORANGE, 'after channel'],
  ['star', C.background1, 'Fréchet mean (equivariant)'], ['dot', DARK, 'distance to mean (invariant)']];
lg.forEach(([kind, col, label], i) => {
  const yy = lgy + i * 0.22;
  if (kind === 'oval') s.addShape(S.OVAL, { x: lgx + 0.04, y: yy + 0.05, w: 0.11, h: 0.11, fill: { color: col },
    line: { color: col }, objectName: nm('legend') });
  if (kind === 'star') s.addShape(S.STAR_5_POINT, { x: lgx + 0.01, y: yy + 0.02, w: 0.17, h: 0.17,
    fill: { color: col }, line: { color: C.text1, width: 0.75 }, objectName: nm('legend') });
  if (kind === 'dot') line(lgx, yy + 0.11, lgx + 0.19, yy + 0.11, { line: { color: col, width: 1.25, dashType: 'sysDot' } });
  txt(label, { x: lgx + 0.26, y: yy, w: 1.65, h: 0.21, fontSize: 10, color: C.text1, valign: 'middle' });
});

// ================================================================ (d) result bars
panelTitle(9.05, PT, 3.8, 'd', 'Accuracy on 6 unseen multipath channel models (65 devices, mean)');
const dx0 = 9.65, dx1 = 12.83, YMAX = 120;
const V = (v) => AY1 - v / YMAX * (AY1 - AY0);
line(dx0, AY1, dx1, AY1, { line: { color: DARK, width: 0.75 } });
line(dx0, V(100), dx0, AY1, { line: { color: DARK, width: 0.75 } });
[0, 20, 40, 60, 80, 100].forEach((v) => {
  txt(String(v), { x: dx0 - 0.42, y: V(v) - 0.1, w: 0.36, h: 0.2, align: 'right', fontSize: 10, color: DARK });
  if (v) line(dx0, V(v), dx1, V(v), { line: { color: C.background2, width: 0.5 } });
});
txt('Top-1 accuracy (%)', { x: dx0 - 1.25, y: (V(100) + AY1) / 2 - 0.15, w: 1.7, h: 0.3, rotate: 270,
  align: 'center', fontSize: 11, color: DARK });
const barW = 0.62, xs = [dx0 + 0.25, dx0 + 0.95, dx0 + 1.85, dx0 + 2.55];
const cols = [GREY, BLUE, GREY, BLUE];
xs.forEach((x, i) => {
  const v = D.ours[i];
  s.addShape(S.RECTANGLE, { x, y: V(v), w: barW, h: AY1 - V(v), fill: { color: cols[i] },
    line: { color: C.text1, width: 0.5 }, objectName: nm('bar') });
  txt(`${Math.round(v)}%`, { x: x - 0.1, y: V(v) - 0.25, w: barW + 0.2, h: 0.22, align: 'center', bold: true, fontSize: 11 });
  s.addShape(S.RECTANGLE, { x: x + 0.08, y: V(D.pristine[i]) - 0.025, w: barW - 0.16, h: 0.05,
    fill: { color: ORANGE }, line: { color: ORANGE, width: 0 }, objectName: nm('pristine-tick') });
  s.addShape(S.DIAMOND, { x: x + barW - 0.03, y: V(D.paper[i]) - 0.08, w: 0.16, h: 0.16,
    fill: { color: C.background1 }, line: { color: C.text1, width: 1 }, objectName: nm('paper-marker') });
  txt(['Baseline', 'ChaRRNet'][i % 2], { x: x - 0.15, y: AY1 + 0.03, w: barW + 0.3, h: 0.22, align: 'center',
    fontSize: 10, color: C.text1 });
});
txt('trained without multipath', { x: xs[0] - 0.1, y: AY1 + 0.25, w: 1.5, h: 0.22, align: 'center', fontSize: 10, color: DARK });
txt('trained with nLOS-200', { x: xs[2] - 0.1, y: AY1 + 0.25, w: 1.5, h: 0.22, align: 'center', fontSize: 10, color: DARK });
// callouts
line(xs[0] + barW / 2, V(D.ours[0]) - 0.35, xs[1] + barW / 2 - 0.05, V(D.ours[1]) - 0.32,
  { line: { color: BLUE, width: 1.5, endArrowType: 'triangle' } });
txt(`×${(D.ours[1] / D.ours[0]).toFixed(1)}`, { x: xs[0] - 0.05, y: V(D.ours[1]) - 0.6, w: 0.6, h: 0.25,
  bold: true, color: BLUE, fontSize: 13 });
txt(`+${Math.round(D.ours[3] - D.ours[2])} pts`, { x: xs[2] + 0.2, y: V(104) - 0.1, w: 1.0, h: 0.25,
  align: 'center', bold: true, color: BLUE, fontSize: 13 });
// legend for d
const dlx = dx0 + 0.1, dly = AY0 - 0.12;
s.addShape(S.DIAMOND, { x: dlx + 0.02, y: dly + 0.03, w: 0.15, h: 0.15, fill: { color: C.background1 },
  line: { color: C.text1, width: 1 }, objectName: nm('legend') });
txt('paper (Table I)', { x: dlx + 0.26, y: dly, w: 1.2, h: 0.21, fontSize: 10, valign: 'middle' });
s.addShape(S.RECTANGLE, { x: dlx + 1.38, y: dly + 0.08, w: 0.2, h: 0.05, fill: { color: ORANGE },
  line: { color: ORANGE, width: 0 }, objectName: nm('legend') });
txt('pristine test set', { x: dlx + 1.64, y: dly, w: 1.3, h: 0.21, fontSize: 10, valign: 'middle' });

// footer
txt('NumPy reproduction of Brown, Mattei & Draganov, arXiv:2105.03568 (simulated 802.11g, independent I/Q IIR '
  + 'fingerprints, reflector-geometry channels). Panel c shows the ideal per-window channel action.', { x: 0.5, y: 7.0, w: 12.33, h: 0.3, fontSize: 10, color: MUTED });

s.addNotes('Panel c: in (theta, ln r) coordinates the channel H(f0) adds a constant vector to every STFT window in '
  + 'that bin, so the weighted Frechet mean shifts by the same vector (equivariance) while each window-to-mean '
  + 'distance is unchanged (invariance). With a physical channel, window boundary effects make this approximate; '
  + 'the Kaiser window roughly halves the error.');

(async () => {
  await pres.writeFile({ fileName: 'charrnet_overview.pptx' });
  await applyTheme('charrnet_overview.pptx', THEME);
  console.log('written');
})();
