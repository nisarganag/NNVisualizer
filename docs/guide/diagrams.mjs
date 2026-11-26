/**
 * Diagrams for the explanation guide, drawn as inline SVG in the site's palette:
 * violet for positive, amber for negative, ink for everything neutral.
 * Where a picture shows behaviour (learning rates, optimizers, sampling), it is
 * computed from the real formulas rather than sketched.
 */

export const C = {
  violet: '#744ed7',
  violetSoft: '#e9e3fa',
  amber: '#cf6d01',
  amberSoft: '#f8e6d4',
  ink: '#1c1e2d',
  ink2: '#4f5161',
  ink3: '#6c6e7b',
  line: 'rgba(28,30,45,.12)',
  lineStrong: 'rgba(28,30,45,.24)',
  well: '#eff0f6',
  paper: '#f3f4f9',
  surface: '#ffffff',
};

const esc = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmt = value => Number(value.toFixed(2));

/** A polyline path through points. */
const path = points => points.map(([x, y], i) => `${i ? 'L' : 'M'}${fmt(x)} ${fmt(y)}`).join(' ');

function arrowDefs(id, color) {
  return `<marker id="${id}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="${color}"/></marker>`;
}

/* ---------- Activation functions ---------- */

const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

/** One small plot of y = f(x) with faint axes. */
function functionPlot(f, { xMin = -3, xMax = 3, yMin, yMax, width = 230, height = 128 }) {
  const pad = 8;
  const sx = x => pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
  const sy = y => height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);
  const points = [];
  for (let i = 0; i <= 160; i++) {
    const x = xMin + (i / 160) * (xMax - xMin);
    points.push([sx(x), sy(Math.max(yMin, Math.min(yMax, f(x))))]);
  }
  const grid = [];
  for (let x = Math.ceil(xMin); x <= xMax; x++) if (x) grid.push(`<line x1="${sx(x)}" x2="${sx(x)}" y1="${pad}" y2="${height - pad}" stroke="${C.line}"/>`);
  for (let y = Math.ceil(yMin); y <= yMax; y++) if (y) grid.push(`<line y1="${sy(y)}" y2="${sy(y)}" x1="${pad}" x2="${width - pad}" stroke="${C.line}"/>`);
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img">
    <rect x="${pad}" y="${pad}" width="${width - 2 * pad}" height="${height - 2 * pad}" rx="6" fill="${C.paper}"/>
    ${grid.join('')}
    <line x1="${sx(0)}" x2="${sx(0)}" y1="${pad}" y2="${height - pad}" stroke="${C.lineStrong}"/>
    <line y1="${sy(0)}" y2="${sy(0)}" x1="${pad}" x2="${width - pad}" stroke="${C.lineStrong}"/>
    <path d="${path(points)}" fill="none" stroke="${C.violet}" stroke-width="2.6" stroke-linejoin="round"/>
  </svg>`;
}

export const ACTIVATIONS = [
  { name: 'Tanh', formula: 'tanh(x)', f: Math.tanh, y: [-1.4, 1.4], text: 'Smooth, from −1 to 1 and centred on zero. The classic choice, and the playground’s default.' },
  { name: 'ReLU', formula: 'max(0, x)', f: x => Math.max(0, x), y: [-0.8, 3], text: 'Zero for negative sums, a straight line for positive ones. Fast and simple, but a neuron whose sum is always negative stops learning (“dies”).' },
  { name: 'Leaky ReLU', formula: 'x if x > 0, otherwise 0.01x', f: x => (x > 0 ? x : 0.1 * x), y: [-0.8, 3], text: 'Like ReLU, with a small slope for negatives so no neuron dies. (The slope is drawn ten times steeper here so you can see it.)' },
  { name: 'GELU', formula: 'x · Φ(x)', f: gelu, y: [-0.8, 3], text: 'A smooth ReLU that lets small negative values through. It is the activation inside GPT-2 on the Transformer page.' },
  { name: 'Sigmoid', formula: '1 / (1 + e^−x)', f: x => 1 / (1 + Math.exp(-x)), y: [-0.4, 1.4], text: 'From 0 to 1. It flattens out at both ends, so gradients get small and deep networks learn slowly.' },
  { name: 'Linear', formula: 'x', f: x => x, y: [-3, 3], text: 'No bend at all. However many layers you stack, the network can only draw straight boundaries.' },
];

export function activationGrid() {
  return `<div class="plot-grid">${ACTIVATIONS.map(a => `<figure class="plot-card">
      ${functionPlot(a.f, { yMin: a.y[0], yMax: a.y[1] })}
      <figcaption><strong>${esc(a.name)}</strong> <code>${esc(a.formula)}</code><span>${esc(a.text)}</span></figcaption>
    </figure>`).join('')}</div>`;
}

/* ---------- Learning rate ---------- */

/** Gradient descent on loss = w², from the same start, with three step sizes. */
export function learningRateSvg() {
  const panels = [
    { title: 'Too small', factor: 0.9, start: 2.3, steps: 9, note: 'Tiny steps: the loss falls very slowly.' },
    { title: 'About right', factor: 0.42, start: 2.3, steps: 5, note: 'Big steps that settle at the bottom.' },
    { title: 'Too large', factor: -1.16, start: 1.35, steps: 5, note: 'Each step overshoots further: training diverges.' },
  ];
  const w = 300;
  const h = 210;
  const gap = 26;
  const body = panels.map((panel, p) => {
    const ox = p * (w + gap);
    const sx = v => ox + 14 + ((v + 3) / 6) * (w - 28);
    const sy = v => 26 + (1 - v / 9) * (h - 70);
    const curve = [];
    for (let i = 0; i <= 120; i++) { const v = -3 + (i / 120) * 6; curve.push([sx(v), sy(v * v)]); }
    const points = [];
    let v = panel.start;
    for (let i = 0; i <= panel.steps; i++) { points.push(v); v *= panel.factor; }
    const marks = points.map((value, i) => `<circle cx="${fmt(sx(value))}" cy="${fmt(sy(value * value))}" r="${i === 0 ? 5 : 4}" fill="${i === 0 ? C.surface : C.violet}" stroke="${C.violet}" stroke-width="2"/>`).join('');
    const hops = points.slice(1).map((value, i) => `<line x1="${fmt(sx(points[i]))}" y1="${fmt(sy(points[i] ** 2))}" x2="${fmt(sx(value))}" y2="${fmt(sy(value * value))}" stroke="${C.violet}" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#lr-arrow)"/>`).join('');
    return `<g>
      <rect x="${ox}" y="0" width="${w}" height="${h}" rx="12" fill="${C.paper}"/>
      <text x="${ox + 16}" y="22" class="svg-title">${panel.title}</text>
      <path d="${path(curve)}" fill="none" stroke="${C.ink3}" stroke-width="1.6"/>
      ${hops}${marks}
      <text x="${ox + 16}" y="${h - 14}" class="svg-note">${panel.note}</text>
    </g>`;
  }).join('');
  return `<svg viewBox="0 0 ${3 * w + 2 * gap} ${h}" width="100%" role="img"><defs>${arrowDefs('lr-arrow', C.violet)}</defs>${body}</svg>`;
}

/* ---------- Optimizers ---------- */

/** SGD, Momentum and Adam on the same narrow valley, f = ½(x² + 25y²), from the same start. */
export function optimizerSvg() {
  const grad = ([x, y]) => [x, 25 * y];
  const start = [-4.6, 1.05];
  const run = step => {
    const points = [start.slice()];
    // Undefined, not null, so each step's default state applies on the first call.
    let state;
    let p = start.slice();
    for (let t = 1; t <= 60; t++) {
      [p, state] = step(p, grad(p), state, t);
      points.push(p.slice());
    }
    return points;
  };
  const sgd = run((p, g) => [[p[0] - 0.074 * g[0], p[1] - 0.074 * g[1]], null]);
  const momentum = run((p, g, v = [0, 0]) => {
    const next = [0.9 * v[0] + g[0], 0.9 * v[1] + g[1]];
    return [[p[0] - 0.012 * next[0], p[1] - 0.012 * next[1]], next];
  });
  const adam = run((p, g, s = { m: [0, 0], v: [0, 0] }, t) => {
    const m = [0.9 * s.m[0] + 0.1 * g[0], 0.9 * s.m[1] + 0.1 * g[1]];
    const v = [0.999 * s.v[0] + 0.001 * g[0] ** 2, 0.999 * s.v[1] + 0.001 * g[1] ** 2];
    const step = k => (0.32 * (m[k] / (1 - 0.9 ** t))) / (Math.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8);
    return [[p[0] - step(0), p[1] - step(1)], { m, v }];
  });
  const width = 760;
  const height = 300;
  const sx = x => 30 + ((x + 5.2) / 6.4) * (width - 60);
  const sy = y => height / 2 - (y / 1.45) * (height / 2 - 24);
  const contours = [0.5, 2, 5, 9, 14, 20, 27].map(c => {
    const rx = Math.sqrt(2 * c);
    const ry = Math.sqrt((2 * c) / 25);
    return `<ellipse cx="${sx(0)}" cy="${sy(0)}" rx="${fmt(sx(rx) - sx(0))}" ry="${fmt(sy(0) - sy(ry))}" fill="none" stroke="${C.lineStrong}"/>`;
  }).join('');
  const line = (points, color, dash, widthPx) => `<path d="${path(points.map(([x, y]) => [sx(x), sy(y)]))}" fill="none" stroke="${color}" stroke-width="${widthPx}" ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linejoin="round"/>`;
  const label = (text, x, y, color) => `<text x="${x}" y="${y}" class="svg-label" fill="${color}">${text}</text>`;
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img">
    <rect width="${width}" height="${height}" rx="14" fill="${C.paper}"/>
    ${contours}
    <circle cx="${sx(0)}" cy="${sy(0)}" r="5" fill="${C.ink}"/>
    ${label('lowest loss', sx(0) - 34, sy(0) + 34, C.ink)}
    ${line(sgd, C.ink3, '5 4', 1.8)}
    ${line(momentum, C.ink, '', 2.2)}
    ${line(adam, C.violet, '', 2.6)}
    <circle cx="${sx(start[0])}" cy="${sy(start[1])}" r="5" fill="${C.surface}" stroke="${C.ink}" stroke-width="2"/>
    ${label('start', sx(start[0]) - 8, sy(start[1]) - 12, C.ink)}
    ${label('SGD zigzags across the valley', 70, height - 36, C.ink3)}
    ${label('Momentum smooths the zigzag', 70, height - 18, C.ink)}
    ${label('Adam heads more directly', 330, 40, C.violet)}
  </svg>`;
}

/* ---------- Regularization ---------- */

export function regularizationSvg() {
  const width = 760;
  const height = 250;
  // Left: the two penalties.
  const pw = 250;
  const sx = w => 24 + ((w + 2) / 4) * (pw - 48);
  const sy = v => height - 34 - (v / 4) * (height - 80);
  const l1 = [];
  const l2 = [];
  for (let i = 0; i <= 100; i++) { const w = -2 + i * 0.04; l1.push([sx(w), sy(Math.abs(w))]); l2.push([sx(w), sy(w * w)]); }
  // Right: the same twelve weights after training with each choice.
  const weights = [1.6, -1.25, 0.9, -0.42, 0.22, 1.12, -0.85, 0.08, -1.5, 0.62, -0.3, 0.95];
  const rows = [
    { name: 'No regularization', values: weights },
    { name: 'L2: every weight smaller', values: weights.map(w => w * 0.55) },
    { name: 'L1: small weights become zero', values: weights.map(w => Math.sign(w) * Math.max(0, Math.abs(w) - 0.5)) },
  ];
  const rx = 300;
  const rowHeight = (height - 30) / 3;
  const bars = rows.map((row, r) => {
    const cy = 30 + r * rowHeight + rowHeight / 2;
    const items = row.values.map((value, i) => {
      const x = rx + 190 + i * 22;
      const hBar = Math.abs(value) * 22;
      const y = value >= 0 ? cy - hBar : cy;
      return value === 0
        ? `<circle cx="${x + 7}" cy="${cy}" r="2.5" fill="${C.ink3}"/>`
        : `<rect x="${x}" y="${fmt(y)}" width="14" height="${fmt(Math.max(1, hBar))}" rx="2" fill="${value >= 0 ? C.violet : C.amber}"/>`;
    }).join('');
    return `<text x="${rx}" y="${cy + 4}" class="svg-label" fill="${C.ink}">${row.name}</text>
      <line x1="${rx + 186}" x2="${rx + 190 + 12 * 22}" y1="${cy}" y2="${cy}" stroke="${C.lineStrong}"/>${items}`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img">
    <rect width="${pw}" height="${height}" rx="14" fill="${C.paper}"/>
    <text x="18" y="24" class="svg-title">Penalty for a weight w</text>
    <line x1="${sx(-2)}" x2="${sx(2)}" y1="${sy(0)}" y2="${sy(0)}" stroke="${C.lineStrong}"/>
    <line x1="${sx(0)}" x2="${sx(0)}" y1="${sy(0)}" y2="${sy(4)}" stroke="${C.lineStrong}"/>
    <path d="${path(l1)}" fill="none" stroke="${C.ink}" stroke-width="2.2"/>
    <path d="${path(l2)}" fill="none" stroke="${C.violet}" stroke-width="2.4"/>
    <text x="${sx(1.25)}" y="${sy(1.25) + 18}" class="svg-label" fill="${C.ink}">L1: |w|</text>
    <text x="${sx(1.05)}" y="${sy(3.6)}" class="svg-label" fill="${C.violet}">L2: w²</text>
    <text x="${sx(-1.9)}" y="${height - 12}" class="svg-note">−2</text><text x="${sx(1.8)}" y="${height - 12}" class="svg-note">+2</text>
    <rect x="${rx - 14}" width="${width - rx + 14}" height="${height}" rx="14" fill="${C.paper}"/>
    <text x="${rx}" y="24" class="svg-title">The same 12 weights after training (violet positive, amber negative)</text>
    ${bars}
  </svg>`;
}

/* ---------- CNN pipeline ---------- */

export function cnnPipelineSvg() {
  const stages = [
    { label: 'Input', size: '28 × 28', maps: 1, side: 56 },
    { label: 'Convolution 1', size: '8 maps, 28 × 28', maps: 8, side: 56, note: '3 × 3 filters, ReLU' },
    { label: 'Max pooling', size: '8 maps, 14 × 14', maps: 8, side: 36 },
    { label: 'Convolution 2', size: '16 maps, 14 × 14', maps: 16, side: 36, note: '3 × 3 filters, ReLU' },
    { label: 'Max pooling', size: '16 maps, 7 × 7', maps: 16, side: 22, note: '= 784 values' },
    { label: 'Dense', size: '64 neurons', dense: 64 },
    { label: 'Output', size: '10 probabilities', dense: 10, output: true },
  ];
  const width = 1000;
  const height = 210;
  const columnWidth = width / stages.length;
  const body = stages.map((stage, i) => {
    const cx = i * columnWidth + columnWidth / 2;
    const cy = 92;
    let shape = '';
    if (stage.maps) {
      const shown = Math.min(stage.maps, 6);
      for (let m = shown - 1; m >= 0; m--) {
        const off = m * 5;
        shape += `<rect x="${fmt(cx - stage.side / 2 + off - 12)}" y="${fmt(cy - stage.side / 2 - off + 12)}" width="${stage.side}" height="${stage.side}" rx="4" fill="${m === 0 ? C.violetSoft : C.surface}" stroke="${C.violet}" stroke-width="1.2"/>`;
      }
    } else {
      const n = stage.output ? 10 : 12;
      for (let k = 0; k < n; k++) {
        const y = cy - (n * 9) / 2 + k * 9;
        shape += `<circle cx="${cx}" cy="${fmt(y + 4)}" r="3.4" fill="${stage.output && k === 3 ? C.violet : C.surface}" stroke="${C.violet}" stroke-width="1.2"/>`;
      }
    }
    const arrow = i < stages.length - 1 ? `<line x1="${fmt(cx + columnWidth * 0.36)}" x2="${fmt(cx + columnWidth * 0.62)}" y1="${cy}" y2="${cy}" stroke="${C.ink3}" stroke-width="1.6" marker-end="url(#cnn-arrow)"/>` : '';
    return `${shape}${arrow}
      <text x="${cx}" y="${height - 46}" text-anchor="middle" class="svg-label" fill="${C.ink}">${stage.label}</text>
      <text x="${cx}" y="${height - 28}" text-anchor="middle" class="svg-note">${stage.size}</text>
      ${stage.note ? `<text x="${cx}" y="${height - 11}" text-anchor="middle" class="svg-note">${stage.note}</text>` : ''}`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img"><defs>${arrowDefs('cnn-arrow', C.ink3)}</defs>
    <rect width="${width}" height="${height}" rx="14" fill="${C.paper}"/>${body}</svg>`;
}

/* ---------- LSTM cell ---------- */

export function lstmCellSvg() {
  const width = 820;
  const height = 300;
  const box = (x, y, label, sub, fill = C.surface) => `<g><rect x="${x - 34}" y="${y - 24}" width="68" height="48" rx="10" fill="${fill}" stroke="${C.violet}" stroke-width="1.5"/>
    <text x="${x}" y="${y - 2}" text-anchor="middle" class="svg-title">${label}</text><text x="${x}" y="${y + 15}" text-anchor="middle" class="svg-note">${sub}</text></g>`;
  const op = (x, y, symbol) => `<circle cx="${x}" cy="${y}" r="14" fill="${C.surface}" stroke="${C.ink}" stroke-width="1.6"/><text x="${x}" y="${y + 5}" text-anchor="middle" class="svg-title">${symbol}</text>`;
  const wire = (d, color = C.ink3, end = true) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.6" ${end ? 'marker-end="url(#lstm-arrow)"' : ''}/>`;
  const memoryY = 64;
  const gatesY = 186;
  const gates = [{ x: 220, label: 'f', sub: 'forget' }, { x: 340, label: 'i', sub: 'input' }, { x: 440, label: 'g', sub: 'candidate' }, { x: 600, label: 'o', sub: 'output' }];
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img"><defs>${arrowDefs('lstm-arrow', C.ink3)}${arrowDefs('lstm-memory', C.violet)}</defs>
    <rect width="${width}" height="${height}" rx="14" fill="${C.paper}"/>
    <text x="24" y="${memoryY - 18}" class="svg-label" fill="${C.violet}">memory (cell state c)</text>
    <path d="M24 ${memoryY} H206" stroke="${C.violet}" stroke-width="3" fill="none"/>
    <path d="M234 ${memoryY} H376" stroke="${C.violet}" stroke-width="3" fill="none"/>
    <path d="M404 ${memoryY} H780" stroke="${C.violet}" stroke-width="3" fill="none" marker-end="url(#lstm-memory)"/>
    <text x="24" y="${memoryY + 22}" class="svg-note">c, old</text><text x="740" y="${memoryY + 22}" class="svg-note">c, new</text>
    ${op(220, memoryY, '×')}${op(390, memoryY, '+')}${op(390, 126, '×')}
    ${wire(`M220 ${gatesY - 24} V${memoryY + 14}`)}
    ${wire(`M340 ${gatesY - 24} V140 H376`)}
    ${wire(`M440 ${gatesY - 24} V140 H404`)}
    ${wire(`M390 112 V${memoryY + 14}`)}
    ${wire(`M560 ${memoryY} V96`, C.violet)}
    <rect x="532" y="96" width="56" height="34" rx="9" fill="${C.surface}" stroke="${C.ink3}"/><text x="560" y="118" text-anchor="middle" class="svg-label" fill="${C.ink}">tanh</text>
    ${op(660, 113, '×')}
    ${wire(`M588 113 H646`)}
    ${wire(`M600 ${gatesY - 24} V150 H660 V127`)}
    ${wire(`M674 113 H780`)}
    <text x="712" y="104" class="svg-label" fill="${C.ink}">output h</text>
    ${gates.map(g => box(g.x, gatesY, g.label, g.sub, g.label === 'g' ? C.violetSoft : C.surface)).join('')}
    <path d="M60 266 H600" stroke="${C.ink3}" stroke-width="1.6" fill="none"/>
    ${gates.map(g => wire(`M${g.x} 266 V${gatesY + 24}`)).join('')}
    <text x="24" y="258" class="svg-label" fill="${C.ink}">new character + previous output h</text>
  </svg>`;
}

/* ---------- Attention ---------- */

export function attentionSvg() {
  const tokens = ['When', 'Mary', 'and', 'John', 'went', '…', 'drink', 'to'];
  const weights = [0.03, 0.53, 0.02, 0.17, 0.02, 0.06, 0.09, 0.08];
  const width = 900;
  const height = 260;
  const step = (width - 160) / tokens.length;
  const cells = tokens.map((token, i) => {
    const x = 140 + i * step;
    const barHeight = weights[i] * 120;
    return `<rect x="${x + 6}" y="26" width="${step - 12}" height="34" rx="8" fill="${i === tokens.length - 1 ? C.ink : C.surface}" stroke="${C.lineStrong}"/>
      <text x="${x + step / 2}" y="48" text-anchor="middle" class="svg-label" fill="${i === tokens.length - 1 ? C.surface : C.ink}">${token}</text>
      <rect x="${x + step / 2 - 14}" y="${fmt(196 - barHeight)}" width="28" height="${fmt(Math.max(2, barHeight))}" rx="4" fill="${C.violet}" opacity="${0.35 + weights[i]}"/>
      <text x="${x + step / 2}" y="214" text-anchor="middle" class="svg-note">${Math.round(weights[i] * 100)}%</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img">
    <rect width="${width}" height="${height}" rx="14" fill="${C.paper}"/>
    <text x="20" y="48" class="svg-label" fill="${C.ink}">Tokens</text>
    <text x="20" y="104" class="svg-label" fill="${C.ink}">Query · key</text>
    <text x="20" y="122" class="svg-note">“to” compares itself</text><text x="20" y="138" class="svg-note">with every token</text>
    <text x="20" y="180" class="svg-label" fill="${C.ink}">Softmax</text>
    <text x="20" y="198" class="svg-note">attention shares</text>
    ${tokens.map((_t, i) => `<line x1="${140 + i * step + step / 2}" x2="${140 + (tokens.length - 1) * step + step / 2}" y1="62" y2="62" stroke="none"/>`).join('')}
    ${tokens.slice(0, -1).map((_t, i) => `<path d="M${fmt(140 + (tokens.length - 1) * step + step / 2)} 64 C ${fmt(140 + (tokens.length - 1) * step + step / 2)} 112, ${fmt(140 + i * step + step / 2)} 82, ${fmt(140 + i * step + step / 2)} 64" fill="none" stroke="${C.violet}" stroke-width="${fmt(0.6 + weights[i] * 7)}" opacity="${0.25 + weights[i]}"/>`).join('')}
    ${cells}
    <text x="20" y="244" class="svg-label" fill="${C.ink}">Output for “to” = 53% of Mary’s value + 17% of John’s + …</text>
    <text x="${width - 20}" y="244" text-anchor="end" class="svg-note">illustrative numbers</text>
  </svg>`;
}

/* ---------- Autoencoder ---------- */

export function autoencoderDiagram(inputSrc, outputSrc) {
  const trapezoid = (narrowRight, label, sub) => `<figure class="ae-net"><svg viewBox="0 0 170 150" width="170"><path d="${narrowRight ? 'M8 8 L162 54 L162 96 L8 142 Z' : 'M8 54 L162 8 L162 142 L8 96 Z'}" fill="${C.violetSoft}" stroke="${C.violet}" stroke-width="1.6"/></svg><figcaption><strong>${label}</strong><span>${sub}</span></figcaption></figure>`;
  return `<div class="ae-diagram">
    <figure class="ae-image"><img src="${inputSrc}" alt=""><figcaption><strong>Digit</strong><span>784 pixels</span></figcaption></figure>
    ${trapezoid(true, 'Encoder', '784 → 256 → 128 → 2')}
    <figure class="ae-code"><div class="ae-numbers"><span>z₁</span><span>z₂</span></div><figcaption><strong>Two numbers</strong><span>a point on the map</span></figcaption></figure>
    ${trapezoid(false, 'Decoder', '2 → 128 → 256 → 784')}
    <figure class="ae-image"><img src="${outputSrc}" alt=""><figcaption><strong>Redrawn</strong><span>784 pixels</span></figcaption></figure>
  </div>`;
}

/* ---------- Sampling ---------- */

const softmax = (logits, temperature) => {
  const scaled = logits.map(l => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(s => Math.exp(s - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / total);
};

export function samplingSvg() {
  const tokens = ['the', 'now', 'a', 'Paris', 'France', 'located'];
  const logits = [3.0, 2.35, 2.2, 1.95, 1.85, 1.0];
  const panels = [
    { title: 'Temperature 0.5: sharper', p: softmax(logits, 0.5) },
    { title: 'Temperature 1: as trained', p: softmax(logits, 1) },
    { title: 'Temperature 2: flatter', p: softmax(logits, 2) },
  ];
  const k = 3;
  const one = softmax(logits, 1);
  const cumulative = one.reduce((acc, p, i) => [...acc, (acc[i - 1] ?? 0) + p], []);
  const keptP = cumulative.findIndex(c => c >= 0.8) + 1;
  const filters = [
    { title: 'Top-k 3: the 3 likeliest', p: one, kept: i => i < k },
    { title: 'Top-p 0.8: up to 80%', p: one, kept: i => i < keptP },
  ];
  const width = 1000;
  const panelWidth = 180;
  const draw = (panel, x, kept = () => true) => {
    const rows = tokens.map((token, i) => {
      const y = 40 + i * 24;
      const barWidth = panel.p[i] * 150 * 1.6;
      return `<text x="${x}" y="${y + 12}" class="svg-note" fill="${C.ink}">${token}</text>
        <rect x="${x + 52}" y="${y + 2}" width="${fmt(Math.max(2, Math.min(panelWidth - 92, barWidth)))}" height="12" rx="4" fill="${kept(i) ? C.ink : C.lineStrong}"/>
        <text x="${x + panelWidth - 6}" y="${y + 12}" text-anchor="end" class="svg-note">${Math.round(panel.p[i] * 100)}%</text>`;
    }).join('');
    return `<text x="${x}" y="22" class="svg-label" fill="${C.ink}">${panel.title}</text>${rows}`;
  };
  return `<svg viewBox="0 0 ${width} 200" width="100%" role="img">
    <rect width="${width}" height="200" rx="14" fill="${C.paper}"/>
    ${panels.map((panel, i) => draw(panel, 20 + i * (panelWidth + 12))).join('')}
    <line x1="${20 + 3 * (panelWidth + 12) - 4}" x2="${20 + 3 * (panelWidth + 12) - 4}" y1="16" y2="184" stroke="${C.lineStrong}"/>
    ${draw(filters[0], 20 + 3 * (panelWidth + 12) + 8, filters[0].kept)}
    ${draw(filters[1], 20 + 4 * (panelWidth + 12) + 8, filters[1].kept)}
  </svg>`;
}

/* ---------- Colors ---------- */

export function colorLegend() {
  const swatch = style => `<i class="swatch" style="${style}"></i>`;
  const rows = [
    { swatch: swatch(`background: linear-gradient(90deg, ${C.amber}, #f4f4f8, ${C.violet})`), title: 'Diverging values: −1 to +1', text: 'Amber is negative, violet positive, pale near zero. Used for weights, neuron and memory heatmaps, the decision boundary, embeddings, Q, K and V.' },
    { swatch: swatch(`background: linear-gradient(90deg, #f6f4fd, ${C.violet})`), title: 'Shares: 0 to 1', text: 'From pale to violet. Attention weights, LSTM gates, feature maps, the dense layer.' },
    { swatch: swatch(`background: linear-gradient(90deg, #fbefe2, ${C.amber})`), title: 'Surprise', text: 'Stronger amber means the network found that character or token more surprising (in bits).' },
    { swatch: `<span class="swatch-pair">${swatch(`background:${C.violet}`)}${swatch(`background:${C.amber}`)}</span>`, title: 'For and against', text: 'In saliency and memory-span bars, violet pushed toward the answer and amber against it. In the backprop view, violet pushed a weight up and amber down.' },
    { swatch: `<span class="swatch-pair">${swatch(`background:${C.violet}`)}${swatch(`background: repeating-linear-gradient(135deg, ${C.amber} 0 3px, #fff 3px 6px)`)}</span>`, title: 'Gained and lost', text: 'With heads switched off, violet is probability gained and hatched amber probability lost.' },
    { swatch: swatch(`background:${C.ink}`), title: 'Ink and grey', text: 'Neutral data and chrome: probability bars, test-digit dots, axes, text. Grey means faded or filtered out.' },
    { swatch: `<svg width="46" height="14"><line x1="2" x2="44" y1="7" y2="7" stroke="${C.ink2}" stroke-width="2.4" stroke-dasharray="5 4"/></svg>`, title: 'Dashed lines', text: 'A second series beside the main one: test loss in the playground, the gateless network on the Recurrent page.' },
    { swatch: swatch(`background: repeating-linear-gradient(135deg, #d9dbe6 0 2px, #fff 2px 6px)`), title: 'Hatching', text: 'Nothing there on purpose: future tokens a transformer may not look at, a switched-off head’s output.' },
  ];
  return `<div class="legend-grid">${rows.map(row => `<div class="legend-row">${row.swatch}<div><strong>${row.title}</strong><span>${row.text}</span></div></div>`).join('')}</div>`;
}
