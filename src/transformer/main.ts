import '@fontsource-variable/mona-sans/wdth.css';
import '../../styles.css';
import './transformer.css';
import * as d3 from 'd3';
import { initializeSite, reducedMotion } from '../site';
import { signalRange, THEME_CHANGE_EVENT } from '../palette';
import { Gpt2Tokenizer } from './tokenizer';
import { applySampling, sample, type SamplingResult, type SamplingSettings } from './sampling';
import { MAX_TOKENS, MODEL_BASE, MODEL_CACHE } from './config';
import type { Backend, EngineState, GpuReason, LensRow, Trace, TransformerRequest, TransformerResponse } from './worker';
import type { Gpt2Manifest } from './gpt2';

initializeSite();

const N_LAYER = 12;
const N_HEAD = 12;
const EMBD = 768;
const HEAD = EMBD / N_HEAD;
const HIDDEN = 4 * EMBD;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const flow = $('transformer-flow');
const promptInput = $<HTMLTextAreaElement>('prompt-input');
const generateButton = $<HTMLButtonElement>('generate-button');
const loadButton = $<HTMLButtonElement>('load-model');
const statusLine = $('transformer-status');

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (message: TransformerRequest) => worker.postMessage(message);

let tokenizer: Gpt2Tokenizer | null = null;
let ready = false;
/** Tokens to show; generation appends to them without re-encoding the text. */
let tokens: number[] = [];
let trace: Trace | null = null;
let traceTokens: number[] = [];
/** The processor that computed the trace shown. */
let traceBackend: Backend = 'cpu';
let layer = 0;
let head = 0;
let focus = -1;
let sampled = -1;
let arriving = -1;
let headWeights: { layer: number; head: number; q: Float32Array; k: Float32Array; v: Float32Array } | null = null;
let requestId = 0;
let pending = false;
/** The run in flight and the latest one asked for meanwhile, each with the processor it should use. */
let inFlight: { tokens: number[]; backend: Backend } | null = null;
let queued: { tokens: number[]; backend: Backend } | null = null;
/**
 * The processor the visitor wants. `?backend=cpu` starts on the CPU; `?backend=gpu` insists on the GPU,
 * accepting a software one (such as SwiftShader), and still falls back when there is none.
 */
const backendParameter = new URLSearchParams(location.search).get('backend');
let backend: Backend = backendParameter === 'cpu' ? 'cpu' : 'gpu';
let engine: EngineState | null = null;
let lastRun: { ms: number; fresh: number } | null = null;
const settings: SamplingSettings = { temperature: .8, mode: 'top-k', k: 5, p: .9 };
let result: SamplingResult | null = null;
/** The latest logit lens, the tokens and position it reads, and the processor that computed it. */
let lens: { tokens: number[]; position: number; rows: LensRow[]; backend: Backend } | null = null;
let lensId = 0;
let lensInFlight = false;
let lensWanted = false;
let lensTimer: ReturnType<typeof setTimeout>;
let typingTimer: ReturnType<typeof setTimeout> | undefined;

function setStatus(message: string, kind: 'info' | 'error' = 'error') {
  statusLine.textContent = message;
  statusLine.dataset.kind = kind;
  statusLine.hidden = !message;
}

/* ---------- Colors ---------- */

let diverging = new Uint8ClampedArray(256 * 3);
let sequential = new Uint8ClampedArray(256 * 3);
function buildRamps() {
  const [negative, neutral, positive] = signalRange();
  const ramp = (scale: (t: number) => string) => {
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const color = d3.rgb(scale(i / 255));
      out.set([color.r, color.g, color.b], i * 3);
    }
    return out;
  };
  diverging = ramp(d3.scale.linear<string, string>().domain([0, .5, 1]).range([negative, neutral, positive]).interpolate(d3.interpolateLab));
  sequential = ramp(d3.scale.linear<string, string>().domain([0, 1]).range([neutral, positive]).interpolate(d3.interpolateLab));
}
buildRamps();

/** A scale that ignores GPT-2's few huge activations so ordinary values stay visible. */
function robustScale(values: ArrayLike<number>, start = 0, length = values.length - start): number {
  const step = Math.max(1, Math.floor(length / 4096));
  const sample: number[] = [];
  for (let i = start; i < start + length; i += step) {
    const value = Math.abs(values[i]);
    if (Number.isFinite(value)) sample.push(value);
  }
  sample.sort((a, b) => a - b);
  return sample[Math.floor(.98 * (sample.length - 1))] || 1;
}

/** Paints one value per pixel; the canvas is scaled up with CSS. NaN cells stay transparent. */
function paint(canvas: HTMLCanvasElement, rows: number, cols: number, value: (row: number, col: number) => number,
    kind: 'diverging' | 'sequential', scale: number) {
  canvas.width = Math.max(1, cols);
  canvas.height = Math.max(1, rows);
  const context = canvas.getContext('2d');
  const image = context.createImageData(canvas.width, canvas.height);
  const lut = kind === 'diverging' ? diverging : sequential;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = value(r, c);
      const p = (r * cols + c) * 4;
      if (Number.isNaN(v)) continue;
      const t = kind === 'diverging' ? (v / scale + 1) / 2 : v / scale;
      const i = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
      image.data[p] = lut[i];
      image.data[p + 1] = lut[i + 1];
      image.data[p + 2] = lut[i + 2];
      image.data[p + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

/* ---------- Layout helpers ---------- */

const rowHeight = (n: number) => Math.max(6, Math.min(20, Math.round(240 / Math.max(n, 1))));
function sizeRows(canvas: HTMLCanvasElement, n: number, square = false) {
  const h = n * rowHeight(n);
  // Square matrices shrink with narrow columns; CSS keeps them square.
  if (square) canvas.style.setProperty('--size', `${h}px`);
  else canvas.style.height = `${h}px`;
}

/** Outlines the focused token's row (and column, for square matrices). */
function markFocus(canvas: HTMLCanvasElement, n: number, square = false) {
  const figure = canvas.parentElement;
  let row = figure.querySelector<HTMLElement>(`.focus-row[data-for="${canvas.id}"]`);
  if (!row) {
    row = document.createElement('span');
    row.className = 'focus-row';
    row.dataset.for = canvas.id;
    figure.append(row);
  }
  const index = focusIndex();
  // Measure the rendered size: square matrices may be scaled to fit.
  const height = canvas.clientHeight / n;
  row.hidden = index < 0;
  row.style.top = `${canvas.offsetTop + index * height}px`;
  row.style.left = `${canvas.offsetLeft}px`;
  row.style.width = `${canvas.clientWidth}px`;
  row.style.height = `${height}px`;
}

const focusIndex = () => (trace ? (focus >= 0 && focus < trace.n ? focus : trace.n - 1) : -1);
const displayToken = (id: number) => tokenizer?.tokenText(id) ?? '';
const quoteToken = (id: number) => `“${displayToken(id).replace(/\n/g, '↵').trim() || '␣'}”`;
const percent = (p: number) => (p < .001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);

/** Tint steps for surprise: none below 2 bits, then one step every 4 bits, strongest from 14. */
const surpriseLevel = (bits: number) => (bits < 2 ? 0 : bits < 6 ? 1 : bits < 10 ? 2 : bits < 14 ? 3 : 4);
/** How unexpected token i was, given the tokens before it. */
function surpriseLabel(i: number): string {
  const bits = trace.surprisal[i];
  return Number.isNaN(bits) ? 'No context yet' : `${bits.toFixed(1)} bits of surprise, rank ${trace.ranks[i].toLocaleString()}`;
}

/* ---------- Rendering ---------- */

function renderTokenList() {
  const list = $('token-list');
  const shown = trace ? traceTokens : tokens;
  $('token-count').textContent = String(shown.length);
  const items = Array.from(list.children) as HTMLElement[];
  shown.forEach((id, i) => {
    let item = items[i];
    if (!item) {
      item = document.createElement('li');
      item.className = 'token';
      item.tabIndex = 0;
      item.innerHTML = '<span class="token-text"></span><span class="token-id readout"></span><span class="token-attention" aria-hidden="true"><i></i></span>';
      item.addEventListener('mouseenter', () => setFocus(Number(item.dataset.index)));
      item.addEventListener('focus', () => setFocus(Number(item.dataset.index)));
      list.append(item);
    }
    item.dataset.index = String(i);
    const text = displayToken(id);
    const label = item.querySelector('.token-text');
    // A leading space belongs to the token; show it as a faint marker.
    label.innerHTML = '';
    if (text.startsWith(' ')) label.append(Object.assign(document.createElement('span'), { className: 'token-space', title: 'Starts with a space' }));
    label.append(text.replace(/^ /, '').replace(/\n/g, '↵') || '·');
    item.querySelector('.token-id').textContent = String(id);
    item.classList.toggle('is-new', i === arriving && !reducedMotion.matches);
  });
  items.slice(shown.length).forEach(item => item.remove());
  renderAttentionBars();
}

function renderAttentionBars() {
  const index = focusIndex();
  $('token-list').querySelectorAll<HTMLElement>('.token').forEach(item => {
    const i = Number(item.dataset.index);
    const weight = trace && index >= 0 && i <= index ? trace.weights[((layer * N_HEAD + head) * trace.n + index) * trace.n + i] : 0;
    item.querySelector<HTMLElement>('.token-attention i').style.width = `${Math.round(weight * 100)}%`;
    item.classList.toggle('is-focus', i === index);
    item.classList.toggle('is-masked', trace !== null && i > index);
    const attention = i > index ? 'After the focused token, so it is masked' : `Gets ${(weight * 100).toFixed(1)}% of the focused token’s attention in this head`;
    item.title = trace ? `${attention}\n${surpriseLabel(i)}` : '';
  });
}

/** The prompt under the input, each token tinted by how unexpected it was. */
function renderSurprise() {
  $('surprise').hidden = !trace;
  if (!trace) return;
  $('surprise-text').replaceChildren(...traceTokens.flatMap((id, i) => {
    const text = displayToken(id).replace(/\n/g, '↵');
    // The space before a word stays untinted, so tints sit on the words.
    const space = text.length > 1 && text.startsWith(' ') ? ' ' : '';
    const span = document.createElement('span');
    span.textContent = text.slice(space.length);
    span.title = surpriseLabel(i);
    const bits = trace.surprisal[i];
    if (!Number.isNaN(bits)) span.dataset.surprise = String(surpriseLevel(bits));
    return space ? [space, span] : [span];
  }));
}

function renderEmbeddings() {
  if (!trace) return;
  const { n, embeddings } = trace;
  const ids = ['token-embedding', 'position-embedding', 'input-embedding'];
  ids.forEach((id, part) => {
    const canvas = $<HTMLCanvasElement>(id);
    const start = part * n * EMBD;
    const scale = robustScale(embeddings, start, n * EMBD);
    paint(canvas, n, EMBD, (r, c) => embeddings[start + r * EMBD + c], 'diverging', scale);
    sizeRows(canvas, n);
    markFocus(canvas, n);
  });
}

/** Mean attention across heads, for the block overview. */
function averageAttention(l: number, n: number) {
  const out = new Float32Array(n * n).fill(Number.NaN);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c <= r; c++) {
      let total = 0;
      for (let h = 0; h < N_HEAD; h++) total += trace.weights[((l * N_HEAD + h) * n + r) * n + c];
      out[r * n + c] = total / N_HEAD;
    }
  }
  return out;
}

function renderBlockStrip() {
  const strip = $('block-strip');
  if (!strip.children.length) {
    for (let l = 0; l < N_LAYER; l++) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'block-tile';
      button.setAttribute('role', 'tab');
      button.innerHTML = `<canvas aria-hidden="true"></canvas><span>${l + 1}</span>`;
      button.setAttribute('aria-label', `Block ${l + 1}`);
      button.addEventListener('click', () => selectLayer(l));
      strip.append(button);
    }
  }
  Array.from(strip.children).forEach((button, l) => {
    button.setAttribute('aria-selected', String(l === layer));
    if (trace) paint(button.querySelector('canvas'), trace.n, trace.n, (() => { const avg = averageAttention(l, trace.n); return (r: number, c: number) => avg[r * trace.n + c]; })(), 'sequential', .6);
  });
}

function renderHeadGrid() {
  const grid = $('head-grid');
  $('heads-caption').textContent = `in block ${layer + 1}`;
  if (!grid.children.length) {
    for (let h = 0; h < N_HEAD; h++) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'head-tile';
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-label', `Head ${h + 1}`);
      button.innerHTML = `<canvas aria-hidden="true"></canvas><span>${h + 1}</span>`;
      button.addEventListener('click', () => selectHead(h));
      grid.append(button);
    }
  }
  Array.from(grid.children).forEach((button, h) => {
    button.setAttribute('aria-selected', String(h === head));
    if (!trace) return;
    const n = trace.n;
    const base = (layer * N_HEAD + h) * n * n;
    paint(button.querySelector('canvas'), n, n, (r, c) => (c > r ? Number.NaN : trace.weights[base + r * n + c]), 'sequential', .8);
  });
}

function renderHeadDetail() {
  $('head-title').textContent = `Block ${layer + 1}, head ${head + 1}`;
  if (!trace) return;
  const { n } = trace;
  const slice = (array: Float32Array) => (r: number, c: number) => array[(layer * n + r) * EMBD + head * HEAD + c];
  const matrices: [string, Float32Array][] = [['query-matrix', trace.q], ['key-matrix', trace.k], ['value-matrix', trace.v], ['head-output', trace.attended]];
  for (const [id, array] of matrices) {
    const canvas = $<HTMLCanvasElement>(id);
    const values = new Float32Array(n * HEAD);
    const read = slice(array);
    for (let r = 0; r < n; r++) for (let c = 0; c < HEAD; c++) values[r * HEAD + c] = read(r, c);
    paint(canvas, n, HEAD, (r, c) => values[r * HEAD + c], 'diverging', robustScale(values));
    sizeRows(canvas, n);
    markFocus(canvas, n);
  }
  const base = (layer * N_HEAD + head) * n * n;
  const scores = trace.scores.subarray(base, base + n * n);
  const scoreCanvas = $<HTMLCanvasElement>('score-matrix');
  paint(scoreCanvas, n, n, (r, c) => scores[r * n + c], 'diverging', robustScale(scores));
  sizeRows(scoreCanvas, n, true);
  markFocus(scoreCanvas, n, true);
  const attentionCanvas = $<HTMLCanvasElement>('attention-matrix');
  paint(attentionCanvas, n, n, (r, c) => (c > r ? Number.NaN : trace.weights[base + r * n + c]), 'sequential', 1);
  sizeRows(attentionCanvas, n, true);
  markFocus(attentionCanvas, n, true);
  renderWeights();
  renderReadout();
}

function renderWeights() {
  const ids: [string, 'q' | 'k' | 'v'][] = [['query-weights', 'q'], ['key-weights', 'k'], ['value-weights', 'v']];
  const current = headWeights && headWeights.layer === layer && headWeights.head === head ? headWeights : null;
  for (const [id, part] of ids) {
    const canvas = $<HTMLCanvasElement>(id);
    canvas.classList.toggle('is-loading', !current);
    if (!current) continue;
    const values = current[part];
    paint(canvas, HEAD, EMBD, (r, c) => values[r * EMBD + c], 'diverging', robustScale(values));
  }
  if (ready && !current) send({ type: 'weights', layer, head });
}

function renderReadout(cell?: { row: number; col: number }) {
  const readout = $('attention-readout');
  if (!trace) { readout.textContent = ''; return; }
  const row = cell?.row ?? focusIndex();
  const n = trace.n;
  const base = (layer * N_HEAD + head) * n * n + row * n;
  let col = cell?.col ?? -1;
  if (col < 0) {
    // The strongest attention from the focused token.
    col = 0;
    for (let c = 1; c <= row; c++) if (trace.weights[base + c] > trace.weights[base + col]) col = c;
  }
  const quote = (i: number) => `“${displayToken(traceTokens[i]).trim() || '␣'}”`;
  readout.textContent = col > row
    ? `${quote(row)} can’t see ${quote(col)}: it comes later, so it’s masked.`
    : `${quote(row)} gives ${(trace.weights[base + col] * 100).toFixed(1)}% of its attention to ${quote(col)}.`;
}

function renderMlp() {
  if (!trace) return;
  const { n, mlp } = trace;
  const canvas = $<HTMLCanvasElement>('mlp-matrix');
  // Pool four neurons per pixel, keeping the strongest, so 3,072 neurons fit the width.
  const cols = HIDDEN / 4;
  const pooled = new Float32Array(n * cols);
  for (let r = 0; r < n; r++) {
    const base = (layer * n + r) * HIDDEN;
    for (let c = 0; c < cols; c++) {
      let best = 0;
      for (let j = 0; j < 4; j++) { const v = mlp[base + c * 4 + j]; if (Math.abs(v) > Math.abs(best)) best = v; }
      pooled[r * cols + c] = best;
    }
  }
  paint(canvas, n, cols, (r, c) => pooled[r * cols + c], 'diverging', robustScale(pooled));
  sizeRows(canvas, n);
  markFocus(canvas, n);
}

function renderCandidates() {
  const list = $('candidate-list');
  if (!trace || !result) { list.innerHTML = ''; $('sampling-summary').textContent = ''; return; }
  const shown = Array.from(result.ranking.subarray(0, 10));
  const max = result.scaled[shown[0]] || 1;
  const kept = new Set(result.kept);
  const items = Array.from(list.children) as HTMLElement[];
  shown.forEach((id, i) => {
    let item = items[i];
    if (!item) {
      item = document.createElement('li');
      item.className = 'candidate';
      item.innerHTML = '<span class="candidate-text"></span><span class="candidate-bar"><i></i></span><span class="candidate-value readout"></span>';
      list.append(item);
    }
    const text = displayToken(id);
    item.querySelector('.candidate-text').textContent = text.replace(/\n/g, '↵').trim() || '␣';
    item.querySelector<HTMLElement>('.candidate-bar i').style.width = `${(result.scaled[id] / max) * 100}%`;
    item.querySelector('.candidate-value').textContent = kept.has(id) ? `${(result.probabilities[id] * 100).toFixed(1)}%` : '—';
    item.classList.toggle('is-filtered', !kept.has(id));
    item.classList.toggle('is-sampled', id === sampled);
    item.title = kept.has(id) ? `${(result.scaled[id] * 100).toFixed(2)}% before sampling filters` : 'Outside the sampling filter';
    // The raw score, so tests can compare the GPU and CPU runs.
    item.dataset.logit = String(trace.logits[id]);
  });
  items.slice(shown.length).forEach(item => item.remove());
  const share = result.kept.reduce((total, id) => total + result.scaled[id], 0);
  $('sampling-summary').textContent = settings.mode === 'top-k'
    ? `Sampling from the ${result.kept.length} most likely tokens, ${(share * 100).toFixed(0)}% of the probability.`
    : `The ${result.kept.length.toLocaleString()} most likely token${result.kept.length === 1 ? '' : 's'} cover ${(share * 100).toFixed(0)}% of the probability.`;
  const finalCanvas = $<HTMLCanvasElement>('final-hidden');
  paint(finalCanvas, 1, EMBD, (_r, c) => trace.finalHidden[c], 'diverging', robustScale(trace.finalHidden));
}

/* ---------- How the guess forms ---------- */

/** Whether a lens reads the current tokens at the focused position. */
const lensIsCurrent = (candidate: { tokens: number[]; position: number; backend: Backend } | null) =>
  candidate !== null && trace !== null && candidate.position === focusIndex() && candidate.tokens.join() === traceTokens.join()
  && candidate.backend === traceBackend;

/**
 * One row per stage: the embedding, then each block (a button that opens it).
 * `reveal` plays the rows in top-down, so the answer visibly forms.
 */
function renderLens(reveal = false) {
  const current = lensIsCurrent(lens);
  $('lens').classList.toggle('is-stale', lens !== null && !current);
  const list = $('lens-list');
  list.setAttribute('aria-busy', String(lens !== null && !current));
  if (!lens) return;
  if (!list.children.length) {
    for (let r = 0; r <= N_LAYER; r++) {
      const item = document.createElement('li');
      const row = document.createElement(r === 0 ? 'div' : 'button');
      row.className = 'lens-row';
      row.innerHTML = `<span class="lens-stage">${r === 0 ? 'Embedding' : `Block ${r}`}</span><span class="lens-token"></span><span class="lens-bar" aria-hidden="true"><i></i></span><span class="lens-value readout"></span>`;
      if (row instanceof HTMLButtonElement) {
        row.type = 'button';
        row.addEventListener('click', () => selectLayer(r - 1));
      }
      item.append(row);
      list.append(item);
    }
  }
  const { rows, tokens: read, position } = lens;
  const final = rows[N_LAYER].ids[0];
  const max = Math.max(...rows.map(row => row.probabilities[0]));
  $('lens-context').textContent = `What could follow ${quoteToken(read[position])}, block by block, before temperature and sampling`;
  const rowElements = Array.from(list.querySelectorAll<HTMLElement>('.lens-row'));
  rowElements.forEach((row, r) => {
    const { ids, probabilities } = rows[r];
    row.querySelector('.lens-token').textContent = displayToken(ids[0]).replace(/\n/g, '↵').trim() || '␣';
    row.querySelector<HTMLElement>('.lens-bar i').style.width = `${(probabilities[0] / max) * 100}%`;
    row.querySelector('.lens-value').textContent = percent(probabilities[0]);
    row.classList.toggle('is-final', ids[0] === final);
    if (r > 0) row.setAttribute('aria-current', String(r - 1 === layer));
    const stage = r === 0 ? 'Straight from the embedding' : r === N_LAYER ? `After block ${r}, the model’s actual prediction` : `After block ${r}`;
    row.title = `${stage}: ${ids.map((id, j) => `${quoteToken(id)} ${percent(probabilities[j])}`).join(', ')}`;
  });
  if (!reveal || reducedMotion.matches) return;
  const easing = 'cubic-bezier(.22, 1, .36, 1)';
  rowElements.forEach((row, r) => {
    const bar = row.querySelector<HTMLElement>('.lens-bar i');
    for (const element of [row, bar]) element.getAnimations().forEach(animation => animation.cancel());
    row.animate([{ opacity: 0, transform: 'translateY(-3px)' }, { opacity: 1, transform: 'none' }], { duration: 260, delay: r * 30, easing, fill: 'backwards' });
    bar.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { duration: 420, delay: r * 30 + 40, easing, fill: 'backwards' });
  });
}

/** Asks for the lens once the focus and tokens settle: thirteen full-vocabulary readouts take about half a second. */
function scheduleLens() {
  renderLens();
  clearTimeout(lensTimer);
  lensTimer = setTimeout(requestLens, 150);
}

function requestLens() {
  // A run (or one about to start after typing) changes the tokens; its result schedules the lens again.
  if (!ready || !trace || pending || typingTimer !== undefined || lensIsCurrent(lens)) return;
  // One at a time, so lenses never queue up ahead of a run; ask again when it returns.
  if (lensInFlight) { lensWanted = true; return; }
  lensInFlight = true;
  lensWanted = false;
  send({ type: 'lens', id: ++lensId, position: focusIndex() });
}

function renderAll() {
  flow.classList.toggle('is-empty', !trace);
  renderTokenList();
  renderSurprise();
  renderEmbeddings();
  renderBlockStrip();
  renderHeadGrid();
  renderHeadDetail();
  renderMlp();
  renderCandidates();
  renderLens();
}

/* ---------- Interaction ---------- */

function setFocus(index: number) {
  if (!trace || index === focusIndex() && focus >= 0) return;
  focus = index;
  renderAttentionBars();
  document.querySelectorAll<HTMLCanvasElement>('canvas.matrix').forEach(canvas => {
    if (canvas.parentElement.querySelector(`.focus-row[data-for="${canvas.id}"]`)) markFocus(canvas, trace.n, canvas.classList.contains('square'));
  });
  renderReadout();
  scheduleLens();
}

function selectLayer(l: number) {
  layer = l;
  renderBlockStrip();
  renderHeadGrid();
  renderHeadDetail();
  renderMlp();
  renderAttentionBars();
  renderLens();
}

function selectHead(h: number) {
  head = h;
  renderHeadGrid();
  renderHeadDetail();
  renderAttentionBars();
}

// Hovering a row of any matrix focuses that token; attention cells name both tokens.
document.querySelectorAll<HTMLCanvasElement>('canvas.matrix').forEach(canvas => {
  canvas.addEventListener('pointermove', event => {
    if (!trace) return;
    const bounds = canvas.getBoundingClientRect();
    const row = Math.floor((event.clientY - bounds.top) / bounds.height * canvas.height);
    if (canvas.height === trace.n && row >= 0 && row < trace.n) setFocus(row);
    if (canvas.classList.contains('square')) {
      const col = Math.floor((event.clientX - bounds.left) / bounds.width * trace.n);
      renderReadout({ row: focusIndex(), col: Math.max(0, Math.min(trace.n - 1, col)) });
    }
  });
});

function request(next: number[]) {
  if (!ready || !next.length) return;
  if (pending) { queued = { tokens: next, backend }; return; }
  pending = true;
  inFlight = { tokens: next, backend };
  flow.setAttribute('aria-busy', 'true');
  send({ type: 'run', id: ++requestId, tokens: next, backend });
}

function updateTokens() {
  if (!tokenizer) return;
  const encoded = tokenizer.encode(promptInput.value);
  tokens = encoded.slice(0, MAX_TOKENS);
  if (encoded.length > MAX_TOKENS) setStatus(`Showing the first ${MAX_TOKENS} tokens so every matrix stays readable.`, 'info');
  else if (statusLine.dataset.kind === 'info') setStatus('');
  sampled = -1;
  arriving = -1;
  if (!trace) renderTokenList();
  generateButton.disabled = !ready || !tokens.length || tokens.length >= MAX_TOKENS;
  request(tokens);
}

/** The prompt box grows with its text, line breaks included, up to its CSS maximum (then it scrolls). */
function fitPrompt() {
  promptInput.style.height = 'auto';
  promptInput.style.height = `${promptInput.scrollHeight + promptInput.offsetHeight - promptInput.clientHeight}px`;
}
fitPrompt();
document.fonts?.ready.then(fitPrompt);

promptInput.addEventListener('input', () => {
  fitPrompt();
  clearTimeout(typingTimer);
  // The next run brings new tokens; a lens now would only delay it.
  clearTimeout(lensTimer);
  typingTimer = setTimeout(() => {
    typingTimer = undefined;
    updateTokens();
  }, 180);
});

$<HTMLSelectElement>('example-select').addEventListener('change', event => {
  const select = event.target as HTMLSelectElement;
  if (!select.value) return;
  promptInput.value = select.value;
  fitPrompt();
  select.value = '';
  updateTokens();
});

generateButton.addEventListener('click', () => {
  if (!result || !trace || pending || traceTokens.length >= MAX_TOKENS) return;
  const next = sample(result);
  sampled = next;
  renderCandidates();
  tokens = [...traceTokens, next];
  arriving = tokens.length - 1;
  focus = -1;
  // A text area keeps the line breaks GPT-2 writes, so editing afterwards re-reads the same text.
  promptInput.value = tokenizer.decode(tokens);
  fitPrompt();
  // Once the box has reached its full height, keep the newest token in view.
  promptInput.scrollTop = promptInput.scrollHeight;
  request(tokens);
});

const temperature = $<HTMLInputElement>('temperature');
const samplingValue = $<HTMLInputElement>('sampling-value');
function syncSamplingControls() {
  $('temperature-value').textContent = settings.temperature.toFixed(1);
  const topK = settings.mode === 'top-k';
  $('sampling-name').textContent = topK ? 'Keep the top k' : 'Keep probability p';
  Object.assign(samplingValue, topK ? { min: '1', max: '20', step: '1' } : { min: '0.05', max: '1', step: '0.05' });
  samplingValue.value = String(topK ? settings.k : settings.p);
  $('sampling-number').textContent = topK ? String(settings.k) : settings.p.toFixed(2);
  document.querySelectorAll<HTMLButtonElement>('.segmented [data-mode]').forEach(button => {
    button.setAttribute('aria-checked', String(button.dataset.mode === settings.mode));
  });
  for (const input of [temperature, samplingValue]) {
    input.style.setProperty('--range-progress', `${100 * (+input.value - +input.min) / (+input.max - +input.min)}%`);
  }
}
function resample() {
  if (trace) result = applySampling(trace.logits, settings);
  sampled = -1;
  renderCandidates();
}
temperature.addEventListener('input', () => { settings.temperature = +temperature.value; syncSamplingControls(); resample(); });
samplingValue.addEventListener('input', () => {
  if (settings.mode === 'top-k') settings.k = +samplingValue.value; else settings.p = +samplingValue.value;
  syncSamplingControls();
  resample();
});
document.querySelectorAll<HTMLButtonElement>('.segmented [data-mode]').forEach(button => {
  button.addEventListener('click', () => {
    settings.mode = button.dataset.mode as SamplingSettings['mode'];
    syncSamplingControls();
    resample();
  });
});
syncSamplingControls();

window.addEventListener(THEME_CHANGE_EVENT, () => { buildRamps(); renderAll(); });
window.addEventListener('resize', () => { fitPrompt(); if (trace) renderAll(); });

/* ---------- Where the model runs ---------- */

const gpuReasons: Record<GpuReason, string> = {
  unsupported: 'because this browser doesn’t support WebGPU',
  'no-adapter': 'because no GPU is available to this page',
  device: 'because the GPU couldn’t start',
  memory: 'because the GPU doesn’t have enough memory for the model',
  mismatch: 'because the GPU’s results didn’t match the CPU’s',
  lost: 'because the GPU stopped responding',
  error: 'because the GPU ran into an error',
};

/** Milliseconds as people read them: one decimal for quick runs, seconds past one second. */
function duration(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** Which processor runs the model and why, how long the last run took, and the choice between them. */
function renderEngine() {
  const box = $('engine');
  box.hidden = !ready || !engine;
  if (box.hidden) return;
  const onGpu = engine.backend === 'gpu';
  $('engine-name').textContent = onGpu ? 'Running on your GPU' : 'Running on the CPU';
  const reason = $('engine-reason');
  reason.textContent = !onGpu && engine.reason ? gpuReasons[engine.reason] : '';
  reason.hidden = !reason.textContent;
  const time = $('engine-time');
  // A shortened prompt computes no new tokens, only the prediction after the new last one.
  const count = lastRun?.fresh === 1 ? '1 token' : `${lastRun?.fresh.toLocaleString()} tokens`;
  time.textContent = !lastRun ? '' : lastRun.fresh ? `${count} in ${duration(lastRun.ms)}` : `Updated in ${duration(lastRun.ms)}`;
  time.title = lastRun ? 'How long the last run took' : '';
  $('engine-choice').hidden = !engine.gpu;
  document.querySelectorAll<HTMLButtonElement>('#engine-choice [data-backend]').forEach(button => {
    button.setAttribute('aria-checked', String(button.dataset.backend === backend));
  });
}

document.querySelectorAll<HTMLButtonElement>('#engine-choice [data-backend]').forEach(button => {
  button.addEventListener('click', () => {
    const next = button.dataset.backend as Backend;
    if (next === backend) return;
    backend = next;
    renderEngine();
    // Rerun the current tokens from scratch on the chosen processor, so the times compare.
    request(trace ? traceTokens : tokens);
  });
});

/* ---------- Worker ---------- */

worker.onmessage = ({ data }: MessageEvent<TransformerResponse>) => {
  if (data.type === 'progress') {
    const megabytes = (bytes: number) => Math.round(bytes / 1e6);
    const percent = Math.round(data.loaded / data.total * 100);
    const track = document.querySelector<HTMLElement>('.progress-track');
    track.setAttribute('aria-valuenow', String(percent));
    track.querySelector('span').style.width = `${percent}%`;
    $('load-status').textContent = data.cached
      ? `Opening from this browser, ${megabytes(data.loaded)} of ${megabytes(data.total)} MB`
      : `Downloading ${megabytes(data.loaded)} of ${megabytes(data.total)} MB`;
  } else if (data.type === 'preparing') {
    $('load-status').textContent = 'Setting up your GPU';
  } else if (data.type === 'ready') {
    ready = true;
    engine = data.engine;
    $('model-loader').classList.add('is-loaded');
    setStatus('');
    renderEngine();
    updateTokens();
  } else if (data.type === 'engine') {
    engine = data.engine;
    renderEngine();
  } else if (data.type === 'result') {
    // The worker can also send a result unasked, after recomputing on the CPU when the GPU stops working.
    if (data.id === requestId) {
      pending = false;
      if (queued && (queued.tokens.join() !== data.tokens.join() || queued.backend !== inFlight?.backend)) {
        const next = queued.tokens;
        queued = null;
        request(next);
      }
      queued = null;
    }
    lastRun = { ms: data.ms, fresh: data.tokens.length - data.reused };
    renderEngine();
    trace = data.trace;
    traceTokens = data.tokens;
    traceBackend = data.backend;
    if (focus >= trace.n) focus = -1;
    result = applySampling(trace.logits, settings);
    generateButton.disabled = traceTokens.length >= MAX_TOKENS;
    flow.setAttribute('aria-busy', String(pending));
    renderAll();
    arriving = -1;
    scheduleLens();
  } else if (data.type === 'lens') {
    lensInFlight = false;
    // Replies for tokens or a position that have since changed are dropped.
    if (data.rows.length && lensIsCurrent(data)) {
      lens = data;
      renderLens(true);
    }
    if (lensWanted) requestLens();
  } else if (data.type === 'weights') {
    headWeights = data;
    renderWeights();
  } else {
    pending = false;
    lensInFlight = false;
    flow.setAttribute('aria-busy', 'false');
    setStatus(data.message);
    loadButton.hidden = false;
    document.querySelector<HTMLElement>('.load-progress').hidden = true;
  }
};
worker.onerror = () => setStatus('The model worker could not start. Reload the page to try again.');

function startLoading() {
  loadButton.hidden = true;
  document.querySelector<HTMLElement>('.load-progress').hidden = false;
  send({ type: 'load', base: MODEL_BASE, backend, softwareGpu: backendParameter === 'gpu' });
}
loadButton.addEventListener('click', startLoading);

/** Fetches through the model cache so the tokenizer also works offline after the first visit. */
async function cachedFetch(url: string): Promise<Response> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return hit;
    const response = await fetch(url);
    if (response.ok) await cache.put(url, response.clone());
    return response;
  } catch {
    return fetch(url);
  }
}

async function initialize() {
  try {
    const [vocabulary, merges] = await Promise.all([
      cachedFetch(`${MODEL_BASE}vocab.json`).then(response => response.json()),
      cachedFetch(`${MODEL_BASE}merges.txt`).then(response => response.text()),
    ]);
    tokenizer = new Gpt2Tokenizer(vocabulary, merges);
    updateTokens();
  } catch {
    setStatus('The tokenizer could not load. Check your connection and reload.');
  }
  // Open the model straight away when every file is already in this browser.
  try {
    const manifest: Gpt2Manifest = await (await fetch(`${MODEL_BASE}manifest.json`)).json();
    const cache = await caches.open(MODEL_CACHE);
    const hits = await Promise.all(manifest.files.map(file => cache.match(`${MODEL_BASE}${file.name}`)));
    if (hits.every(Boolean)) startLoading();
  } catch {
    // Without Cache Storage the visitor chooses when to download.
  }
}
initialize();
