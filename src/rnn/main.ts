import '@fontsource-variable/mona-sans/wdth.css';
import '../../styles.css';
import './rnn.css';
import * as d3 from 'd3';
import { initializeSite, reducedMotion } from '../site';
import { signalRange, THEME_CHANGE_EVENT } from '../palette';
import { applySampling, sample, type SamplingResult, type SamplingSettings } from '../transformer/sampling';
import { CharLstmRun, encodeText, parseCharLstm, spanReach, type CharLstm, type CharLstmMeta } from './model';
import { createGates, type Gates } from './gates';
import { IN_WORD, LETTER, languageFrom } from './languages';
import {
  NEARLY_CERTAIN, SPAN_REACH, characterName, characters, correlationText, glyph, paintSpanBars, percent, prepareCanvas, surpriseLevel,
  type ChartColors,
} from './views';

initializeSite();

/** English or Hindi, from the address; each has its own networks. */
const language = languageFrom(location.search);
/** The longest text the page reads; generation stops there. */
const MAX_CHARACTERS = 400;
/** Characters one press of Generate adds. */
const GENERATE_LENGTH = 120;
/** About 30 characters a second. */
const GENERATE_INTERVAL_MS = 33;
/** Cell states beyond ±CELL_SCALE get the strongest color. */
const CELL_SCALE = 2;
/** Cells whose correlation with a text feature is weaker than this are not pointed out. */
const NOTABLE_CORRELATION = .5;
/** Generate picks each new character from this many of the likeliest. */
const GENERATE_TOP = 5;
/** How the next-character list names characters that don't start a word. */
const MARK_NAMES: Record<string, string> = {
  ' ': 'space', '\n': 'new paragraph', ',': 'comma', '.': 'period', '"': 'quote', "'": 'apostrophe',
  '!': 'exclamation', '?': 'question mark', ';': 'semicolon', ':': 'colon', '-': 'dash', '(': 'open bracket', ')': 'close bracket',
  ...language.marks,
};

type Detail = 'hidden' | 'input' | 'forget' | 'output';
/** The second map's choices; `gate` is the gate's block in the model's order (−1 for the hidden state). */
const DETAILS: Record<Detail, { name: string; note: string; gate: number }> = {
  hidden: { name: 'Hidden state', note: 'what each cell reveals to the prediction, h = output gate × tanh(c)', gate: -1 },
  input: { name: 'Input gate', note: 'how much new information each cell writes, from 0 to 1', gate: 0 },
  forget: { name: 'Forget gate', note: 'how much of its memory each cell keeps, from 0 (forget) to 1 (keep)', gate: 1 },
  output: { name: 'Output gate', note: 'how much of its memory each cell reveals, from 0 to 1', gate: 3 },
};
/** How the page names the features the trainer looks for (see notableCells in the model's metadata). */
const NOTABLE: Record<string, { label: string; note: (r: number) => string }> = {
  quote: { label: 'Quotation cell', note: r => `${r < 0 ? 'amber' : 'violet'} inside a quotation, ${r < 0 ? 'violet' : 'amber'} outside` },
  word: { label: 'Word-length cell', note: r => `${r > 0 ? 'rises' : 'falls'} with every letter of a word` },
  parenthesis: { label: 'Parenthesis cell', note: r => `${r < 0 ? 'amber' : 'violet'} inside parentheses` },
  paragraph: { label: 'Paragraph cell', note: r => `${r > 0 ? 'rises' : 'falls'} through each paragraph` },
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const workbench = $('rnn-workbench');
const promptInput = $<HTMLTextAreaElement>('prompt-input');
const generateButton = $<HTMLButtonElement>('generate-button');
const temperature = $<HTMLInputElement>('temperature');
const timeline = $('timeline');
const track = $('timeline-track');
const cellList = $('char-cells');
const cellMap = $<HTMLCanvasElement>('cell-map');
const detailMap = $<HTMLCanvasElement>('detail-map');
const unitChart = $<HTMLCanvasElement>('unit-chart');
const spanBars = $<HTMLCanvasElement>('span-bars');
const statusLine = $('rnn-status');

let model: CharLstm | null = null;
let run: CharLstmRun | null = null;
/** The comparison with the gateless network, below the memory panel. */
let gates: Gates | null = null;
/** The step whose prediction is shown; −1 follows the end of the text. */
let chosenStep = -1;
/** The memory cell traced below the cell-state map; picked automatically until someone chooses one. */
let unit = 0;
let unitChosen = false;
/** Cells the trainer found tracking something on held-out text. */
let notableCells: { cell: number; label: string; note: string; correlation: number }[] = [];
let detail: Detail = 'forget';
/** Temperature, and Generate picks from the five likeliest characters, so it rarely makes up words. */
const settings: SamplingSettings = { temperature: .6, mode: 'top-k', k: GENERATE_TOP, p: 1 };
let generation: { timer: number; remaining: number } | null = null;
/** A character that just arrived from Generate, so it can pop in. */
let arriving = -1;
/** The span view's last result, kept while the text and step stay the same. */
let influence: { key: string; values: Float32Array; target: number } | null = null;
/** The word each candidate character would lead to, for the text up to one step. */
let previews: { key: string; words: Map<number, string> } | null = null;

function setStatus(message: string) {
  statusLine.textContent = message;
  statusLine.dataset.kind = 'error';
  statusLine.hidden = !message;
}

/* ---------- Colors ---------- */

let diverging = new Uint8ClampedArray(256 * 3);
let sequential = new Uint8ClampedArray(256 * 3);
let colors: ChartColors = { positive: '#744ed7', negative: '#cf6d01', ink: '#1c1e2d', line: 'rgb(28 30 45 / .16)', well: '#eff0f6' };
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
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  colors = { positive, negative, ink: read('--ink', colors.ink), line: read('--line-strong', colors.line), well: read('--well', colors.well) };
}
buildRamps();

const alpha = (color: string, opacity: number) => {
  const { r, g, b } = d3.rgb(color);
  return `rgb(${r} ${g} ${b} / ${opacity})`;
};

/* ---------- Text helpers ---------- */

const steps = () => run?.length ?? 0;
/** The step the side panels describe: the chosen one, or the last. */
const focusStep = () => (chosenStep >= 0 && chosenStep < steps() ? chosenStep : steps() - 1);
/** A predicted character in a sentence: “e”, or a space. */
const guessName = (index: number) => {
  const c = model.vocabulary[index];
  return c === ' ' ? 'of a space' : c === '\n' ? 'of a new paragraph' : `“${c}”`;
};
const ordinal = (n: number) => {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th';
  return `${n}${suffix}`;
};

/** The characters read up to and including step t, as a key for caching what depends on them. */
const prefixKey = (t: number) => Array.from(run.indices.subarray(0, t + 1)).join(',');

/** The letters of the word being written at step t (empty after a space or punctuation). */
function wordBefore(t: number): string {
  let word = '';
  for (let s = t; s >= 0 && IN_WORD.test(model.vocabulary[run.indices[s]]); s--) word = model.vocabulary[run.indices[s]] + word;
  return word.replace(/^'+/, '');
}

/** The text up to and including step t, shortened from the left, for quoting. */
function textBefore(t: number, length = 14): string {
  const start = Math.max(0, t + 1 - length);
  let text = '';
  for (let s = start; s <= t; s++) text += model.vocabulary[run.indices[s]].replace('\n', '↵');
  return `${start > 0 ? '…' : ''}${text}`;
}

function surpriseLabel(t: number): string {
  const bits = run.surprisal(t);
  if (Number.isNaN(bits)) return 'Nothing read before it, so nothing predicted it';
  const p = 2 ** -bits;
  return `${bits.toFixed(1)} bits of surprise: the network gave it ${percent(p)}`;
}

/* ---------- Character cells ---------- */

function renderCells(from = 0) {
  const n = steps();
  $('step-count').textContent = String(n);
  const items = cellList.children as HTMLCollectionOf<HTMLElement>;
  for (let t = Math.min(from, items.length); t < n; t++) {
    let item = items[t];
    if (!item) {
      item = document.createElement('li');
      item.className = 'char-cell';
      item.setAttribute('role', 'option');
      item.dataset.index = String(t);
      item.tabIndex = -1;
      cellList.append(item);
    }
    const character = model.vocabulary[run.indices[t]];
    item.textContent = character === ' ' ? '' : glyph(character);
    item.classList.toggle('is-space', character === ' ');
    item.classList.toggle('is-break', character === '\n');
    const bits = run.surprisal(t);
    if (Number.isNaN(bits)) delete item.dataset.surprise;
    else item.dataset.surprise = String(surpriseLevel(bits));
    item.setAttribute('aria-label', `${characterName(character)}, ${Number.isNaN(bits) ? 'first character' : `${bits.toFixed(1)} bits of surprise`}`);
    item.title = `${characterName(character) === character ? `“${character}”` : characterName(character)}: ${surpriseLabel(t)}`;
    item.classList.toggle('is-new', t === arriving && !reducedMotion.matches);
  }
  while (items.length > n) items[items.length - 1].remove();
}

/** Selection state, the roving tab stop and the column marker for the focused step, here and in the comparison. */
function markStep() {
  gates?.markStep();
  const t = focusStep();
  const items = cellList.children as HTMLCollectionOf<HTMLElement>;
  cellList.querySelectorAll('[aria-selected="true"]').forEach(item => item.setAttribute('aria-selected', 'false'));
  cellList.querySelectorAll<HTMLElement>('[tabindex="0"]').forEach(item => { item.tabIndex = -1; });
  const marker = $('step-marker');
  marker.hidden = t < 0;
  if (t < 0) return;
  const item = items[t];
  item.setAttribute('aria-selected', 'true');
  item.tabIndex = 0;
  marker.style.setProperty('--at', String(t));
}

/* ---------- Painting ---------- */

/** One pixel per step and cell; CSS scales the canvas so columns line up with the characters. */
function paintMap(canvas: HTMLCanvasElement, value: (t: number, k: number) => number, kind: 'diverging' | 'sequential', scale: number) {
  const n = Math.max(1, steps());
  const H = model?.hiddenSize ?? 256;
  canvas.width = n;
  canvas.height = H;
  const context = canvas.getContext('2d');
  const image = context.createImageData(n, H);
  const lut = kind === 'diverging' ? diverging : sequential;
  for (let t = 0; t < steps(); t++) {
    for (let k = 0; k < H; k++) {
      const v = value(t, k);
      const x = kind === 'diverging' ? (v / scale + 1) / 2 : v / scale;
      const i = Math.round(Math.max(0, Math.min(1, x)) * 255) * 3;
      const p = (k * n + t) * 4;
      image.data[p] = lut[i];
      image.data[p + 1] = lut[i + 1];
      image.data[p + 2] = lut[i + 2];
      image.data[p + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function paintCellMap() {
  if (!run) return;
  const H = model.hiddenSize;
  paintMap(cellMap, (t, k) => run.cells[t * H + k], 'diverging', CELL_SCALE);
}

function paintDetailMap() {
  if (!run) return;
  const H = model.hiddenSize;
  const { gate } = DETAILS[detail];
  if (gate < 0) paintMap(detailMap, (t, k) => run.hidden[t * H + k], 'diverging', 1);
  else paintMap(detailMap, (t, k) => run.gates[t * 4 * H + gate * H + k], 'sequential', 1);
  $('detail-note').textContent = DETAILS[detail].note;
  detailMap.setAttribute('aria-label', `${DETAILS[detail].name} of all ${model.hiddenSize} memory cells at every step. Use the up and down arrow keys to choose a cell, left and right to move along the text.`);
  document.querySelectorAll<HTMLButtonElement>('[data-detail]').forEach(button => {
    button.setAttribute('aria-checked', String(button.dataset.detail === detail));
  });
}

/** The traced cell's state over the text: a line around zero, violet above and amber below. */
function paintUnitChart() {
  const { context, width, height } = prepareCanvas(unitChart);
  const n = steps();
  if (!n) return;
  const H = model.hiddenSize;
  const step = width / n;
  // The range always includes zero and at least ±0.5, so small wiggles stay small.
  let low = -.5;
  let high = .5;
  for (let t = 0; t < n; t++) {
    low = Math.min(low, run.cells[t * H + unit]);
    high = Math.max(high, run.cells[t * H + unit]);
  }
  const pad = 5;
  const y = (v: number) => pad + ((high - v) / (high - low)) * (height - 2 * pad);
  const x = (t: number) => (t + .5) * step;
  const zero = y(0);
  context.strokeStyle = colors.line;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, Math.round(zero) + .5);
  context.lineTo(width, Math.round(zero) + .5);
  context.stroke();
  for (const [sign, color] of [[1, colors.positive], [-1, colors.negative]] as const) {
    context.save();
    context.beginPath();
    if (sign > 0) context.rect(0, 0, width, zero);
    else context.rect(0, zero, width, height - zero);
    context.clip();
    context.beginPath();
    context.moveTo(x(0), zero);
    for (let t = 0; t < n; t++) context.lineTo(x(t), y(run.cells[t * H + unit]));
    context.lineTo(x(n - 1), zero);
    context.closePath();
    context.fillStyle = alpha(color, .22);
    context.fill();
    context.restore();
  }
  context.strokeStyle = colors.ink;
  context.lineWidth = 1.5;
  context.lineJoin = 'round';
  context.beginPath();
  for (let t = 0; t < n; t++) {
    if (t === 0) context.moveTo(x(t), y(run.cells[t * H + unit]));
    else context.lineTo(x(t), y(run.cells[t * H + unit]));
  }
  context.stroke();
  if (n === 1) {
    context.fillStyle = colors.ink;
    context.beginPath();
    context.arc(x(0), y(run.cells[unit]), 2.5, 0, 2 * Math.PI);
    context.fill();
  }
}

/** The traced cell's input, forget and output gates, one strip each. */
function paintGateStrips() {
  const n = Math.max(1, steps());
  const H = model?.hiddenSize ?? 256;
  document.querySelectorAll<HTMLCanvasElement>('.gate-strip').forEach(canvas => {
    const gate = { input: 0, forget: 1, output: 3 }[canvas.dataset.gate];
    canvas.width = n;
    canvas.height = 1;
    const context = canvas.getContext('2d');
    const image = context.createImageData(n, 1);
    for (let t = 0; t < steps(); t++) {
      const i = Math.round(Math.max(0, Math.min(1, run.gates[t * 4 * H + gate * H + unit])) * 255) * 3;
      image.data.set([sequential[i], sequential[i + 1], sequential[i + 2], 255], t * 4);
    }
    context.putImageData(image, 0, 0);
  });
}

/* ---------- Memory span ---------- */

/** What shaped the guess after step t: the LSTM's influence, cached for the current text and step. */
function lstmSpan(t: number) {
  const key = prefixKey(t);
  if (influence?.key !== key) {
    const target = run.topCharacter(t);
    influence = { key, values: run.influence(t, target, SPAN_REACH), target };
  }
  return influence;
}

/**
 * How much each character up to the focused step shaped its prediction: bars over the
 * characters, violet where a character pushed toward the most likely next character
 * and amber where it pushed away.
 */
function renderSpan() {
  const t = focusStep();
  const summary = $('span-summary');
  if (t < 0) {
    paintSpanBars(spanBars, null, t, 0, colors);
    summary.textContent = '';
    return;
  }
  const { values, target } = lstmSpan(t);
  paintSpanBars(spanBars, values, t, steps(), colors);
  // How many of the latest characters hold half, and nine tenths, of the total influence.
  const half = spanReach(values, t, .5);
  const most = spanReach(values, t, .9);
  const guess = guessName(target);
  const certainty = run.probabilities[t * model.inputSize + target];
  const shares = `Half came from the last ${characters(half)}, 90% from the last ${characters(most)}`;
  const reading = t === 0 ? 'only the first character so far'
    : certainty >= NEARLY_CERTAIN ? `it is ${percent(certainty)} sure, so no single character moves it much`
      : shares;
  summary.textContent = t === 0 || certainty >= NEARLY_CERTAIN
    ? `what shaped the guess ${guess} after “${textBefore(t, 8)}”: ${reading}.`
    : `what shaped the guess ${guess} after “${textBefore(t, 8)}”. ${reading}.`;
  spanBars.setAttribute('aria-label', `What shaped the guess ${guess}: ${t === 0 ? 'the first character alone' : reading.charAt(0).toLowerCase() + reading.slice(1)}`);
}

/* ---------- Next character ---------- */

let sampling: { step: number; result: SamplingResult } | null = null;
function samplingAt(t: number): SamplingResult {
  if (sampling?.step !== t) sampling = { step: t, result: applySampling(run.step(t).logits, settings) };
  return sampling.result;
}

function renderCandidates() {
  const list = $('candidate-list');
  const t = focusStep();
  if (t < 0) {
    list.replaceChildren();
    $('next-context').textContent = 'Type something to see what the network expects next.';
    $('next-actual').textContent = '';
    return;
  }
  const result = samplingAt(t);
  const shown = Array.from(result.ranking.subarray(0, 10));
  const max = result.scaled[shown[0]] || 1;
  const actual = t + 1 < steps() ? run.indices[t + 1] : -1;
  const items = Array.from(list.children) as HTMLElement[];
  // The word each letter would start or finish, if the network kept taking its likeliest letter.
  const key = prefixKey(t);
  if (previews?.key !== key) previews = { key, words: new Map() };
  const written = wordBefore(t);
  const kept = new Set(result.kept);
  shown.forEach((id, i) => {
    let item = items[i];
    if (!item) {
      item = document.createElement('li');
      item.className = 'candidate';
      item.innerHTML = '<span class="candidate-text"></span><span class="candidate-word"></span><span class="candidate-bar" aria-hidden="true"><i></i></span><span class="candidate-value readout"></span>';
      list.append(item);
    }
    const character = model.vocabulary[id];
    const text = item.querySelector<HTMLElement>('.candidate-text');
    text.textContent = glyph(character);
    text.classList.toggle('is-whitespace', character === ' ' || character === '\n');
    text.setAttribute('aria-label', characterName(character));
    const preview = item.querySelector<HTMLElement>('.candidate-word');
    // Unlikely letters lead to made-up words, which say little; only plausible ones get a preview.
    if (LETTER.test(character) && result.scaled[id] >= .01) {
      if (!previews.words.has(id)) previews.words.set(id, run.completeWord(t, id).map(index => model.vocabulary[index]).join(''));
      preview.replaceChildren(Object.assign(document.createElement('span'), { className: 'known', textContent: written }), previews.words.get(id));
      preview.title = `${written}${previews.words.get(id)}`;
    } else {
      preview.replaceChildren(LETTER.test(character) ? '' : MARK_NAMES[character] ?? '');
      preview.title = '';
    }
    preview.classList.toggle('is-mark', !LETTER.test(character));
    item.querySelector<HTMLElement>('.candidate-bar i').style.width = `${(result.scaled[id] / max) * 100}%`;
    item.querySelector('.candidate-value').textContent = percent(result.scaled[id]);
    item.classList.toggle('is-actual', id === actual);
    item.classList.toggle('is-filtered', !kept.has(id));
    item.title = id === actual ? 'What the text has next' : kept.has(id) ? '' : `Generate only picks from the ${GENERATE_TOP} likeliest`;
  });
  items.slice(shown.length).forEach(item => item.remove());
  $('next-context').replaceChildren('What could follow ', Object.assign(document.createElement('q'), { textContent: textBefore(t) }));
  const note = $('next-actual');
  if (actual < 0) {
    note.textContent = t === steps() - 1 && steps() < MAX_CHARACTERS
      ? `Generate picks each new character from the ${GENERATE_TOP} likeliest, weighted by these chances.` : '';
  } else {
    const rank = result.ranking.indexOf(actual) + 1;
    note.textContent = `The text goes on with “${glyph(model.vocabulary[actual])}”, its ${ordinal(rank)} choice at ${percent(result.scaled[actual])}.`;
  }
}

/* ---------- How often it's right ---------- */

/** How often the network's first guess, and its top three, held the character the text has next. */
function renderScore() {
  const n = steps();
  $('scorecard').hidden = n < 2;
  if (n < 2) return;
  const V = model.inputSize;
  let first = 0;
  let three = 0;
  const kinds = { start: [0, 0], inside: [0, 0] };
  for (let t = 0; t + 1 < n; t++) {
    const actual = run.indices[t + 1];
    const offset = t * V;
    let rank = 0;
    for (let v = 0; v < V; v++) if (run.probabilities[offset + v] > run.probabilities[offset + actual]) rank++;
    if (rank === 0) first++;
    if (rank < 3) three++;
    if (LETTER.test(model.vocabulary[actual])) {
      const kind = IN_WORD.test(model.vocabulary[run.indices[t]]) ? kinds.inside : kinds.start;
      kind[0]++;
      if (rank === 0) kind[1]++;
    }
  }
  const share = (count: number, of: number) => `${Math.round((100 * count) / of)}%`;
  $('score-first').textContent = share(first, n - 1);
  $('score-three').textContent = share(three, n - 1);
  const [starts, startsRight] = kinds.start;
  const [inside, insideRight] = kinds.inside;
  $('score-note').textContent = starts >= 3 && inside >= 3
    ? `Inside a word it guesses right ${share(insideRight, inside)} of the time; at the start of one only ${share(startsRight, starts)}, because that means choosing the next word.`
    : '';
}

/* ---------- The traced cell ---------- */

/**
 * The cell whose memory, as the prediction sees it (tanh c), varies most over the
 * text: one that switches on and off rather than one that slowly piles up.
 */
function liveliestUnit(): number {
  const n = steps();
  const H = model.hiddenSize;
  if (!n) return 0;
  let best = 0;
  let bestVariance = -1;
  for (let k = 0; k < H; k++) {
    let sum = 0;
    let squares = 0;
    for (let t = 0; t < n; t++) {
      const v = Math.tanh(run.cells[t * H + k]);
      sum += v;
      squares += v * v;
    }
    const variance = squares / n - (sum / n) ** 2;
    if (variance > bestVariance) { bestVariance = variance; best = k; }
  }
  return best;
}

function renderUnit() {
  $('unit-title').textContent = `Cell ${unit + 1}`;
  const notable = notableCells.find(entry => entry.cell === unit);
  $('unit-note').textContent = notable ? `${notable.note}, and its gates`
    : unitChosen ? 'its memory over the text, and its gates' : 'the cell that changes most over this text, and its gates';
  $('cell-picks').querySelectorAll<HTMLButtonElement>('button').forEach(button => {
    button.setAttribute('aria-pressed', String(Number(button.dataset.cell) === unit));
  });
  document.querySelectorAll<HTMLElement>('.unit-marker').forEach(marker => marker.style.setProperty('--unit', String(unit)));
  for (const canvas of [cellMap, detailMap]) {
    canvas.setAttribute('aria-valuenow', String(unit + 1));
    canvas.setAttribute('aria-valuetext', `Cell ${unit + 1}`);
  }
  if (!run) return;
  paintUnitChart();
  paintGateStrips();
  renderReadout();
}

/** Exact values of the traced cell at the focused step, or at a hovered step. */
function renderReadout(at = focusStep()) {
  const readout = $('map-readout');
  if (!run || at < 0) { readout.textContent = ''; return; }
  const H = model.hiddenSize;
  const value = (array: Float32Array, width: number, offset = 0) => array[at * width + offset + unit].toFixed(2);
  readout.textContent = `Cell ${unit + 1} after “${textBefore(at, 10)}”: memory ${value(run.cells, H)}, revealed ${value(run.hidden, H)}; `
    + `gates input ${value(run.gates, 4 * H)}, forget ${value(run.gates, 4 * H, H)}, output ${value(run.gates, 4 * H, 3 * H)}.`;
}

/** Traces another memory cell; once someone chooses one, new texts keep it. */
function chooseUnit(next: number) {
  next = Math.max(0, Math.min((model?.hiddenSize ?? 256) - 1, next));
  unitChosen = true;
  if (next === unit) return;
  unit = next;
  renderUnit();
}

/* ---------- Rendering ---------- */

/** Everything that depends on the focused step. */
function renderStep() {
  markStep();
  renderSpan();
  renderCandidates();
  renderReadout();
  gates?.renderStep();
}

function renderAll(from = 0) {
  const n = steps();
  workbench.classList.toggle('is-empty', n === 0);
  track.style.setProperty('--steps', String(n));
  renderCells(from);
  paintCellMap();
  paintDetailMap();
  renderUnit();
  gates?.render(from);
  renderStep();
  renderScore();
  syncGenerateButton();
}

/**
 * Drawing happens once per frame however many changes arrive: `from` is the first
 * step whose character cell changed, `step` asks only for the views of the focused
 * step, and `follow` scrolls the end of the text into view afterwards.
 */
const pending = { from: Infinity, step: false, follow: false };
let frame = 0;
function schedule(part: { from?: number; step?: boolean; follow?: boolean }) {
  pending.from = Math.min(pending.from, part.from ?? Infinity);
  pending.step ||= Boolean(part.step);
  pending.follow ||= Boolean(part.follow);
  frame ||= requestAnimationFrame(() => {
    frame = 0;
    const { from, step, follow } = pending;
    Object.assign(pending, { from: Infinity, step: false, follow: false });
    if (from !== Infinity) renderAll(from);
    else if (step) renderStep();
    arriving = -1;
    if (follow) {
      timeline.scrollLeft = timeline.scrollWidth;
      gates?.followEnd();
    }
  });
}

/** Whether the end of the text is in view, so it can stay there as the text grows. */
const atEnd = () => timeline.scrollLeft + timeline.clientWidth >= timeline.scrollWidth - 24;

/** Reads the text box with the model, computing only the steps that changed. */
function update({ fresh = false } = {}) {
  if (!model) return;
  const wasAtEnd = atEnd();
  const encoded = encodeText(model, promptInput.value);
  const note = $('skipped-note');
  note.hidden = !encoded.skipped.length;
  note.textContent = encoded.skipped.length
    ? `Left out, since the network never learned ${encoded.skipped.length === 1 ? 'it' : 'them'}: ${encoded.skipped.map(c => `“${c}”`).join(' ')}`
    : '';
  const indices = encoded.indices.slice(0, MAX_CHARACTERS);
  const reused = run.setIndices(indices);
  gates?.sync(indices);
  if (chosenStep >= indices.length) chosenStep = -1;
  if (fresh && !unitChosen) unit = liveliestUnit();
  sampling = null;
  schedule({ from: Math.max(0, reused - 1), follow: wasAtEnd || fresh });
}

/* ---------- Interaction ---------- */

/** Focuses a step; its views follow on the next frame, so quick pointer moves compute once. */
function chooseStep(t: number) {
  if (!run || t < 0 || t >= steps() || t === focusStep() && chosenStep >= 0) return;
  chosenStep = t;
  markStep();
  schedule({ step: true });
}

cellList.addEventListener('pointerover', event => {
  const item = (event.target as HTMLElement).closest<HTMLElement>('.char-cell');
  if (item) chooseStep(Number(item.dataset.index));
});
cellList.addEventListener('focusin', event => {
  const item = (event.target as HTMLElement).closest<HTMLElement>('.char-cell');
  if (item) chooseStep(Number(item.dataset.index));
});
cellList.addEventListener('keydown', event => {
  const t = focusStep();
  const moves: Record<string, number> = { ArrowLeft: t - 1, ArrowRight: t + 1, Home: 0, End: steps() - 1, PageUp: t - 10, PageDown: t + 10 };
  if (!(event.key in moves) || t < 0) return;
  event.preventDefault();
  const next = Math.max(0, Math.min(steps() - 1, moves[event.key]));
  chooseStep(next);
  const item = cellList.children[next] as HTMLElement;
  item.focus({ preventScroll: true });
  item.scrollIntoView({ block: 'nearest', inline: 'nearest' });
});

/** Hovering (or tapping) a row of either map follows that cell; the readout names the value under the pointer. */
for (const canvas of [cellMap, detailMap]) {
  const follow = (event: PointerEvent) => {
    if (!run || !steps()) return;
    const bounds = canvas.getBoundingClientRect();
    const row = Math.floor(((event.clientY - bounds.top) / bounds.height) * model.hiddenSize);
    const column = Math.floor(((event.clientX - bounds.left) / bounds.width) * steps());
    chooseUnit(row);
    renderReadout(Math.max(0, Math.min(steps() - 1, column)));
  };
  canvas.addEventListener('pointermove', follow);
  canvas.addEventListener('pointerdown', follow);
  canvas.addEventListener('pointerleave', () => renderReadout());
  // Up and down choose the cell; left and right move along the text.
  canvas.addEventListener('keydown', event => {
    const cells: Record<string, number> = { ArrowUp: unit - 1, ArrowDown: unit + 1, PageUp: unit - 8, PageDown: unit + 8, Home: 0, End: (model?.hiddenSize ?? 256) - 1 };
    const moves: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1 };
    if (event.key in cells) chooseUnit(cells[event.key]);
    else if (event.key in moves && steps()) chooseStep(Math.max(0, Math.min(steps() - 1, focusStep() + moves[event.key])));
    else return;
    event.preventDefault();
  });
}

/** Hovering (or tapping) the traced cell's chart chooses the step under the pointer. */
const stepAt = (event: PointerEvent) => {
  if (!run || !steps()) return;
  const bounds = unitChart.getBoundingClientRect();
  chooseStep(Math.max(0, Math.min(steps() - 1, Math.floor(((event.clientX - bounds.left) / bounds.width) * steps()))));
};
unitChart.addEventListener('pointermove', stepAt);
unitChart.addEventListener('pointerdown', stepAt);

document.querySelectorAll<HTMLButtonElement>('[data-detail]').forEach(button => {
  button.addEventListener('click', () => {
    detail = button.dataset.detail as Detail;
    paintDetailMap();
  });
});

promptInput.addEventListener('input', () => {
  stopGenerating();
  chosenStep = -1;
  update();
});

$<HTMLSelectElement>('example-select').addEventListener('change', event => {
  const select = event.target as HTMLSelectElement;
  if (!select.value) return;
  stopGenerating();
  promptInput.value = select.value;
  select.value = '';
  chosenStep = -1;
  update({ fresh: true });
});

/* ---------- Generate ---------- */

function syncGenerateButton() {
  const running = generation !== null;
  generateButton.disabled = !run || !steps() || (!running && steps() >= MAX_CHARACTERS);
  generateButton.setAttribute('aria-pressed', String(running));
  generateButton.classList.toggle('is-running', running);
  $('generate-label').textContent = running ? 'Stop' : 'Generate';
}

/** Samples one character after the end of the text and reads it. */
function appendSample() {
  const t = steps() - 1;
  const next = sample(applySampling(run.step(t).logits, settings));
  promptInput.value += model.vocabulary[next];
  // Once the box has reached its full height, keep the newest characters in view.
  promptInput.scrollTop = promptInput.scrollHeight;
  arriving = t + 1;
}

function generateStep() {
  if (!generation || !run) return;
  if (generation.remaining <= 0 || steps() >= MAX_CHARACTERS) { stopGenerating(); return; }
  generation.remaining--;
  appendSample();
  update();
  if (generation.remaining <= 0 || steps() >= MAX_CHARACTERS) stopGenerating();
}

function startGenerating() {
  if (!run || !steps() || generation) return;
  chosenStep = -1;
  const remaining = Math.min(GENERATE_LENGTH, MAX_CHARACTERS - steps());
  if (reducedMotion.matches) {
    // No animation: the whole continuation arrives at once.
    for (let i = 0; i < remaining; i++) {
      appendSample();
      run.setIndices(encodeText(model, promptInput.value).indices.slice(0, MAX_CHARACTERS));
    }
    update();
    return;
  }
  generation = { timer: window.setInterval(generateStep, GENERATE_INTERVAL_MS), remaining };
  syncGenerateButton();
}

function stopGenerating() {
  if (!generation) return;
  window.clearInterval(generation.timer);
  generation = null;
  syncGenerateButton();
}

generateButton.addEventListener('click', () => {
  if (generation) stopGenerating();
  else startGenerating();
});

function syncTemperature() {
  $('temperature-value').textContent = settings.temperature.toFixed(1);
  temperature.style.setProperty('--range-progress', `${100 * (+temperature.value - +temperature.min) / (+temperature.max - +temperature.min)}%`);
}
temperature.addEventListener('input', () => {
  settings.temperature = +temperature.value;
  syncTemperature();
  sampling = null;
  renderCandidates();
});
syncTemperature();

window.addEventListener(THEME_CHANGE_EVENT, () => {
  buildRamps();
  // The character tints follow the theme in CSS; every canvas is repainted.
  if (run) schedule({ from: steps() });
});
// The charts are drawn at the screen's resolution and the track's width.
new ResizeObserver(() => {
  if (!run) return;
  paintUnitChart();
  renderSpan();
}).observe(timeline);

/** Buttons for the cells that learned something recognizable, with what each tracks. */
function renderCellPicks(meta: CharLstmMeta) {
  notableCells = (meta.notableCells ?? [])
    .filter(entry => NOTABLE[entry.feature] && Math.abs(entry.correlation) >= NOTABLE_CORRELATION
      && Number.isInteger(entry.cell) && entry.cell >= 0 && entry.cell < model.hiddenSize)
    .map(entry => ({ cell: entry.cell, label: NOTABLE[entry.feature].label, note: NOTABLE[entry.feature].note(entry.correlation), correlation: entry.correlation }));
  const picks = $('cell-picks');
  picks.hidden = !notableCells.length;
  picks.replaceChildren('Try', ...notableCells.map(entry => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'button subtle small';
    button.dataset.cell = String(entry.cell);
    button.textContent = entry.label;
    button.title = `Cell ${entry.cell + 1}: ${entry.note} (correlation ${correlationText(entry.correlation)} on text it never trained on)`;
    button.addEventListener('click', () => chooseUnit(entry.cell));
    return button;
  }));
}

/* ---------- Language ---------- */

/** The switch shows which networks are loaded; in Hindi, the texts and credits are Hindi's. */
function applyLanguage() {
  document.querySelectorAll<HTMLAnchorElement>('[data-language]').forEach(link => {
    if (link.dataset.language === language.code) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  if (language.code === 'en') return;
  document.body.dataset.language = language.code;
  // Text in the page's own language, so it gets a Devanagari font and is read out as Hindi.
  for (const id of ['prompt-input', 'char-cells', 'candidate-list', 'next-context', 'gates-lstm-chars', 'gates-rnn-chars', 'continue-results']) {
    $(id)?.setAttribute('lang', language.code);
  }
  $('rnn-intro').textContent = language.intro;
  promptInput.value = language.text;
  const select = $<HTMLSelectElement>('example-select');
  select.replaceChildren(new Option('Choose…', ''), ...language.examples.map(text => new Option(text, text)));
  $('model-credits').innerHTML = language.credits
    .replace('{lstm-bits}', '<span id="validation-bits" class="readout">about 2</span>')
    .replace('{rnn-bits}', '<span id="gates-validation-bits" class="readout">about 2</span>');
}
applyLanguage();

/* ---------- Loading ---------- */

async function initialize() {
  try {
    const [meta, weights] = await Promise.all([
      fetch(`${language.lstm}.json`).then(response => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json() as Promise<CharLstmMeta>;
      }),
      fetch(`${language.lstm}.bin`).then(response => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.arrayBuffer();
      }),
    ]);
    model = parseCharLstm(meta, weights);
    run = new CharLstmRun(model);
    // The maps have one row per memory cell.
    const cells = String(model.hiddenSize);
    track.style.setProperty('--cells', cells);
    $('cell-count').textContent = cells;
    for (const map of [cellMap, detailMap]) map.setAttribute('aria-valuemax', cells);
    cellMap.setAttribute('aria-label', `Cell state of all ${cells} memory cells at every step. Use the up and down arrow keys to choose a cell, left and right to move along the text.`);
    renderCellPicks(meta);
    $('validation-bits').textContent = meta.validationBitsPerChar.toFixed(2);
    workbench.setAttribute('aria-busy', 'false');
    update({ fresh: true });
    gates = createGates({ run, meta, focusStep, chooseStep, lstmSpan, colors: () => colors, settings, modelBase: language.rnn });
  } catch {
    setStatus('The network could not load. Reload the page to try again.');
    // The comparison needs the LSTM too.
    $('gates-status').textContent = 'The comparison could not start without the LSTM.';
    $('gates-panel').setAttribute('aria-busy', 'false');
  }
}
initialize();
