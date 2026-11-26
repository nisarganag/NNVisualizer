/**
 * The Recurrent page's "Why gates matter" section. A gateless network (src/rnn/model.ts,
 * CharRnn), trained on the same books as the LSTM for as long and with about as many
 * parameters, reads the same text, and the section compares the two: how surprised
 * each was by every character, what shaped each one's guess at the focused step, how
 * fast their memory fades, how often they are right, and whether either keeps track
 * of quotations.
 *
 * The gateless network loads when the section comes near the screen. Its memory span
 * and this text's fade curves are only computed while the section is on screen: the
 * span for the focused step, the curves a guess at a time when the page is idle.
 */
import { applySampling, sample, type SamplingSettings } from '../transformer/sampling';
import {
  CharRnnRun, addInfluenceShares, guessScores, lstmCell, parseCharRnn, readout, rnnCell, spanReach,
  type CharLstmMeta, type CharLstmRun, type CharRnn, type CharRnnMeta, type CharRun, type HeldOutReference, type NotableCell,
} from './model';
import {
  NEARLY_CERTAIN, SPAN_REACH, characterName, characters, correlationText, glyph, paintSpanBars, percent, prepareCanvas,
  surpriseLevel, type ChartColors,
} from './views';

/** The quotation views appear only when the LSTM's quotation cell follows quotations at least this closely. */
const QUOTE_CORRELATION = .5;
/** The gateless network starts loading when the section is less than a screen below the view. */
const LOAD_MARGIN = '0px 0px 100% 0px';
/** This text's fade curves average the guesses after every FADE_EVERY-th step, from the (FADE_FIRST + 1)th character on. */
const FADE_FIRST = 19;
const FADE_EVERY = 10;
/**
 * This text gets its own fade curves from this many characters on, and each distance
 * needs FADE_MIN_GUESSES traced guesses: shorter texts give too few guesses that far
 * back to say anything the held-out curves don't.
 */
const FADE_MIN_TEXT = 150;
const FADE_MIN_GUESSES = 5;
/** Milliseconds of idle time a trace needs before it starts. */
const FADE_IDLE_MS = 30;
/** The fade chart's axes: characters back from 1 to FADE_DISTANCE (the span's reach), and a guess's share of pull from FADE_FLOOR to 100%, logarithmic. */
const FADE_DISTANCE = SPAN_REACH;
const FADE_FLOOR = 1e-6;
/** Characters each network writes when asked to continue the text. */
const CONTINUE_LENGTH = 120;
const SVG = 'http://www.w3.org/2000/svg';

export interface GatesHost {
  /** The LSTM's run; the gateless network reads the same characters. */
  readonly run: CharLstmRun;
  readonly meta: CharLstmMeta;
  /** The step the page describes, or −1 before any text. */
  focusStep(): number;
  /** Focuses a step everywhere on the page. */
  chooseStep(t: number): void;
  /** The LSTM's memory span after step t, as the memory panel draws it. */
  lstmSpan(t: number): { values: Float32Array; target: number };
  colors(): ChartColors;
  /** How Generate samples: the page's temperature, from the five likeliest characters. */
  readonly settings: SamplingSettings;
  /** The gateless network's files for the page's language, without .json and .bin. */
  readonly modelBase: string;
}

export interface Gates {
  /** Keeps the gateless network on the LSTM's characters, after the LSTM has read them. */
  sync(indices: ArrayLike<number>): void;
  /** Redraws the character rows from step `from` on, and every chart. */
  render(from: number): void;
  /** Redraws what depends on the focused step. */
  renderStep(): void;
  /** Moves the step marker and the selection. */
  markStep(): void;
  /** Scrolls the section's characters to the end of the text. */
  followEnd(): void;
}

type Network = 'lstm' | 'rnn';
/** One traced guess on this text: each network's share of the pull from each distance back (empty if nothing pulled), once computed. */
interface Sample { lstm: Float64Array | null; rnn: Float64Array | null }

/** The quotation feature the trainer correlates cells with: quotes toggle it, a new paragraph ends it. */
function insideQuotation(text: ArrayLike<string>): Uint8Array {
  const inside = new Uint8Array(text.length);
  let quote = 0;
  for (let t = 0; t < text.length; t++) {
    if (text[t] === '"') quote = 1 - quote;
    if (text[t] === '\n') quote = 0;
    inside[t] = quote;
  }
  return inside;
}

/** Pearson correlation, or NaN when either series is constant. */
function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : Number.NaN;
}

const share = (count: number, of: number) => `${Math.round((100 * count) / of)}%`;

/** Two networks lean on a distance "alike" while their average shares there are within this factor of each other. */
const ALIKE = 1.5;
/** How many times one share is another's: 10, 110, 1,200. */
const times = (ratio: number) => Number(ratio.toPrecision(2)).toLocaleString('en-US');

export function createGates(host: GatesHost): Gates {
  const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
  const panel = $('gates-panel');
  const status = $('gates-status');
  const timeline = $('gates-timeline');
  const track = $('gates-track');
  const rows: Record<Network, HTMLElement> = { lstm: $('gates-lstm-chars'), rnn: $('gates-rnn-chars') };
  const spans: Record<Network, HTMLCanvasElement> = { lstm: $('gates-lstm-span'), rnn: $('gates-rnn-span') };
  const notes: Record<Network, HTMLElement> = { lstm: $('gates-lstm-note'), rnn: $('gates-rnn-note') };
  const marker = $('gates-marker');
  const quoteChart = $<HTMLCanvasElement>('quote-chart');
  const fadeChart = $<SVGSVGElement>('fade-chart');
  const vocabulary = Array.from(host.meta.vocabulary);

  let model: CharRnn | null = null;
  let run: CharRnnRun | null = null;
  let loading = false;
  let visible = false;
  /** The step views wait for the section to come into view. */
  let stepStale = true;
  /** The gateless network's memory span for the focused step, kept while the text and step stay the same. */
  let span: { key: string; values: Float32Array; target: number } | null = null;
  /** The text up to the step whose spans are drawn, or null when none are. */
  let shownKey: string | null = null;
  /** This text's traced guesses, by step: each network's share of the pull from each distance back. */
  const samples = new Map<number, Sample>();
  let fadeTask = 0;
  /** The fade chart's lines and scales, built once per size. */
  let fadeLines: Record<string, SVGPathElement> | null = null;
  let fadeScale: { x: (d: number) => number; y: (value: number) => number; bottom: number } | null = null;
  let fadeWidth = 0;

  const runs = (): Record<Network, CharRun> => ({ lstm: host.run, rnn: run });
  const prefixKey = (t: number) => Array.from(host.run.indices.subarray(0, t + 1)).join(',');
  const references = (): Record<Network, HeldOutReference | undefined> => ({ lstm: host.meta.reference, rnn: model?.meta.reference });
  const quoteCell = (meta: CharLstmMeta | CharRnnMeta): NotableCell | undefined => meta?.notableCells?.find(entry => entry.feature === 'quote');
  /**
   * The quotation views need an LSTM cell that clearly follows quotations, as the English
   * one does (r = -0.74). The Hindi books mark dialogue less consistently, and no Hindi
   * cell does (best r = 0.16), so there the views are left out rather than overstated.
   */
  const quotesShown = () => Math.abs(quoteCell(host.meta)?.correlation ?? 0) >= QUOTE_CORRELATION && !!quoteCell(model?.meta);

  /* ---------- Loading ---------- */

  async function load() {
    if (loading || model) return;
    loading = true;
    try {
      const [meta, weights] = await Promise.all([
        fetch(`${host.modelBase}.json`).then(response => {
          if (!response.ok) throw new Error(`${response.status}`);
          return response.json() as Promise<CharRnnMeta>;
        }),
        fetch(`${host.modelBase}.bin`).then(response => {
          if (!response.ok) throw new Error(`${response.status}`);
          return response.arrayBuffer();
        }),
      ]);
      const parsed = parseCharRnn(meta, weights);
      // Both networks read the same characters, one step per character.
      if (parsed.meta.vocabulary !== host.meta.vocabulary) throw new Error('The networks have different vocabularies.');
      model = parsed;
    } catch {
      status.textContent = 'The network without gates could not load. Reload the page to try again.';
      status.dataset.kind = 'error';
      panel.setAttribute('aria-busy', 'false');
      return;
    }
    run = new CharRnnRun(model);
    run.setIndices(host.run.indices.subarray(0, host.run.length));
    describeModels();
    status.hidden = true;
    panel.classList.remove('is-loading');
    panel.setAttribute('aria-busy', 'false');
    render(0);
    renderStep();
  }

  /** What the metadata says about both networks: sizes, held-out scores, quotation cells. */
  function describeModels() {
    $('gates-units').textContent = String(model.hiddenSize);
    $('gates-validation-bits').textContent = model.meta.validationBitsPerChar.toFixed(2);
    const [lstm, rnn] = [host.meta.reference, model.meta.reference];
    if (lstm && rnn) {
      const exact = (value: number) => `${(value * 100).toFixed(1)}%`;
      $('scores-held-out').textContent = `On text neither trained on, the LSTM’s first guess is right ${exact(lstm.accuracy.firstGuess)} of the time `
        + `and the gateless network’s ${exact(rnn.accuracy.firstGuess)}; within three guesses, ${exact(lstm.accuracy.topThree)} and ${exact(rnn.accuracy.topThree)}.`;
    }
    const cells = { lstm: quoteCell(host.meta), rnn: quoteCell(model.meta) };
    const evidence = $('quote-evidence');
    const shown = quotesShown();
    evidence.hidden = !shown;
    quoteChart.hidden = !shown;
    $('quote-note').parentElement.hidden = !shown;
    const callout = document.getElementById('quote-callout');
    if (callout) callout.hidden = !shown;
    if (!shown) return;
    $('quote-lstm-cell').textContent = `cell ${cells.lstm.cell + 1}`;
    $('quote-rnn-cell').textContent = `unit ${cells.rnn.cell + 1}`;
    for (const network of ['lstm', 'rnn'] as const) {
      const r = cells[network].correlation;
      $(`quote-${network}-r`).textContent = correlationText(r);
      $(`quote-${network}-bar`).style.setProperty('--strength', String(Math.min(1, Math.abs(r))));
    }
    $('quote-summary').textContent = `With gates, cell ${cells.lstm.cell + 1} of the LSTM follows whether the text is inside a quotation `
      + `(r = ${correlationText(cells.lstm.correlation)} on text it never trained on). Without gates, the closest of the ${model.hiddenSize} units, `
      + `unit ${cells.rnn.cell + 1}, reaches only r = ${correlationText(cells.rnn.correlation)}.`;
  }

  /* ---------- Keeping up with the text ---------- */

  function sync(indices: ArrayLike<number>) {
    if (!run) return;
    const length = run.length;
    const kept = run.setIndices(indices);
    for (const t of [...samples.keys()]) if (t >= kept) samples.delete(t);
    if (kept !== length || run.length !== length) clearContinuations();
  }

  function render(from: number) {
    if (!run) return;
    const n = run.length;
    track.style.setProperty('--steps', String(n));
    panel.classList.toggle('is-empty', n === 0);
    renderRow('lstm', from);
    renderRow('rnn', from);
    renderScores();
    $<HTMLButtonElement>('continue-button').disabled = n === 0;
    paintQuotes();
    renderFade();
    scheduleFade();
  }

  /** One row of characters tinted by how surprising each was to one network, on the memory panel's scale. */
  function renderRow(network: Network, from: number) {
    const source = runs()[network];
    const other = runs()[network === 'lstm' ? 'rnn' : 'lstm'];
    const list = rows[network];
    const n = source.length;
    const items = list.children as HTMLCollectionOf<HTMLElement>;
    for (let t = Math.min(from, items.length); t < n; t++) {
      let item = items[t];
      if (!item) {
        item = document.createElement('li');
        item.className = 'char-cell';
        item.dataset.index = String(t);
        if (network === 'rnn') {
          item.setAttribute('role', 'option');
          item.tabIndex = -1;
        }
        list.append(item);
      }
      const character = vocabulary[source.indices[t]];
      item.textContent = character === ' ' ? '' : glyph(character);
      item.classList.toggle('is-space', character === ' ');
      item.classList.toggle('is-break', character === '\n');
      const bits = source.surprisal(t);
      if (Number.isNaN(bits)) delete item.dataset.surprise;
      else item.dataset.surprise = String(surpriseLevel(bits));
      const name = characterName(character) === character ? `“${character}”` : characterName(character);
      const bitsText = (value: number) => `${value.toFixed(1)} bits`;
      if (Number.isNaN(bits)) {
        item.title = `${name}: nothing read before it, so nothing predicted it`;
      } else if (network === 'rnn') {
        item.title = `${name}: ${bitsText(bits)} of surprise to the gateless network, ${bitsText(other.surprisal(t))} to the LSTM`;
      } else {
        item.title = `${name}: ${bitsText(bits)} of surprise to the LSTM, ${bitsText(other.surprisal(t))} to the gateless network`;
      }
      if (network === 'rnn') {
        item.setAttribute('aria-label', Number.isNaN(bits) ? `${characterName(character)}, first character`
          : `${characterName(character)}, ${bitsText(bits)} of surprise without gates, ${bitsText(other.surprisal(t))} with`);
      }
    }
    while (items.length > n) items[items.length - 1].remove();
  }

  /** How often each network's first guess, and its top three, held the next character. */
  function renderScores() {
    const n = run.length;
    for (const network of ['lstm', 'rnn'] as const) {
      const { guesses, first, three } = guessScores(runs()[network]);
      $(`score-${network}-first`).textContent = guesses ? share(first, guesses) : '–';
      $(`score-${network}-three`).textContent = guesses ? share(three, guesses) : '–';
    }
    $('gates-scores').classList.toggle('is-empty', n < 2);
  }

  /* ---------- The focused step ---------- */

  function markStep() {
    const t = host.focusStep();
    for (const list of Object.values(rows)) {
      list.querySelectorAll('[aria-selected="true"]').forEach(item => item.setAttribute('aria-selected', 'false'));
    }
    rows.rnn.querySelectorAll<HTMLElement>('[tabindex="0"]').forEach(item => { item.tabIndex = -1; });
    const shown = Boolean(run) && t >= 0 && t < run.length;
    marker.hidden = !shown;
    if (!shown) return;
    rows.lstm.children[t]?.setAttribute('aria-selected', 'true');
    const item = rows.rnn.children[t] as HTMLElement | undefined;
    if (item) {
      item.setAttribute('aria-selected', 'true');
      item.tabIndex = 0;
    }
    marker.style.setProperty('--at', String(t));
  }

  /** In words, what shaped one network's guess after step t. */
  function describeSpan(source: CharRun, t: number, values: Float32Array, target: number): string {
    const c = vocabulary[target];
    const guess = c === ' ' ? 'a space' : c === '\n' ? 'a new paragraph' : `“${c}”`;
    const certainty = source.probabilities[t * source.model.inputSize + target];
    if (t === 0) return `guesses ${guess} from the first character alone`;
    if (certainty >= NEARLY_CERTAIN) return `is ${percent(certainty)} sure of ${guess}, so no single character moves it much`;
    return `guesses ${guess}: half of what shaped it came from the last ${characters(spanReach(values, t, .5))}, `
      + `90% from the last ${characters(spanReach(values, t, .9))}`;
  }

  /** Empties both span rows, so they never describe another step than the focused one. */
  function clearSpans() {
    shownKey = null;
    for (const network of ['lstm', 'rnn'] as const) {
      paintSpanBars(spans[network], null, -1, 0, host.colors());
      notes[network].textContent = '';
      spans[network].setAttribute('aria-label', `What shaped the ${network === 'lstm' ? 'LSTM' : 'gateless network'}’s guess`);
    }
  }

  /**
   * Both networks' memory spans for the focused step, aligned to the same characters.
   * They are traced only while the section is on screen; until then the rows stay empty.
   */
  function renderStep() {
    markStep();
    if (!run) return;
    const t = host.focusStep();
    const key = t < 0 ? null : prefixKey(t);
    if (!visible || key === null) {
      stepStale = !visible;
      if (key !== shownKey) clearSpans();
      return;
    }
    stepStale = false;
    const colors = host.colors();
    if (span?.key !== key) {
      const target = run.topCharacter(t);
      span = { key, values: run.influence(t, target, SPAN_REACH), target };
    }
    const shown = { lstm: host.lstmSpan(t), rnn: span };
    const names = { lstm: 'The LSTM', rnn: 'The gateless network' };
    for (const network of ['lstm', 'rnn'] as const) {
      const { values, target } = shown[network];
      paintSpanBars(spans[network], values, t, run.length, colors);
      const reading = describeSpan(runs()[network], t, values, target);
      notes[network].textContent = reading;
      spans[network].setAttribute('aria-label', `${names[network]} ${reading}`);
    }
    shownKey = key;
  }

  /* ---------- Quotations ---------- */

  /**
   * Each network's quotation cell over the text, scaled to the text's range and turned
   * so that up means inside, over bands where the text is inside a quotation.
   */
  function paintQuotes() {
    const { context, width, height } = prepareCanvas(quoteChart);
    const n = run.length;
    const cells = { lstm: quoteCell(host.meta), rnn: quoteCell(model.meta) };
    const note = $('quote-note');
    if (!n || !quotesShown()) { note.textContent = ''; return; }
    const colors = host.colors();
    const text = Array.from(run.indices.subarray(0, n), index => vocabulary[index]);
    const inside = insideQuotation(text);
    const step = width / n;
    context.fillStyle = colors.well;
    for (let t = 0; t < n; t++) {
      if (!inside[t]) continue;
      let end = t;
      while (end + 1 < n && inside[end + 1]) end++;
      context.fillRect(t * step, 0, (end + 1 - t) * step, height);
      t = end;
    }
    const traces = {
      lstm: Float64Array.from({ length: n }, (_, t) => host.run.cells[t * host.run.model.hiddenSize + cells.lstm.cell]),
      rnn: Float64Array.from({ length: n }, (_, t) => run.hidden[t * model.hiddenSize + cells.rnn.cell]),
    };
    const pad = 5;
    for (const network of ['rnn', 'lstm'] as const) {
      const sign = Math.sign(cells[network].correlation) || 1;
      let low = Infinity;
      let high = -Infinity;
      for (const value of traces[network]) { low = Math.min(low, sign * value); high = Math.max(high, sign * value); }
      const range = Math.max(high - low, 1e-6);
      const y = (value: number) => pad + (1 - (sign * value - low) / range) * (height - 2 * pad);
      context.strokeStyle = network === 'lstm' ? colors.positive : colors.ink;
      context.lineWidth = network === 'lstm' ? 2 : 1.5;
      context.setLineDash(network === 'lstm' ? [] : [4, 3]);
      context.lineJoin = 'round';
      context.beginPath();
      for (let t = 0; t < n; t++) {
        if (t === 0) context.moveTo((t + .5) * step, y(traces[network][t]));
        else context.lineTo((t + .5) * step, y(traces[network][t]));
      }
      context.stroke();
    }
    context.setLineDash([]);
    const r = { lstm: correlation(traces.lstm, inside), rnn: correlation(traces.rnn, inside) };
    note.textContent = Number.isNaN(r.lstm) || Number.isNaN(r.rnn)
      ? 'the LSTM’s quotation cell and the gateless network’s closest unit; add a quotation to compare them'
      : `on this text, r = ${correlationText(r.lstm)} for the LSTM’s cell ${cells.lstm.cell + 1} and ${correlationText(r.rnn)} for the gateless network’s unit ${cells.rnn.cell + 1}, each drawn so that up means inside the shaded quotes`;
    quoteChart.setAttribute('aria-label', `The LSTM’s quotation cell and the gateless network’s closest unit over the text, with the quoted passages shaded: ${note.textContent}.`);
  }

  /* ---------- How fast memory fades ---------- */

  /** This text's average share of the pull from each distance back, or NaN where too few guesses reach. */
  function textCurve(network: Network): Float64Array | null {
    const traced = [...samples.values()].filter(sample => sample.lstm?.length && sample.rnn?.length);
    if (run.length < FADE_MIN_TEXT || traced.length < FADE_MIN_GUESSES) return null;
    const curve = new Float64Array(FADE_DISTANCE).fill(Number.NaN);
    for (let d = 1; d <= FADE_DISTANCE; d++) {
      let sum = 0;
      let count = 0;
      for (const sample of traced) {
        const shares = sample[network];
        if (shares.length < d) continue;
        sum += shares[d - 1];
        count++;
      }
      if (count >= FADE_MIN_GUESSES) curve[d - 1] = sum / count;
    }
    return curve;
  }

  /**
   * Traces one more of this text's guesses, one network at a time, while the page is
   * idle and the section on screen. A trace takes a few tens of milliseconds, so it
   * waits for an idle period with room for it.
   */
  function scheduleFade() {
    if (fadeTask || !visible || !run || run.length < FADE_MIN_TEXT) return;
    // Safari has no idle callbacks; a short timeout stands in.
    const next = (callback: (deadline?: IdleDeadline) => void) => (typeof window.requestIdleCallback === 'function'
      ? window.requestIdleCallback(callback)
      : window.setTimeout(callback, 40));
    fadeTask = next(deadline => {
      fadeTask = 0;
      if (!visible || !run) return;
      if (deadline && deadline.timeRemaining() < FADE_IDLE_MS) { scheduleFade(); return; }
      let t = -1;
      for (let s = FADE_FIRST; s < run.length; s += FADE_EVERY) {
        const sample = samples.get(s);
        if (!sample?.lstm || !sample.rnn) { t = s; break; }
      }
      if (t < 0) return;
      const sample = samples.get(t) ?? { lstm: null, rnn: null };
      samples.set(t, sample);
      const network: Network = sample.lstm ? 'rnn' : 'lstm';
      const source = runs()[network];
      const shares = new Float64Array(Math.min(SPAN_REACH, t + 1));
      sample[network] = addInfluenceShares(source.influence(t, source.topCharacter(t), SPAN_REACH), t, shares) ? shares : new Float64Array(0);
      if (network === 'rnn') renderFade();
      scheduleFade();
    });
  }

  /** The chart's frame, grid and lines, rebuilt when its width changes. */
  function buildFade(width: number, height: number) {
    fadeWidth = width;
    fadeChart.setAttribute('viewBox', `0 0 ${width} ${height}`);
    fadeChart.replaceChildren();
    const margin = { top: 10, right: 14, bottom: 30, left: 50 };
    const x = (d: number) => margin.left + ((d - 1) / (FADE_DISTANCE - 1)) * (width - margin.left - margin.right);
    const decades = Math.round(-Math.log10(FADE_FLOOR));
    const y = (value: number) => margin.top + (-Math.log10(value) / decades) * (height - margin.top - margin.bottom);
    const add = (name: string, attributes: Record<string, string | number>, parent: Element = fadeChart, text?: string) => {
      const element = document.createElementNS(SVG, name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
      if (text !== undefined) element.textContent = text;
      parent.append(element);
      return element;
    };
    const grid = add('g', { class: 'fade-grid' });
    for (let k = 0; k <= decades; k++) {
      const value = 10 ** -k;
      add('line', { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value) }, grid);
      const label = value >= .01 ? `${value * 100}%` : `${(value * 100).toFixed(k - 2)}%`;
      add('text', { x: margin.left - 8, y: y(value), dy: '.32em', 'text-anchor': 'end', class: 'fade-tick' }, fadeChart, label);
    }
    for (const d of [1, 50, 100, 150, 200].filter(d => d <= FADE_DISTANCE)) {
      add('line', { x1: x(d), x2: x(d), y1: height - margin.bottom, y2: height - margin.bottom + 4, class: 'fade-axis' }, fadeChart);
      add('text', { x: x(d), y: height - margin.bottom + 17, 'text-anchor': 'middle', class: 'fade-tick' }, fadeChart, String(d));
    }
    add('text', { x: width - margin.right, y: height - 2, 'text-anchor': 'end', class: 'fade-tick fade-title' }, fadeChart, 'characters back');
    const clip = `fade-clip-${Math.round(width)}`;
    add('rect', { x: margin.left, y: margin.top - 2, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom + 2 },
      add('clipPath', { id: clip }, add('defs', {})));
    const lines = add('g', { 'clip-path': `url(#${clip})` });
    fadeLines = {};
    for (const kind of ['reference', 'text']) {
      for (const network of ['lstm', 'rnn'] as const) {
        fadeLines[`${kind}-${network}`] = add('path', { class: `fade-line is-${kind} is-${network}` }, lines) as SVGPathElement;
      }
    }
    return { x, y, bottom: height - margin.bottom };
  }

  /** A fade curve as a path: shares below the floor run off the bottom of the chart and stop. */
  function fadePath(curve: ArrayLike<number> | null): string {
    if (!curve || !fadeScale) return '';
    const { x, y, bottom } = fadeScale;
    let path = '';
    let drawing = false;
    for (let d = 1; d <= Math.min(FADE_DISTANCE, curve.length); d++) {
      const value = curve[d - 1];
      if (Number.isNaN(value)) { drawing = false; continue; }
      if (!(value >= FADE_FLOOR)) {
        if (drawing) path += `L${x(d).toFixed(1)},${(bottom + 6).toFixed(1)}`;
        break;
      }
      path += `${drawing ? 'L' : 'M'}${x(d).toFixed(1)},${y(value).toFixed(1)}`;
      drawing = true;
    }
    return path;
  }

  /**
   * The held-out curves in words: how far back the networks lean on the text alike, and
   * how differently they fade beyond that, measured 100 and 150 characters back.
   */
  function describeFade(lstm: HeldOutReference['memorySpan'] | undefined, rnn: HeldOutReference['memorySpan'] | undefined): string {
    if (!lstm || !rnn) return '';
    const share = (curve: number[], d: number) => curve[d - 1];
    let alike = 0;
    while (alike < FADE_DISTANCE && share(lstm.byDistance, alike + 1) < ALIKE * share(rnn.byDistance, alike + 1)
      && share(rnn.byDistance, alike + 1) < ALIKE * share(lstm.byDistance, alike + 1)) alike++;
    const near = `half of a guess’s pull comes from the last ${characters(lstm.halfReach)} and 90% from the last ${lstm.ninetyReach} with gates, `
      + `the last ${characters(rnn.halfReach)} and the last ${rnn.ninetyReach} without`;
    if (alike < 20 || alike >= 150 || !(share(rnn.byDistance, 150) > 0)) return `On text neither network trained on, ${near}.`;
    // Without gates, the share shrinks by a steady factor beyond the stretch they share: a straight line on the chart.
    const step = (share(rnn.byDistance, 150) / share(rnn.byDistance, 50)) ** (1 / 100);
    const shrink = Math.round((1 - step) * 100);
    const ratio = (d: number) => share(lstm.byDistance, d) / share(rnn.byDistance, d);
    return `On text neither network trained on, the two lean on the last ${Math.floor(alike / 10) * 10} or so characters almost alike: ${near}. `
      + `Further back they part. Without gates, a character’s share shrinks by about ${shrink}% with every character that follows; `
      + `the LSTM’s fades far more slowly, so 100 characters back it is ${times(ratio(100))} times larger, and 150 back ${times(ratio(150))} times.`;
  }

  function renderFade() {
    if (!run) return;
    const { width, height } = fadeChart.getBoundingClientRect();
    if (!width || !height) return;
    if (!fadeLines || Math.abs(width - fadeWidth) > .5) fadeScale = buildFade(width, height);
    const reference = references();
    const text = { lstm: textCurve('lstm'), rnn: textCurve('rnn') };
    for (const network of ['lstm', 'rnn'] as const) {
      fadeLines[`reference-${network}`].setAttribute('d', fadePath(reference[network]?.memorySpan.byDistance ?? null));
      fadeLines[`text-${network}`].setAttribute('d', fadePath(text[network]));
    }
    const traced = [...samples.values()].filter(sample => sample.lstm?.length && sample.rnn?.length).length;
    $('fade-text-key').hidden = !text.lstm;
    $('fade-text-key').title = `${traced} guesses traced on this text`;
    $('fade-text-hint').hidden = run.length >= FADE_MIN_TEXT;
    const summary = $('fade-summary');
    summary.textContent = describeFade(reference.lstm?.memorySpan, reference.rnn?.memorySpan);
    fadeChart.setAttribute('aria-label', `How a guess’s pull fades with distance, on a logarithmic scale. ${summary.textContent}`);
  }

  /* ---------- Continuing the text ---------- */

  /** CONTINUE_LENGTH characters one network writes after the text, sampled as Generate samples; the runs stay unchanged. */
  function continuation(network: Network): string {
    const source = runs()[network];
    const n = source.length;
    const H = source.model.hiddenSize;
    const V = source.model.inputSize;
    const logits = source.logits.slice((n - 1) * V, n * V);
    let h = source.hidden.slice((n - 1) * H, n * H);
    let hNext = new Float32Array(H);
    let read: (index: number) => void;
    if (network === 'lstm') {
      const lstm = host.run.model;
      let c = host.run.cells.slice((n - 1) * H, n * H);
      let cNext = new Float32Array(H);
      const gates = new Float32Array(4 * H);
      read = index => {
        lstmCell(lstm, lstm.inputWeights, index * 4 * H, h, 0, c, 0, gates, 0, cNext, 0, hNext, 0);
        [c, cNext] = [cNext, c];
      };
    } else {
      read = index => rnnCell(model, model.inputWeights, index * H, h, 0, hNext, 0);
    }
    let text = '';
    for (let k = 0; k < CONTINUE_LENGTH; k++) {
      const next = sample(applySampling(logits, host.settings));
      text += vocabulary[next];
      read(next);
      [h, hNext] = [hNext, h];
      readout(source.model, h, 0, logits, 0);
    }
    return text;
  }

  function writeContinuations() {
    const n = run?.length ?? 0;
    if (!n) return;
    let tail = '';
    for (let t = Math.max(0, n - 28); t < n; t++) tail += vocabulary[run.indices[t]];
    for (const network of ['lstm', 'rnn'] as const) {
      const known = Object.assign(document.createElement('span'), { className: 'known', textContent: `${n > 28 ? '…' : ''}${tail}` });
      $(`continue-${network}`).replaceChildren(known, continuation(network));
    }
    $('continue-results').hidden = false;
  }

  function clearContinuations() {
    $('continue-results').hidden = true;
    for (const network of ['lstm', 'rnn'] as const) $(`continue-${network}`).replaceChildren();
  }

  /* ---------- Interaction ---------- */

  $('continue-button').addEventListener('click', writeContinuations);

  for (const list of Object.values(rows)) {
    list.addEventListener('pointerover', event => {
      const item = (event.target as HTMLElement).closest<HTMLElement>('.char-cell');
      if (item) host.chooseStep(Number(item.dataset.index));
    });
  }
  rows.rnn.addEventListener('focusin', event => {
    const item = (event.target as HTMLElement).closest<HTMLElement>('.char-cell');
    if (item) host.chooseStep(Number(item.dataset.index));
  });
  rows.rnn.addEventListener('keydown', event => {
    const t = host.focusStep();
    const n = run?.length ?? 0;
    const moves: Record<string, number> = { ArrowLeft: t - 1, ArrowRight: t + 1, Home: 0, End: n - 1, PageUp: t - 10, PageDown: t + 10 };
    if (!(event.key in moves) || t < 0) return;
    event.preventDefault();
    const next = Math.max(0, Math.min(n - 1, moves[event.key]));
    host.chooseStep(next);
    const item = rows.rnn.children[next] as HTMLElement;
    item.focus({ preventScroll: true });
    item.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  });

  new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    if (!visible || !run) return;
    if (stepStale) renderStep();
    scheduleFade();
  }).observe(panel);
  const loader = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    loader.disconnect();
    load();
  }, { rootMargin: LOAD_MARGIN });
  loader.observe(panel);
  // The charts are drawn at the screen's resolution and their boxes' widths.
  const resized = new ResizeObserver(() => {
    if (!run) return;
    paintQuotes();
    renderFade();
    renderStep();
  });
  resized.observe(timeline);
  resized.observe(fadeChart);

  return {
    sync,
    render,
    renderStep,
    markStep,
    followEnd: () => { timeline.scrollLeft = timeline.scrollWidth; },
  };
}
