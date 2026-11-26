/**
 * The Recurrent page's character-level language models. Both read text one character
 * at a time, carry a memory forward, and predict the next character.
 *
 * The LSTM, the page's main network:
 *
 *   x_t                                one-hot character (V values, V = vocabulary size)
 *   z_t = W_x[x_t] + U·h_{t-1} + b     4H pre-activations, gates in the order
 *                                      input, forget, cell candidate, output
 *   i_t = σ(z_i)     input gate: how much of the candidate to write
 *   f_t = σ(z_f)     forget gate: how much of the old memory to keep
 *   g_t = tanh(z_g)  cell candidate: what could be written
 *   o_t = σ(z_o)     output gate: how much of the memory to reveal
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t    cell state, the memory
 *   h_t = o_t ⊙ tanh(c_t)              hidden state
 *   p_t = softmax(W_y·h_t + b_y)       probabilities of the character after x_t
 *
 * The gateless network it is compared with, an Elman RNN whose hidden state is its
 * whole memory, squashed through the same tanh at every step:
 *
 *   h_t = tanh(W_x[x_t] + U·h_{t-1} + b)   H values
 *   p_t = softmax(W_y·h_t + b_y)
 *
 * with h_{-1} = c_{-1} = 0, so nothing is predicted before the first character.
 *
 * `char-lstm-v1.bin` holds little-endian float32 values in this order (shapes are
 * [rows][columns], row-major):
 *   inputWeights      [V][4H]  row x is W_x[x], the gate inputs a character adds
 *   recurrentWeights  [4H][H]  row = gate unit (gate-major: i, f, g, o), column = h_{t-1} unit
 *   bias              [4H]
 *   outputWeights     [V][H]
 *   outputBias        [V]
 * `char-rnn-v1.bin` has the same tensors without the gates: inputWeights [V][H],
 * recurrentWeights [H][H] (row = new unit, column = h_{t-1} unit), bias [H],
 * outputWeights [V][H], outputBias [V]. Each `.json` holds the vocabulary, the sizes
 * and the layout.
 *
 * CharLstmRun and CharRnnRun keep every step's activations for drawing and reuse the
 * longest shared prefix when the text changes. Their influence() backpropagates
 * through time from those activations: the exact gradient of log p_t(target) with
 * respect to each earlier one-hot input, times that input. encodeText maps typed text
 * onto the vocabulary the way the training text was normalized.
 *
 * Pure TypeScript with no DOM dependencies and no imports: it runs in the browser
 * and in Node. scripts/train-char-lstm.mjs imports lstmCell, rnnCell, readout and
 * softmaxInto from this file (Node strips the types), so training, evaluation and the
 * page share one forward pass. Keep the syntax erasable: no enums, namespaces or
 * parameter properties. Every stored activation is a float32, and each step reads only
 * stored values, so a run is reproducible bit for bit however it is split into calls.
 */

export const GATE_ORDER = ['input', 'forget', 'cell', 'output'] as const;
export type GateName = (typeof GATE_ORDER)[number];
export const MODEL_FORMAT = 'char-lstm-v1';
/** The gateless network's file format. */
export const RNN_FORMAT = 'char-rnn-v1';

export interface TensorLayout {
  name: 'inputWeights' | 'recurrentWeights' | 'bias' | 'outputWeights' | 'outputBias';
  shape: number[];
  /** Offset in float32 values from the start of the file. */
  offset: number;
}

export interface CorpusBook {
  title: string;
  author: string;
  gutenbergId: number;
}

/** A memory cell whose state follows a text feature on held-out text. */
export interface NotableCell {
  feature: string;
  description: string;
  /** 0-based. */
  cell: number;
  correlation: number;
}

/**
 * Statistics measured on the held-out text (the last 5% of every book) with the
 * shipped weights, so the page can compare the two networks on text neither trained on.
 */
export interface HeldOutReference {
  note: string;
  /** How often the character after each step was the likeliest one, or among the three likeliest. */
  accuracy: { guesses: number; firstGuess: number; topThree: number };
  /** How far back the characters that shaped a guess lie, measured as on the page (see influence()). */
  memorySpan: {
    /** Guesses traced: evenly spaced, each with at least `reach` characters before it. */
    guesses: number;
    /** Characters traced back from each guess. */
    reach: number;
    /** Medians over the guesses of how many of the latest characters hold half, and 90%, of the pull (spanReach). */
    halfReach: number;
    ninetyReach: number;
    /** The average share of a guess's pull from d characters back, d = 1 … reach (addInfluenceShares); sums to 1. */
    byDistance: number[];
  };
}

/** Contents of `char-lstm-v1.json`. */
export interface CharLstmMeta {
  format: typeof MODEL_FORMAT;
  /** The characters the model knows, in index order. */
  vocabulary: string;
  inputSize: number;
  hiddenSize: number;
  gateOrder: string[];
  parameters: number;
  tensors: TensorLayout[];
  corpus: CorpusBook[];
  /** Bits per character on the held-out text, measured on the exported file. */
  validationBitsPerChar: number;
  /** For a few text features, the cell whose state correlates with it most on held-out text (0-based). */
  notableCells?: NotableCell[];
  reference?: HeldOutReference;
  [key: string]: unknown;
}

/** Contents of `char-rnn-v1.json`: like the LSTM's, without the gates. */
export interface CharRnnMeta {
  format: typeof RNN_FORMAT;
  vocabulary: string;
  inputSize: number;
  hiddenSize: number;
  parameters: number;
  tensors: TensorLayout[];
  corpus: CorpusBook[];
  validationBitsPerChar: number;
  /** For a few text features, the hidden unit whose state correlates with it most on held-out text (0-based). */
  notableCells?: NotableCell[];
  reference?: HeldOutReference;
  [key: string]: unknown;
}

/** What both networks share: the vocabulary, the sizes and the readout. */
export interface CharModel {
  /** Index → character. */
  vocabulary: string[];
  /** Character → index. */
  lookup: Map<string, number>;
  /** V, the vocabulary size. */
  inputSize: number;
  /** H, the number of memory cells (LSTM) or hidden units (gateless network). */
  hiddenSize: number;
  /** [V][H] */
  outputWeights: Float32Array;
  /** [V] */
  outputBias: Float32Array;
}

export interface CharLstm extends CharModel {
  meta: CharLstmMeta;
  /** [V][4H] */
  inputWeights: Float32Array;
  /** [4H][H] */
  recurrentWeights: Float32Array;
  /** [4H] */
  bias: Float32Array;
}

/** The gateless network. */
export interface CharRnn extends CharModel {
  meta: CharRnnMeta;
  /** [V][H] */
  inputWeights: Float32Array;
  /** [H][H] */
  recurrentWeights: Float32Array;
  /** [H] */
  bias: Float32Array;
}

function layoutOf(shapes: [TensorLayout['name'], number[]][]): TensorLayout[] {
  let offset = 0;
  return shapes.map(([name, shape]) => {
    const entry = { name, shape, offset };
    offset += shape.reduce((a, b) => a * b, 1);
    return entry;
  });
}

/** The tensors of a model with V characters and H cells, in file order. */
export function tensorLayout(inputSize: number, hiddenSize: number): TensorLayout[] {
  const gates = 4 * hiddenSize;
  return layoutOf([
    ['inputWeights', [inputSize, gates]],
    ['recurrentWeights', [gates, hiddenSize]],
    ['bias', [gates]],
    ['outputWeights', [inputSize, hiddenSize]],
    ['outputBias', [inputSize]],
  ]);
}

export function parameterCount(inputSize: number, hiddenSize: number): number {
  return 4 * hiddenSize * (inputSize + hiddenSize + 1) + inputSize * (hiddenSize + 1);
}

/** The tensors of a gateless network with V characters and H hidden units, in file order. */
export function rnnTensorLayout(inputSize: number, hiddenSize: number): TensorLayout[] {
  return layoutOf([
    ['inputWeights', [inputSize, hiddenSize]],
    ['recurrentWeights', [hiddenSize, hiddenSize]],
    ['bias', [hiddenSize]],
    ['outputWeights', [inputSize, hiddenSize]],
    ['outputBias', [inputSize]],
  ]);
}

export function rnnParameterCount(inputSize: number, hiddenSize: number): number {
  return hiddenSize * (inputSize + hiddenSize + 1) + inputSize * (hiddenSize + 1);
}

/** The vocabulary and sizes both formats share, checked; `kind` names the network in errors. */
function readVocabulary(kind: string, meta: CharLstmMeta | CharRnnMeta) {
  const vocabulary = Array.from(meta.vocabulary ?? '');
  const { inputSize, hiddenSize } = meta;
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) throw new Error(`Invalid ${kind} metadata: hidden size ${hiddenSize}.`);
  if (inputSize !== vocabulary.length || inputSize < 2) {
    throw new Error(`Invalid ${kind} metadata: input size ${inputSize} but ${vocabulary.length} characters.`);
  }
  const lookup = new Map(vocabulary.map((character, index) => [character, index]));
  if (lookup.size !== vocabulary.length) throw new Error(`Invalid ${kind} metadata: repeated characters in the vocabulary.`);
  return { vocabulary, lookup, inputSize, hiddenSize };
}

/** The tensors of `layout` from `buffer`, after checking the metadata describes exactly them. */
function readTensors(kind: string, meta: CharLstmMeta | CharRnnMeta, layout: TensorLayout[], count: number, buffer: ArrayBuffer) {
  if (JSON.stringify(meta.tensors) !== JSON.stringify(layout)) throw new Error(`Invalid ${kind} metadata: unexpected tensor layout.`);
  if (meta.parameters !== count) throw new Error(`Invalid ${kind} metadata: ${meta.parameters} parameters, expected ${count}.`);
  const bytes = buffer?.byteLength;
  if (bytes !== count * 4) {
    throw new Error(`Invalid ${kind} weights: expected ${count * 4} bytes (${count} float32 values) but got ${bytes}.`);
  }
  const view = new DataView(buffer);
  const tensors = {} as Record<TensorLayout['name'], Float32Array>;
  for (const { name, shape, offset } of layout) {
    const values = new Float32Array(shape.reduce((a, b) => a * b, 1));
    for (let i = 0; i < values.length; i++) {
      const value = view.getFloat32((offset + i) * 4, true);
      if (!Number.isFinite(value)) throw new Error(`Invalid ${kind} weights: ${name}[${i}] is ${value}.`);
      values[i] = value;
    }
    tensors[name] = values;
  }
  return tensors;
}

/**
 * Builds a model from its metadata and `char-lstm-v1.bin`. Values are copied, so the
 * buffer can be reused.
 * @throws Error if the metadata is inconsistent, the buffer has the wrong size, or a
 * value is NaN or infinite.
 */
export function parseCharLstm(meta: CharLstmMeta, buffer: ArrayBuffer): CharLstm {
  if (!meta || meta.format !== MODEL_FORMAT) throw new Error(`Invalid LSTM metadata: expected format ${MODEL_FORMAT}.`);
  const { vocabulary, lookup, inputSize, hiddenSize } = readVocabulary('LSTM', meta);
  if (JSON.stringify(meta.gateOrder) !== JSON.stringify(GATE_ORDER)) {
    throw new Error(`Invalid LSTM metadata: gate order must be ${GATE_ORDER.join(', ')}.`);
  }
  const tensors = readTensors('LSTM', meta, tensorLayout(inputSize, hiddenSize), parameterCount(inputSize, hiddenSize), buffer);
  return { meta, vocabulary, lookup, inputSize, hiddenSize, ...tensors };
}

/**
 * Builds the gateless network from its metadata and `char-rnn-v1.bin`, with the same
 * checks as parseCharLstm.
 * @throws Error if the metadata is inconsistent, the buffer has the wrong size, or a
 * value is NaN or infinite.
 */
export function parseCharRnn(meta: CharRnnMeta, buffer: ArrayBuffer): CharRnn {
  if (!meta || meta.format !== RNN_FORMAT) throw new Error(`Invalid RNN metadata: expected format ${RNN_FORMAT}.`);
  const { vocabulary, lookup, inputSize, hiddenSize } = readVocabulary('RNN', meta);
  const tensors = readTensors('RNN', meta, rnnTensorLayout(inputSize, hiddenSize), rnnParameterCount(inputSize, hiddenSize), buffer);
  return { meta, vocabulary, lookup, inputSize, hiddenSize, ...tensors };
}

/* ---------- Text ---------- */

/** Typographic characters written the way the training text writes them. */
const REPLACEMENTS: Record<string, string> = {
  '‘': "'", '’': "'", '‚': "'", '‛': "'", '′': "'", '´': "'", '`': "'",
  '“': '"', '”': '"', '„': '"', '‟': '"', '″': '"', '«': '"', '»': '"',
  '‐': '-', '‑': '-', '‒': '-', '–': '-', '−': '-', '—': '--', '―': '--',
  '…': '...', ' ': ' ', ' ': ' ', ' ': ' ', '\t': ' ',
  'Æ': 'Ae', 'æ': 'ae', 'Œ': 'Oe', 'œ': 'oe', 'ß': 'ss',
};
const COMBINING_MARKS = /[̀-ͯ]/g;

/**
 * Straight quotes, simple dashes, plain spaces and unaccented letters: the spelling
 * the corpus was normalized to before training, and what typed text is mapped to.
 */
export function simplifyCharacters(text: string): string {
  let out = '';
  for (const character of text.replace(/\r\n?/g, '\n')) {
    const replaced = REPLACEMENTS[character];
    if (replaced !== undefined) out += replaced;
    else if (character.charCodeAt(0) < 128) out += character;
    else out += character.normalize('NFKD').replace(COMBINING_MARKS, '');
  }
  return out;
}

export interface EncodedText {
  /** Vocabulary indices, one per step. */
  indices: number[];
  /** The text as the model reads it, one character per index. */
  text: string;
  /** Characters that could not be mapped to the vocabulary and were left out, each listed once. */
  skipped: string[];
}

/** Maps text to vocabulary indices, simplifying typography and accents and skipping anything else. */
export function encodeText(model: Pick<CharModel, 'lookup'>, text: string): EncodedText {
  const indices: number[] = [];
  const skipped: string[] = [];
  let read = '';
  for (const original of text.replace(/\r\n?/g, '\n')) {
    const known = model.lookup.get(original);
    if (known !== undefined) {
      indices.push(known);
      read += original;
      continue;
    }
    const simplified = simplifyCharacters(original);
    const mapped = Array.from(simplified).map(character => model.lookup.get(character));
    if (simplified && mapped.every(index => index !== undefined)) {
      indices.push(...mapped);
      read += simplified;
    } else if (!skipped.includes(original)) {
      skipped.push(original);
    }
  }
  return { indices, text: read, skipped };
}

/* ---------- Forward pass ---------- */

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * One LSTM step for the input `input[inputOffset..+4H]` (for a one-hot character,
 * the model's inputWeights row: pass `model.inputWeights` and `index * 4H`). Reads
 * the previous hidden and cell state and writes the activated gates (i, f, g, o,
 * H values each), the new cell state and the new hidden state at the given offsets.
 * Every written value is rounded to float32 before it is used, so later steps and
 * the backward pass see exactly what is stored.
 */
export function lstmCell(
  model: CharLstm,
  input: Float32Array, inputOffset: number,
  hPrev: Float32Array, hPrevOffset: number,
  cPrev: Float32Array, cPrevOffset: number,
  gates: Float32Array, gatesOffset: number,
  c: Float32Array, cOffset: number,
  h: Float32Array, hOffset: number,
): void {
  const H = model.hiddenSize;
  const G = 4 * H;
  const U = model.recurrentWeights;
  const b = model.bias;
  for (let j = 0; j < G; j++) {
    const row = j * H;
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    let k = 0;
    for (; k + 3 < H; k += 4) {
      s0 += U[row + k] * hPrev[hPrevOffset + k];
      s1 += U[row + k + 1] * hPrev[hPrevOffset + k + 1];
      s2 += U[row + k + 2] * hPrev[hPrevOffset + k + 2];
      s3 += U[row + k + 3] * hPrev[hPrevOffset + k + 3];
    }
    for (; k < H; k++) s0 += U[row + k] * hPrev[hPrevOffset + k];
    const z = b[j] + input[inputOffset + j] + ((s0 + s1) + (s2 + s3));
    gates[gatesOffset + j] = j >= 2 * H && j < 3 * H ? Math.tanh(z) : sigmoid(z);
  }
  for (let k = 0; k < H; k++) {
    const i = gates[gatesOffset + k];
    const f = gates[gatesOffset + H + k];
    const g = gates[gatesOffset + 2 * H + k];
    const o = gates[gatesOffset + 3 * H + k];
    const cell = Math.fround(f * cPrev[cPrevOffset + k] + i * g);
    c[cOffset + k] = cell;
    h[hOffset + k] = o * Math.tanh(cell);
  }
}

/**
 * One step of the gateless network, h = tanh(input + U·h_prev + b), for the input
 * `input[inputOffset..+H]` (for a one-hot character, the model's inputWeights row:
 * pass `model.inputWeights` and `index * H`). Writes the new hidden state, rounded to
 * float32, at `h[hOffset..+H]`, which must not overlap `hPrev`.
 */
export function rnnCell(
  model: CharRnn,
  input: Float32Array, inputOffset: number,
  hPrev: Float32Array, hPrevOffset: number,
  h: Float32Array, hOffset: number,
): void {
  const H = model.hiddenSize;
  const U = model.recurrentWeights;
  const b = model.bias;
  // Four units at a time, so each value of h_prev is read once for all four.
  let j = 0;
  for (; j + 3 < H; j += 4) {
    const r0 = j * H;
    const r1 = r0 + H;
    const r2 = r1 + H;
    const r3 = r2 + H;
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let k = 0; k < H; k++) {
      const x = hPrev[hPrevOffset + k];
      s0 += U[r0 + k] * x;
      s1 += U[r1 + k] * x;
      s2 += U[r2 + k] * x;
      s3 += U[r3 + k] * x;
    }
    h[hOffset + j] = Math.tanh(b[j] + input[inputOffset + j] + s0);
    h[hOffset + j + 1] = Math.tanh(b[j + 1] + input[inputOffset + j + 1] + s1);
    h[hOffset + j + 2] = Math.tanh(b[j + 2] + input[inputOffset + j + 2] + s2);
    h[hOffset + j + 3] = Math.tanh(b[j + 3] + input[inputOffset + j + 3] + s3);
  }
  for (; j < H; j++) {
    const row = j * H;
    let s = 0;
    for (let k = 0; k < H; k++) s += U[row + k] * hPrev[hPrevOffset + k];
    h[hOffset + j] = Math.tanh(b[j] + input[inputOffset + j] + s);
  }
}

/** Next-character scores W_y·h + b_y for the hidden state at `h[hOffset..+H]`. */
export function readout(
  model: Pick<CharModel, 'inputSize' | 'hiddenSize' | 'outputWeights' | 'outputBias'>,
  h: Float32Array, hOffset: number, logits: Float32Array, logitsOffset: number,
): void {
  const H = model.hiddenSize;
  const W = model.outputWeights;
  for (let v = 0; v < model.inputSize; v++) {
    const row = v * H;
    let sum = model.outputBias[v];
    for (let k = 0; k < H; k++) sum += W[row + k] * h[hOffset + k];
    logits[logitsOffset + v] = sum;
  }
}

/** Softmax of `length` scores into `out`. Returns log of the normalizer (log Σ e^score). */
export function softmaxInto(logits: ArrayLike<number>, offset: number, length: number, out: Float32Array, outOffset: number): number {
  let max = -Infinity;
  for (let i = 0; i < length; i++) max = Math.max(max, logits[offset + i]);
  let total = 0;
  for (let i = 0; i < length; i++) total += Math.exp(logits[offset + i] - max);
  for (let i = 0; i < length; i++) out[outOffset + i] = Math.exp(logits[offset + i] - max) / total;
  return max + Math.log(total);
}

/* ---------- Runs ---------- */

/** Everything the LSTM computed at one step, as views into the run's arrays. */
export interface StepTrace {
  /** Vocabulary index of the character read at this step. */
  index: number;
  character: string;
  input: Float32Array;
  forget: Float32Array;
  /** The cell candidate g. */
  candidate: Float32Array;
  output: Float32Array;
  /** Cell state c after this step. */
  cell: Float32Array;
  /** Hidden state h after this step. */
  hidden: Float32Array;
  /** Scores for the next character. */
  logits: Float32Array;
  /** Probabilities of the next character, summing to 1. */
  probabilities: Float32Array;
}

/** Everything the gateless network computed at one step, as views into the run's arrays. */
export interface RnnStepTrace {
  index: number;
  character: string;
  /** Hidden state h after this step: the network's whole memory. */
  hidden: Float32Array;
  logits: Float32Array;
  probabilities: Float32Array;
}

/** Letters, with the vowel signs and other marks Devanagari writes with them; English adds the apostrophe inside words. */
const LETTER = /[\p{L}\p{M}]/u;
const IN_WORD = /[\p{L}\p{M}']/u;

/**
 * A run of either network over a sequence, with every step's activations kept in flat
 * arrays (step-major) for drawing. `setIndices` reuses the longest shared prefix, so
 * typing or generating only computes the new steps. Subclasses compute the steps and
 * keep whatever else their network stores.
 */
export abstract class CharRun<M extends CharModel = CharModel> {
  readonly model: M;
  /** Steps computed. */
  length = 0;
  /** Vocabulary index read at each step. */
  indices = new Int32Array(0);
  /** [step][H] */
  hidden = new Float32Array(0);
  /** [step][V] */
  logits = new Float32Array(0);
  /** [step][V] */
  probabilities = new Float32Array(0);
  private capacity = 0;
  protected readonly zeros: Float32Array;

  constructor(model: M) {
    this.model = model;
    this.zeros = new Float32Array(model.hiddenSize);
  }

  /** Reads one more character. */
  abstract push(index: number): void;

  /**
   * The word the network would write after step t if it read `first` next and then
   * kept taking its likeliest character: `first` followed by every letter it then
   * expects, stopping before anything that isn't a letter or an apostrophe, after
   * `maxLength` characters at most. Just `first` when that isn't a letter. Leaves the
   * run unchanged.
   */
  abstract completeWord(t: number, first: number, maxLength?: number): number[];

  /**
   * How much each character read so far shaped the prediction after step t: for every
   * s ≤ t, the exact gradient of log p_t(target) with respect to the one-hot input x_s,
   * times x_s (so only the component of the character actually present remains),
   * computed by backpropagation through time from the stored activations. Positive
   * values pushed the prediction toward `target`, negative ones away from it.
   * `target` defaults to the most likely next character after step t. With `reach`,
   * only the last `reach` characters are traced (earlier ones stay 0), which bounds the
   * cost on long texts; the influence of characters that far back is negligible anyway.
   */
  abstract influence(t: number, target?: number, reach?: number): Float32Array;

  /** Grows the subclass's own arrays to `capacity` steps, keeping the first `length`. */
  protected abstract growState(capacity: number): void;

  /**
   * Makes the run hold exactly `indices`, keeping the steps of the longest shared
   * prefix and computing the rest. Returns how many steps were reused.
   */
  setIndices(indices: ArrayLike<number>): number {
    const { inputSize } = this.model;
    for (let t = 0; t < indices.length; t++) {
      const index = indices[t];
      if (!Number.isInteger(index) || index < 0 || index >= inputSize) throw new Error(`Unknown character index ${index}.`);
    }
    let shared = 0;
    while (shared < this.length && shared < indices.length && this.indices[shared] === indices[shared]) shared++;
    this.length = shared;
    this.reserve(indices.length);
    for (let t = shared; t < indices.length; t++) this.push(indices[t]);
    return shared;
  }

  /** The character the model finds most likely after step t. */
  topCharacter(t: number): number {
    this.check(t);
    const V = this.model.inputSize;
    let best = 0;
    for (let v = 1; v < V; v++) if (this.logits[t * V + v] > this.logits[t * V + best]) best = v;
    return best;
  }

  /**
   * How unexpected the character at step t was, in bits: −log₂ of the probability
   * the model gave it after step t − 1. NaN for the first step, which nothing predicts.
   */
  surprisal(t: number): number {
    this.check(t);
    if (t === 0) return Number.NaN;
    const V = this.model.inputSize;
    return -Math.log2(Math.max(this.probabilities[(t - 1) * V + this.indices[t]], 1e-30));
  }

  /** Checks an index and the step it would be read at, before anything is computed. */
  protected beginStep(index: number): number {
    if (!Number.isInteger(index) || index < 0 || index >= this.model.inputSize) throw new Error(`Unknown character index ${index}.`);
    const t = this.length;
    this.reserve(t + 1);
    this.indices[t] = index;
    return t;
  }

  /** Scores and probabilities of the next character from step t's hidden state, completing the step. */
  protected finishStep(t: number) {
    const V = this.model.inputSize;
    readout(this.model, this.hidden, t * this.model.hiddenSize, this.logits, t * V);
    softmaxInto(this.logits, t * V, V, this.probabilities, t * V);
    this.length = t + 1;
  }

  /** The gradient of log p_t(target) with respect to the hidden state h_t, back through W_y. */
  protected outputGradient(t: number, target: number): Float64Array {
    const H = this.model.hiddenSize;
    const V = this.model.inputSize;
    const Wy = this.model.outputWeights;
    const dh = new Float64Array(H);
    // d log p(target) / d logits = onehot(target) − p, then back through W_y.
    for (let v = 0; v < V; v++) {
      const d = (v === target ? 1 : 0) - this.probabilities[t * V + v];
      if (d === 0) continue;
      const row = v * H;
      for (let k = 0; k < H; k++) dh[k] += d * Wy[row + k];
    }
    return dh;
  }

  /** Steps a copy of the hidden state forward from step t, as completeWord describes. */
  protected readAhead(first: number, maxLength: number, step: (index: number, logits: Float32Array) => void): number[] {
    const { vocabulary, inputSize: V } = this.model;
    const word = [first];
    if (!LETTER.test(vocabulary[first])) return word;
    const logits = new Float32Array(V);
    let index = first;
    while (word.length < maxLength) {
      step(index, logits);
      let best = 0;
      for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
      if (!IN_WORD.test(vocabulary[best])) break;
      word.push(best);
      index = best;
    }
    // An apostrophe can close a quotation as well as join a word ("dogs'", "can't").
    while (word.length > 1 && vocabulary[word[word.length - 1]] === "'") word.pop();
    return word;
  }

  protected check(t: number) {
    if (!Number.isInteger(t) || t < 0 || t >= this.length) throw new Error(`Step ${t} is outside the run (${this.length} steps).`);
  }

  /** `array` grown to `capacity` rows of `width` values, keeping the computed steps. */
  protected grow(array: Float32Array, width: number, capacity: number) {
    const next = new Float32Array(capacity * width);
    next.set(array.subarray(0, this.length * width));
    return next;
  }

  private reserve(steps: number) {
    if (steps <= this.capacity) return;
    const capacity = Math.max(steps, Math.ceil(this.capacity * 1.5), 64);
    const H = this.model.hiddenSize;
    const V = this.model.inputSize;
    const indices = new Int32Array(capacity);
    indices.set(this.indices.subarray(0, this.length));
    this.indices = indices;
    this.hidden = this.grow(this.hidden, H, capacity);
    this.logits = this.grow(this.logits, V, capacity);
    this.probabilities = this.grow(this.probabilities, V, capacity);
    this.growState(capacity);
    this.capacity = capacity;
  }
}

/** A run of the LSTM, which also keeps every step's gates and cell state. */
export class CharLstmRun extends CharRun<CharLstm> {
  /** [step][4H]: activated gates i, f, g, o. */
  gates = new Float32Array(0);
  /** [step][H] */
  cells = new Float32Array(0);

  /** Reads one more character. */
  push(index: number): void {
    const t = this.beginStep(index);
    const { model } = this;
    const H = model.hiddenSize;
    const first = t === 0;
    lstmCell(model,
      model.inputWeights, index * 4 * H,
      first ? this.zeros : this.hidden, first ? 0 : (t - 1) * H,
      first ? this.zeros : this.cells, first ? 0 : (t - 1) * H,
      this.gates, t * 4 * H,
      this.cells, t * H,
      this.hidden, t * H);
    this.finishStep(t);
  }

  /** The activations of step t as views (valid until the run grows). */
  step(t: number): StepTrace {
    this.check(t);
    const H = this.model.hiddenSize;
    const V = this.model.inputSize;
    const gates = this.gates.subarray(t * 4 * H, (t + 1) * 4 * H);
    return {
      index: this.indices[t],
      character: this.model.vocabulary[this.indices[t]],
      input: gates.subarray(0, H),
      forget: gates.subarray(H, 2 * H),
      candidate: gates.subarray(2 * H, 3 * H),
      output: gates.subarray(3 * H, 4 * H),
      cell: this.cells.subarray(t * H, (t + 1) * H),
      hidden: this.hidden.subarray(t * H, (t + 1) * H),
      logits: this.logits.subarray(t * V, (t + 1) * V),
      probabilities: this.probabilities.subarray(t * V, (t + 1) * V),
    };
  }

  completeWord(t: number, first: number, maxLength = 14): number[] {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    let h = this.hidden.slice(t * H, (t + 1) * H);
    let c = this.cells.slice(t * H, (t + 1) * H);
    let hNext = new Float32Array(H);
    let cNext = new Float32Array(H);
    const gates = new Float32Array(4 * H);
    return this.readAhead(first, maxLength, (index, logits) => {
      lstmCell(model, model.inputWeights, index * 4 * H, h, 0, c, 0, gates, 0, cNext, 0, hNext, 0);
      readout(model, hNext, 0, logits, 0);
      [h, hNext] = [hNext, h];
      [c, cNext] = [cNext, c];
    });
  }

  influence(t: number, target = this.topCharacter(t), reach = Infinity): Float32Array {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    const G = 4 * H;
    const transposed = transposeOnce(model.recurrentWeights, G, H);
    const Wx = model.inputWeights;
    const out = new Float32Array(t + 1);
    const dh = this.outputGradient(t, target);
    const dc = new Float64Array(H);
    const dz = new Float64Array(G);
    const first = Math.max(0, t + 1 - reach);
    for (let s = t; s >= first; s--) {
      const gate = s * G;
      const cPrevOffset = (s - 1) * H;
      for (let k = 0; k < H; k++) {
        const i = this.gates[gate + k];
        const f = this.gates[gate + H + k];
        const g = this.gates[gate + 2 * H + k];
        const o = this.gates[gate + 3 * H + k];
        const tanhC = Math.tanh(this.cells[s * H + k]);
        const cPrev = s > 0 ? this.cells[cPrevOffset + k] : 0;
        const dCell = dc[k] + dh[k] * o * (1 - tanhC * tanhC);
        dz[k] = dCell * g * i * (1 - i);
        dz[H + k] = dCell * cPrev * f * (1 - f);
        dz[2 * H + k] = dCell * i * (1 - g * g);
        dz[3 * H + k] = dh[k] * tanhC * o * (1 - o);
        dc[k] = dCell * f;
      }
      const row = this.indices[s] * G;
      let total = 0;
      for (let j = 0; j < G; j++) total += Wx[row + j] * dz[j];
      out[s] = total;
      if (s === first) break;
      // dh_{s−1} = Uᵀ dz_s, as dot products over the rows of Uᵀ.
      multiplyRows(transposed, H, G, dz, dh);
    }
    return out;
  }

  protected growState(capacity: number) {
    const H = this.model.hiddenSize;
    this.gates = this.grow(this.gates, 4 * H, capacity);
    this.cells = this.grow(this.cells, H, capacity);
  }
}

/** A run of the gateless network, whose hidden state is its whole memory. */
export class CharRnnRun extends CharRun<CharRnn> {
  /** Reads one more character. */
  push(index: number): void {
    const t = this.beginStep(index);
    const { model } = this;
    const H = model.hiddenSize;
    const first = t === 0;
    rnnCell(model, model.inputWeights, index * H, first ? this.zeros : this.hidden, first ? 0 : (t - 1) * H, this.hidden, t * H);
    this.finishStep(t);
  }

  /** The activations of step t as views (valid until the run grows). */
  step(t: number): RnnStepTrace {
    this.check(t);
    const H = this.model.hiddenSize;
    const V = this.model.inputSize;
    return {
      index: this.indices[t],
      character: this.model.vocabulary[this.indices[t]],
      hidden: this.hidden.subarray(t * H, (t + 1) * H),
      logits: this.logits.subarray(t * V, (t + 1) * V),
      probabilities: this.probabilities.subarray(t * V, (t + 1) * V),
    };
  }

  completeWord(t: number, first: number, maxLength = 14): number[] {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    let h = this.hidden.slice(t * H, (t + 1) * H);
    let hNext = new Float32Array(H);
    return this.readAhead(first, maxLength, (index, logits) => {
      rnnCell(model, model.inputWeights, index * H, h, 0, hNext, 0);
      readout(model, hNext, 0, logits, 0);
      [h, hNext] = [hNext, h];
    });
  }

  /**
   * As for the LSTM (see CharRun.influence). Each step back multiplies the gradient by
   * the tanh slope 1 − h² and by Uᵀ: there is no gate to let it pass unchanged.
   */
  influence(t: number, target = this.topCharacter(t), reach = Infinity): Float32Array {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    const transposed = transposeOnce(model.recurrentWeights, H, H);
    const Wx = model.inputWeights;
    const out = new Float32Array(t + 1);
    const dh = this.outputGradient(t, target);
    const dz = new Float64Array(H);
    const first = Math.max(0, t + 1 - reach);
    for (let s = t; s >= first; s--) {
      const offset = s * H;
      const row = this.indices[s] * H;
      let total = 0;
      for (let k = 0; k < H; k++) {
        const h = this.hidden[offset + k];
        dz[k] = dh[k] * (1 - h * h);
        total += Wx[row + k] * dz[k];
      }
      out[s] = total;
      if (s === first) break;
      // dh_{s−1} = Uᵀ dz_s.
      multiplyRows(transposed, H, H, dz, dh);
    }
    return out;
  }

  protected growState() {
    // The hidden state, kept by every run, is all this network stores.
  }
}

const transposes = new WeakMap<Float32Array, Float32Array>();
/** The transpose ([columns][rows]) of a weight matrix, computed once per model, so backpropagation reads it row by row. */
function transposeOnce(matrix: Float32Array, rows: number, columns: number): Float32Array {
  let transposed = transposes.get(matrix);
  if (!transposed) {
    transposed = new Float32Array(rows * columns);
    for (let j = 0; j < rows; j++) for (let k = 0; k < columns; k++) transposed[k * rows + j] = matrix[j * columns + k];
    transposes.set(matrix, transposed);
  }
  return transposed;
}

/**
 * out[r] = Σ_j matrix[r][j] · vector[j] for a [rows][columns] matrix, four rows at a
 * time so each vector value is read once for all four.
 */
function multiplyRows(matrix: Float32Array, rows: number, columns: number, vector: Float64Array, out: Float64Array) {
  let r = 0;
  for (; r + 3 < rows; r += 4) {
    const r0 = r * columns;
    const r1 = r0 + columns;
    const r2 = r1 + columns;
    const r3 = r2 + columns;
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let j = 0; j < columns; j++) {
      const x = vector[j];
      s0 += matrix[r0 + j] * x;
      s1 += matrix[r1 + j] * x;
      s2 += matrix[r2 + j] * x;
      s3 += matrix[r3 + j] * x;
    }
    out[r] = s0;
    out[r + 1] = s1;
    out[r + 2] = s2;
    out[r + 3] = s3;
  }
  for (; r < rows; r++) {
    const row = r * columns;
    let s = 0;
    for (let j = 0; j < columns; j++) s += matrix[row + j] * vector[j];
    out[r] = s;
  }
}

/**
 * Average surprise in bits per character over a sequence read from the zero state:
 * −log₂ p(x_t | x_<t) for t ≥ 1. Uses constant memory, for long held-out texts.
 */
export function bitsPerCharacter(model: CharLstm, indices: ArrayLike<number>): { bits: number; count: number } {
  const H = model.hiddenSize;
  const V = model.inputSize;
  let h = new Float32Array(H);
  let c = new Float32Array(H);
  let hNext = new Float32Array(H);
  let cNext = new Float32Array(H);
  const gates = new Float32Array(4 * H);
  const logits = new Float32Array(V);
  const probabilities = new Float32Array(V);
  let bits = 0;
  for (let t = 0; t + 1 < indices.length; t++) {
    lstmCell(model, model.inputWeights, indices[t] * 4 * H, h, 0, c, 0, gates, 0, cNext, 0, hNext, 0);
    readout(model, hNext, 0, logits, 0);
    softmaxInto(logits, 0, V, probabilities, 0);
    bits -= Math.log2(Math.max(probabilities[indices[t + 1]], 1e-30));
    [h, hNext] = [hNext, h];
    [c, cNext] = [cNext, c];
  }
  const count = Math.max(0, indices.length - 1);
  return { bits: count ? bits / count : Number.NaN, count };
}

/** bitsPerCharacter for the gateless network. */
export function rnnBitsPerCharacter(model: CharRnn, indices: ArrayLike<number>): { bits: number; count: number } {
  const H = model.hiddenSize;
  const V = model.inputSize;
  let h = new Float32Array(H);
  let hNext = new Float32Array(H);
  const logits = new Float32Array(V);
  const probabilities = new Float32Array(V);
  let bits = 0;
  for (let t = 0; t + 1 < indices.length; t++) {
    rnnCell(model, model.inputWeights, indices[t] * H, h, 0, hNext, 0);
    readout(model, hNext, 0, logits, 0);
    softmaxInto(logits, 0, V, probabilities, 0);
    bits -= Math.log2(Math.max(probabilities[indices[t + 1]], 1e-30));
    [h, hNext] = [hNext, h];
  }
  const count = Math.max(0, indices.length - 1);
  return { bits: count ? bits / count : Number.NaN, count };
}

/* ---------- Memory span and accuracy, as the page and the trainer measure them ---------- */

/**
 * How many of the latest characters up to step t hold `share` (0 to 1) of the total
 * |influence| on the guess after step t, given influence() values.
 */
export function spanReach(values: ArrayLike<number>, t: number, share: number): number {
  let total = 0;
  for (let s = 0; s <= t; s++) total += Math.abs(values[s]);
  let sum = 0;
  for (let s = t; s >= 0; s--) {
    sum += Math.abs(values[s]);
    if (sum >= share * total) return t - s + 1;
  }
  return t + 1;
}

/**
 * Adds each character's share of the total |influence| on the guess after step t to
 * `shares`: shares[0] for the latest character (step t), shares[d − 1] for the one d
 * characters back, as far back as `shares` is long. Returns false, adding nothing, if
 * no character had any influence.
 */
export function addInfluenceShares(values: ArrayLike<number>, t: number, shares: Float64Array): boolean {
  const far = Math.min(shares.length, t + 1);
  let total = 0;
  for (let d = 1; d <= far; d++) total += Math.abs(values[t + 1 - d]);
  if (!(total > 0)) return false;
  for (let d = 1; d <= far; d++) shares[d - 1] += Math.abs(values[t + 1 - d]) / total;
  return true;
}

/**
 * How often the character after each step from `from` to `to` (inclusive, before the
 * last step) was the run's likeliest guess, or among its three likeliest. A character
 * ranks below every character given a higher probability.
 */
export function guessScores(run: CharRun, from = 0, to = run.length - 2): { guesses: number; first: number; three: number } {
  const V = run.model.inputSize;
  let guesses = 0;
  let first = 0;
  let three = 0;
  for (let t = Math.max(0, from); t <= Math.min(to, run.length - 2); t++) {
    const actual = run.indices[t + 1];
    const offset = t * V;
    const p = run.probabilities[offset + actual];
    let rank = 0;
    for (let v = 0; v < V; v++) if (run.probabilities[offset + v] > p) rank++;
    guesses++;
    if (rank === 0) first++;
    if (rank < 3) three++;
  }
  return { guesses, first, three };
}
