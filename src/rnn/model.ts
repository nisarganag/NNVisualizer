/**
 * The Recurrent page's character-level language model: one LSTM layer that reads
 * text one character at a time, carries a memory forward, and predicts the next
 * character.
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
 * with h_{-1} = c_{-1} = 0, so nothing is predicted before the first character.
 *
 * `char-lstm-v1.bin` holds little-endian float32 values in this order (shapes are
 * [rows][columns], row-major):
 *   inputWeights      [V][4H]  row x is W_x[x], the gate inputs a character adds
 *   recurrentWeights  [4H][H]  row = gate unit (gate-major: i, f, g, o), column = h_{t-1} unit
 *   bias              [4H]
 *   outputWeights     [V][H]
 *   outputBias        [V]
 * `char-lstm-v1.json` holds the vocabulary, the sizes and this layout.
 *
 * CharLstmRun keeps every step's activations for drawing and reuses the longest shared
 * prefix when the text changes. Its influence() backpropagates through time from those
 * activations: the exact gradient of log p_t(target) with respect to each earlier
 * one-hot input, times that input. encodeText maps typed text onto the vocabulary the
 * way the training text was normalized.
 *
 * Pure TypeScript with no DOM dependencies and no imports: it runs in the browser
 * and in Node. scripts/train-char-lstm.mjs imports lstmCell, readout and softmaxInto
 * from this file (Node strips the types), so training, evaluation and the page share
 * one forward pass. Keep the syntax erasable: no enums, namespaces or parameter
 * properties. Every stored activation is a float32, and each step reads only stored
 * values, so a run is reproducible bit for bit however it is split into calls.
 */

export const GATE_ORDER = ['input', 'forget', 'cell', 'output'] as const;
export type GateName = (typeof GATE_ORDER)[number];
export const MODEL_FORMAT = 'char-lstm-v1';

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
  notableCells?: { feature: string; description: string; cell: number; correlation: number }[];
  [key: string]: unknown;
}

export interface CharLstm {
  meta: CharLstmMeta;
  /** Index → character. */
  vocabulary: string[];
  /** Character → index. */
  lookup: Map<string, number>;
  /** V, the vocabulary size. */
  inputSize: number;
  /** H, the number of memory cells. */
  hiddenSize: number;
  /** [V][4H] */
  inputWeights: Float32Array;
  /** [4H][H] */
  recurrentWeights: Float32Array;
  /** [4H] */
  bias: Float32Array;
  /** [V][H] */
  outputWeights: Float32Array;
  /** [V] */
  outputBias: Float32Array;
}

/** The tensors of a model with V characters and H cells, in file order. */
export function tensorLayout(inputSize: number, hiddenSize: number): TensorLayout[] {
  const gates = 4 * hiddenSize;
  const shapes: [TensorLayout['name'], number[]][] = [
    ['inputWeights', [inputSize, gates]],
    ['recurrentWeights', [gates, hiddenSize]],
    ['bias', [gates]],
    ['outputWeights', [inputSize, hiddenSize]],
    ['outputBias', [inputSize]],
  ];
  let offset = 0;
  return shapes.map(([name, shape]) => {
    const entry = { name, shape, offset };
    offset += shape.reduce((a, b) => a * b, 1);
    return entry;
  });
}

export function parameterCount(inputSize: number, hiddenSize: number): number {
  return 4 * hiddenSize * (inputSize + hiddenSize + 1) + inputSize * (hiddenSize + 1);
}

/**
 * Builds a model from its metadata and `char-lstm-v1.bin`. Values are copied, so the
 * buffer can be reused.
 * @throws Error if the metadata is inconsistent, the buffer has the wrong size, or a
 * value is NaN or infinite.
 */
export function parseCharLstm(meta: CharLstmMeta, buffer: ArrayBuffer): CharLstm {
  if (!meta || meta.format !== MODEL_FORMAT) throw new Error(`Invalid LSTM metadata: expected format ${MODEL_FORMAT}.`);
  const vocabulary = Array.from(meta.vocabulary ?? '');
  const { inputSize, hiddenSize } = meta;
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) throw new Error(`Invalid LSTM metadata: hidden size ${hiddenSize}.`);
  if (inputSize !== vocabulary.length || inputSize < 2) {
    throw new Error(`Invalid LSTM metadata: input size ${inputSize} but ${vocabulary.length} characters.`);
  }
  const lookup = new Map(vocabulary.map((character, index) => [character, index]));
  if (lookup.size !== vocabulary.length) throw new Error('Invalid LSTM metadata: repeated characters in the vocabulary.');
  if (JSON.stringify(meta.gateOrder) !== JSON.stringify(GATE_ORDER)) {
    throw new Error(`Invalid LSTM metadata: gate order must be ${GATE_ORDER.join(', ')}.`);
  }
  const layout = tensorLayout(inputSize, hiddenSize);
  if (JSON.stringify(meta.tensors) !== JSON.stringify(layout)) throw new Error('Invalid LSTM metadata: unexpected tensor layout.');
  const count = parameterCount(inputSize, hiddenSize);
  if (meta.parameters !== count) throw new Error(`Invalid LSTM metadata: ${meta.parameters} parameters, expected ${count}.`);
  const bytes = buffer?.byteLength;
  if (bytes !== count * 4) {
    throw new Error(`Invalid LSTM weights: expected ${count * 4} bytes (${count} float32 values) but got ${bytes}.`);
  }
  const view = new DataView(buffer);
  const tensors = {} as Record<TensorLayout['name'], Float32Array>;
  for (const { name, shape, offset } of layout) {
    const values = new Float32Array(shape.reduce((a, b) => a * b, 1));
    for (let i = 0; i < values.length; i++) {
      const value = view.getFloat32((offset + i) * 4, true);
      if (!Number.isFinite(value)) throw new Error(`Invalid LSTM weights: ${name}[${i}] is ${value}.`);
      values[i] = value;
    }
    tensors[name] = values;
  }
  return { meta, vocabulary, lookup, inputSize, hiddenSize, ...tensors };
}

/* ---------- Text ---------- */

/** Typographic characters written the way the training text writes them. */
const REPLACEMENTS: Record<string, string> = {
  '\u2018': "'", '\u2019': "'", '\u201A': "'", '\u201B': "'", '\u2032': "'", '\u00B4': "'", '`': "'",
  '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u201F': '"', '\u2033': '"', '\u00AB': '"', '\u00BB': '"',
  '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2212': '-', '\u2014': '--', '\u2015': '--',
  '\u2026': '...', '\u00A0': ' ', '\u2009': ' ', '\u202F': ' ', '\t': ' ',
  '\u00C6': 'Ae', '\u00E6': 'ae', '\u0152': 'Oe', '\u0153': 'oe', '\u00DF': 'ss',
};
const COMBINING_MARKS = /[\u0300-\u036F]/g;

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
export function encodeText(model: Pick<CharLstm, 'lookup'>, text: string): EncodedText {
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

/** Next-character scores W_y·h + b_y for the hidden state at `h[hOffset..+H]`. */
export function readout(model: CharLstm, h: Float32Array, hOffset: number, logits: Float32Array, logitsOffset: number): void {
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

/** Everything the model computed at one step, as views into the run's arrays. */
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

/**
 * A run of the model over a sequence, with every step's activations kept in flat
 * arrays (step-major) for drawing. `setIndices` reuses the longest shared prefix,
 * so typing or generating only computes the new steps.
 */
export class CharLstmRun {
  readonly model: CharLstm;
  /** Steps computed. */
  length = 0;
  /** Vocabulary index read at each step. */
  indices = new Int32Array(0);
  /** [step][4H]: activated gates i, f, g, o. */
  gates = new Float32Array(0);
  /** [step][H] */
  cells = new Float32Array(0);
  /** [step][H] */
  hidden = new Float32Array(0);
  /** [step][V] */
  logits = new Float32Array(0);
  /** [step][V] */
  probabilities = new Float32Array(0);
  private capacity = 0;
  private readonly zeros: Float32Array;

  constructor(model: CharLstm) {
    this.model = model;
    this.zeros = new Float32Array(model.hiddenSize);
  }

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

  /** Reads one more character. */
  push(index: number): void {
    const { model } = this;
    if (!Number.isInteger(index) || index < 0 || index >= model.inputSize) throw new Error(`Unknown character index ${index}.`);
    const H = model.hiddenSize;
    const V = model.inputSize;
    const t = this.length;
    this.reserve(t + 1);
    this.indices[t] = index;
    const first = t === 0;
    lstmCell(model,
      model.inputWeights, index * 4 * H,
      first ? this.zeros : this.hidden, first ? 0 : (t - 1) * H,
      first ? this.zeros : this.cells, first ? 0 : (t - 1) * H,
      this.gates, t * 4 * H,
      this.cells, t * H,
      this.hidden, t * H);
    readout(model, this.hidden, t * H, this.logits, t * V);
    softmaxInto(this.logits, t * V, V, this.probabilities, t * V);
    this.length = t + 1;
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

  /**
   * The word the model would write after step t if it read `first` next and then kept
   * taking its likeliest character: `first` followed by every letter it then expects,
   * stopping before anything that isn't a letter or an apostrophe, after `maxLength`
   * characters at most. Just `first` when that isn't a letter. Leaves the run unchanged.
   */
  completeWord(t: number, first: number, maxLength = 14): number[] {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    const V = model.inputSize;
    const inWord = (index: number) => /[A-Za-z']/.test(model.vocabulary[index]);
    const word = [first];
    if (!/[A-Za-z]/.test(model.vocabulary[first])) return word;
    let h = this.hidden.slice(t * H, (t + 1) * H);
    let c = this.cells.slice(t * H, (t + 1) * H);
    let hNext = new Float32Array(H);
    let cNext = new Float32Array(H);
    const gates = new Float32Array(4 * H);
    const logits = new Float32Array(V);
    let index = first;
    while (word.length < maxLength) {
      lstmCell(model, model.inputWeights, index * 4 * H, h, 0, c, 0, gates, 0, cNext, 0, hNext, 0);
      readout(model, hNext, 0, logits, 0);
      let best = 0;
      for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
      if (!inWord(best)) break;
      word.push(best);
      index = best;
      [h, hNext] = [hNext, h];
      [c, cNext] = [cNext, c];
    }
    // An apostrophe can close a quotation as well as join a word ("dogs'", "can't").
    while (word.length > 1 && model.vocabulary[word[word.length - 1]] === "'") word.pop();
    return word;
  }

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
  influence(t: number, target = this.topCharacter(t), reach = Infinity): Float32Array {
    this.check(t);
    const { model } = this;
    const H = model.hiddenSize;
    const V = model.inputSize;
    const G = 4 * H;
    const transposed = transposedRecurrent(model);
    const Wx = model.inputWeights;
    const Wy = model.outputWeights;
    const out = new Float32Array(t + 1);
    const dh = new Float64Array(H);
    const dc = new Float64Array(H);
    const dz = new Float64Array(G);
    // d log p(target) / d logits = onehot(target) − p, then back through W_y.
    for (let v = 0; v < V; v++) {
      const d = (v === target ? 1 : 0) - this.probabilities[t * V + v];
      if (d === 0) continue;
      const row = v * H;
      for (let k = 0; k < H; k++) dh[k] += d * Wy[row + k];
    }
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
      for (let k = 0; k < H; k++) {
        const row = k * G;
        let s0 = 0;
        let s1 = 0;
        let s2 = 0;
        let s3 = 0;
        for (let j = 0; j < G; j += 4) {
          s0 += transposed[row + j] * dz[j];
          s1 += transposed[row + j + 1] * dz[j + 1];
          s2 += transposed[row + j + 2] * dz[j + 2];
          s3 += transposed[row + j + 3] * dz[j + 3];
        }
        dh[k] = (s0 + s1) + (s2 + s3);
      }
    }
    return out;
  }

  private check(t: number) {
    if (!Number.isInteger(t) || t < 0 || t >= this.length) throw new Error(`Step ${t} is outside the run (${this.length} steps).`);
  }

  private reserve(steps: number) {
    if (steps <= this.capacity) return;
    const capacity = Math.max(steps, Math.ceil(this.capacity * 1.5), 64);
    const H = this.model.hiddenSize;
    const V = this.model.inputSize;
    const grow = (array: Float32Array, width: number) => {
      const next = new Float32Array(capacity * width);
      next.set(array.subarray(0, this.length * width));
      return next;
    };
    const indices = new Int32Array(capacity);
    indices.set(this.indices.subarray(0, this.length));
    this.indices = indices;
    this.gates = grow(this.gates, 4 * H);
    this.cells = grow(this.cells, H);
    this.hidden = grow(this.hidden, H);
    this.logits = grow(this.logits, V);
    this.probabilities = grow(this.probabilities, V);
    this.capacity = capacity;
  }
}

const transposes = new WeakMap<Float32Array, Float32Array>();
/** Uᵀ ([H][4H]), computed once per model, so backpropagation reads it row by row. */
function transposedRecurrent(model: CharLstm): Float32Array {
  let transposed = transposes.get(model.recurrentWeights);
  if (!transposed) {
    const H = model.hiddenSize;
    const G = 4 * H;
    const U = model.recurrentWeights;
    transposed = new Float32Array(G * H);
    for (let j = 0; j < G; j++) for (let k = 0; k < H; k++) transposed[k * G + j] = U[j * H + k];
    transposes.set(model.recurrentWeights, transposed);
  }
  return transposed;
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
