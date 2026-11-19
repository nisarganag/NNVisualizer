/**
 * GPT-2 small inference, written to expose every intermediate value.
 *
 * Weights come from scripts/convert-gpt2.mjs: matrices are stored as
 * [outputs][inputs], either int8 with one scale per block of `groupSize`
 * consecutive inputs (browser) or float32 (verification). The session keeps per-token keys and values, so appending
 * tokens only computes the new positions; earlier activations never change in a
 * causal model, so they are reused as-is.
 */

export interface Gpt2Config {
  nLayer: number;
  nHead: number;
  nEmbd: number;
  vocabSize: number;
  nCtx: number;
  layerNormEpsilon: number;
}

export interface Matrix {
  rows: number;
  cols: number;
  /** Row-major values. Int8 data is multiplied by its block's scale. */
  data: Int8Array | Float32Array;
  /** One scale per block: index row * (cols / groupSize) + block. Null for float32 data. */
  scales: Float32Array | null;
  /** Inputs per scale block; equals `cols` when a whole row shares one scale. */
  groupSize: number;
}

export interface Norm {
  gamma: Float32Array;
  beta: Float32Array;
}

export interface Linear {
  weight: Matrix;
  bias: Float32Array;
}

export interface LayerWeights {
  ln1: Norm;
  /** Query, key and value projections stacked: rows 0–767 Q, 768–1535 K, 1536–2303 V. */
  attention: Linear;
  attentionOutput: Linear;
  ln2: Norm;
  mlpUp: Linear;
  mlpDown: Linear;
}

/**
 * Exact token-embedding values for the few final-hidden dimensions that are far
 * larger than the rest, applied when scoring next tokens with quantized weights.
 */
export interface OutputCorrection {
  dims: number[];
  /** [vocabSize][dims.length] */
  values: Float32Array;
}

export interface Gpt2Weights {
  config: Gpt2Config;
  tokenEmbedding: Matrix;
  positionEmbedding: Matrix;
  layers: LayerWeights[];
  finalNorm: Norm;
  outputCorrection?: OutputCorrection | null;
}

export interface TensorEntry {
  dtype: 'int8' | 'float32' | 'float16';
  shape: number[];
  offset: number;
  /** Byte offset of the float16 block scales. */
  scales?: number;
  groupSize?: number;
  /** Hidden dimensions covered by an output correction. */
  dims?: number[];
}

/** Expands IEEE half-precision values to float32. */
export function halfToFloat(halves: Uint16Array): Float32Array {
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i++) {
    const h = halves[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    out[i] = exponent === 0 ? sign * fraction * 2 ** -24
      : exponent === 31 ? (fraction ? NaN : sign * Infinity)
      : sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
  }
  return out;
}

export interface Gpt2Manifest {
  format: string;
  version: number;
  config: Gpt2Config;
  totalBytes: number;
  files: { name: string; bytes: number; sha256: string }[];
  tensors: Record<string, TensorEntry>;
}

/** Builds weight views over the concatenated weight files (no copying). */
export function weightsFromBuffer(manifest: Gpt2Manifest, buffer: ArrayBuffer): Gpt2Weights {
  if (buffer.byteLength !== manifest.totalBytes) {
    throw new Error(`Expected ${manifest.totalBytes} bytes of weights, received ${buffer.byteLength}.`);
  }
  const entry = (name: string) => {
    const found = manifest.tensors[name];
    if (!found) throw new Error(`The weights are missing ${name}.`);
    return found;
  };
  const vector = (name: string) => {
    const { offset, shape } = entry(name);
    return new Float32Array(buffer, offset, shape[0]);
  };
  const matrix = (name: string): Matrix => {
    const { offset, shape, scales, groupSize = shape[1] } = entry(name);
    const [rows, cols] = shape;
    const blocks = rows * (cols / groupSize);
    return { rows, cols, groupSize, data: new Int8Array(buffer, offset, rows * cols), scales: halfToFloat(new Uint16Array(buffer, scales, blocks)) };
  };
  const norm = (prefix: string): Norm => ({ gamma: vector(`${prefix}.weight`), beta: vector(`${prefix}.bias`) });
  const linear = (prefix: string): Linear => ({ weight: matrix(`${prefix}.weight`), bias: vector(`${prefix}.bias`) });
  const config = manifest.config;
  const layers: LayerWeights[] = [];
  for (let i = 0; i < config.nLayer; i++) {
    layers.push({
      ln1: norm(`h.${i}.ln_1`),
      attention: linear(`h.${i}.attn.c_attn`),
      attentionOutput: linear(`h.${i}.attn.c_proj`),
      ln2: norm(`h.${i}.ln_2`),
      mlpUp: linear(`h.${i}.mlp.c_fc`),
      mlpDown: linear(`h.${i}.mlp.c_proj`),
    });
  }
  const outliers = manifest.tensors['wte.outliers'];
  return {
    config,
    tokenEmbedding: matrix('wte'),
    positionEmbedding: matrix('wpe'),
    layers,
    finalNorm: norm('ln_f'),
    outputCorrection: outliers ? {
      dims: outliers.dims,
      values: halfToFloat(new Uint16Array(buffer, outliers.offset, outliers.shape[0] * outliers.shape[1])),
    } : null,
  };
}

/** One output row of a matrix as float32 values. */
export function matrixRow(matrix: Matrix, row: number, out = new Float32Array(matrix.cols)): Float32Array {
  const { cols, data, scales, groupSize } = matrix;
  const base = row * cols;
  const blocks = cols / groupSize;
  for (let c = 0; c < cols; c++) {
    out[c] = data[base + c] * (scales ? scales[row * blocks + Math.floor(c / groupSize)] : 1);
  }
  return out;
}

/** y_t = W x_t + b for every input vector, reading each weight row once. */
export function linear(inputs: Float32Array[], layer: Linear | Matrix, firstRow = 0, rowCount?: number): Float32Array[] {
  const weight = 'weight' in layer ? layer.weight : layer;
  const bias = 'weight' in layer ? layer.bias : null;
  const { cols, data, scales, groupSize } = weight;
  const blocks = cols / groupSize;
  const rows = rowCount ?? weight.rows - firstRow;
  const outputs = inputs.map(() => new Float32Array(rows));
  const n = inputs.length;
  for (let r = 0; r < rows; r++) {
    const row = firstRow + r;
    const base = row * cols;
    const scaleBase = row * blocks;
    const offset = bias ? bias[row] : 0;
    for (let t = 0; t < n; t++) {
      const x = inputs[t];
      let total = 0;
      for (let b = 0; b < blocks; b++) {
        const start = b * groupSize;
        const end = start + groupSize;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        // Group sizes are multiples of four.
        for (let k = start; k < end; k += 4) {
          s0 += data[base + k] * x[k];
          s1 += data[base + k + 1] * x[k + 1];
          s2 += data[base + k + 2] * x[k + 2];
          s3 += data[base + k + 3] * x[k + 3];
        }
        total += scales ? (s0 + s1 + s2 + s3) * scales[scaleBase + b] : s0 + s1 + s2 + s3;
      }
      outputs[t][r] = total + offset;
    }
  }
  return outputs;
}

export function layerNorm(x: Float32Array, norm: Norm, epsilon: number): Float32Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(variance / n + epsilon);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * norm.gamma[i] + norm.beta[i];
  return out;
}

/** GPT-2's tanh approximation of GELU. */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

/** Everything computed for one layer at one position. */
export interface LayerStep {
  /** Input after layer norm 1 (768). */
  normed: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  /** Per head: scaled dot products with positions 0..t (causal), before softmax. */
  scores: Float32Array[];
  /** Per head: attention weights over positions 0..t. */
  weights: Float32Array[];
  /** Concatenated head outputs before the output projection (768). */
  attended: Float32Array;
  /** Residual stream after attention (768). */
  afterAttention: Float32Array;
  /** MLP hidden activations after GELU (3072). */
  mlp: Float32Array;
  /** Residual stream leaving the block (768). */
  output: Float32Array;
}

export interface PositionTrace {
  token: number;
  tokenEmbedding: Float32Array;
  positionEmbedding: Float32Array;
  /** Token plus position embedding: the block input (768). */
  input: Float32Array;
  layers: LayerStep[];
}

export class Gpt2Session {
  readonly weights: Gpt2Weights;
  readonly positions: PositionTrace[] = [];

  constructor(weights: Gpt2Weights) {
    this.weights = weights;
  }

  get tokens(): number[] {
    return this.positions.map(position => position.token);
  }

  /**
   * Makes the session hold exactly `tokens`, reusing the longest shared prefix
   * and computing only what changed. Returns how many positions were reused.
   */
  setTokens(tokens: number[]): number {
    const { nCtx } = this.weights.config;
    if (tokens.length > nCtx) throw new Error(`GPT-2 reads at most ${nCtx} tokens.`);
    let shared = 0;
    while (shared < tokens.length && shared < this.positions.length && tokens[shared] === this.positions[shared].token) shared++;
    this.positions.length = shared;
    if (tokens.length > shared) this.extend(tokens.slice(shared));
    return shared;
  }

  private extend(tokens: number[]): void {
    const { config, tokenEmbedding, positionEmbedding, layers, } = this.weights;
    const { nEmbd, nHead, layerNormEpsilon } = config;
    const headSize = nEmbd / nHead;
    const start = this.positions.length;
    const fresh: PositionTrace[] = tokens.map((token, index) => {
      if (token < 0 || token >= config.vocabSize) throw new Error(`Unknown token ${token}.`);
      const tokenVector = matrixRow(tokenEmbedding, token);
      const positionVector = matrixRow(positionEmbedding, start + index);
      const input = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) input[i] = tokenVector[i] + positionVector[i];
      return { token, tokenEmbedding: tokenVector, positionEmbedding: positionVector, input, layers: [] };
    });
    this.positions.push(...fresh);

    let stream = fresh.map(position => position.input);
    layers.forEach((layer, l) => {
      const normed = stream.map(x => layerNorm(x, layer.ln1, layerNormEpsilon));
      const q = linear(normed, layer.attention, 0, nEmbd);
      const k = linear(normed, layer.attention, nEmbd, nEmbd);
      const v = linear(normed, layer.attention, 2 * nEmbd, nEmbd);
      // Store keys and values first: a new position attends to itself and every earlier one.
      const steps: LayerStep[] = fresh.map((_position, i) => ({
        normed: normed[i], q: q[i], k: k[i], v: v[i], scores: [], weights: [],
        attended: new Float32Array(nEmbd), afterAttention: null, mlp: null, output: null,
      }));
      fresh.forEach((position, i) => { position.layers[l] = steps[i]; });

      const scale = 1 / Math.sqrt(headSize);
      fresh.forEach((_position, i) => {
        const t = start + i;
        const step = steps[i];
        for (let h = 0; h < nHead; h++) {
          const offset = h * headSize;
          const scores = new Float32Array(t + 1);
          let max = -Infinity;
          for (let s = 0; s <= t; s++) {
            const key = this.positions[s].layers[l].k;
            let dot = 0;
            for (let d = 0; d < headSize; d++) dot += step.q[offset + d] * key[offset + d];
            scores[s] = dot * scale;
            if (scores[s] > max) max = scores[s];
          }
          const weights = new Float32Array(t + 1);
          let total = 0;
          for (let s = 0; s <= t; s++) { weights[s] = Math.exp(scores[s] - max); total += weights[s]; }
          for (let s = 0; s <= t; s++) weights[s] /= total;
          for (let s = 0; s <= t; s++) {
            const value = this.positions[s].layers[l].v;
            const w = weights[s];
            for (let d = 0; d < headSize; d++) step.attended[offset + d] += w * value[offset + d];
          }
          step.scores[h] = scores;
          step.weights[h] = weights;
        }
      });

      const projected = linear(steps.map(step => step.attended), layer.attentionOutput);
      const afterAttention = stream.map((x, i) => {
        const out = new Float32Array(nEmbd);
        for (let d = 0; d < nEmbd; d++) out[d] = x[d] + projected[i][d];
        steps[i].afterAttention = out;
        return out;
      });
      const normed2 = afterAttention.map(x => layerNorm(x, layer.ln2, layerNormEpsilon));
      const hidden = linear(normed2, layer.mlpUp);
      hidden.forEach((values, i) => {
        for (let d = 0; d < values.length; d++) values[d] = gelu(values[d]);
        steps[i].mlp = values;
      });
      const down = linear(hidden, layer.mlpDown);
      stream = afterAttention.map((x, i) => {
        const out = new Float32Array(nEmbd);
        for (let d = 0; d < nEmbd; d++) out[d] = x[d] + down[i][d];
        steps[i].output = out;
        return out;
      });
    });
  }

  /** The final layer-normed vector at the last position. */
  finalHidden(): Float32Array | null {
    const last = this.positions.at(-1);
    if (!last) return null;
    const { config, finalNorm } = this.weights;
    return layerNorm(last.layers[config.nLayer - 1].output, finalNorm, config.layerNormEpsilon);
  }

  /** Next-token logits after the last position (GPT-2 ties output and token embeddings). */
  logits(): Float32Array | null {
    const hidden = this.finalHidden();
    if (!hidden) return null;
    const { tokenEmbedding, outputCorrection } = this.weights;
    const logits = linear([hidden], tokenEmbedding)[0];
    if (outputCorrection && tokenEmbedding.scales) {
      // Replace the rounded embedding values on the outsized dimensions with exact ones.
      const { dims, values } = outputCorrection;
      const { cols, data, scales, groupSize } = tokenEmbedding;
      const blocks = cols / groupSize;
      const k = dims.length;
      for (let t = 0; t < logits.length; t++) {
        let delta = 0;
        for (let j = 0; j < k; j++) {
          const d = dims[j];
          delta += (values[t * k + j] - data[t * cols + d] * scales[t * blocks + Math.floor(d / groupSize)]) * hidden[d];
        }
        logits[t] += delta;
      }
    }
    return logits;
  }
}
