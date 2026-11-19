/**
 * The Digits page's MNIST classifier: a small convolutional network whose every
 * intermediate activation is returned so the page can draw it.
 *
 *   input 1×28×28 (ink = 1, background = 0)
 *   → conv1 8 filters 3×3, stride 1, zero padding 1, ReLU   → 8×28×28
 *   → max pool 2×2, stride 2                                 → 8×14×14
 *   → conv2 16 filters 3×3 over 8 channels, padding 1, ReLU  → 16×14×14
 *   → max pool 2×2, stride 2                                 → 16×7×7
 *   → flatten channel-major (c, y, x)                        → 784
 *   → dense 64 + ReLU → dense 10 → softmax                   (52,138 parameters)
 *
 * Every tensor is a flat Float32Array in row-major order with the channel first,
 * e.g. conv1[c * 784 + y * 28 + x]. inputGradient runs the exact backward pass from
 * those activations to the input, for the saliency view and the pixel attack.
 *
 * Pure TypeScript with no DOM dependencies: it runs in the browser, in workers and in
 * Node. scripts/train-mnist-cnn.mjs imports runCnn from this file (Node strips the
 * types), so training, evaluation and the page share one forward pass. Keep the
 * syntax erasable: no enums, namespaces or parameter properties.
 */

/** Layer sizes of the network, in the order the page draws them. */
export const CNN_LAYOUT = {
  input: { channels: 1, size: 28 },
  conv1: { filters: 8, kernel: 3, size: 28 },
  pool1: { channels: 8, size: 14 },
  conv2: { filters: 16, kernel: 3, size: 14 },
  pool2: { channels: 16, size: 7 },
  dense: { units: 64 },
  output: { units: 10 },
} as const;

/**
 * Trained parameters. Weight arrays are row-major with the output unit first:
 * `conv1.weights[f * 9 + ky * 3 + kx]`, `conv2.weights[f * 72 + c * 9 + ky * 3 + kx]`,
 * `dense.weights[unit * 784 + flatIndex]`, `output.weights[digit * 64 + unit]`.
 */
export interface CnnWeights {
  /** weights [8][1][3][3], biases [8]. */
  conv1: { weights: Float32Array; biases: Float32Array };
  /** weights [16][8][3][3], biases [16]. */
  conv2: { weights: Float32Array; biases: Float32Array };
  /** weights [64][784] (row = output unit, column = flattened pool2 index), biases [64]. */
  dense: { weights: Float32Array; biases: Float32Array };
  /** weights [10][64], biases [10]. */
  output: { weights: Float32Array; biases: Float32Array };
}

/** Everything the network computes for one input. Each array is freshly allocated. */
export interface CnnActivations {
  /** 784: the 28×28 input, a copy of what was passed in. */
  input: Float32Array;
  /** 8×28×28 after ReLU, channel-major. */
  conv1: Float32Array;
  /** 8×14×14. */
  pool1: Float32Array;
  /** 16×14×14 after ReLU. */
  conv2: Float32Array;
  /** 16×7×7; also the flattened input of the dense layer. */
  pool2: Float32Array;
  /** 64 after ReLU. */
  dense: Float32Array;
  /** 10 raw scores. */
  logits: Float32Array;
  /** 10 softmax probabilities that sum to 1. */
  probabilities: Float32Array;
}

const SIZE = 28;
const PADDED = SIZE + 2;
const C1 = 8;
const P1 = 14;
const P1_PADDED = P1 + 2;
const C2 = 16;
const P2 = 7;
const FLAT = C2 * P2 * P2;
const HIDDEN = 64;
const CLASSES = 10;

/** Float counts of each tensor, in file order. */
const TENSOR_SIZES = [
  ['conv1', 'weights', C1 * 9], ['conv1', 'biases', C1],
  ['conv2', 'weights', C2 * C1 * 9], ['conv2', 'biases', C2],
  ['dense', 'weights', HIDDEN * FLAT], ['dense', 'biases', HIDDEN],
  ['output', 'weights', CLASSES * HIDDEN], ['output', 'biases', CLASSES],
] as const;
const PARAMETER_COUNT = TENSOR_SIZES.reduce((sum, entry) => sum + entry[2], 0); // 52,138
const SAMPLE_COUNT = 10;
const SAMPLE_BYTES = SIZE * SIZE;

/**
 * Parses `mnist-cnn-v1.bin`: exactly 52,138 little-endian float32 values in the order
 * conv1.weights, conv1.biases, conv2.weights, conv2.biases, dense.weights, dense.biases,
 * output.weights, output.biases. Returns copies, so the buffer can be reused.
 * @throws Error if the buffer has the wrong size or contains NaN/Infinity.
 */
export function parseCnnWeights(buffer: ArrayBuffer): CnnWeights {
  const bytes = byteLengthOf(buffer, 'parseCnnWeights');
  const expected = PARAMETER_COUNT * 4;
  if (bytes !== expected) {
    throw new Error(`Invalid CNN weights: expected ${expected} bytes (${PARAMETER_COUNT} float32 values) but got ${bytes}.`);
  }
  const view = new DataView(buffer);
  const parsed: Record<string, Record<string, Float32Array>> = {};
  let offset = 0;
  for (const [layer, kind, count] of TENSOR_SIZES) {
    const values = new Float32Array(count);
    for (let i = 0; i < count; i++, offset += 4) {
      const value = view.getFloat32(offset, true);
      if (!Number.isFinite(value)) throw new Error(`Invalid CNN weights: ${layer}.${kind}[${i}] is ${value}.`);
      values[i] = value;
    }
    (parsed[layer] ??= {})[kind] = values;
  }
  return parsed as unknown as CnnWeights;
}

/**
 * Parses `mnist-samples-v1.bin`: ten 28×28 uint8 MNIST test digits (0 = background,
 * 255 = ink), one per class in order 0..9.
 * @returns ten network inputs with values in 0..1; index = digit.
 * @throws Error if the buffer is not exactly 7,840 bytes.
 */
export function parseSamples(buffer: ArrayBuffer): Float32Array[] {
  const bytes = byteLengthOf(buffer, 'parseSamples');
  const expected = SAMPLE_COUNT * SAMPLE_BYTES;
  if (bytes !== expected) {
    throw new Error(`Invalid digit samples: expected ${expected} bytes (${SAMPLE_COUNT} × 28×28 uint8) but got ${bytes}.`);
  }
  const data = new Uint8Array(buffer);
  const samples: Float32Array[] = [];
  for (let digit = 0; digit < SAMPLE_COUNT; digit++) {
    const sample = new Float32Array(SAMPLE_BYTES);
    for (let i = 0; i < SAMPLE_BYTES; i++) sample[i] = data[digit * SAMPLE_BYTES + i] / 255;
    samples.push(sample);
  }
  return samples;
}

function byteLengthOf(buffer: ArrayBuffer, caller: string): number {
  if (ArrayBuffer.isView(buffer) || typeof buffer?.byteLength !== 'number') {
    throw new TypeError(`${caller} expects an ArrayBuffer (e.g. from response.arrayBuffer()), not a typed array or other value.`);
  }
  return buffer.byteLength;
}

// Scratch buffers shared by every runCnn call. runCnn is synchronous, so a call never
// sees another call's data; the zero borders implement the convolutions' padding.
const paddedInput = new Float32Array(PADDED * PADDED);
const paddedPool1 = new Float32Array(C1 * P1_PADDED * P1_PADDED);
const conv2Sums = new Float64Array(P1 * P1);

/**
 * Runs the network on one 28×28 input (784 values, 0..1, ink = 1) and returns every
 * layer's activations. About 330k multiply-adds; well under a millisecond.
 * @throws Error if the input or a weight array has the wrong length.
 */
export function runCnn(weights: CnnWeights, input: Float32Array): CnnActivations {
  if (!input || input.length !== SIZE * SIZE) {
    throw new Error(`runCnn expects a 784-value input but got ${input?.length ?? input}.`);
  }
  checkWeights(weights);

  const inputCopy = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const value = input[y * SIZE + x];
      inputCopy[y * SIZE + x] = value;
      paddedInput[(y + 1) * PADDED + x + 1] = value;
    }
  }

  // conv1 + ReLU: 1 → 8 channels, 28×28.
  const conv1 = new Float32Array(C1 * SIZE * SIZE);
  const w1 = weights.conv1.weights;
  const b1 = weights.conv1.biases;
  const pin = paddedInput;
  for (let f = 0; f < C1; f++) {
    const k = f * 9;
    const w00 = w1[k], w01 = w1[k + 1], w02 = w1[k + 2];
    const w10 = w1[k + 3], w11 = w1[k + 4], w12 = w1[k + 5];
    const w20 = w1[k + 6], w21 = w1[k + 7], w22 = w1[k + 8];
    const bias = b1[f];
    for (let y = 0; y < SIZE; y++) {
      const r0 = y * PADDED;
      const r1 = r0 + PADDED;
      const r2 = r1 + PADDED;
      const out = f * SIZE * SIZE + y * SIZE;
      for (let x = 0; x < SIZE; x++) {
        const sum = bias
          + w00 * pin[r0 + x] + w01 * pin[r0 + x + 1] + w02 * pin[r0 + x + 2]
          + w10 * pin[r1 + x] + w11 * pin[r1 + x + 1] + w12 * pin[r1 + x + 2]
          + w20 * pin[r2 + x] + w21 * pin[r2 + x + 1] + w22 * pin[r2 + x + 2];
        conv1[out + x] = sum > 0 ? sum : 0;
      }
    }
  }

  // pool1: 2×2 max, also written into the padded buffer that feeds conv2.
  const pool1 = new Float32Array(C1 * P1 * P1);
  for (let c = 0; c < C1; c++) {
    for (let y = 0; y < P1; y++) {
      for (let x = 0; x < P1; x++) {
        const i = c * SIZE * SIZE + 2 * y * SIZE + 2 * x;
        const best = Math.max(conv1[i], conv1[i + 1], conv1[i + SIZE], conv1[i + SIZE + 1]);
        pool1[c * P1 * P1 + y * P1 + x] = best;
        paddedPool1[c * P1_PADDED * P1_PADDED + (y + 1) * P1_PADDED + x + 1] = best;
      }
    }
  }

  // conv2 + ReLU: 8 → 16 channels, 14×14.
  const conv2 = new Float32Array(C2 * P1 * P1);
  const w2 = weights.conv2.weights;
  const b2 = weights.conv2.biases;
  const pp = paddedPool1;
  const sums = conv2Sums;
  for (let f = 0; f < C2; f++) {
    sums.fill(b2[f]);
    for (let c = 0; c < C1; c++) {
      const k = f * C1 * 9 + c * 9;
      const w00 = w2[k], w01 = w2[k + 1], w02 = w2[k + 2];
      const w10 = w2[k + 3], w11 = w2[k + 4], w12 = w2[k + 5];
      const w20 = w2[k + 6], w21 = w2[k + 7], w22 = w2[k + 8];
      const plane = c * P1_PADDED * P1_PADDED;
      for (let y = 0; y < P1; y++) {
        const r0 = plane + y * P1_PADDED;
        const r1 = r0 + P1_PADDED;
        const r2 = r1 + P1_PADDED;
        const row = y * P1;
        for (let x = 0; x < P1; x++) {
          sums[row + x] += w00 * pp[r0 + x] + w01 * pp[r0 + x + 1] + w02 * pp[r0 + x + 2]
            + w10 * pp[r1 + x] + w11 * pp[r1 + x + 1] + w12 * pp[r1 + x + 2]
            + w20 * pp[r2 + x] + w21 * pp[r2 + x + 1] + w22 * pp[r2 + x + 2];
        }
      }
    }
    const out = f * P1 * P1;
    for (let i = 0; i < P1 * P1; i++) conv2[out + i] = sums[i] > 0 ? sums[i] : 0;
  }

  // pool2: 2×2 max → 16×7×7, already in the dense layer's flatten order.
  const pool2 = new Float32Array(FLAT);
  for (let c = 0; c < C2; c++) {
    for (let y = 0; y < P2; y++) {
      for (let x = 0; x < P2; x++) {
        const i = c * P1 * P1 + 2 * y * P1 + 2 * x;
        pool2[c * P2 * P2 + y * P2 + x] = Math.max(conv2[i], conv2[i + 1], conv2[i + P1], conv2[i + P1 + 1]);
      }
    }
  }

  // dense + ReLU: 784 → 64.
  const dense = new Float32Array(HIDDEN);
  const wd = weights.dense.weights;
  const bd = weights.dense.biases;
  for (let j = 0; j < HIDDEN; j++) {
    const row = j * FLAT;
    let sum = bd[j];
    for (let i = 0; i < FLAT; i++) sum += wd[row + i] * pool2[i];
    dense[j] = sum > 0 ? sum : 0;
  }

  // output: 64 → 10, then a numerically stable softmax.
  const logits = new Float32Array(CLASSES);
  const wo = weights.output.weights;
  const bo = weights.output.biases;
  let maxLogit = -Infinity;
  for (let k = 0; k < CLASSES; k++) {
    const row = k * HIDDEN;
    let sum = bo[k];
    for (let j = 0; j < HIDDEN; j++) sum += wo[row + j] * dense[j];
    logits[k] = sum;
    if (logits[k] > maxLogit) maxLogit = logits[k];
  }
  const probabilities = new Float32Array(CLASSES);
  let total = 0;
  const exps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let k = 0; k < CLASSES; k++) {
    exps[k] = Math.exp(logits[k] - maxLogit);
    total += exps[k];
  }
  for (let k = 0; k < CLASSES; k++) probabilities[k] = exps[k] / total;

  return { input: inputCopy, conv1, pool1, conv2, pool2, dense, logits, probabilities };
}

function checkWeights(weights: CnnWeights, caller = 'runCnn'): void {
  for (const [layer, kind, count] of TENSOR_SIZES) {
    const values = weights?.[layer]?.[kind];
    if (!values || values.length !== count) {
      throw new Error(`${caller}: ${layer}.${kind} should hold ${count} values but has ${values?.length ?? 'none'}.`);
    }
  }
}

// ---------------------------------------------------------------------------------
// Backward pass: how much each input pixel moves a score
// ---------------------------------------------------------------------------------

// Scratch buffers for inputGradient, which is synchronous like runCnn. The padded
// borders collect gradient for the convolutions' zero padding, which is then dropped.
const gradientPool2 = new Float64Array(FLAT);
const gradientPool1 = new Float64Array(C1 * P1_PADDED * P1_PADDED);
const gradientInput = new Float64Array(PADDED * PADDED);

const ACTIVATION_SIZES = [
  ['conv1', C1 * SIZE * SIZE], ['pool1', C1 * P1 * P1], ['conv2', C2 * P1 * P1], ['pool2', FLAT], ['dense', HIDDEN],
] as const;

/**
 * The exact gradient of one score with respect to the 784 input pixels, by
 * backpropagation through the activations runCnn returned for that input:
 *
 *   ∂ logits[digit] / ∂ input,  or with `against`:  ∂ (logits[digit] − logits[against]) / ∂ input.
 *
 * A ReLU passes gradient where its output is > 0. A max-pool window sends its gradient
 * to the first of its positions (top left, top right, bottom left, bottom right) that
 * holds the pooled value: the same choice the trainer's backward pass makes, in the
 * order runCnn's Math.max reads the window. The convolutions are cross-correlations with
 * zero padding 1, so the gradient flows back through each kernel in the orientation the
 * forward pass applied it. The network is piecewise linear, so this is the derivative
 * itself wherever no ReLU input is exactly 0 and no pool window is tied.
 *
 * At most about 120k multiply-adds, a third of a forward pass.
 * @param activations runCnn's result for the input of interest.
 * @param digit the score to differentiate, 0..9.
 * @param against optional second score, 0..9, to subtract.
 * @returns 784 values, row-major 28×28: how fast the score grows as each pixel gains ink.
 * @throws Error if a digit is out of range or an array has the wrong length.
 */
export function inputGradient(weights: CnnWeights, activations: CnnActivations, digit: number, against?: number): Float32Array {
  checkDigit(digit, 'digit');
  if (against !== undefined) checkDigit(against, 'against');
  checkWeights(weights, 'inputGradient');
  for (const [name, count] of ACTIVATION_SIZES) {
    const values = activations?.[name];
    if (!values || values.length !== count) {
      throw new Error(`inputGradient: activations.${name} should hold ${count} values but has ${values?.length ?? 'none'}; pass runCnn's result.`);
    }
  }
  const { conv1, pool1, conv2, pool2, dense } = activations;

  // Output and dense layers: the score's gradient reaches dense unit j through row j of
  // the output weights, passes its ReLU where the unit is active, and spreads over
  // every flattened pool2 value through the unit's dense weights.
  const wo = weights.output.weights;
  const wd = weights.dense.weights;
  const toPool2 = gradientPool2;
  toPool2.fill(0);
  for (let j = 0; j < HIDDEN; j++) {
    if (!(dense[j] > 0)) continue;
    let d = wo[digit * HIDDEN + j];
    if (against !== undefined) d -= wo[against * HIDDEN + j];
    if (d === 0) continue;
    const row = j * FLAT;
    for (let i = 0; i < FLAT; i++) toPool2[i] += d * wd[row + i];
  }

  // pool2 → conv2 → pool1: each window's gradient goes to its max, passes the ReLU if
  // that max is > 0, then back through the 8×3×3 weights onto the pool1 values that
  // conv2[f][y][x] read, pool1[c][y + ky − 1][x + kx − 1] (padded index y + ky, x + kx).
  const w2 = weights.conv2.weights;
  const toPool1 = gradientPool1;
  toPool1.fill(0);
  for (let f = 0; f < C2; f++) {
    for (let py = 0; py < P2; py++) {
      for (let px = 0; px < P2; px++) {
        const index = f * P2 * P2 + py * P2 + px;
        const d = toPool2[index];
        const max = pool2[index];
        if (d === 0 || !(max > 0)) continue;
        const position = windowArgmax(conv2, f * P1 * P1 + 2 * py * P1 + 2 * px, P1, max);
        const y = 2 * py + (position >> 1);
        const x = 2 * px + (position & 1);
        for (let c = 0; c < C1; c++) {
          const k = f * C1 * 9 + c * 9;
          const p = c * P1_PADDED * P1_PADDED + y * P1_PADDED + x;
          for (let ky = 0; ky < 3; ky++) {
            const row = p + ky * P1_PADDED;
            toPool1[row] += d * w2[k + ky * 3];
            toPool1[row + 1] += d * w2[k + ky * 3 + 1];
            toPool1[row + 2] += d * w2[k + ky * 3 + 2];
          }
        }
      }
    }
  }

  // pool1 → conv1 → input, the same way with the 3×3 conv1 kernels.
  const w1 = weights.conv1.weights;
  const toInput = gradientInput;
  toInput.fill(0);
  for (let f = 0; f < C1; f++) {
    for (let py = 0; py < P1; py++) {
      for (let px = 0; px < P1; px++) {
        const d = toPool1[f * P1_PADDED * P1_PADDED + (py + 1) * P1_PADDED + px + 1];
        const max = pool1[f * P1 * P1 + py * P1 + px];
        if (d === 0 || !(max > 0)) continue;
        const position = windowArgmax(conv1, f * SIZE * SIZE + 2 * py * SIZE + 2 * px, SIZE, max);
        const y = 2 * py + (position >> 1);
        const x = 2 * px + (position & 1);
        const k = f * 9;
        for (let ky = 0; ky < 3; ky++) {
          const row = (y + ky) * PADDED + x;
          toInput[row] += d * w1[k + ky * 3];
          toInput[row + 1] += d * w1[k + ky * 3 + 1];
          toInput[row + 2] += d * w1[k + ky * 3 + 2];
        }
      }
    }
  }

  const gradient = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) gradient[y * SIZE + x] = toInput[(y + 1) * PADDED + x + 1];
  }
  return gradient;
}

/**
 * Which position of the 2×2 window starting at flat index `start` holds `max`:
 * 0 top left, 1 top right, 2 bottom left, 3 bottom right; ties go to the first.
 */
function windowArgmax(values: Float32Array, start: number, stride: number, max: number): number {
  if (values[start] === max) return 0;
  if (values[start + 1] === max) return 1;
  return values[start + stride] === max ? 2 : 3;
}

function checkDigit(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value >= CLASSES) {
    throw new RangeError(`inputGradient: ${name} must be an integer from 0 to 9 (got ${value}).`);
  }
}

// ---------------------------------------------------------------------------------
// Drawing → MNIST-style input
// ---------------------------------------------------------------------------------

/** Alpha below this (about 10%) is antialiasing fringe or noise and does not define the ink box. */
const INK_THRESHOLD = 26;
/** Drawings with less ink than this share of the canvas (e.g. 39 solid pixels of 280×280) count as blank. */
const MIN_INK_SHARE = 0.0005;
/** MNIST fits each digit's bounding box into 20×20 and centers its mass in the 28×28 field. */
const FIT_BOX = 20;
const CENTER = 14;

/**
 * Turns a drawing into the 28×28 input MNIST digits look like, following MNIST's own
 * preprocessing:
 *
 * 1. Bounding box of the ink, ignoring alpha below ~10% (antialiasing fringe, specks).
 *    Returns null when there is no ink or less than 0.05% of the canvas area of it.
 * 2. Scale the box (aspect ratio preserved) so its longer side is 20 pixels. Resampling
 *    is exact area averaging: every output pixel gets the mean alpha of the canvas
 *    region it covers, so shrinking a 280×280 or 560×560 canvas antialiases properly.
 * 3. Position it so the intensity-weighted center of mass sits at pixel (14, 14) of
 *    the 28×28 field (index coordinates, i.e. Σx·v/Σv = 14, as in MNIST), with
 *    sub-pixel precision; the offset is clamped so the box stays inside the field.
 * 4. Rescale intensities so the darkest pixel is 1 (every MNIST digit peaks at full
 *    ink), which keeps thin or translucent strokes readable. Output is in 0..1.
 *
 * Cost is linear in the canvas size: about a millisecond for 560×560.
 *
 * @param alpha one byte per pixel of the drawing (0 = empty, 255 = full ink), row-major
 *   width×height. From a canvas: `alpha[i] = imageData.data[4 * i + 3]`.
 * @returns 784 values (row-major 28×28, 0..1) or null for a blank / near-blank drawing.
 * @throws RangeError if the dimensions are invalid or do not match alpha.length.
 */
export function preprocessDrawing(alpha: Uint8ClampedArray | Uint8Array, width: number, height: number): Float32Array | null {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`preprocessDrawing: width and height must be positive integers (got ${width}×${height}).`);
  }
  if (!alpha || alpha.length !== width * height) {
    throw new RangeError(`preprocessDrawing expects one alpha byte per pixel (${width * height}) but got ${alpha?.length}; `
      + 'for ImageData pass data[4 * i + 3].');
  }

  // 1. Ink bounding box.
  let left = width;
  let right = -1;
  let top = -1;
  let bottom = -1;
  let ink = 0;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let first = -1;
    let last = -1;
    for (let x = 0; x < width; x++) {
      const a = alpha[row + x];
      if (a >= INK_THRESHOLD) {
        if (first < 0) first = x;
        last = x;
        ink += a;
      }
    }
    if (first >= 0) {
      if (top < 0) top = y;
      bottom = y;
      if (first < left) left = first;
      if (last > right) right = last;
    }
  }
  if (top < 0 || ink < MIN_INK_SHARE * 255 * width * height) return null;

  // Center of mass of everything inside the box, fringe included.
  let mass = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = top; y <= bottom; y++) {
    const row = y * width;
    let rowMass = 0;
    let rowX = 0;
    for (let x = left; x <= right; x++) {
      const a = alpha[row + x];
      rowMass += a;
      rowX += a * x;
    }
    mass += rowMass;
    sumX += rowX;
    sumY += rowMass * y;
  }

  // 2–3. Scale and placement. Source pixel x covers [x, x + 1); its center is x + 0.5.
  const boxWidth = right - left + 1;
  const boxHeight = bottom - top + 1;
  const scale = FIT_BOX / Math.max(boxWidth, boxHeight);
  const massX = (sumX / mass - left + 0.5) * scale;
  const massY = (sumY / mass - top + 0.5) * scale;
  const offsetX = clamp(CENTER + 0.5 - massX, 0, SIZE - boxWidth * scale);
  const offsetY = clamp(CENTER + 0.5 - massY, 0, SIZE - boxHeight * scale);
  const columns = overlapWeights(boxWidth, scale, offsetX);
  const rows = overlapWeights(boxHeight, scale, offsetY);

  // Separable area averaging: canvas rows → 28 columns, then box rows → 28 rows.
  const narrowed = new Float64Array(boxHeight * SIZE);
  for (let y = 0; y < boxHeight; y++) {
    const source = (top + y) * width + left;
    const target = y * SIZE;
    for (let t = 0; t < columns.count; t++) {
      narrowed[target + columns.to[t]] += alpha[source + columns.from[t]] * columns.weight[t];
    }
  }
  const field = new Float64Array(SIZE * SIZE);
  for (let t = 0; t < rows.count; t++) {
    const source = rows.from[t] * SIZE;
    const target = rows.to[t] * SIZE;
    const weight = rows.weight[t];
    for (let x = 0; x < SIZE; x++) field[target + x] += narrowed[source + x] * weight;
  }

  // 4. Normalize so the strongest pixel is full ink.
  let peak = 0;
  for (let i = 0; i < field.length; i++) if (field[i] > peak) peak = field[i];
  if (!(peak > 0)) return null;
  const gain = 1 / Math.min(peak, 255);
  const result = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < field.length; i++) result[i] = Math.min(1, field[i] * gain);
  return result;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/**
 * 1-D area-resampling weights. Source pixel i spans [offset + i·scale, offset + (i+1)·scale)
 * in output pixels; each (from, to, weight) entry is its overlap with output pixel `to`.
 */
function overlapWeights(count: number, scale: number, offset: number) {
  const capacity = count * (Math.ceil(scale) + 1);
  const from = new Int32Array(capacity);
  const to = new Int32Array(capacity);
  const weight = new Float64Array(capacity);
  let entries = 0;
  for (let i = 0; i < count; i++) {
    let start = offset + i * scale;
    const end = start + scale;
    for (let j = Math.floor(start); start < end && j < SIZE; j++) {
      const stop = Math.min(end, j + 1);
      if (j >= 0 && stop > start) {
        from[entries] = i;
        to[entries] = j;
        weight[entries] = stop - start;
        entries++;
      }
      start = stop;
    }
  }
  return { from, to, weight, count: entries };
}
