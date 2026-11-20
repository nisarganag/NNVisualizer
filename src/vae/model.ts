/**
 * The Autoencoder page's model: a variational autoencoder (VAE) trained on MNIST. The
 * encoder squeezes a 28×28 digit into two numbers; the decoder draws a digit from any
 * two numbers, so every point of the plane decodes to a digit.
 *
 *   encoder  x: 784 pixels, ink = 1, background = 0
 *            → dense 256 + ReLU → dense 128 + ReLU
 *            → mean μ (2) and log-variance log σ² (2)
 *   decoder  z: 2 numbers
 *            → dense 128 + ReLU → dense 256 + ReLU
 *            → dense 784 → sigmoid: how likely each pixel is to be ink, 0..1
 *
 * (The widths are read from `mnist-vae-v1.json`; these are the shipped ones.)
 *
 * The encoder describes a digit by a small Gaussian, N(μ, σ²). Training decodes a random
 * sample of it, z = μ + σ·ε with ε ~ N(0, I), and minimizes the negative evidence lower
 * bound (ELBO), in nats per image:
 *   reconstructionLoss  binary cross-entropy of the digit under the decoded image
 *   + klDivergence      KL(N(μ, σ²) ‖ N(0, I)), what the code costs
 * The noise makes nearby points decode to similar digits, and the KL term packs every
 * code close to the origin, so the plane has no gaps. The page places each digit at its
 * mean μ and decodes whatever point the visitor picks.
 *
 * `mnist-vae-v1.bin` holds little-endian float32 values: the dense layers of VAE_LAYERS
 * in order, each as weights then biases. Weights are stored input-major,
 * [inputs][outputs]: weights[i * outputs + j] connects input i to output j. A layer then
 * adds one contiguous row per nonzero input, and skips the rest: most pixels, and about
 * half of the ReLU outputs, are exactly zero. `mnist-vae-v1.json` holds the sizes, this
 * layout, the map's extent and class centers, the scores and the training details.
 *
 * Pure TypeScript with no DOM dependencies and no imports: it runs in the browser and in
 * Node. scripts/train-mnist-vae.mjs imports the forward passes and losses from this file
 * (Node strips the types), so training, evaluation and the page share one forward pass.
 * Keep the syntax erasable: no enums, namespaces or parameter properties.
 */

export const VAE_FORMAT = 'mnist-vae-v1';
export const IMAGE_SIZE = 28;
export const PIXELS = IMAGE_SIZE * IMAGE_SIZE;
export const CLASSES = 10;

/** The dense layers, in file order. */
export const VAE_LAYERS = [
  'encoder.hidden1', 'encoder.hidden2', 'encoder.mean', 'encoder.logVariance',
  'decoder.hidden1', 'decoder.hidden2', 'decoder.output',
] as const;
export type VaeLayerName = (typeof VAE_LAYERS)[number];

export interface TensorLayout {
  /** e.g. `encoder.hidden1.weights` (shape [inputs, outputs]) or `encoder.hidden1.biases` ([outputs]). */
  name: string;
  shape: number[];
  /** Offset in float32 values from the start of the file. */
  offset: number;
}

/** Contents of `mnist-vae-v1.json`. Only the fields parseVae checks are required. */
export interface VaeMeta {
  format: typeof VAE_FORMAT;
  imageSize: number;
  latentSize: number;
  /** Widths of the encoder's two hidden layers; the decoder mirrors them. */
  hiddenSizes: number[];
  activation: 'relu';
  parameters: number;
  tensors: TensorLayout[];
  /** The square of the plane the page shows: both axes run from min to max. */
  view?: { min: number; max: number };
  /** For each digit 0..9, the point the page labels with it and morphs from or to. */
  classCenters?: number[][];
  codes?: { file: string; count: number; [key: string]: unknown };
  examples?: { file: string; count: number; labels: number[]; testIndices: number[]; [key: string]: unknown };
  scores?: Record<string, unknown>;
  training?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DenseLayer {
  inputs: number;
  outputs: number;
  /** [inputs][outputs], input-major. */
  weights: Float32Array;
  /** [outputs] */
  biases: Float32Array;
}

export interface Vae {
  latentSize: number;
  /** Widths of the encoder's hidden layers, [first, second]. */
  hiddenSizes: [number, number];
  encoder: { hidden1: DenseLayer; hidden2: DenseLayer; mean: DenseLayer; logVariance: DenseLayer };
  decoder: { hidden1: DenseLayer; hidden2: DenseLayer; output: DenseLayer };
  /** Every parameter in file order; the layers' arrays are views into it. */
  parameters: Float32Array;
}

/** What encode computes for one image. */
export interface Encoding {
  hidden1: Float32Array;
  hidden2: Float32Array;
  /** μ: where the digit sits on the map. */
  mean: Float32Array;
  /** log σ²: how uncertain the encoder is about it. */
  logVariance: Float32Array;
}

/** What decode computes for one point. */
export interface Decoding {
  hidden1: Float32Array;
  hidden2: Float32Array;
  /** 784 pre-sigmoid scores. */
  logits: Float32Array;
  /** 784 ink probabilities, 0..1, row-major 28×28. */
  image: Float32Array;
}

/** [inputs, outputs] of each layer for a model with these sizes, in file order. */
function layerShapes(latentSize: number, [first, second]: number[]): [VaeLayerName, number, number][] {
  return [
    ['encoder.hidden1', PIXELS, first],
    ['encoder.hidden2', first, second],
    ['encoder.mean', second, latentSize],
    ['encoder.logVariance', second, latentSize],
    ['decoder.hidden1', latentSize, second],
    ['decoder.hidden2', second, first],
    ['decoder.output', first, PIXELS],
  ];
}

function checkSizes(latentSize: number, hiddenSizes: number[], caller: string): void {
  const valid = (value: number) => Number.isInteger(value) && value >= 1 && value <= 4096;
  if (!valid(latentSize) || !Array.isArray(hiddenSizes) || hiddenSizes.length !== 2 || !hiddenSizes.every(valid)) {
    throw new Error(`${caller}: expected a positive latent size and two hidden widths (got ${latentSize} and ${JSON.stringify(hiddenSizes)}).`);
  }
}

/** The tensors of a model with these sizes, in file order. */
export function vaeLayout(latentSize: number, hiddenSizes: number[]): TensorLayout[] {
  checkSizes(latentSize, hiddenSizes, 'vaeLayout');
  const layout: TensorLayout[] = [];
  let offset = 0;
  for (const [name, inputs, outputs] of layerShapes(latentSize, hiddenSizes)) {
    layout.push({ name: `${name}.weights`, shape: [inputs, outputs], offset });
    offset += inputs * outputs;
    layout.push({ name: `${name}.biases`, shape: [outputs], offset });
    offset += outputs;
  }
  return layout;
}

export function vaeParameterCount(latentSize: number, hiddenSizes: number[]): number {
  checkSizes(latentSize, hiddenSizes, 'vaeParameterCount');
  return layerShapes(latentSize, hiddenSizes).reduce((sum, [, inputs, outputs]) => sum + (inputs + 1) * outputs, 0);
}

/**
 * A model whose layers are views into `parameters` (file order), without copying, so
 * training can update the values in place.
 * @throws Error if `parameters` has the wrong length.
 */
export function vaeFromParameters(parameters: Float32Array, latentSize: number, hiddenSizes: number[]): Vae {
  const count = vaeParameterCount(latentSize, hiddenSizes);
  if (!(parameters instanceof Float32Array) || parameters.length !== count) {
    throw new Error(`vaeFromParameters: expected ${count} float32 parameters but got ${parameters?.length}.`);
  }
  const layers = {} as Record<VaeLayerName, DenseLayer>;
  let offset = 0;
  for (const [name, inputs, outputs] of layerShapes(latentSize, hiddenSizes)) {
    const weights = parameters.subarray(offset, offset + inputs * outputs);
    offset += inputs * outputs;
    const biases = parameters.subarray(offset, offset + outputs);
    offset += outputs;
    layers[name] = { inputs, outputs, weights, biases };
  }
  return {
    latentSize,
    hiddenSizes: [hiddenSizes[0], hiddenSizes[1]],
    encoder: {
      hidden1: layers['encoder.hidden1'], hidden2: layers['encoder.hidden2'],
      mean: layers['encoder.mean'], logVariance: layers['encoder.logVariance'],
    },
    decoder: { hidden1: layers['decoder.hidden1'], hidden2: layers['decoder.hidden2'], output: layers['decoder.output'] },
    parameters,
  };
}

/**
 * Builds a model from its metadata and `mnist-vae-v1.bin`. Values are copied, so the
 * buffer can be reused.
 * @throws Error if the metadata is inconsistent, the buffer has the wrong size, or a value
 * is NaN or infinite.
 */
export function parseVae(meta: VaeMeta, buffer: ArrayBuffer): Vae {
  if (!meta || meta.format !== VAE_FORMAT) throw new Error(`Invalid VAE metadata: expected format ${VAE_FORMAT}.`);
  if (meta.imageSize !== IMAGE_SIZE) throw new Error(`Invalid VAE metadata: image size ${meta.imageSize}, expected ${IMAGE_SIZE}.`);
  if (meta.activation !== 'relu') throw new Error(`Invalid VAE metadata: activation ${meta.activation}, expected relu.`);
  const { latentSize, hiddenSizes } = meta;
  checkSizes(latentSize, hiddenSizes, 'Invalid VAE metadata');
  if (JSON.stringify(meta.tensors) !== JSON.stringify(vaeLayout(latentSize, hiddenSizes))) {
    throw new Error('Invalid VAE metadata: unexpected tensor layout.');
  }
  const count = vaeParameterCount(latentSize, hiddenSizes);
  if (meta.parameters !== count) throw new Error(`Invalid VAE metadata: ${meta.parameters} parameters, expected ${count}.`);
  if (ArrayBuffer.isView(buffer) || typeof buffer?.byteLength !== 'number') {
    throw new TypeError('parseVae expects an ArrayBuffer (e.g. from response.arrayBuffer()), not a typed array or other value.');
  }
  if (buffer.byteLength !== count * 4) {
    throw new Error(`Invalid VAE weights: expected ${count * 4} bytes (${count} float32 values) but got ${buffer.byteLength}.`);
  }
  const view = new DataView(buffer);
  const parameters = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value)) throw new Error(`Invalid VAE weights: value ${i} is ${value}.`);
    parameters[i] = value;
  }
  return vaeFromParameters(parameters, latentSize, hiddenSizes);
}

/** Buffers for encode, to reuse across calls. */
export function createEncoding(model: Vae): Encoding {
  return {
    hidden1: new Float32Array(model.encoder.hidden1.outputs),
    hidden2: new Float32Array(model.encoder.hidden2.outputs),
    mean: new Float32Array(model.latentSize),
    logVariance: new Float32Array(model.latentSize),
  };
}

/** Buffers for decode, to reuse across calls. */
export function createDecoding(model: Vae): Decoding {
  return {
    hidden1: new Float32Array(model.decoder.hidden1.outputs),
    hidden2: new Float32Array(model.decoder.hidden2.outputs),
    logits: new Float32Array(PIXELS),
    image: new Float32Array(PIXELS),
  };
}

// Shared accumulator for dense(); encode and decode are synchronous, so calls never overlap.
let sums = new Float64Array(PIXELS);

/**
 * output = b + Σ_i input[i] · W[i] (then ReLU if asked), adding one weight row per
 * nonzero input. Sums are kept in float64 and stored as float32.
 */
function dense(layer: DenseLayer, input: ArrayLike<number>, output: Float32Array, relu: boolean): void {
  const { inputs, outputs, weights, biases } = layer;
  if (sums.length < outputs) sums = new Float64Array(outputs);
  const acc = sums;
  for (let j = 0; j < outputs; j++) acc[j] = biases[j];
  for (let i = 0; i < inputs; i++) {
    const x = input[i];
    if (x === 0) continue;
    const row = i * outputs;
    for (let j = 0; j < outputs; j++) acc[j] += x * weights[row + j];
  }
  if (relu) for (let j = 0; j < outputs; j++) output[j] = acc[j] > 0 ? acc[j] : 0;
  else for (let j = 0; j < outputs; j++) output[j] = acc[j];
}

/**
 * Encodes one 28×28 image (784 values, 0..1, ink = 1): its code's mean μ and log-variance.
 * Pass `out` (from createEncoding) to reuse buffers; otherwise new ones are allocated.
 * About 60k multiply-adds for a typical digit.
 * @throws Error if the input does not have 784 values.
 */
export function encode(model: Vae, input: ArrayLike<number>, out: Encoding = createEncoding(model)): Encoding {
  if (!input || input.length !== PIXELS) throw new Error(`encode expects a ${PIXELS}-value image but got ${input?.length ?? input}.`);
  const { hidden1, hidden2, mean, logVariance } = model.encoder;
  dense(hidden1, input, out.hidden1, true);
  dense(hidden2, out.hidden1, out.hidden2, true);
  dense(mean, out.hidden2, out.mean, false);
  dense(logVariance, out.hidden2, out.logVariance, false);
  return out;
}

/**
 * Decodes one point of the latent plane into a 28×28 image of ink probabilities (0..1).
 * Pass `out` (from createDecoding) to reuse buffers. About 120k multiply-adds: well
 * under a millisecond, so the page decodes every frame.
 * @throws Error if `z` does not have latentSize finite values.
 */
export function decode(model: Vae, z: ArrayLike<number>, out: Decoding = createDecoding(model)): Decoding {
  if (!z || z.length !== model.latentSize) throw new Error(`decode expects ${model.latentSize} numbers but got ${z?.length ?? z}.`);
  for (let k = 0; k < z.length; k++) {
    if (!Number.isFinite(z[k])) throw new Error(`decode expects finite numbers but got ${z[k]}.`);
  }
  const { hidden1, hidden2, output } = model.decoder;
  dense(hidden1, z, out.hidden1, true);
  dense(hidden2, out.hidden1, out.hidden2, true);
  dense(output, out.hidden2, out.logits, false);
  const { logits, image } = out;
  for (let i = 0; i < PIXELS; i++) image[i] = 1 / (1 + Math.exp(-logits[i]));
  return out;
}

/**
 * Binary cross-entropy of a target image (0..1 per pixel) under decoded logits, summed
 * over pixels, in nats: −Σ [x log p + (1 − x) log(1 − p)] with p = sigmoid(logit),
 * computed as softplus(logit) − x·logit so it never overflows. With grey-level targets
 * it equals the expected loss of the image binarized at random (each pixel ink with
 * probability x).
 */
export function reconstructionLoss(logits: ArrayLike<number>, target: ArrayLike<number>): number {
  let loss = 0;
  for (let i = 0; i < target.length; i++) {
    const l = logits[i];
    loss += (l > 0 ? l : 0) + Math.log1p(Math.exp(-Math.abs(l))) - target[i] * l;
  }
  return loss;
}

/** KL(N(μ, σ²) ‖ N(0, I)) in nats: ½ Σ (σ² + μ² − 1 − log σ²). */
export function klDivergence(mean: ArrayLike<number>, logVariance: ArrayLike<number>): number {
  let kl = 0;
  for (let k = 0; k < mean.length; k++) kl += 0.5 * (Math.exp(logVariance[k]) + mean[k] * mean[k] - 1 - logVariance[k]);
  return kl;
}

/* ---------- Map data ---------- */

/** Codes of test digits for the map: `codes[2n]`, `codes[2n + 1]` is digit n's mean μ. */
export interface DigitCodes {
  count: number;
  codes: Float32Array;
  labels: Uint8Array;
}

/**
 * Parses `mnist-vae-codes-v1.bin`: `count` pairs of little-endian float32 (μ₁, μ₂),
 * then `count` uint8 labels.
 * @throws Error if the size is wrong, a code is not finite or a label is not a digit.
 */
export function parseCodes(buffer: ArrayBuffer, count: number): DigitCodes {
  if (!Number.isInteger(count) || count < 1) throw new Error(`parseCodes: invalid count ${count}.`);
  if (buffer?.byteLength !== count * 9) {
    throw new Error(`Invalid digit codes: expected ${count * 9} bytes (${count} × 2 float32 + ${count} labels) but got ${buffer?.byteLength}.`);
  }
  const view = new DataView(buffer);
  const codes = new Float32Array(count * 2);
  for (let i = 0; i < count * 2; i++) {
    codes[i] = view.getFloat32(i * 4, true);
    if (!Number.isFinite(codes[i])) throw new Error(`Invalid digit codes: value ${i} is ${codes[i]}.`);
  }
  const labels = new Uint8Array(buffer.slice(count * 8));
  if (labels.some(label => label >= CLASSES)) throw new Error('Invalid digit codes: a label is not a digit.');
  return { count, codes, labels };
}

/**
 * Parses `mnist-vae-examples-v1.bin`: `count` unmodified 28×28 uint8 MNIST test digits
 * (0 = background, 255 = ink). Returns images with values in 0..1.
 * @throws Error if the size is wrong.
 */
export function parseExamples(buffer: ArrayBuffer, count: number): Float32Array[] {
  if (!Number.isInteger(count) || count < 1) throw new Error(`parseExamples: invalid count ${count}.`);
  if (buffer?.byteLength !== count * PIXELS) {
    throw new Error(`Invalid example digits: expected ${count * PIXELS} bytes (${count} × 28×28 uint8) but got ${buffer?.byteLength}.`);
  }
  const bytes = new Uint8Array(buffer);
  return Array.from({ length: count }, (_, n) => {
    const image = new Float32Array(PIXELS);
    for (let i = 0; i < PIXELS; i++) image[i] = bytes[n * PIXELS + i] / 255;
    return image;
  });
}

/**
 * The labels of the k codes nearest to (x, y), nearest first, skipping index `exclude`.
 * A linear scan: about 0.1 ms for 5,000 codes.
 */
export function nearestLabels(map: DigitCodes, x: number, y: number, k: number, exclude = -1): number[] {
  const distances = new Float64Array(k).fill(Infinity);
  const labels: number[] = new Array(k).fill(-1);
  const { codes, count } = map;
  for (let n = 0; n < count; n++) {
    if (n === exclude) continue;
    const dx = codes[2 * n] - x;
    const dy = codes[2 * n + 1] - y;
    const d = dx * dx + dy * dy;
    if (d >= distances[k - 1]) continue;
    let slot = k - 1;
    while (slot > 0 && distances[slot - 1] > d) {
      distances[slot] = distances[slot - 1];
      labels[slot] = labels[slot - 1];
      slot--;
    }
    distances[slot] = d;
    labels[slot] = map.labels[n];
  }
  return labels.filter(label => label >= 0);
}

/** The most common label among `labels` (nearest first); a tie goes to the one with the nearest code. */
export function majorityLabel(labels: number[]): number {
  const votes = new Array(CLASSES).fill(0);
  for (const label of labels) votes[label]++;
  const top = Math.max(...votes);
  return labels.find(label => votes[label] === top) ?? -1;
}
