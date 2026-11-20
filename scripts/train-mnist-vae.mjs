#!/usr/bin/env node
/**
 * Trains the Autoencoder page's variational autoencoder on MNIST and exports it with the
 * codes of test digits for the page's map and example digits.
 *
 * Usage:
 *   node scripts/train-mnist-vae.mjs <mnist-data-dir> <output-dir> [options]
 *     --epochs N      training epochs (default 60)
 *     --seed N        PRNG seed (default 20261120)
 *     --workers N     worker threads (default 4); results do not depend on it
 *     --kl-warmup N   epochs over which the KL term's weight rises from 0 to 1 (default 0: none)
 *     --hidden A,B    encoder widths, mirrored by the decoder (default 256,128)
 *     --holdout N     tuning mode: train on the first 60000-N training images, report
 *                     scores on the last N and write only the weights (test set unused)
 *   node scripts/train-mnist-vae.mjs --check-gradients
 *     compares backpropagation with finite differences and exits
 *
 * <mnist-data-dir> must contain the four gzipped MNIST IDX files, e.g. downloaded from
 * https://storage.googleapis.com/cvdf-datasets/mnist/ or https://ossci-datasets.s3.amazonaws.com/mnist/:
 *   train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz  t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz
 * They are treated as untrusted input: sizes, headers and SHA-256 digests of the
 * decompressed contents are verified before use.
 *
 * Data: the MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and
 * Christopher J. C. Burges, licensed under CC BY-SA 3.0
 * (https://creativecommons.org/licenses/by-sa/3.0/). The exported example digits are
 * unmodified MNIST test images and remain under that license.
 *
 * Writes to <output-dir>:
 *   mnist-vae-v1.bin           little-endian float32 parameters (layout in src/vae/model.ts)
 *   mnist-vae-v1.json          sizes, tensor layout, the map's extent and class centers,
 *                              scores and training details
 *   mnist-vae-codes-v1.bin     the code (mean μ) and label of every second test digit
 *   mnist-vae-examples-v1.bin  ten random test digits per class, 28×28 uint8
 *
 * Model and objective: encoder 784 → 256 → 128 → (μ, log σ²) ∈ R², decoder 2 → 128 →
 * 256 → 784 with ReLU and a sigmoid output, a Bernoulli likelihood (binary cross-entropy
 * on the grey levels, i.e. the expected loss of a randomly binarized image) plus
 * KL(N(μ, σ²) ‖ N(0, I)). The widths are half the usual 512/256 so the weights take
 * 1.9 MB instead of 4.3 MB. That costs some fidelity: trained for 15 epochs and scored on
 * 10,000 held-out training digits (--holdout 10000), 512/256 reached an ELBO of −140.75
 * nats against −144.85 for 256/128 (reconstruction from the mean code 133.3 against
 * 137.5), and separated the classes about as well (77.6% against 76.6% 5-NN accuracy).
 * The objective is the plain ELBO from the first step: a KL warm-up (--kl-warmup 3) was
 * tried the same way and made no difference to the ELBO (−144.83 against −144.85) or to
 * how well the classes separate (76.6% 5-NN accuracy for both), so it was left out.
 *
 * The forward passes and losses are imported from src/vae/model.ts (Node strips its
 * types), so the trainer, the reported scores and the browser all run the same code.
 * This file adds data loading, the reparameterization (z = μ + σ·ε), backpropagation
 * computed from encode's and decode's activations, and Adam (mini-batches of 128,
 * learning rate warming up to 1e-3 and cosine-decaying to 1e-5). Training is
 * data-parallel over worker threads: each mini-batch is split into fixed shards whose
 * gradients are summed in a fixed order, and every sample's noise comes from its own
 * seed, so a given seed reproduces the exact same weights whatever the worker count.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import {
  CLASSES, IMAGE_SIZE, PIXELS, VAE_FORMAT, createDecoding, createEncoding, decode, encode, klDivergence, majorityLabel,
  nearestLabels, parseCodes, parseExamples, parseVae, reconstructionLoss, vaeFromParameters, vaeLayout, vaeParameterCount,
} from '../src/vae/model.ts';
import { parseCnnWeights, runCnn } from '../src/cnn/model.ts';

// ---------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------

const LATENT = 2;
const BATCH = 128;
const SHARD = 8; // samples per gradient shard; 16 shards per batch
const SHARDS = BATCH / SHARD;
const UPDATE_CHUNK = 8192; // parameters per optimizer task
const EVAL_CHUNK = 250; // images per evaluation task
const LEARNING_RATE = 1e-3; // peak; cosine decay to FINAL_LEARNING_RATE
const FINAL_LEARNING_RATE = 1e-5;
const WARMUP_STEPS = 200;
const ADAM = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
const EVAL_SEED = 0x5eed1234; // the noise each evaluated image gets, the same every epoch
const KNN = 5; // neighbors for the latent k-NN scores
const CODE_STRIDE = 2; // every second test digit goes on the map
const EXAMPLES_PER_CLASS = 10;
const CENTER_RADIUS = 0.25; // neighborhood for choosing class centers
const DEFAULTS = { epochs: 60, seed: 20261120, workers: Math.min(4, availableParallelism()), holdout: 0, klWarmup: 0, hidden: [256, 128] };

const FILES = {
  trainImages: { name: 'train-images-idx3-ubyte', sha256: 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db' },
  trainLabels: { name: 'train-labels-idx1-ubyte', sha256: '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5' },
  testImages: { name: 't10k-images-idx3-ubyte', sha256: '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7' },
  testLabels: { name: 't10k-labels-idx1-ubyte', sha256: 'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2' },
};

// Shared control block used to run phases on the workers.
const CTRL = { generation: 0, command: 1, tasks: 2, next: 3, done: 4, failed: 5, batchSize: 6, shardsUsed: 7, length: 8 };
const COMMAND = { train: 1, update: 2, evaluate: 3, exit: 4 };
// hyper: learning rate, Adam bias corrections, gradient scale (1 / batch size), KL weight.
const HYPER = { rate: 0, correction1: 1, correction2: 2, scale: 3, beta: 4, length: 5 };

// ---------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------

/** mulberry32: small, fast, seedable PRNG returning floats in [0, 1). */
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fills `out` with independent standard normal samples (Box–Muller). */
function gaussians(random, out) {
  for (let k = 0; k < out.length; k += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    out[k] = radius * Math.cos(angle);
    if (k + 1 < out.length) out[k + 1] = radius * Math.sin(angle);
  }
}

/** The noise of evaluated image n: fixed, so scores from different epochs compare. */
const evalSeed = n => (EVAL_SEED ^ Math.imul(n + 1, 0x9e3779b1)) >>> 0;

/** z = μ + σ·ε, σ = exp(½ log σ²): the reparameterization. */
function sampleCode(encoding, epsilon, z) {
  for (let k = 0; k < z.length; k++) z[k] = encoding.mean[k] + Math.exp(0.5 * encoding.logVariance[k]) * epsilon[k];
}

function offsetsOf(hidden) {
  const offsets = {};
  for (const { name, offset } of vaeLayout(LATENT, hidden)) offsets[name] = offset;
  return offsets;
}

/**
 * Backpropagation of the negative ELBO for one sample, computed from encode's and
 * decode's activations: a ReLU unit is active iff its output is > 0. Accumulates
 * d(reconstruction + β·KL)/d(parameter) into `grads` (flat, file order).
 */
function createBackprop(model) {
  const offsets = offsetsOf(model.hiddenSizes);
  const { encoder, decoder } = model;
  const dLogits = new Float64Array(PIXELS);
  const dDecoder2 = new Float64Array(decoder.hidden2.outputs);
  const dDecoder1 = new Float64Array(decoder.hidden1.outputs);
  const dZ = new Float64Array(LATENT);
  const dMean = new Float64Array(LATENT);
  const dLogVariance = new Float64Array(LATENT);
  const fromMean = new Float64Array(encoder.mean.inputs);
  const fromLogVariance = new Float64Array(encoder.logVariance.inputs);
  const dEncoder2 = new Float64Array(encoder.hidden2.outputs);
  const dEncoder1 = new Float64Array(encoder.hidden1.outputs);

  /**
   * One dense layer y = b + Σ_i x_i·W[i]: adds dL/dW and dL/db, and writes dL/dx into
   * `dx` if given. When x is a ReLU output, a zero input is an inactive unit whose
   * gradient is zero, so its row is skipped entirely.
   */
  function layer(name, layerWeights, x, dy, dx, reluInput, grads) {
    const { inputs, outputs, weights } = layerWeights;
    const w = offsets[`${name}.weights`];
    const b = offsets[`${name}.biases`];
    for (let j = 0; j < outputs; j++) grads[b + j] += dy[j];
    for (let i = 0; i < inputs; i++) {
      const xi = x[i];
      const row = i * outputs;
      if (xi === 0) {
        if (dx) {
          let sum = 0;
          if (!reluInput) for (let j = 0; j < outputs; j++) sum += weights[row + j] * dy[j];
          dx[i] = sum;
        }
        continue;
      }
      const g = w + row;
      if (dx) {
        let sum = 0;
        for (let j = 0; j < outputs; j++) {
          grads[g + j] += xi * dy[j];
          sum += weights[row + j] * dy[j];
        }
        dx[i] = sum;
      } else {
        for (let j = 0; j < outputs; j++) grads[g + j] += xi * dy[j];
      }
    }
  }

  return function backward(input, encoding, decoding, z, epsilon, beta, grads) {
    // Sigmoid + binary cross-entropy: d/dlogit = p − x.
    for (let i = 0; i < PIXELS; i++) dLogits[i] = decoding.image[i] - input[i];
    layer('decoder.output', decoder.output, decoding.hidden2, dLogits, dDecoder2, true, grads);
    layer('decoder.hidden2', decoder.hidden2, decoding.hidden1, dDecoder2, dDecoder1, true, grads);
    layer('decoder.hidden1', decoder.hidden1, z, dDecoder1, dZ, false, grads);
    // z = μ + exp(½ s)·ε and KL = ½ Σ (exp(s) + μ² − 1 − s), with s = log σ².
    for (let k = 0; k < LATENT; k++) {
      const sigma = Math.exp(0.5 * encoding.logVariance[k]);
      dMean[k] = dZ[k] + beta * encoding.mean[k];
      dLogVariance[k] = 0.5 * dZ[k] * epsilon[k] * sigma + 0.5 * beta * (sigma * sigma - 1);
    }
    layer('encoder.mean', encoder.mean, encoding.hidden2, dMean, fromMean, true, grads);
    layer('encoder.logVariance', encoder.logVariance, encoding.hidden2, dLogVariance, fromLogVariance, true, grads);
    for (let j = 0; j < dEncoder2.length; j++) dEncoder2[j] = fromMean[j] + fromLogVariance[j];
    layer('encoder.hidden2', encoder.hidden2, encoding.hidden1, dEncoder2, dEncoder1, true, grads);
    layer('encoder.hidden1', encoder.hidden1, input, dEncoder1, null, false, grads);
  };
}

// ---------------------------------------------------------------------------------
// Worker: trains gradient shards, applies Adam to parameter chunks, evaluates
// ---------------------------------------------------------------------------------

function workerMain({ shared, workerCount, hidden, params: paramCount }) {
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const model = vaeFromParameters(params, LATENT, hidden);
  const adamM = new Float64Array(shared.adamM);
  const adamV = new Float64Array(shared.adamV);
  const hyper = new Float64Array(shared.hyper);
  const grads = new Float32Array(shared.grads);
  const batch = new Uint32Array(shared.batch);
  const shardStats = new Float64Array(shared.shardStats);
  const trainImages = new Uint8Array(shared.trainImages);
  const evalImages = new Uint8Array(shared.evalImages);
  const evalStats = new Float64Array(shared.evalStats);
  const backward = createBackprop(model);
  const encoding = createEncoding(model);
  const decoding = createDecoding(model);
  const input = new Float32Array(PIXELS);
  const epsilon = new Float64Array(LATENT);
  const z = new Float64Array(LATENT);
  const sums = new Float64Array(UPDATE_CHUNK);

  function trainShard(shard) {
    const size = Atomics.load(control, CTRL.batchSize);
    const first = shard * SHARD;
    const last = Math.min(first + SHARD, size);
    const shardGrads = grads.subarray(shard * paramCount, (shard + 1) * paramCount);
    shardGrads.fill(0);
    const beta = hyper[HYPER.beta];
    let reconstruction = 0;
    let kl = 0;
    for (let n = first; n < last; n++) {
      const index = batch[2 * n];
      for (let i = 0; i < PIXELS; i++) input[i] = trainImages[index * PIXELS + i] / 255;
      encode(model, input, encoding);
      gaussians(mulberry32(batch[2 * n + 1]), epsilon);
      sampleCode(encoding, epsilon, z);
      decode(model, z, decoding);
      reconstruction += reconstructionLoss(decoding.logits, input);
      kl += klDivergence(encoding.mean, encoding.logVariance);
      backward(input, encoding, decoding, z, epsilon, beta, shardGrads);
    }
    shardStats[2 * shard] = reconstruction;
    shardStats[2 * shard + 1] = kl;
  }

  function updateChunk(chunk) {
    const lo = chunk * UPDATE_CHUNK;
    const hi = Math.min(lo + UPDATE_CHUNK, paramCount);
    const shards = Atomics.load(control, CTRL.shardsUsed);
    const rate = hyper[HYPER.rate];
    const correction1 = hyper[HYPER.correction1];
    const correction2 = hyper[HYPER.correction2];
    const scale = hyper[HYPER.scale];
    sums.fill(0);
    for (let s = 0; s < shards; s++) {
      const base = s * paramCount;
      for (let i = lo; i < hi; i++) sums[i - lo] += grads[base + i];
    }
    const { beta1, beta2, epsilon: eps } = ADAM;
    for (let i = lo; i < hi; i++) {
      const g = sums[i - lo] * scale;
      const m = beta1 * adamM[i] + (1 - beta1) * g;
      const v = beta2 * adamV[i] + (1 - beta2) * g * g;
      adamM[i] = m;
      adamV[i] = v;
      params[i] -= (rate * (m / correction1)) / (Math.sqrt(v / correction2) + eps);
    }
  }

  function evaluateChunk(chunk) {
    const count = evalImages.length / PIXELS;
    const lo = chunk * EVAL_CHUNK;
    const hi = Math.min(lo + EVAL_CHUNK, count);
    let reconstruction = 0;
    let kl = 0;
    let atMean = 0;
    for (let n = lo; n < hi; n++) {
      for (let i = 0; i < PIXELS; i++) input[i] = evalImages[n * PIXELS + i] / 255;
      encode(model, input, encoding);
      gaussians(mulberry32(evalSeed(n)), epsilon);
      sampleCode(encoding, epsilon, z);
      reconstruction += reconstructionLoss(decode(model, z, decoding).logits, input);
      kl += klDivergence(encoding.mean, encoding.logVariance);
      atMean += reconstructionLoss(decode(model, encoding.mean, decoding).logits, input);
    }
    evalStats[3 * chunk] = reconstruction;
    evalStats[3 * chunk + 1] = kl;
    evalStats[3 * chunk + 2] = atMean;
  }

  let generation = 0;
  for (;;) {
    Atomics.wait(control, CTRL.generation, generation);
    generation = Atomics.load(control, CTRL.generation);
    const command = Atomics.load(control, CTRL.command);
    if (command === COMMAND.exit) break;
    try {
      const tasks = Atomics.load(control, CTRL.tasks);
      for (let task = Atomics.add(control, CTRL.next, 1); task < tasks; task = Atomics.add(control, CTRL.next, 1)) {
        if (command === COMMAND.train) trainShard(task);
        else if (command === COMMAND.update) updateChunk(task);
        else if (command === COMMAND.evaluate) evaluateChunk(task);
      }
    } catch (error) {
      Atomics.store(control, CTRL.failed, 1);
      parentPort.postMessage(String(error?.stack ?? error));
    }
    if (Atomics.add(control, CTRL.done, 1) + 1 === workerCount) Atomics.notify(control, CTRL.done);
  }
}

// ---------------------------------------------------------------------------------
// Main thread
// ---------------------------------------------------------------------------------

function parseArguments(argv) {
  const positional = [];
  const options = { ...DEFAULTS, checkGradients: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--check-gradients') options.checkGradients = true;
    else if (['--epochs', '--seed', '--workers', '--holdout', '--kl-warmup'].includes(arg)) {
      const value = Number(argv[++i]);
      if (!Number.isInteger(value) || value < 0) throw new Error(`${arg} needs a non-negative integer`);
      options[arg === '--kl-warmup' ? 'klWarmup' : arg.slice(2)] = value;
    } else if (arg === '--hidden') {
      const widths = String(argv[++i]).split(',').map(Number);
      if (widths.length !== 2 || !widths.every(w => Number.isInteger(w) && w > 0 && w <= 4096)) {
        throw new Error('--hidden needs two positive widths, e.g. 256,128');
      }
      options.hidden = widths;
    } else if (arg.startsWith('--')) throw new Error(`Unknown option ${arg}`);
    else positional.push(arg);
  }
  [options.dataDir, options.outDir] = positional;
  options.workers = Math.max(1, options.workers);
  return options;
}

/** Reads, decompresses and validates one IDX file. */
function readIdx(dir, { name, sha256 }, expectedMagic, expectedCount) {
  const path = join(dir, `${name}.gz`);
  if (!existsSync(path)) throw new Error(`Missing ${path}; see the usage notes at the top of this script.`);
  if (statSync(path).size > 20_000_000) throw new Error(`${path} is unexpectedly large`);
  const bytes = gunzipSync(readFileSync(path), { maxOutputLength: 50_000_000 });
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== sha256) throw new Error(`${path}: contents do not match the official MNIST file (sha256 ${digest})`);
  const magic = bytes.readUInt32BE(0);
  const count = bytes.readUInt32BE(4);
  if (magic !== expectedMagic || count !== expectedCount) throw new Error(`${path}: bad IDX header`);
  if (magic === 2051) {
    if (bytes.readUInt32BE(8) !== IMAGE_SIZE || bytes.readUInt32BE(12) !== IMAGE_SIZE || bytes.length !== 16 + count * PIXELS) {
      throw new Error(`${path}: expected ${count} images of 28×28`);
    }
    return bytes.subarray(16);
  }
  if (bytes.length !== 8 + count) throw new Error(`${path}: expected ${count} labels`);
  const labels = bytes.subarray(8);
  if (labels.some(label => label >= CLASSES)) throw new Error(`${path}: label out of range`);
  return labels;
}

function loadMnist(dir) {
  return {
    trainImages: readIdx(dir, FILES.trainImages, 2051, 60000),
    trainLabels: readIdx(dir, FILES.trainLabels, 2049, 60000),
    testImages: readIdx(dir, FILES.testImages, 2051, 10000),
    testLabels: readIdx(dir, FILES.testLabels, 2049, 10000),
  };
}

function sharedCopy(bytes) {
  const copy = new Uint8Array(new SharedArrayBuffer(bytes.length));
  copy.set(bytes);
  return copy;
}

/**
 * He-normal weights (std √(2 / fan-in)) for layers followed by a ReLU, √(1 / fan-in) for
 * the code and output layers, zero biases, and the output biases set to the logit of the
 * mean training image, so the decoder starts by drawing the average digit.
 */
function initialize(params, hidden, random, trainImages) {
  const model = vaeFromParameters(params, LATENT, hidden);
  const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const layers = [
    [model.encoder.hidden1, 2], [model.encoder.hidden2, 2], [model.encoder.mean, 1], [model.encoder.logVariance, 1],
    [model.decoder.hidden1, 2], [model.decoder.hidden2, 2], [model.decoder.output, 1],
  ];
  for (const [layer, gain] of layers) {
    const std = Math.sqrt(gain / layer.inputs);
    for (let i = 0; i < layer.weights.length; i++) layer.weights[i] = std * normal();
    layer.biases.fill(0);
  }
  const count = trainImages.length / PIXELS;
  const mean = new Float64Array(PIXELS);
  for (let n = 0; n < count; n++) for (let i = 0; i < PIXELS; i++) mean[i] += trainImages[n * PIXELS + i];
  for (let i = 0; i < PIXELS; i++) {
    const p = Math.min(1 - 1e-3, Math.max(1e-3, mean[i] / (255 * count)));
    model.decoder.output.biases[i] = Math.log(p / (1 - p));
  }
}

function encodeFloats(values) {
  const bytes = new ArrayBuffer(values.length * 4);
  const view = new DataView(bytes);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i], true);
  return new Uint8Array(bytes);
}

function toArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

const percent = value => `${(value * 100).toFixed(2)}%`;
const round = (value, digits = 4) => Number(value.toFixed(digits));

function metaFor(hidden) {
  return {
    format: VAE_FORMAT,
    imageSize: IMAGE_SIZE,
    latentSize: LATENT,
    hiddenSizes: hidden,
    activation: 'relu',
    parameters: vaeParameterCount(LATENT, hidden),
    tensors: vaeLayout(LATENT, hidden),
  };
}

/** Means (μ) of every image, as a DigitCodes map for nearestLabels. */
function codesOf(model, images, labels) {
  const encoding = createEncoding(model);
  const input = new Float32Array(PIXELS);
  const codes = new Float32Array(labels.length * 2);
  for (let n = 0; n < labels.length; n++) {
    for (let i = 0; i < PIXELS; i++) input[i] = images[n * PIXELS + i] / 255;
    encode(model, input, encoding);
    codes[2 * n] = encoding.mean[0];
    codes[2 * n + 1] = encoding.mean[1];
  }
  return { count: labels.length, codes, labels: Uint8Array.from(labels) };
}

/** Share of `queries` whose k nearest codes in `reference` vote for their own label, overall and per class. */
function knnAccuracy(queries, reference, k, leaveOneOut = false) {
  const perClass = Array.from({ length: CLASSES }, () => ({ total: 0, correct: 0 }));
  let correct = 0;
  for (let n = 0; n < queries.count; n++) {
    const votes = nearestLabels(reference, queries.codes[2 * n], queries.codes[2 * n + 1], k, leaveOneOut ? n : -1);
    const label = queries.labels[n];
    perClass[label].total++;
    if (majorityLabel(votes) === label) {
      correct++;
      perClass[label].correct++;
    }
  }
  return { accuracy: correct / queries.count, perClass: perClass.map(c => c.correct / c.total) };
}

async function train(options) {
  const startTime = performance.now();
  const data = loadMnist(options.dataDir);
  const { hidden, holdout } = options;
  if (holdout >= 60000) throw new Error('--holdout must be below 60000');
  const PARAMS = vaeParameterCount(LATENT, hidden);
  const trainCount = 60000 - holdout;
  const evalImages = holdout ? data.trainImages.subarray(trainCount * PIXELS) : data.testImages;
  const evalLabels = holdout ? data.trainLabels.subarray(trainCount) : data.testLabels;
  const evalName = holdout ? `holdout (${holdout})` : 'test';
  const evalCount = evalLabels.length;
  const evalChunks = Math.ceil(evalCount / EVAL_CHUNK);

  const shared = {
    control: new SharedArrayBuffer(CTRL.length * 4),
    params: new SharedArrayBuffer(PARAMS * 4),
    adamM: new SharedArrayBuffer(PARAMS * 8),
    adamV: new SharedArrayBuffer(PARAMS * 8),
    hyper: new SharedArrayBuffer(HYPER.length * 8),
    grads: new SharedArrayBuffer(SHARDS * PARAMS * 4),
    batch: new SharedArrayBuffer(BATCH * 2 * 4),
    shardStats: new SharedArrayBuffer(SHARDS * 2 * 8),
    trainImages: sharedCopy(data.trainImages.subarray(0, trainCount * PIXELS)).buffer,
    evalImages: sharedCopy(evalImages).buffer,
    evalStats: new SharedArrayBuffer(evalChunks * 3 * 8),
  };
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const hyper = new Float64Array(shared.hyper);
  const batch = new Uint32Array(shared.batch);
  const shardStats = new Float64Array(shared.shardStats);
  const evalStats = new Float64Array(shared.evalStats);

  const random = mulberry32(options.seed);
  initialize(params, hidden, random, data.trainImages.subarray(0, trainCount * PIXELS));

  const workerCount = options.workers;
  const workerErrors = [];
  const workers = Array.from({ length: workerCount }, () => {
    const worker = new Worker(new URL(import.meta.url), { workerData: { shared, workerCount, hidden, params: PARAMS } });
    worker.on('message', message => workerErrors.push(message));
    worker.on('error', error => workerErrors.push(String(error?.stack ?? error)));
    return worker;
  });

  async function runPhase(command, tasks) {
    Atomics.store(control, CTRL.command, command);
    Atomics.store(control, CTRL.tasks, tasks);
    Atomics.store(control, CTRL.next, 0);
    Atomics.store(control, CTRL.done, 0);
    Atomics.add(control, CTRL.generation, 1);
    Atomics.notify(control, CTRL.generation);
    if (command === COMMAND.exit) return;
    for (;;) {
      const done = Atomics.load(control, CTRL.done);
      if (done === workerCount) break;
      if (Atomics.wait(control, CTRL.done, done, 30_000) === 'timed-out') {
        await new Promise(resolve => setTimeout(resolve, 10)); // let worker errors arrive
        if (workerErrors.length) break;
      }
    }
    if (Atomics.load(control, CTRL.failed) || workerErrors.length) {
      await new Promise(resolve => setTimeout(resolve, 100));
      throw new Error(`Worker failed:\n${workerErrors.join('\n')}`);
    }
  }

  async function evaluate() {
    await runPhase(COMMAND.evaluate, evalChunks);
    let reconstruction = 0;
    let kl = 0;
    let atMean = 0;
    for (let c = 0; c < evalChunks; c++) {
      reconstruction += evalStats[3 * c];
      kl += evalStats[3 * c + 1];
      atMean += evalStats[3 * c + 2];
    }
    return { reconstruction: reconstruction / evalCount, kl: kl / evalCount, atMean: atMean / evalCount };
  }

  const order = Uint32Array.from({ length: trainCount }, (_, i) => i);
  const stepsPerEpoch = Math.ceil(trainCount / BATCH);
  const totalSteps = options.epochs * stepsPerEpoch;
  const warmupSteps = options.klWarmup * stepsPerEpoch;
  const learningRate = step => Math.min(1, step / WARMUP_STEPS)
    * (FINAL_LEARNING_RATE + 0.5 * (LEARNING_RATE - FINAL_LEARNING_RATE) * (1 + Math.cos((Math.PI * step) / totalSteps)));
  const klWeight = step => (warmupSteps ? Math.min(1, step / warmupSteps) : 1);

  console.log(`MNIST VAE: 784 → ${hidden.join(' → ')} → 2 → ${[...hidden].reverse().join(' → ')} → 784, ${PARAMS} parameters, `
    + `${trainCount} training images, ${options.epochs} epochs, batch ${BATCH}, KL warm-up ${options.klWarmup} epochs, `
    + `seed ${options.seed}, ${workerCount} workers`);
  let step = 0;
  let scores = null;
  const history = [];
  try {
    for (let epoch = 1; epoch <= options.epochs; epoch++) {
      const epochStart = performance.now();
      for (let i = trainCount - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      let reconstruction = 0;
      let kl = 0;
      for (let start = 0; start < trainCount; start += BATCH) {
        const size = Math.min(BATCH, trainCount - start);
        for (let n = 0; n < size; n++) {
          batch[2 * n] = order[start + n];
          batch[2 * n + 1] = Math.floor(random() * 4294967296);
        }
        const shards = Math.ceil(size / SHARD);
        Atomics.store(control, CTRL.batchSize, size);
        hyper[HYPER.beta] = klWeight(step);
        await runPhase(COMMAND.train, shards);
        for (let s = 0; s < shards; s++) {
          reconstruction += shardStats[2 * s];
          kl += shardStats[2 * s + 1];
        }
        step++;
        hyper[HYPER.rate] = learningRate(step);
        hyper[HYPER.correction1] = 1 - ADAM.beta1 ** step;
        hyper[HYPER.correction2] = 1 - ADAM.beta2 ** step;
        hyper[HYPER.scale] = 1 / size;
        Atomics.store(control, CTRL.shardsUsed, shards);
        await runPhase(COMMAND.update, Math.ceil(PARAMS / UPDATE_CHUNK));
      }
      if (!Number.isFinite(reconstruction + kl)) throw new Error(`Training diverged in epoch ${epoch}`);
      scores = await evaluate();
      const elbo = -(scores.reconstruction + scores.kl);
      history.push({ epoch, elbo: round(elbo, 2), reconstructionAtMean: round(scores.atMean, 2) });
      console.log(`epoch ${String(epoch).padStart(3)}/${options.epochs}  lr ${hyper[HYPER.rate].toExponential(2)}  `
        + `KL weight ${hyper[HYPER.beta].toFixed(2)}  train reconstruction ${(reconstruction / trainCount).toFixed(2)} `
        + `KL ${(kl / trainCount).toFixed(2)}  ${evalName} ELBO ${elbo.toFixed(2)} (reconstruction ${scores.reconstruction.toFixed(2)}, `
        + `KL ${scores.kl.toFixed(2)}, at the mean ${scores.atMean.toFixed(2)})  ${((performance.now() - epochStart) / 1000).toFixed(1)} s`);
    }
  } finally {
    await runPhase(COMMAND.exit, 0); // idle workers leave their loop; terminate() covers the rest
    await Promise.all(workers.map(worker => worker.terminate()));
  }
  const seconds = (performance.now() - startTime) / 1000;
  mkdirSync(options.outDir, { recursive: true });
  if (holdout) {
    const weightsPath = join(options.outDir, 'mnist-vae-v1.bin');
    writeFileSync(weightsPath, encodeFloats(params));
    const model = parseVae(metaFor(hidden), toArrayBuffer(readFileSync(weightsPath)));
    const reference = codesOf(model, data.trainImages.subarray(0, 20000 * PIXELS), data.trainLabels.subarray(0, 20000));
    const held = codesOf(model, evalImages, evalLabels);
    const knn = knnAccuracy(held, reference, KNN);
    console.log(`Holdout: ELBO ${(-(scores.reconstruction + scores.kl)).toFixed(2)}, reconstruction at the mean `
      + `${scores.atMean.toFixed(2)}, ${KNN}-NN accuracy ${percent(knn.accuracy)} (against 20,000 training codes) `
      + `after ${seconds.toFixed(0)} s; wrote ${weightsPath} only.`);
    return;
  }
  exportModel(options, data, params, { seconds, history, workerCount });
}

/**
 * Writes every file, then re-reads the weights and scores them with model.ts exactly as
 * the page will.
 */
function exportModel(options, data, params, details) {
  const { hidden, outDir } = options;
  const meta = metaFor(hidden);
  const weightsPath = join(outDir, 'mnist-vae-v1.bin');
  writeFileSync(weightsPath, encodeFloats(params));
  const model = parseVae(meta, toArrayBuffer(readFileSync(weightsPath)));
  const { testImages, testLabels, trainImages, trainLabels } = data;
  const testCount = testLabels.length;

  // Scores on all 10,000 test digits: the ELBO with the same noise as during training's
  // evaluations, and the reconstruction from the mean code, which is what the page shows.
  const encoding = createEncoding(model);
  const decoding = createDecoding(model);
  const input = new Float32Array(PIXELS);
  const epsilon = new Float64Array(LATENT);
  const z = new Float64Array(LATENT);
  const testCodes = { count: testCount, codes: new Float32Array(testCount * 2), labels: Uint8Array.from(testLabels) };
  const cnnPath = new URL('../public/models/mnist-cnn-v1.bin', import.meta.url);
  const cnn = existsSync(cnnPath) ? parseCnnWeights(toArrayBuffer(readFileSync(cnnPath))) : null;
  const meanImage = new Float64Array(PIXELS);
  for (let n = 0; n < trainLabels.length; n++) for (let i = 0; i < PIXELS; i++) meanImage[i] += trainImages[n * PIXELS + i];
  const meanLogits = Float64Array.from(meanImage, total => {
    const p = Math.min(1 - 1e-6, Math.max(1e-6, total / (255 * trainLabels.length)));
    return Math.log(p / (1 - p));
  });
  let reconstruction = 0;
  let kl = 0;
  let atMean = 0;
  let baseline = 0;
  let readCorrectly = 0;
  const readPerClass = Array.from({ length: CLASSES }, () => ({ total: 0, correct: 0 }));
  for (let n = 0; n < testCount; n++) {
    for (let i = 0; i < PIXELS; i++) input[i] = testImages[n * PIXELS + i] / 255;
    encode(model, input, encoding);
    testCodes.codes[2 * n] = encoding.mean[0];
    testCodes.codes[2 * n + 1] = encoding.mean[1];
    gaussians(mulberry32(evalSeed(n)), epsilon);
    sampleCode(encoding, epsilon, z);
    reconstruction += reconstructionLoss(decode(model, z, decoding).logits, input);
    kl += klDivergence(encoding.mean, encoding.logVariance);
    atMean += reconstructionLoss(decode(model, encoding.mean, decoding).logits, input);
    baseline += reconstructionLoss(meanLogits, input);
    if (cnn) {
      const label = testLabels[n];
      readPerClass[label].total++;
      if (argmax(runCnn(cnn, decoding.image).probabilities) === label) {
        readCorrectly++;
        readPerClass[label].correct++;
      }
    }
  }
  reconstruction /= testCount;
  kl /= testCount;
  atMean /= testCount;
  baseline /= testCount;

  // How well the classes separate: k-NN on the codes, against all training codes, and
  // leave-one-out among the codes shipped for the map (what tests/vae.test.ts checks).
  const trainCodes = codesOf(model, trainImages, trainLabels);
  const knnTest = knnAccuracy(testCodes, trainCodes, KNN);
  const mapIndices = Array.from({ length: Math.ceil(testCount / CODE_STRIDE) }, (_, i) => i * CODE_STRIDE);
  const map = {
    count: mapIndices.length,
    codes: Float32Array.from(mapIndices.flatMap(n => [testCodes.codes[2 * n], testCodes.codes[2 * n + 1]])),
    labels: Uint8Array.from(mapIndices, n => testLabels[n]),
  };
  const knnMap = knnAccuracy(map, map, KNN, true);

  // The map's extent: a square around the origin holding 99.5% of the test codes.
  const reach = Float64Array.from({ length: testCount }, (_, n) =>
    Math.max(Math.abs(testCodes.codes[2 * n]), Math.abs(testCodes.codes[2 * n + 1]))).sort();
  const extent = Math.ceil(reach[Math.floor(0.995 * (testCount - 1))] * 2) / 2 + 0.5;

  // Class centers: each digit's purest dense spot, the test code with the most codes of
  // its own class and the fewest of others within CENTER_RADIUS, whose decoded image the
  // Digits page's CNN also reads as that digit.
  const classCenters = [];
  const centerReadings = [];
  for (let digit = 0; digit < CLASSES; digit++) {
    const candidates = [];
    for (let n = 0; n < testCount; n++) {
      if (testLabels[n] !== digit) continue;
      const x = testCodes.codes[2 * n];
      const y = testCodes.codes[2 * n + 1];
      let score = 0;
      for (let m = 0; m < testCount; m++) {
        const dx = testCodes.codes[2 * m] - x;
        const dy = testCodes.codes[2 * m + 1] - y;
        if (dx * dx + dy * dy < CENTER_RADIUS * CENTER_RADIUS) score += testLabels[m] === digit ? 1 : -1;
      }
      candidates.push({ x, y, score });
    }
    candidates.sort((a, b) => b.score - a.score);
    const chosen = candidates.find(({ x, y }) => {
      if (!cnn) return true;
      const p = runCnn(cnn, decode(model, [x, y], decoding).image).probabilities;
      return argmax(p) === digit && p[digit] >= 0.9;
    }) ?? candidates[0];
    classCenters.push([round(chosen.x), round(chosen.y)]);
    if (cnn) centerReadings.push(round(runCnn(cnn, decode(model, [chosen.x, chosen.y], decoding).image).probabilities[digit], 3));
  }

  // Codes for the map, and example digits: ten per class, drawn at random from the
  // digits on the map, so each example's code is also in the codes file.
  const codesPath = join(outDir, 'mnist-vae-codes-v1.bin');
  const codeBytes = new Uint8Array(map.count * 9);
  codeBytes.set(encodeFloats(map.codes));
  codeBytes.set(map.labels, map.count * 8);
  writeFileSync(codesPath, codeBytes);
  const exampleIndices = [];
  for (let digit = 0; digit < CLASSES; digit++) {
    const pool = mapIndices.filter(n => testLabels[n] === digit);
    const random = mulberry32(options.seed + 1 + digit);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    exampleIndices.push(...pool.slice(0, EXAMPLES_PER_CLASS).sort((a, b) => a - b));
  }
  const examplesPath = join(outDir, 'mnist-vae-examples-v1.bin');
  const exampleBytes = new Uint8Array(exampleIndices.length * PIXELS);
  exampleIndices.forEach((n, e) => exampleBytes.set(testImages.subarray(n * PIXELS, (n + 1) * PIXELS), e * PIXELS));
  writeFileSync(examplesPath, exampleBytes);

  // Read the map files back the way the page does.
  const shippedCodes = parseCodes(toArrayBuffer(readFileSync(codesPath)), map.count);
  const examples = parseExamples(toArrayBuffer(readFileSync(examplesPath)), exampleIndices.length);
  let worstExample = 0;
  let exampleReconstruction = 0;
  examples.forEach((image, e) => {
    const { mean } = encode(model, image, encoding);
    const slot = exampleIndices[e] / CODE_STRIDE;
    worstExample = Math.max(worstExample, Math.abs(mean[0] - shippedCodes.codes[2 * slot]), Math.abs(mean[1] - shippedCodes.codes[2 * slot + 1]));
    exampleReconstruction += reconstructionLoss(decode(model, mean, decoding).logits, image);
  });
  if (worstExample > 1e-6) throw new Error(`Example codes differ from the map's by ${worstExample}`);

  const fullMeta = {
    ...meta,
    view: { min: -extent, max: extent },
    classCenters,
    codes: {
      file: 'mnist-vae-codes-v1.bin',
      count: map.count,
      layout: `${map.count} pairs of little-endian float32 (the mean code μ₁, μ₂), then ${map.count} uint8 labels`,
      testIndices: `every ${CODE_STRIDE === 2 ? 'second' : `${CODE_STRIDE}th`} MNIST test digit: 0, ${CODE_STRIDE}, ${2 * CODE_STRIDE}, …`,
    },
    examples: {
      file: 'mnist-vae-examples-v1.bin',
      count: exampleIndices.length,
      layout: `${exampleIndices.length} unmodified 28×28 uint8 MNIST test digits (CC BY-SA 3.0), ${EXAMPLES_PER_CLASS} random ones per class`,
      labels: exampleIndices.map(n => testLabels[n]),
      testIndices: exampleIndices,
    },
    scores: {
      testImages: testCount,
      note: 'Nats per test image, from model.ts reading the exported file. The ELBO decodes one random sample of each code '
        + '(fixed noise); the page decodes the mean code.',
      elbo: round(-(reconstruction + kl), 2),
      reconstruction: round(reconstruction, 2),
      kl: round(kl, 3),
      reconstructionAtMean: round(atMean, 2),
      meanImageReconstruction: round(baseline, 2),
      knn: {
        k: KNN,
        testAgainstTrain: round(knnTest.accuracy),
        testAgainstTrainPerClass: knnTest.perClass.map(value => round(value, 3)),
        leaveOneOutOnMap: round(knnMap.accuracy),
      },
      ...(cnn ? {
        readByDigitsNetwork: {
          note: 'Share of test digits whose reconstruction (decoded from the mean code) the Digits page\'s CNN reads as the true digit.',
          overall: round(readCorrectly / testCount),
          perClass: readPerClass.map(c => round(c.correct / c.total, 3)),
          classCenters: centerReadings,
        },
      } : {}),
    },
    training: {
      data: 'MNIST training set, 60,000 digits, grey levels scaled to 0..1, no augmentation',
      likelihood: 'Bernoulli: binary cross-entropy of the grey levels against the decoded probabilities',
      prior: options.klWarmup
        ? `standard normal; the KL weight rises linearly from 0 to 1 over the first ${options.klWarmup} epochs, then stays 1`
        : 'standard normal, with the full KL term from the first step (no warm-up)',
      epochs: options.epochs,
      klWarmupEpochs: options.klWarmup,
      batch: BATCH,
      optimizer: `Adam (β₁ ${ADAM.beta1}, β₂ ${ADAM.beta2}), learning rate warming up over ${WARMUP_STEPS} steps to ${LEARNING_RATE}, cosine decay to ${FINAL_LEARNING_RATE}`,
      initialization: 'He-normal before ReLUs, √(1 / fan-in) for the code and output layers, output biases at the logit of the mean training image',
      seed: options.seed,
      workers: details.workerCount,
      seconds: Math.round(details.seconds),
      node: process.version,
      history: details.history,
    },
  };
  const metaPath = join(outDir, 'mnist-vae-v1.json');
  writeFileSync(metaPath, `${JSON.stringify(fullMeta, null, 2)}\n`);
  parseVae(JSON.parse(readFileSync(metaPath, 'utf8')), toArrayBuffer(readFileSync(weightsPath)));

  console.log(`\nTest set (model.ts on the exported file): ELBO ${(-(reconstruction + kl)).toFixed(2)} nats `
    + `(reconstruction ${reconstruction.toFixed(2)}, KL ${kl.toFixed(2)}); reconstruction from the mean code ${atMean.toFixed(2)}; `
    + `predicting the mean training image ${baseline.toFixed(2)}`);
  console.log(`${KNN}-NN accuracy of test codes against training codes ${percent(knnTest.accuracy)} `
    + `(per class ${knnTest.perClass.map((a, d) => `${d} ${percent(a)}`).join(', ')}); leave-one-out on the map's `
    + `${map.count} codes ${percent(knnMap.accuracy)}`);
  if (cnn) {
    console.log(`The Digits CNN reads ${percent(readCorrectly / testCount)} of reconstructions as their true digit `
      + `(per class ${readPerClass.map((c, d) => `${d} ${percent(c.correct / c.total)}`).join(', ')})`);
  }
  console.log(`Map extent ±${extent}; class centers ${classCenters.map((c, d) => `${d} (${c[0].toFixed(2)}, ${c[1].toFixed(2)})`).join(', ')}`
    + (cnn ? `; CNN reads the centers at ${centerReadings.join(', ')}` : ''));
  console.log(`Examples: ${exampleIndices.length}, mean reconstruction from the mean code ${(exampleReconstruction / examples.length).toFixed(2)}`);
  for (const path of [weightsPath, metaPath, codesPath, examplesPath]) console.log(`Wrote ${path} (${statSync(path).size} bytes)`);
  console.log(`Training time ${details.seconds.toFixed(0)} s.`);
}

// ---------------------------------------------------------------------------------
// Gradient check: backprop vs. central differences of the loss computed with model.ts
// ---------------------------------------------------------------------------------

function checkGradients() {
  const random = mulberry32(7);
  const hidden = [24, 12];
  const count = vaeParameterCount(LATENT, hidden);
  const params = new Float32Array(count);
  const fakeImages = new Uint8Array(4 * PIXELS).map(() => (random() < 0.3 ? Math.floor(random() * 256) : 0));
  initialize(params, hidden, random, fakeImages);
  const model = vaeFromParameters(params, LATENT, hidden);
  for (const layer of [model.encoder.hidden1, model.encoder.hidden2, model.decoder.hidden1, model.decoder.hidden2]) {
    for (let j = 0; j < layer.biases.length; j++) layer.biases[j] = (random() - 0.3) * 0.2;
  }
  const backward = createBackprop(model);
  const beta = 0.7;
  const samples = Array.from({ length: 4 }, (_, n) => ({
    input: Float32Array.from(fakeImages.subarray(n * PIXELS, (n + 1) * PIXELS), v => v / 255),
    epsilon: (() => { const e = new Float64Array(LATENT); gaussians(random, e); return e; })(),
  }));
  const encoding = createEncoding(model);
  const decoding = createDecoding(model);
  const z = new Float64Array(LATENT);
  const loss = () => samples.reduce((sum, { input, epsilon }) => {
    encode(model, input, encoding);
    sampleCode(encoding, epsilon, z);
    decode(model, z, decoding);
    return sum + reconstructionLoss(decoding.logits, input) + beta * klDivergence(encoding.mean, encoding.logVariance);
  }, 0);
  const grads = new Float32Array(count);
  for (const { input, epsilon } of samples) {
    encode(model, input, encoding);
    sampleCode(encoding, epsilon, z);
    decode(model, z, decoding);
    backward(input, encoding, decoding, z, epsilon, beta, grads);
  }

  let worst = 0;
  for (const { name, shape, offset } of vaeLayout(LATENT, hidden)) {
    const size = shape.reduce((a, b) => a * b, 1);
    // Directional derivative along a random unit direction within this tensor (small
    // enough that no ReLU flips).
    const direction = Float64Array.from({ length: size }, () => random() * 2 - 1);
    const length = Math.hypot(...direction);
    for (let i = 0; i < size; i++) direction[i] /= length;
    const original = params.slice(offset, offset + size);
    const step = 1e-3;
    const at = sign => {
      for (let i = 0; i < size; i++) params[offset + i] = original[i] + sign * step * direction[i];
      return loss();
    };
    const numeric = (at(1) - at(-1)) / (2 * step);
    params.set(original, offset);
    let analytic = 0;
    for (let i = 0; i < size; i++) analytic += grads[offset + i] * direction[i];
    const error = Math.abs(numeric - analytic) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic));
    worst = Math.max(worst, error);
    console.log(`${name.padEnd(30)} analytic ${analytic.toExponential(5).padStart(12)}  `
      + `numeric ${numeric.toExponential(5).padStart(12)}  relative error ${error.toExponential(2)}`);
  }
  if (worst > 1e-2) throw new Error(`Gradient check failed (worst relative error ${worst})`);
  console.log('Gradient check passed.');
}

if (isMainThread) {
  try {
    const options = parseArguments(process.argv.slice(2));
    if (options.checkGradients) checkGradients();
    else if (!options.dataDir || !options.outDir) {
      throw new Error('Usage: node scripts/train-mnist-vae.mjs <mnist-data-dir> <output-dir> '
        + '[--epochs N] [--seed N] [--workers N] [--kl-warmup N] [--hidden A,B] [--holdout N] | --check-gradients');
    } else await train(options);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
} else {
  workerMain(workerData);
}
