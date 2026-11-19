#!/usr/bin/env node
/**
 * Trains the Digits page's MNIST CNN and exports its weights and example digits.
 *
 * Usage:
 *   node scripts/train-mnist-cnn.mjs <mnist-data-dir> <output-dir> [options]
 *     --epochs N     training epochs (default 45)
 *     --seed N       PRNG seed (default 20261119)
 *     --workers N    worker threads (default: all cores); results do not depend on it
 *     --holdout N    tuning mode: train on the first 60000-N training images, report
 *                    accuracy on the last N and write only the weights (test set unused)
 *   node scripts/train-mnist-cnn.mjs --check-gradients
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
 * ten unmodified MNIST test images and remain under that license.
 *
 * Writes to <output-dir>:
 *   mnist-cnn-v1.bin      52,138 little-endian float32 parameters (order in src/cnn/model.ts)
 *   mnist-samples-v1.bin  ten 28×28 uint8 MNIST test digits, classes 0..9 in order
 *
 * The forward pass is imported from src/cnn/model.ts (Node strips its types), so the
 * trainer, the reported test accuracy and the browser all run the same code. This file
 * adds only data loading, augmentation (random rotation ±15°, scale 0.85–1.15, shift
 * ±3 px, occasional stroke thickening/thinning), backpropagation computed from runCnn's
 * activations, and Adam (He init, mini-batches of 64, cross-entropy, learning rate
 * warming up to 2e-3 and cosine-decaying to 1e-5; about 3 minutes on a 10-core laptop).
 * Training is data-parallel over worker threads: each mini-batch is split into fixed
 * shards whose gradients are summed in a fixed order, so a given seed reproduces the
 * exact same weights whatever the worker count.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { CNN_LAYOUT, parseCnnWeights, parseSamples, runCnn } from '../src/cnn/model.ts';

// ---------------------------------------------------------------------------------
// Network layout (must match src/cnn/model.ts)
// ---------------------------------------------------------------------------------

const SIZE = CNN_LAYOUT.input.size; // 28
const PIXELS = SIZE * SIZE;
const C1 = CNN_LAYOUT.conv1.filters; // 8
const P1 = CNN_LAYOUT.pool1.size; // 14
const C2 = CNN_LAYOUT.conv2.filters; // 16
const P2 = CNN_LAYOUT.pool2.size; // 7
const FLAT = C2 * P2 * P2; // 784
const HIDDEN = CNN_LAYOUT.dense.units; // 64
const CLASSES = CNN_LAYOUT.output.units; // 10
const TENSORS = [
  ['conv1', 'weights', C1 * 9, 9], ['conv1', 'biases', C1, 0],
  ['conv2', 'weights', C2 * C1 * 9, C1 * 9], ['conv2', 'biases', C2, 0],
  ['dense', 'weights', HIDDEN * FLAT, FLAT], ['dense', 'biases', HIDDEN, 0],
  ['output', 'weights', CLASSES * HIDDEN, HIDDEN], ['output', 'biases', CLASSES, 0],
]; // [layer, kind, count, fan-in for He init]
const OFFSET = {};
let PARAMS = 0;
for (const [layer, kind, count] of TENSORS) {
  OFFSET[`${layer}.${kind}`] = PARAMS;
  PARAMS += count;
}
if (PARAMS !== 52138) throw new Error(`Unexpected parameter count ${PARAMS}`);
const C1W = OFFSET['conv1.weights'], C1B = OFFSET['conv1.biases'];
const C2W = OFFSET['conv2.weights'], C2B = OFFSET['conv2.biases'];
const DW = OFFSET['dense.weights'], DB = OFFSET['dense.biases'];
const OW = OFFSET['output.weights'], OB = OFFSET['output.biases'];

// ---------------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------------

const BATCH = 64;
const SHARD = 4; // samples per gradient shard; 16 shards per batch
const SHARDS = BATCH / SHARD;
const UPDATE_CHUNK = 2048; // parameters per optimizer task
const EVAL_CHUNK = 250; // images per evaluation task
const LEARNING_RATE = 2e-3; // peak; cosine decay to FINAL_LEARNING_RATE (tuned with --holdout)
const FINAL_LEARNING_RATE = 1e-5;
const WARMUP_STEPS = 300;
const ADAM = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
const AUGMENT = {
  clean: 0.1, // share of samples left untouched
  rotation: (15 * Math.PI) / 180,
  minScale: 0.85,
  maxScale: 1.15,
  shift: 3,
  thicken: 0.2, // share with a 3×3 dilation blended in
  thin: 0.1, // share with a 3×3 erosion blended in
};
const DEFAULTS = { epochs: 45, seed: 20261119, workers: availableParallelism(), holdout: 0 };

const FILES = {
  trainImages: { name: 'train-images-idx3-ubyte', sha256: 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db' },
  trainLabels: { name: 'train-labels-idx1-ubyte', sha256: '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5' },
  testImages: { name: 't10k-images-idx3-ubyte', sha256: '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7' },
  testLabels: { name: 't10k-labels-idx1-ubyte', sha256: 'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2' },
};

// Shared control block used to run phases on the workers.
const CTRL = { generation: 0, command: 1, tasks: 2, next: 3, done: 4, failed: 5, batchSize: 6, shardsUsed: 7, length: 8 };
const COMMAND = { train: 1, update: 2, evaluate: 3, exit: 4 };

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

/** CnnWeights views over one flat parameter array in file order. */
function weightViews(params) {
  const views = {};
  for (const [layer, kind, count] of TENSORS) {
    const start = OFFSET[`${layer}.${kind}`];
    (views[layer] ??= {})[kind] = params.subarray(start, start + count);
  }
  return views;
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * Backpropagation for one sample, computed entirely from runCnn's activations:
 * a ReLU unit is active iff its output is > 0, and a max-pool window routes its
 * gradient to the first position holding the pooled value. Accumulates
 * d(cross-entropy)/d(parameter) into `grads` (flat, file order).
 */
function createBackprop(params) {
  const dLogits = new Float64Array(CLASSES);
  const dPool2 = new Float64Array(FLAT);
  const pool1Padded = new Float32Array(C1 * 16 * 16);
  const dPool1Padded = new Float64Array(C1 * 16 * 16);
  const inputPadded = new Float32Array(30 * 30);

  return function backward(acts, label, grads) {
    const { input, conv1, pool1, conv2, pool2, dense, probabilities } = acts;

    // Output layer: softmax + cross-entropy gives p - onehot.
    for (let k = 0; k < CLASSES; k++) {
      const d = probabilities[k] - (k === label ? 1 : 0);
      dLogits[k] = d;
      grads[OB + k] += d;
      const row = OW + k * HIDDEN;
      for (let j = 0; j < HIDDEN; j++) grads[row + j] += d * dense[j];
    }

    // Dense layer.
    dPool2.fill(0);
    for (let j = 0; j < HIDDEN; j++) {
      if (dense[j] <= 0) continue;
      let d = 0;
      for (let k = 0; k < CLASSES; k++) d += params[OW + k * HIDDEN + j] * dLogits[k];
      if (d === 0) continue;
      grads[DB + j] += d;
      const row = DW + j * FLAT;
      for (let i = 0; i < FLAT; i++) {
        grads[row + i] += d * pool2[i];
        dPool2[i] += d * params[row + i];
      }
    }

    // pool2 → conv2 → (conv2 weights, pool1).
    pool1Padded.fill(0);
    for (let c = 0; c < C1; c++) {
      for (let y = 0; y < P1; y++) {
        for (let x = 0; x < P1; x++) pool1Padded[c * 256 + (y + 1) * 16 + x + 1] = pool1[c * P1 * P1 + y * P1 + x];
      }
    }
    dPool1Padded.fill(0);
    for (let f = 0; f < C2; f++) {
      for (let py = 0; py < P2; py++) {
        for (let px = 0; px < P2; px++) {
          const index = f * P2 * P2 + py * P2 + px;
          const value = pool2[index];
          const d = dPool2[index];
          if (value <= 0 || d === 0) continue;
          let y = 2 * py;
          let x = 2 * px;
          const plane = f * P1 * P1;
          if (conv2[plane + y * P1 + x] !== value) {
            if (conv2[plane + y * P1 + x + 1] === value) x++;
            else if (conv2[plane + (y + 1) * P1 + x] === value) y++;
            else { x++; y++; }
          }
          grads[C2B + f] += d;
          for (let c = 0; c < C1; c++) {
            const w = C2W + f * C1 * 9 + c * 9;
            const p = c * 256 + y * 16 + x;
            for (let ky = 0; ky < 3; ky++) {
              for (let kx = 0; kx < 3; kx++) {
                grads[w + ky * 3 + kx] += d * pool1Padded[p + ky * 16 + kx];
                dPool1Padded[p + ky * 16 + kx] += d * params[w + ky * 3 + kx];
              }
            }
          }
        }
      }
    }

    // pool1 → conv1 → conv1 weights.
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) inputPadded[(y + 1) * 30 + x + 1] = input[y * SIZE + x];
    }
    for (let f = 0; f < C1; f++) {
      for (let py = 0; py < P1; py++) {
        for (let px = 0; px < P1; px++) {
          const value = pool1[f * P1 * P1 + py * P1 + px];
          const d = dPool1Padded[f * 256 + (py + 1) * 16 + px + 1];
          if (value <= 0 || d === 0) continue;
          let y = 2 * py;
          let x = 2 * px;
          const plane = f * PIXELS;
          if (conv1[plane + y * SIZE + x] !== value) {
            if (conv1[plane + y * SIZE + x + 1] === value) x++;
            else if (conv1[plane + (y + 1) * SIZE + x] === value) y++;
            else { x++; y++; }
          }
          grads[C1B + f] += d;
          const w = C1W + f * 9;
          const p = y * 30 + x;
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) grads[w + ky * 3 + kx] += d * inputPadded[p + ky * 30 + kx];
          }
        }
      }
    }
  };
}

/**
 * Random augmentation of one MNIST image into `out` (values 0..1), driven by `seed`:
 * rotation, scale and shift about the digit's center of mass with bilinear sampling,
 * then occasionally a blended 3×3 dilation (thicker strokes) or erosion (thinner).
 * Augmented images are rescaled so their peak is 1, like preprocessDrawing's output.
 */
function createAugmenter() {
  const source = new Float32Array(PIXELS);
  const morphed = new Float32Array(PIXELS);
  const pixel = (image, x, y) => (x >= 0 && x < SIZE && y >= 0 && y < SIZE ? image[y * SIZE + x] : 0);

  function morph(image, dilate, blend) {
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        let best = dilate ? 0 : 1;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const v = pixel(image, x + dx, y + dy);
            best = dilate ? Math.max(best, v) : Math.min(best, v);
          }
        }
        morphed[y * SIZE + x] = best;
      }
    }
    for (let i = 0; i < PIXELS; i++) image[i] += blend * (morphed[i] - image[i]);
  }

  return function augment(images, offset, seed, out) {
    const random = mulberry32(seed);
    let mass = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = images[offset + y * SIZE + x] / 255;
        source[y * SIZE + x] = v;
        mass += v;
        sumX += v * x;
        sumY += v * y;
      }
    }
    if (random() < AUGMENT.clean || mass === 0) {
      out.set(source);
      return;
    }
    const cx = sumX / mass;
    const cy = sumY / mass;
    const angle = (random() * 2 - 1) * AUGMENT.rotation;
    const scale = AUGMENT.minScale + random() * (AUGMENT.maxScale - AUGMENT.minScale);
    const shiftX = (random() * 2 - 1) * AUGMENT.shift;
    const shiftY = (random() * 2 - 1) * AUGMENT.shift;
    const cos = Math.cos(angle) / scale;
    const sin = Math.sin(angle) / scale;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        // Inverse map: output pixel → source position.
        const dx = x - cx - shiftX;
        const dy = y - cy - shiftY;
        const sx = cos * dx + sin * dy + cx;
        const sy = -sin * dx + cos * dy + cy;
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        const fx = sx - x0;
        const fy = sy - y0;
        out[y * SIZE + x] = (pixel(source, x0, y0) * (1 - fx) + pixel(source, x0 + 1, y0) * fx) * (1 - fy)
          + (pixel(source, x0, y0 + 1) * (1 - fx) + pixel(source, x0 + 1, y0 + 1) * fx) * fy;
      }
    }
    const stroke = random();
    if (stroke < AUGMENT.thicken) morph(out, true, 0.5 + 0.5 * random());
    else if (stroke < AUGMENT.thicken + AUGMENT.thin) morph(out, false, 0.3 + 0.5 * random());
    let peak = 0;
    for (let i = 0; i < PIXELS; i++) if (out[i] > peak) peak = out[i];
    if (peak > 0 && peak < 1) for (let i = 0; i < PIXELS; i++) out[i] = Math.min(1, out[i] / peak);
  };
}

// ---------------------------------------------------------------------------------
// Worker: trains gradient shards, applies Adam to parameter chunks, evaluates
// ---------------------------------------------------------------------------------

function workerMain({ shared, workerCount }) {
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const weights = weightViews(params);
  const adamM = new Float64Array(shared.adamM);
  const adamV = new Float64Array(shared.adamV);
  const hyper = new Float64Array(shared.hyper);
  const grads = new Float32Array(shared.grads);
  const batch = new Uint32Array(shared.batch);
  const shardStats = new Float64Array(shared.shardStats);
  const trainImages = new Uint8Array(shared.trainImages);
  const trainLabels = new Uint8Array(shared.trainLabels);
  const evalImages = new Uint8Array(shared.evalImages);
  const evalLabels = new Uint8Array(shared.evalLabels);
  const evalStats = new Float64Array(shared.evalStats);
  const backward = createBackprop(params);
  const augment = createAugmenter();
  const input = new Float32Array(PIXELS);
  const sums = new Float64Array(UPDATE_CHUNK);

  function trainShard(shard) {
    const size = Atomics.load(control, CTRL.batchSize);
    const first = shard * SHARD;
    const last = Math.min(first + SHARD, size);
    const shardGrads = grads.subarray(shard * PARAMS, (shard + 1) * PARAMS);
    shardGrads.fill(0);
    let loss = 0;
    let correct = 0;
    for (let n = first; n < last; n++) {
      const index = batch[2 * n];
      augment(trainImages, index * PIXELS, batch[2 * n + 1], input);
      const acts = runCnn(weights, input);
      const label = trainLabels[index];
      loss -= Math.log(Math.max(acts.probabilities[label], 1e-30));
      if (argmax(acts.probabilities) === label) correct++;
      backward(acts, label, shardGrads);
    }
    shardStats[2 * shard] = loss;
    shardStats[2 * shard + 1] = correct;
  }

  function updateChunk(chunk) {
    const lo = chunk * UPDATE_CHUNK;
    const hi = Math.min(lo + UPDATE_CHUNK, PARAMS);
    const shards = Atomics.load(control, CTRL.shardsUsed);
    const [rate, correction1, correction2, scale] = hyper;
    sums.fill(0);
    for (let s = 0; s < shards; s++) {
      const base = s * PARAMS;
      for (let i = lo; i < hi; i++) sums[i - lo] += grads[base + i];
    }
    const { beta1, beta2, epsilon } = ADAM;
    for (let i = lo; i < hi; i++) {
      const g = sums[i - lo] * scale;
      const m = beta1 * adamM[i] + (1 - beta1) * g;
      const v = beta2 * adamV[i] + (1 - beta2) * g * g;
      adamM[i] = m;
      adamV[i] = v;
      params[i] -= (rate * (m / correction1)) / (Math.sqrt(v / correction2) + epsilon);
    }
  }

  function evaluateChunk(chunk) {
    const lo = chunk * EVAL_CHUNK;
    const hi = Math.min(lo + EVAL_CHUNK, evalLabels.length);
    let correct = 0;
    let loss = 0;
    for (let n = lo; n < hi; n++) {
      for (let i = 0; i < PIXELS; i++) input[i] = evalImages[n * PIXELS + i] / 255;
      const p = runCnn(weights, input).probabilities;
      loss -= Math.log(Math.max(p[evalLabels[n]], 1e-30));
      if (argmax(p) === evalLabels[n]) correct++;
    }
    evalStats[2 * chunk] = correct;
    evalStats[2 * chunk + 1] = loss;
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
    else if (['--epochs', '--seed', '--workers', '--holdout'].includes(arg)) {
      const value = Number(argv[++i]);
      if (!Number.isInteger(value) || value < 0) throw new Error(`${arg} needs a non-negative integer`);
      options[arg.slice(2)] = value;
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
    if (bytes.readUInt32BE(8) !== SIZE || bytes.readUInt32BE(12) !== SIZE || bytes.length !== 16 + count * PIXELS) {
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

/** He-normal initialization (std = sqrt(2 / fan-in)) with zero biases. */
function initialize(params, random) {
  for (const [layer, kind, count, fanIn] of TENSORS) {
    if (!fanIn) continue;
    const start = OFFSET[`${layer}.${kind}`];
    const std = Math.sqrt(2 / fanIn);
    for (let i = 0; i < count; i++) {
      const u = 1 - random();
      const v = random();
      params[start + i] = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }
}

function encodeWeights(params) {
  const bytes = new ArrayBuffer(PARAMS * 4);
  const view = new DataView(bytes);
  for (let i = 0; i < PARAMS; i++) view.setFloat32(i * 4, params[i], true);
  return new Uint8Array(bytes);
}

function toArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

const percent = value => `${(value * 100).toFixed(2)}%`;

async function train(options) {
  const startTime = performance.now();
  const data = loadMnist(options.dataDir);
  const holdout = options.holdout;
  if (holdout >= 60000) throw new Error('--holdout must be below 60000');
  const trainCount = 60000 - holdout;
  const evalImages = holdout ? data.trainImages.subarray(trainCount * PIXELS) : data.testImages;
  const evalLabels = holdout ? data.trainLabels.subarray(trainCount) : data.testLabels;
  const evalName = holdout ? `holdout (${holdout})` : 'test';

  const shared = {
    control: new SharedArrayBuffer(CTRL.length * 4),
    params: new SharedArrayBuffer(PARAMS * 4),
    adamM: new SharedArrayBuffer(PARAMS * 8),
    adamV: new SharedArrayBuffer(PARAMS * 8),
    hyper: new SharedArrayBuffer(4 * 8),
    grads: new SharedArrayBuffer(SHARDS * PARAMS * 4),
    batch: new SharedArrayBuffer(BATCH * 2 * 4),
    shardStats: new SharedArrayBuffer(SHARDS * 2 * 8),
    trainImages: sharedCopy(data.trainImages.subarray(0, trainCount * PIXELS)).buffer,
    trainLabels: sharedCopy(data.trainLabels.subarray(0, trainCount)).buffer,
    evalImages: sharedCopy(evalImages).buffer,
    evalLabels: sharedCopy(evalLabels).buffer,
    evalStats: new SharedArrayBuffer(Math.ceil(evalLabels.length / EVAL_CHUNK) * 2 * 8),
  };
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const hyper = new Float64Array(shared.hyper);
  const batch = new Uint32Array(shared.batch);
  const shardStats = new Float64Array(shared.shardStats);
  const evalStats = new Float64Array(shared.evalStats);

  const random = mulberry32(options.seed);
  initialize(params, random);

  const workerCount = options.workers;
  const workerErrors = [];
  const workers = Array.from({ length: workerCount }, () => {
    const worker = new Worker(new URL(import.meta.url), { workerData: { shared, workerCount } });
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
    const chunks = Math.ceil(evalLabels.length / EVAL_CHUNK);
    await runPhase(COMMAND.evaluate, chunks);
    let correct = 0;
    for (let c = 0; c < chunks; c++) correct += evalStats[2 * c];
    return correct / evalLabels.length;
  }

  const order = Uint32Array.from({ length: trainCount }, (_, i) => i);
  const stepsPerEpoch = Math.ceil(trainCount / BATCH);
  const totalSteps = options.epochs * stepsPerEpoch;
  const learningRate = step => Math.min(1, step / WARMUP_STEPS)
    * (FINAL_LEARNING_RATE + 0.5 * (LEARNING_RATE - FINAL_LEARNING_RATE) * (1 + Math.cos((Math.PI * step) / totalSteps)));

  console.log(`MNIST CNN: ${PARAMS} parameters, ${trainCount} training images, ${options.epochs} epochs, `
    + `batch ${BATCH}, seed ${options.seed}, ${workerCount} workers`);
  let step = 0;
  let accuracy = 0;
  try {
    for (let epoch = 1; epoch <= options.epochs; epoch++) {
      const epochStart = performance.now();
      for (let i = trainCount - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      let loss = 0;
      let correct = 0;
      for (let start = 0; start < trainCount; start += BATCH) {
        const size = Math.min(BATCH, trainCount - start);
        for (let n = 0; n < size; n++) {
          batch[2 * n] = order[start + n];
          batch[2 * n + 1] = Math.floor(random() * 4294967296);
        }
        const shards = Math.ceil(size / SHARD);
        Atomics.store(control, CTRL.batchSize, size);
        await runPhase(COMMAND.train, shards);
        for (let s = 0; s < shards; s++) {
          loss += shardStats[2 * s];
          correct += shardStats[2 * s + 1];
        }
        step++;
        hyper[0] = learningRate(step);
        hyper[1] = 1 - ADAM.beta1 ** step;
        hyper[2] = 1 - ADAM.beta2 ** step;
        hyper[3] = 1 / size;
        Atomics.store(control, CTRL.shardsUsed, shards);
        await runPhase(COMMAND.update, Math.ceil(PARAMS / UPDATE_CHUNK));
      }
      accuracy = await evaluate();
      console.log(`epoch ${String(epoch).padStart(2)}/${options.epochs}  lr ${hyper[0].toExponential(2)}  `
        + `train loss ${(loss / trainCount).toFixed(4)} acc ${percent(correct / trainCount)} (augmented)  `
        + `${evalName} acc ${percent(accuracy)}  ${((performance.now() - epochStart) / 1000).toFixed(1)} s`);
    }
  } finally {
    await runPhase(COMMAND.exit, 0); // idle workers leave their loop; terminate() covers the rest
    await Promise.all(workers.map(worker => worker.terminate()));
  }
  const seconds = (performance.now() - startTime) / 1000;
  if (holdout) {
    mkdirSync(options.outDir, { recursive: true });
    const weightsPath = join(options.outDir, 'mnist-cnn-v1.bin');
    writeFileSync(weightsPath, encodeWeights(params));
    console.log(`Holdout accuracy ${percent(accuracy)} after ${seconds.toFixed(0)} s; wrote ${weightsPath} only.`);
    return;
  }
  exportModel(options, data, params, seconds);
}

/** Writes the binaries, then re-reads them and evaluates with model.ts exactly as the page will. */
function exportModel(options, data, params, trainSeconds) {
  mkdirSync(options.outDir, { recursive: true });
  const weightsPath = join(options.outDir, 'mnist-cnn-v1.bin');
  writeFileSync(weightsPath, encodeWeights(params));
  const weights = parseCnnWeights(toArrayBuffer(readFileSync(weightsPath)));

  const { testImages, testLabels, trainImages, trainLabels } = data;
  const input = new Float32Array(PIXELS);
  const predictions = new Uint8Array(testLabels.length);
  const confidence = new Float32Array(testLabels.length);
  const perClass = Array.from({ length: CLASSES }, () => ({ total: 0, correct: 0 }));
  const activeConv1 = new Uint32Array(C1);
  const activeConv2 = new Uint32Array(C2);
  let correct = 0;
  for (let n = 0; n < testLabels.length; n++) {
    for (let i = 0; i < PIXELS; i++) input[i] = testImages[n * PIXELS + i] / 255;
    const acts = runCnn(weights, input);
    const predicted = argmax(acts.probabilities);
    predictions[n] = predicted;
    confidence[n] = acts.probabilities[predicted];
    perClass[testLabels[n]].total++;
    if (predicted === testLabels[n]) {
      correct++;
      perClass[testLabels[n]].correct++;
    }
    for (let f = 0; f < C1; f++) if (acts.conv1.subarray(f * PIXELS, (f + 1) * PIXELS).some(v => v > 0)) activeConv1[f]++;
    for (let f = 0; f < C2; f++) if (acts.conv2.subarray(f * P1 * P1, (f + 1) * P1 * P1).some(v => v > 0)) activeConv2[f]++;
  }

  // Example digits: per class, among test images with a typical stroke weight (ink
  // between the class's 35th and 65th percentiles; the mean image alone favors bold
  // blobs) that are classified correctly with p ≥ 0.999, the one most similar (cosine)
  // to the class's mean training image.
  const means = Array.from({ length: CLASSES }, () => new Float64Array(PIXELS));
  for (let n = 0; n < trainLabels.length; n++) {
    const mean = means[trainLabels[n]];
    for (let i = 0; i < PIXELS; i++) mean[i] += trainImages[n * PIXELS + i];
  }
  const ink = Float64Array.from({ length: testLabels.length }, (_, n) =>
    testImages.subarray(n * PIXELS, (n + 1) * PIXELS).reduce((sum, v) => sum + v, 0));
  const samples = new Uint8Array(CLASSES * PIXELS);
  const chosen = [];
  for (let digit = 0; digit < CLASSES; digit++) {
    const mean = means[digit];
    const meanNorm = Math.hypot(...mean);
    const inks = ink.filter((_, n) => testLabels[n] === digit).sort();
    const [minInk, maxInk] = [inks[Math.floor(inks.length * 0.35)], inks[Math.floor(inks.length * 0.65)]];
    let best = -1;
    let bestScore = -Infinity;
    for (let n = 0; n < testLabels.length; n++) {
      if (testLabels[n] !== digit || predictions[n] !== digit || confidence[n] < 0.999) continue;
      if (ink[n] < minInk || ink[n] > maxInk) continue;
      let dot = 0;
      let norm = 0;
      for (let i = 0; i < PIXELS; i++) {
        const v = testImages[n * PIXELS + i];
        dot += v * mean[i];
        norm += v * v;
      }
      const score = dot / (Math.sqrt(norm) * meanNorm);
      if (score > bestScore) {
        bestScore = score;
        best = n;
      }
    }
    if (best < 0) throw new Error(`No confidently classified test image for digit ${digit}`);
    samples.set(testImages.subarray(best * PIXELS, (best + 1) * PIXELS), digit * PIXELS);
    chosen.push(best);
  }
  const samplesPath = join(options.outDir, 'mnist-samples-v1.bin');
  writeFileSync(samplesPath, samples);
  const sampleChecks = parseSamples(toArrayBuffer(readFileSync(samplesPath))).map((sample, digit) => {
    const p = runCnn(weights, sample).probabilities;
    if (argmax(p) !== digit) throw new Error(`Sample ${digit} is misclassified`);
    return `${digit}: #${chosen[digit]} p=${p[digit].toFixed(4)}`;
  });

  const accuracy = correct / testLabels.length;
  console.log(`\nTest accuracy (model.ts on the exported file): ${percent(accuracy)} `
    + `(${testLabels.length - correct} errors of ${testLabels.length})`);
  console.log(`Per class: ${perClass.map((c, d) => `${d} ${percent(c.correct / c.total)}`).join(', ')}`);
  console.log(`Filters active on test images: conv1 min ${percent(Math.min(...activeConv1) / testLabels.length)}, `
    + `conv2 min ${percent(Math.min(...activeConv2) / testLabels.length)}`);
  console.log(`Samples (MNIST test index, confidence): ${sampleChecks.join(', ')}`);
  console.log(`Wrote ${weightsPath} (${statSync(weightsPath).size} bytes) and ${samplesPath} (${statSync(samplesPath).size} bytes)`);
  console.log(`Training time ${trainSeconds.toFixed(0)} s.`);
}

// ---------------------------------------------------------------------------------
// Gradient check: backprop vs. central differences of the loss computed by runCnn
// ---------------------------------------------------------------------------------

function checkGradients() {
  const random = mulberry32(7);
  const params = new Float32Array(PARAMS);
  initialize(params, random);
  for (const [layer, kind, count] of TENSORS) {
    if (kind !== 'biases') continue;
    const start = OFFSET[`${layer}.${kind}`];
    for (let i = 0; i < count; i++) params[start + i] = (random() - 0.3) * 0.2;
  }
  const weights = weightViews(params);
  const backward = createBackprop(params);
  const inputs = Array.from({ length: 4 }, () => Float32Array.from({ length: PIXELS }, () => (random() < 0.3 ? random() : 0)));
  const labels = [3, 1, 4, 1];
  const loss = () => inputs.reduce((sum, input, n) => {
    const logits = runCnn(weights, input).logits;
    const max = Math.max(...logits);
    return sum + Math.log(logits.reduce((s, v) => s + Math.exp(v - max), 0)) + max - logits[labels[n]];
  }, 0);
  const grads = new Float32Array(PARAMS);
  inputs.forEach((input, n) => backward(runCnn(weights, input), labels[n], grads));

  let worst = 0;
  for (const [layer, kind, count] of TENSORS) {
    const start = OFFSET[`${layer}.${kind}`];
    // Directional derivative along a random unit direction within this tensor (small
    // enough that no ReLU or max-pool decision flips).
    const direction = Float64Array.from({ length: count }, () => random() * 2 - 1);
    const length = Math.hypot(...direction);
    for (let i = 0; i < count; i++) direction[i] /= length;
    const original = params.slice(start, start + count);
    const epsilon = 1e-3;
    const at = sign => {
      for (let i = 0; i < count; i++) params[start + i] = original[i] + sign * epsilon * direction[i];
      return loss();
    };
    const numeric = (at(1) - at(-1)) / (2 * epsilon);
    params.set(original, start);
    let analytic = 0;
    for (let i = 0; i < count; i++) analytic += grads[start + i] * direction[i];
    const error = Math.abs(numeric - analytic) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic));
    worst = Math.max(worst, error);
    console.log(`${`${layer}.${kind}`.padEnd(15)} analytic ${analytic.toExponential(5).padStart(12)}  `
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
      throw new Error('Usage: node scripts/train-mnist-cnn.mjs <mnist-data-dir> <output-dir> '
        + '[--epochs N] [--seed N] [--workers N] [--holdout N] | --check-gradients');
    } else await train(options);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
} else {
  workerMain(workerData);
}
