#!/usr/bin/env node
/**
 * Trains the Recurrent page's character-level LSTM and exports its weights and metadata.
 *
 * Usage:
 *   node scripts/train-char-lstm.mjs <corpus-dir> <output-dir> [options]
 *     --epochs N     training epochs (default 8)
 *     --seed N       PRNG seed (default 20261119)
 *     --workers N    worker threads (default: all cores); results do not depend on it
 *     --corpus-only  write the normalized training and held-out text to <output-dir>
 *                    (corpus-train.txt, corpus-held-out.txt) and stop
 *   node scripts/train-char-lstm.mjs --check-gradients
 *     compares backpropagation through time with finite differences and exits
 *
 * <corpus-dir> must contain the nineteen Project Gutenberg plain-text books in BOOKS
 * below (UTF-8), saved as pg<id>.txt, e.g. from
 * https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt or the mirror file <id>-0.txt on
 * https://gutenberg.pglaf.org/ (pg11.txt is Alice's Adventures in Wonderland, and so
 * on). The books are public domain in the United States. They are treated as untrusted
 * input: each file must be small, valid UTF-8 and framed by the Project Gutenberg
 * START and END lines; only characters that survive normalization are kept. Their
 * SHA-256 digests are printed and recorded in the metadata.
 *
 * Preprocessing: keep the text between the START and END lines; drop [Illustration]
 * and [Picture] tags, footnote marks and the underscores around italics; join the
 * wrapped lines of each paragraph with single spaces and separate paragraphs with one
 * "\n"; write quotes, dashes, spaces and accented letters as simplifyCharacters in
 * src/rnn/model.ts does (curly quotes become straight, an em dash becomes "--"). Then
 * drop every paragraph that isn't prose (headings, tables of contents, titles, bylines
 * and credits; see isFrontMatter) and every paragraph containing one of the period
 * slurs some of these books use (OFFENSIVE), so the page can never write them. The
 * vocabulary is the letters plus the digits and common punctuation in ALLOWED that
 * occur at least MIN_COUNT times. The last 5% of each book (from a paragraph boundary)
 * is held out for validation and never trained on.
 *
 * Model (src/rnn/model.ts): one-hot characters into one LSTM layer of 256 cells, gates
 * in the order input, forget, cell candidate, output, and a softmax over the vocabulary.
 * The forget-gate bias starts at 1, recurrent weights start orthogonal per gate, and the
 * output bias starts at the log character frequencies.
 *
 * Training: truncated backpropagation through time over 100 steps on 40 parallel
 * streams through the training text. Each stream carries its memory from one window to
 * the next, but starts again from zero with probability 0.1 per window (and at every
 * epoch), so the model also learns to read from a blank memory, as it does on the page.
 * Adam, gradient norm clipped at 1, learning rate warming up to 2e-3 and cosine-decaying
 * to 5e-5. Validation bits per character are measured after every epoch and the best
 * epoch's weights are exported.
 *
 * Training is data-parallel over worker threads: every stream's gradients go to their
 * own buffer and are summed in a fixed order, so a given seed reproduces the exact same
 * weights whatever the worker count. The shipped model (8 epochs, the default) took
 * 105 minutes on a 10-core M4 and scores 1.77 bits per character on the held-out text.
 *
 * The forward pass (lstmCell, readout, softmaxInto, bitsPerCharacter) is imported from
 * src/rnn/model.ts (Node strips its types), so training, the reported validation score
 * and the browser all run the same code. Writes to <output-dir>:
 *   char-lstm-v1.bin   little-endian float32 parameters, layout in src/rnn/model.ts
 *   char-lstm-v1.json  vocabulary, sizes, tensor layout, gate order, corpus, scores
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import {
  CharLstmRun, GATE_ORDER, MODEL_FORMAT, bitsPerCharacter, lstmCell, parameterCount, parseCharLstm,
  readout, simplifyCharacters, softmaxInto, tensorLayout,
} from '../src/rnn/model.ts';

// ---------------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------------

const BOOKS = [
  { gutenbergId: 11, title: 'Alice’s Adventures in Wonderland', author: 'Lewis Carroll' },
  { gutenbergId: 12, title: 'Through the Looking-Glass, and What Alice Found There', author: 'Lewis Carroll' },
  { gutenbergId: 55, title: 'The Wonderful Wizard of Oz', author: 'L. Frank Baum' },
  { gutenbergId: 54, title: 'The Marvelous Land of Oz', author: 'L. Frank Baum' },
  { gutenbergId: 16, title: 'Peter Pan', author: 'J. M. Barrie' },
  { gutenbergId: 45, title: 'Anne of Green Gables', author: 'L. M. Montgomery' },
  { gutenbergId: 113, title: 'The Secret Garden', author: 'Frances Hodgson Burnett' },
  { gutenbergId: 146, title: 'A Little Princess', author: 'Frances Hodgson Burnett' },
  { gutenbergId: 120, title: 'Treasure Island', author: 'Robert Louis Stevenson' },
  { gutenbergId: 236, title: 'The Jungle Book', author: 'Rudyard Kipling' },
  { gutenbergId: 2781, title: 'Just So Stories', author: 'Rudyard Kipling' },
  { gutenbergId: 271, title: 'Black Beauty', author: 'Anna Sewell' },
  { gutenbergId: 289, title: 'The Wind in the Willows', author: 'Kenneth Grahame' },
  { gutenbergId: 500, title: 'The Adventures of Pinocchio', author: 'Carlo Collodi' },
  { gutenbergId: 708, title: 'The Princess and the Goblin', author: 'George MacDonald' },
  { gutenbergId: 902, title: 'The Happy Prince and Other Tales', author: 'Oscar Wilde' },
  { gutenbergId: 1597, title: 'Andersen’s Fairy Tales', author: 'Hans Christian Andersen' },
  { gutenbergId: 2591, title: 'Grimms’ Fairy Tales', author: 'Jacob and Wilhelm Grimm' },
  { gutenbergId: 1874, title: 'The Railway Children', author: 'E. Nesbit' },
];
const MAX_FILE_BYTES = 3_000_000;
const MIN_COUNT = 50; // rarer punctuation is removed rather than given a vocabulary slot
/** Characters the model may learn besides letters, if common enough: space, newline and plain punctuation (digits are too rare). */
const ALLOWED = /^[ \n!"'(),\-.:;?]$/;
/**
 * Period slurs and terms some of these books use about peoples, which the page must
 * never write: every paragraph containing one is dropped before training.
 */
const OFFENSIVE = /\b(nigg\w*|negro\w*|darkey\w*|darky|darkies|injuns?|redskins?|piccaninn\w*|pickaninn\w*|squaws?|coolies?|chinam[ae]n|chinks?|japs?|kikes?|wops?|jews?|jewess|gypsy|gypsies|gipsy|gipsies|savages?|half-breeds?|mulattos?|blackamoors?)\b/i;
const VALIDATION_SHARE = 0.05;

// ---------------------------------------------------------------------------------
// Model and training configuration
// ---------------------------------------------------------------------------------

const HIDDEN = 256;
const STEPS = 100; // truncated BPTT window
const STREAMS = 40; // sequences per update, one gradient buffer each (a multiple of the 10 cores)
const RESET_PROBABILITY = 0.1;
const LEARNING_RATE = 2e-3;
const FINAL_LEARNING_RATE = 5e-5;
const WARMUP_STEPS = 150;
const CLIP_NORM = 1;
const ADAM = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
const REDUCE_CHUNK = 4096; // parameters per reduction task
const DEFAULTS = { epochs: 8, seed: 20261119, workers: availableParallelism() };

// Shared control block used to run phases on the workers.
const CTRL = { generation: 0, command: 1, tasks: 2, next: 3, done: 4, failed: 5, length: 8 };
const COMMAND = { train: 1, reduce: 2, evaluate: 3, exit: 4 };

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

function gaussian(random) {
  const u = 1 - random();
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** A CharLstm (see src/rnn/model.ts) whose tensors are views into one flat parameter array. */
function modelViews(params, vocabulary, hidden) {
  const inputSize = vocabulary.length;
  const model = { vocabulary, lookup: new Map(vocabulary.map((c, i) => [c, i])), inputSize, hiddenSize: hidden };
  for (const { name, shape, offset } of tensorLayout(inputSize, hidden)) {
    model[name] = params.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
  }
  return model;
}

/**
 * Forward and backward passes over one window of `steps` characters for a model whose
 * parameters live in `params`. `run(text, start, h0, c0, grads)` reads text[start ..
 * start + steps − 1], predicts text[start + 1 .. start + steps], adds d(total loss in
 * nats)/d(parameter) to `grads` (flat, file order), leaves the final state in
 * `finalH`/`finalC` and returns the total loss in nats.
 */
function createSequenceTrainer(model, steps) {
  const H = model.hiddenSize;
  const V = model.inputSize;
  const G = 4 * H;
  const layout = Object.fromEntries(tensorLayout(V, H).map(entry => [entry.name, entry.offset]));
  const { inputWeights: Wx, recurrentWeights: U, outputWeights: Wy } = model;
  const hs = new Float32Array((steps + 1) * H);
  const cs = new Float32Array((steps + 1) * H);
  const gates = new Float32Array(steps * G);
  const logits = new Float32Array(V);
  const probs = new Float32Array(steps * V);
  const dh = new Float64Array(H);
  const dhNext = new Float64Array(H);
  const dcNext = new Float64Array(H);
  const dz = new Float64Array(G);

  function run(text, start, h0, c0, grads) {
    hs.set(h0, 0);
    cs.set(c0, 0);
    let loss = 0;
    for (let t = 0; t < steps; t++) {
      lstmCell(model, Wx, text[start + t] * G, hs, t * H, cs, t * H, gates, t * G, cs, (t + 1) * H, hs, (t + 1) * H);
      readout(model, hs, (t + 1) * H, logits, 0);
      loss += softmaxInto(logits, 0, V, probs, t * V) - logits[text[start + t + 1]];
    }

    dhNext.fill(0);
    dcNext.fill(0);
    const gWx = layout.inputWeights;
    const gU = layout.recurrentWeights;
    const gB = layout.bias;
    const gWy = layout.outputWeights;
    const gBy = layout.outputBias;
    for (let t = steps - 1; t >= 0; t--) {
      const target = text[start + t + 1];
      const hOffset = (t + 1) * H;
      // Softmax with cross-entropy: d loss / d logits = p − onehot(target).
      for (let k = 0; k < H; k++) dh[k] = dhNext[k];
      for (let v = 0; v < V; v++) {
        const d = probs[t * V + v] - (v === target ? 1 : 0);
        grads[gBy + v] += d;
        const row = v * H;
        for (let k = 0; k < H; k++) {
          grads[gWy + row + k] += d * hs[hOffset + k];
          dh[k] += d * Wy[row + k];
        }
      }
      // Through h = o ⊙ tanh(c) and c = f ⊙ c_prev + i ⊙ g to the pre-activations.
      const gate = t * G;
      for (let k = 0; k < H; k++) {
        const i = gates[gate + k];
        const f = gates[gate + H + k];
        const g = gates[gate + 2 * H + k];
        const o = gates[gate + 3 * H + k];
        const tanhC = Math.tanh(cs[hOffset + k]);
        const dCell = dcNext[k] + dh[k] * o * (1 - tanhC * tanhC);
        dz[k] = dCell * g * i * (1 - i);
        dz[H + k] = dCell * cs[t * H + k] * f * (1 - f);
        dz[2 * H + k] = dCell * i * (1 - g * g);
        dz[3 * H + k] = dh[k] * tanhC * o * (1 - o);
        dcNext[k] = dCell * f;
      }
      const inputRow = gWx + text[start + t] * G;
      for (let j = 0; j < G; j++) {
        grads[inputRow + j] += dz[j];
        grads[gB + j] += dz[j];
      }
      const hPrev = t * H;
      if (t > 0) dhNext.fill(0);
      for (let j = 0; j < G; j++) {
        const d = dz[j];
        if (d === 0) continue;
        const row = j * H;
        if (t > 0) {
          for (let k = 0; k < H; k++) {
            grads[gU + row + k] += d * hs[hPrev + k];
            dhNext[k] += d * U[row + k];
          }
        } else {
          for (let k = 0; k < H; k++) grads[gU + row + k] += d * hs[hPrev + k];
        }
      }
    }
    return loss;
  }

  return { run, finalH: hs.subarray(steps * H), finalC: cs.subarray(steps * H) };
}

// ---------------------------------------------------------------------------------
// Worker: trains streams, sums gradients, evaluates
// ---------------------------------------------------------------------------------

function workerMain({ shared, vocabulary, workerCount }) {
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const grads = new Float32Array(shared.grads);
  const total = new Float64Array(shared.total);
  const norms = new Float64Array(shared.norms);
  const streams = new Int32Array(shared.streams);
  const state = new Float32Array(shared.state);
  const losses = new Float64Array(shared.losses);
  const text = new Uint8Array(shared.text);
  const validation = new Uint8Array(shared.validation);
  const validationBounds = new Int32Array(shared.validationBounds);
  const evalStats = new Float64Array(shared.evalStats);
  const model = modelViews(params, vocabulary, HIDDEN);
  const PARAMS = params.length;
  const H = HIDDEN;
  const trainer = createSequenceTrainer(model, STEPS);
  const zeros = new Float32Array(H);

  function trainStream(s) {
    const start = streams[2 * s];
    const reset = streams[2 * s + 1] === 1;
    const memory = state.subarray(s * 2 * H, (s + 1) * 2 * H);
    const streamGrads = grads.subarray(s * PARAMS, (s + 1) * PARAMS);
    streamGrads.fill(0);
    losses[s] = trainer.run(text, start, reset ? zeros : memory.subarray(0, H), reset ? zeros : memory.subarray(H), streamGrads);
    memory.set(trainer.finalH, 0);
    memory.set(trainer.finalC, H);
  }

  function reduceChunk(chunk) {
    const lo = chunk * REDUCE_CHUNK;
    const hi = Math.min(lo + REDUCE_CHUNK, PARAMS);
    for (let i = lo; i < hi; i++) total[i] = 0;
    for (let s = 0; s < STREAMS; s++) {
      const base = s * PARAMS;
      for (let i = lo; i < hi; i++) total[i] += grads[base + i];
    }
    let squares = 0;
    for (let i = lo; i < hi; i++) squares += total[i] * total[i];
    norms[chunk] = squares;
  }

  function evaluateStream(index) {
    const { bits, count } = bitsPerCharacter(model, validation.subarray(validationBounds[index], validationBounds[index + 1]));
    evalStats[2 * index] = bits * count;
    evalStats[2 * index + 1] = count;
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
        if (command === COMMAND.train) trainStream(task);
        else if (command === COMMAND.reduce) reduceChunk(task);
        else if (command === COMMAND.evaluate) evaluateStream(task);
      }
    } catch (error) {
      Atomics.store(control, CTRL.failed, 1);
      parentPort.postMessage(String(error?.stack ?? error));
    }
    if (Atomics.add(control, CTRL.done, 1) + 1 === workerCount) Atomics.notify(control, CTRL.done);
  }
}

// ---------------------------------------------------------------------------------
// Main thread: corpus
// ---------------------------------------------------------------------------------

function parseArguments(argv) {
  const positional = [];
  const options = { ...DEFAULTS, checkGradients: false, corpusOnly: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--check-gradients') options.checkGradients = true;
    else if (arg === '--corpus-only') options.corpusOnly = true;
    else if (['--epochs', '--seed', '--workers'].includes(arg)) {
      const value = Number(argv[++i]);
      if (!Number.isInteger(value) || value < 0) throw new Error(`${arg} needs a non-negative integer`);
      options[arg.slice(2)] = value;
    } else if (arg.startsWith('--')) throw new Error(`Unknown option ${arg}`);
    else positional.push(arg);
  }
  [options.corpusDir, options.outDir] = positional;
  options.workers = Math.max(1, options.workers);
  options.epochs = Math.max(1, options.epochs);
  return options;
}

/** One book as normalized paragraphs, from its Project Gutenberg plain-text file. */
function readBook(dir, book) {
  const path = join(dir, `pg${book.gutenbergId}.txt`);
  if (!existsSync(path)) throw new Error(`Missing ${path}; see the usage notes at the top of this script.`);
  if (statSync(path).size > MAX_FILE_BYTES) throw new Error(`${path} is unexpectedly large`);
  const bytes = readFileSync(path);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  let text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  text = text.replace(/\r\n?/g, '\n');
  const start = text.match(/^\*\*\* ?START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK[^\n]*$/m);
  const end = text.match(/^\*\*\* ?END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK[^\n]*$/m);
  if (!start || !end || end.index < start.index) throw new Error(`${path}: missing the Project Gutenberg START or END line`);
  const body = text.slice(start.index + start[0].length, end.index)
    .replace(/\[(?:Illustration|Picture)[^\]]*\]/g, '')
    .replace(/\[\d+\]/g, '');
  let offensive = 0;
  const paragraphs = body.split(/\n[ \t]*\n/)
    .map(paragraph => simplifyCharacters(paragraph.replace(/_/g, '')).replace(/\s+/g, ' ').trim())
    .filter(paragraph => paragraph && !isFrontMatter(paragraph))
    .filter(paragraph => {
      if (!OFFENSIVE.test(paragraph)) return true;
      offensive++;
      return false;
    });
  return { ...book, sha256, paragraphs, offensive };
}

/** Paragraphs that are headings, contents, titles or credits rather than prose. */
function isFrontMatter(paragraph) {
  const letters = paragraph.match(/[A-Za-z]/g)?.length ?? 0;
  const lower = paragraph.match(/[a-z]/g)?.length ?? 0;
  if (letters === 0 || lower < letters / 2) return true; // "CHAPTER IV", "THE GOLDEN BIRD", "* * *", "THE END"
  if ((paragraph.match(/\bchapter\b/gi) ?? []).length >= 2) return true; // a table of contents
  if (/\bchapter\b/i.test(paragraph) && paragraph.length < 120) return true; // "Chapter I. The Cyclone"
  if (paragraph.length < 40 && !/[.!?"')]$/.test(paragraph)) return true; // titles and bylines: "Down the Rabbit-Hole"
  return /^(?:produced by|transcriber|e-?text|this e-?book|contents\b|illustrated by|copyright|author of|end of (?:the )?project gutenberg)/i.test(paragraph);
}

/** Normalized books, the vocabulary, and each book split into training and held-out text. */
function loadCorpus(dir) {
  const books = BOOKS.map(book => readBook(dir, book));
  const counts = new Map();
  for (const book of books) {
    for (const paragraph of book.paragraphs) for (const c of paragraph) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  counts.set('\n', books.reduce((n, book) => n + book.paragraphs.length, 0));
  const printable = c => c === '\n' || (c >= ' ' && c <= '~');
  const letter = c => /[A-Za-z]/.test(c);
  const vocabulary = [...counts.keys()].filter(c => printable(c) && (letter(c) || (ALLOWED.test(c) && counts.get(c) >= MIN_COUNT))).sort();
  const removed = [...counts.entries()].filter(([c]) => !vocabulary.includes(c)).sort((a, b) => b[1] - a[1]);
  const keep = new Set(vocabulary);
  for (const book of books) {
    book.paragraphs = book.paragraphs.map(p => Array.from(p).filter(c => keep.has(c)).join('').replace(/ +/g, ' ').trim()).filter(Boolean);
    const length = book.paragraphs.reduce((n, p) => n + p.length + 1, 0);
    let seen = 0;
    let split = book.paragraphs.length;
    for (let i = 0; i < book.paragraphs.length; i++) {
      if (seen >= length * (1 - VALIDATION_SHARE)) { split = i; break; }
      seen += book.paragraphs[i].length + 1;
    }
    book.train = book.paragraphs.slice(0, split).join('\n');
    book.validation = book.paragraphs.slice(split).join('\n');
    if (book.train.length < 50_000 || book.validation.length < 2_000) throw new Error(`${book.title}: unexpectedly short text`);
  }
  return { books, vocabulary, removed };
}

const encode = (text, lookup) => Uint8Array.from(text, c => lookup.get(c));

// ---------------------------------------------------------------------------------
// Main thread: training
// ---------------------------------------------------------------------------------

/** Orthogonal recurrent blocks, small random input and output weights, forget bias 1, output bias log-frequencies. */
function initialize(params, model, trainIndices, random) {
  const H = model.hiddenSize;
  const V = model.inputSize;
  for (let i = 0; i < model.inputWeights.length; i++) model.inputWeights[i] = 0.3 * gaussian(random);
  for (let gate = 0; gate < 4; gate++) {
    // Modified Gram–Schmidt on a Gaussian matrix gives a random orthogonal H×H block.
    const rows = Array.from({ length: H }, () => Float64Array.from({ length: H }, () => gaussian(random)));
    for (let r = 0; r < H; r++) {
      for (let q = 0; q < r; q++) {
        let dot = 0;
        for (let k = 0; k < H; k++) dot += rows[r][k] * rows[q][k];
        for (let k = 0; k < H; k++) rows[r][k] -= dot * rows[q][k];
      }
      const norm = Math.hypot(...rows[r]);
      for (let k = 0; k < H; k++) rows[r][k] /= norm;
    }
    for (let r = 0; r < H; r++) model.recurrentWeights.set(rows[r], (gate * H + r) * H);
  }
  model.bias.fill(0);
  model.bias.fill(1, H, 2 * H); // forget gate
  for (let i = 0; i < model.outputWeights.length; i++) model.outputWeights[i] = gaussian(random) / Math.sqrt(H);
  const frequency = new Float64Array(V).fill(1);
  for (const index of trainIndices) frequency[index]++;
  for (let v = 0; v < V; v++) model.outputBias[v] = Math.log(frequency[v] / (trainIndices.length + V));
  if (params.some(value => !Number.isFinite(value))) throw new Error('Initialization produced a non-finite value');
}

function sharedBytes(bytes) {
  const copy = new Uint8Array(new SharedArrayBuffer(Math.max(1, bytes.length)));
  copy.set(bytes);
  return copy;
}

async function train(options) {
  const startTime = performance.now();
  const { books, vocabulary, removed } = loadCorpus(options.corpusDir);
  const lookup = new Map(vocabulary.map((c, i) => [c, i]));
  const V = vocabulary.length;
  const PARAMS = parameterCount(V, HIDDEN);
  const trainText = books.map(book => book.train).join('\n');
  const trainIndices = encode(trainText, lookup);
  const validationIndices = books.map(book => encode(book.validation, lookup));
  const validationBounds = Int32Array.from({ length: books.length + 1 }, (_, i) =>
    validationIndices.slice(0, i).reduce((n, part) => n + part.length, 0));

  for (const book of books) {
    console.log(`${book.title}: ${book.train.length.toLocaleString()} training and ${book.validation.length.toLocaleString()} held-out characters, `
      + `${book.offensive} paragraph${book.offensive === 1 ? '' : 's'} dropped for offensive terms (sha256 ${book.sha256})`);
  }
  console.log(`Vocabulary: ${V} characters ${JSON.stringify(vocabulary.join(''))}`);
  console.log(`Removed as too rare: ${removed.map(([c, n]) => `${JSON.stringify(c)}×${n}`).join(' ') || 'none'}`);
  if (options.corpusOnly) {
    mkdirSync(options.outDir, { recursive: true });
    writeFileSync(join(options.outDir, 'corpus-train.txt'), trainText);
    writeFileSync(join(options.outDir, 'corpus-held-out.txt'), books.map(book => book.validation).join('\n\n'));
    console.log(`Wrote the normalized text to ${options.outDir}`);
    return;
  }

  const shared = {
    control: new SharedArrayBuffer(CTRL.length * 4),
    params: new SharedArrayBuffer(PARAMS * 4),
    grads: new SharedArrayBuffer(STREAMS * PARAMS * 4),
    total: new SharedArrayBuffer(PARAMS * 8),
    norms: new SharedArrayBuffer(Math.ceil(PARAMS / REDUCE_CHUNK) * 8),
    streams: new SharedArrayBuffer(STREAMS * 2 * 4),
    state: new SharedArrayBuffer(STREAMS * 2 * HIDDEN * 4),
    losses: new SharedArrayBuffer(STREAMS * 8),
    text: sharedBytes(trainIndices).buffer,
    validation: sharedBytes(Uint8Array.from(validationIndices.flatMap(part => Array.from(part)))).buffer,
    validationBounds: (() => { const b = new Int32Array(new SharedArrayBuffer(validationBounds.length * 4)); b.set(validationBounds); return b.buffer; })(),
    evalStats: new SharedArrayBuffer(books.length * 2 * 8),
  };
  const control = new Int32Array(shared.control);
  const params = new Float32Array(shared.params);
  const total = new Float64Array(shared.total);
  const norms = new Float64Array(shared.norms);
  const streams = new Int32Array(shared.streams);
  const losses = new Float64Array(shared.losses);
  const evalStats = new Float64Array(shared.evalStats);
  const model = modelViews(params, vocabulary, HIDDEN);

  const random = mulberry32(options.seed);
  initialize(params, model, trainIndices, random);
  const adamM = new Float64Array(PARAMS);
  const adamV = new Float64Array(PARAMS);

  const workerCount = options.workers;
  const workerErrors = [];
  const workers = Array.from({ length: workerCount }, () => {
    const worker = new Worker(new URL(import.meta.url), { workerData: { shared, vocabulary, workerCount } });
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
    await runPhase(COMMAND.evaluate, books.length);
    let bits = 0;
    let count = 0;
    for (let b = 0; b < books.length; b++) {
      bits += evalStats[2 * b];
      count += evalStats[2 * b + 1];
    }
    return bits / count;
  }

  const N = trainIndices.length;
  const windowsPerEpoch = Math.floor((Math.floor((N - STEPS) / STREAMS) - 1) / STEPS);
  const totalSteps = options.epochs * windowsPerEpoch;
  const learningRate = step => Math.min(1, step / WARMUP_STEPS)
    * (FINAL_LEARNING_RATE + 0.5 * (LEARNING_RATE - FINAL_LEARNING_RATE) * (1 + Math.cos(Math.PI * Math.min(1, step / totalSteps))));
  const chunks = Math.ceil(PARAMS / REDUCE_CHUNK);

  console.log(`LSTM ${HIDDEN} cells, ${PARAMS.toLocaleString()} parameters; ${N.toLocaleString()} training characters, `
    + `${options.epochs} epochs of ${windowsPerEpoch} updates (${STREAMS} streams × ${STEPS} steps), seed ${options.seed}, ${workerCount} workers`);
  let step = 0;
  let best = { bits: Infinity, epoch: 0, params: null };
  const history = [];
  try {
    for (let epoch = 1; epoch <= options.epochs; epoch++) {
      const epochStart = performance.now();
      const offset = Math.floor(random() * STEPS);
      const span = Math.floor((N - 1 - offset) / STREAMS);
      const windows = Math.floor((span - 1) / STEPS);
      let loss = 0;
      let clipped = 0;
      let normSum = 0;
      const progressEvery = Math.max(1, Math.ceil(windows / 10));
      for (let w = 0; w < windows; w++) {
        if (w > 0 && w % progressEvery === 0) {
          const elapsed = (performance.now() - epochStart) / 1000;
          console.log(`  epoch ${epoch}: ${w}/${windows} updates, train ${(loss / (w * STREAMS * STEPS) / Math.LN2).toFixed(3)} bits/char, `
            + `${(w * STREAMS * STEPS / elapsed).toFixed(0)} characters/s, about ${((windows - w) * elapsed / w / 60).toFixed(1)} min left in this epoch`);
        }
        for (let s = 0; s < STREAMS; s++) {
          streams[2 * s] = offset + s * span + w * STEPS;
          streams[2 * s + 1] = w === 0 || random() < RESET_PROBABILITY ? 1 : 0;
        }
        await runPhase(COMMAND.train, STREAMS);
        for (let s = 0; s < STREAMS; s++) loss += losses[s];
        await runPhase(COMMAND.reduce, chunks);
        let squares = 0;
        for (let c = 0; c < chunks; c++) squares += norms[c];
        const count = STREAMS * STEPS;
        const norm = Math.sqrt(squares) / count;
        normSum += norm;
        const scale = (norm > CLIP_NORM ? CLIP_NORM / norm : 1) / count;
        if (norm > CLIP_NORM) clipped++;
        step++;
        const rate = learningRate(step);
        const correction1 = 1 - ADAM.beta1 ** step;
        const correction2 = 1 - ADAM.beta2 ** step;
        for (let i = 0; i < PARAMS; i++) {
          const g = total[i] * scale;
          const m = ADAM.beta1 * adamM[i] + (1 - ADAM.beta1) * g;
          const v = ADAM.beta2 * adamV[i] + (1 - ADAM.beta2) * g * g;
          adamM[i] = m;
          adamV[i] = v;
          params[i] -= (rate * (m / correction1)) / (Math.sqrt(v / correction2) + ADAM.epsilon);
        }
      }
      const trainBits = loss / (windows * STREAMS * STEPS) / Math.LN2;
      const validationBits = await evaluate();
      history.push({ epoch, trainBits: +trainBits.toFixed(4), validationBits: +validationBits.toFixed(4) });
      if (validationBits < best.bits) best = { bits: validationBits, epoch, params: params.slice() };
      console.log(`epoch ${String(epoch).padStart(2)}/${options.epochs}  lr ${learningRate(step).toExponential(2)}  `
        + `train ${trainBits.toFixed(3)} bits/char  held-out ${validationBits.toFixed(3)} bits/char  `
        + `grad norm ${(normSum / windows).toFixed(3)} (clipped ${clipped}/${windows})  `
        + `${((performance.now() - epochStart) / 1000).toFixed(1)} s`);
    }
  } finally {
    await runPhase(COMMAND.exit, 0); // idle workers leave their loop; terminate() covers the rest
    await Promise.all(workers.map(worker => worker.terminate()));
  }
  const seconds = (performance.now() - startTime) / 1000;
  exportModel(options, { books, vocabulary, validationIndices, trainIndices }, best, history, seconds);
}

// ---------------------------------------------------------------------------------
// Export and analysis
// ---------------------------------------------------------------------------------

function encodeWeights(params) {
  const bytes = new ArrayBuffer(params.length * 4);
  const view = new DataView(bytes);
  for (let i = 0; i < params.length; i++) view.setFloat32(i * 4, params[i], true);
  return new Uint8Array(bytes);
}

function toArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

/** Pearson correlation of two equally long series. */
function correlation(a, b) {
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
    const da = a[i] - ma;
    const db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

/**
 * Text features some memory cells might track, one value per character read:
 * inside a quotation (quotes toggle within a paragraph), inside parentheses, letters
 * since the word began, and characters since the paragraph began.
 */
function textFeatures(text) {
  const features = { quote: [], parenthesis: [], word: [], paragraph: [] };
  let quote = 0;
  let parenthesis = 0;
  let word = 0;
  let paragraph = 0;
  for (const c of text) {
    if (c === '"') quote = 1 - quote;
    if (c === '(') parenthesis = 1;
    if (c === ')') parenthesis = 0;
    word = /[A-Za-z']/.test(c) ? word + 1 : 0;
    paragraph = c === '\n' ? 0 : paragraph + 1;
    if (c === '\n') { quote = 0; parenthesis = 0; }
    features.quote.push(quote);
    features.parenthesis.push(parenthesis);
    features.word.push(word);
    features.paragraph.push(Math.min(paragraph, 400));
  }
  return features;
}

const FEATURE_NAMES = {
  quote: 'inside a quotation',
  parenthesis: 'inside parentheses',
  word: 'letters into the current word',
  paragraph: 'characters into the paragraph',
};

/** For each feature, the memory cell whose state follows it most closely on the held-out text. */
function notableCells(model, texts) {
  const H = model.hiddenSize;
  const cells = Array.from({ length: H }, () => []);
  const all = { quote: [], parenthesis: [], word: [], paragraph: [] };
  for (const text of texts) {
    const run = new CharLstmRun(model);
    run.setIndices(Array.from(text, c => model.lookup.get(c)));
    for (let t = 0; t < run.length; t++) for (let k = 0; k < H; k++) cells[k].push(run.cells[t * H + k]);
    const features = textFeatures(text);
    for (const name of Object.keys(all)) all[name].push(...features[name]);
  }
  return Object.entries(all).map(([name, values]) => {
    let bestCell = 0;
    let bestR = 0;
    for (let k = 0; k < H; k++) {
      const r = correlation(cells[k], values);
      if (Math.abs(r) > Math.abs(bestR)) { bestR = r; bestCell = k; }
    }
    return { feature: name, description: FEATURE_NAMES[name], cell: bestCell, correlation: +bestR.toFixed(3) };
  });
}

/** Greedy or sampled continuation, for a look at what the model writes. */
function continueText(model, prompt, length, temperature, random) {
  const run = new CharLstmRun(model);
  const indices = Array.from(prompt, c => model.lookup.get(c));
  run.setIndices(indices);
  const V = model.inputSize;
  let out = prompt;
  for (let n = 0; n < length; n++) {
    const t = run.length - 1;
    let next = run.topCharacter(t);
    if (temperature > 0) {
      const scaled = new Float32Array(V);
      for (let v = 0; v < V; v++) scaled[v] = run.logits[t * V + v] / temperature;
      const p = new Float32Array(V);
      softmaxInto(scaled, 0, V, p, 0);
      let r = random();
      next = V - 1;
      for (let v = 0; v < V; v++) { r -= p[v]; if (r <= 0) { next = v; break; } }
    }
    run.push(next);
    out += model.vocabulary[next];
  }
  return out;
}

/** Writes the binary and metadata, then re-reads them and measures held-out bits per character with model.ts. */
function exportModel(options, corpus, best, history, trainSeconds) {
  const { books, vocabulary, validationIndices } = corpus;
  const V = vocabulary.length;
  mkdirSync(options.outDir, { recursive: true });
  const weightsPath = join(options.outDir, 'char-lstm-v1.bin');
  const metaPath = join(options.outDir, 'char-lstm-v1.json');
  writeFileSync(weightsPath, encodeWeights(best.params));

  const draft = {
    format: MODEL_FORMAT,
    vocabulary: vocabulary.join(''),
    inputSize: V,
    hiddenSize: HIDDEN,
    gateOrder: [...GATE_ORDER],
    parameters: parameterCount(V, HIDDEN),
    tensors: tensorLayout(V, HIDDEN),
    corpus: books.map(({ title, author, gutenbergId }) => ({ title, author, gutenbergId })),
    validationBitsPerChar: 0,
  };
  const model = parseCharLstm(draft, toArrayBuffer(readFileSync(weightsPath)));
  let bits = 0;
  let count = 0;
  const perBook = books.map((book, b) => {
    const score = bitsPerCharacter(model, validationIndices[b]);
    bits += score.bits * score.count;
    count += score.count;
    return { title: book.title, bitsPerChar: +score.bits.toFixed(4), characters: score.count + 1 };
  });
  const validationBits = bits / count;
  const cells = notableCells(model, books.map(book => book.validation));
  const meta = {
    ...draft,
    validationBitsPerChar: +validationBits.toFixed(4),
    validation: { characters: count + books.length, share: VALIDATION_SHARE, perBook, note: 'The last 5% of each book, read from a blank memory; never trained on.' },
    training: {
      characters: corpus.trainIndices.length,
      epochs: options.epochs,
      bestEpoch: best.epoch,
      streams: STREAMS,
      steps: STEPS,
      resetProbability: RESET_PROBABILITY,
      learningRate: LEARNING_RATE,
      finalLearningRate: FINAL_LEARNING_RATE,
      clipNorm: CLIP_NORM,
      seed: options.seed,
      seconds: Math.round(trainSeconds),
      history,
    },
    sources: books.map(({ gutenbergId, sha256 }) => ({ gutenbergId, sha256 })),
    notableCells: cells,
  };
  writeFileSync(metaPath, `${JSON.stringify(meta, null, 2)}\n`);
  // The page parses exactly what was written.
  parseCharLstm(JSON.parse(readFileSync(metaPath, 'utf8')), toArrayBuffer(readFileSync(weightsPath)));

  console.log(`\nHeld-out bits per character (model.ts on the exported file): ${validationBits.toFixed(4)} `
    + `(${perBook.map(book => `${book.title} ${book.bitsPerChar.toFixed(3)}`).join(', ')}); best epoch ${best.epoch}`);
  for (const cell of cells) console.log(`Cell ${cell.cell + 1} follows "${cell.description}" (r = ${cell.correlation})`);
  const random = mulberry32(options.seed + 1);
  for (const prompt of ['Alice was beginning to', '"Where are you going?" asked the', 'Peter']) {
    console.log(`\nGreedy: ${JSON.stringify(continueText(model, prompt, 120, 0, random))}`);
    console.log(`T=0.8:  ${JSON.stringify(continueText(model, prompt, 120, 0.8, random))}`);
  }
  console.log(`\nWrote ${weightsPath} (${statSync(weightsPath).size} bytes) and ${metaPath}`);
  console.log(`Training time ${trainSeconds.toFixed(0)} s.`);
}

// ---------------------------------------------------------------------------------
// Gradient check: backprop through time vs. central differences of the loss
// ---------------------------------------------------------------------------------

function checkGradients() {
  const random = mulberry32(7);
  const vocabulary = ['a', 'b', 'c', 'd', 'e', ' '];
  const V = vocabulary.length;
  const H = 6;
  const steps = 12;
  const PARAMS = parameterCount(V, H);
  const params = new Float32Array(PARAMS);
  for (let i = 0; i < PARAMS; i++) params[i] = (random() * 2 - 1) * 0.6;
  const model = modelViews(params, vocabulary, H);
  const text = Uint8Array.from({ length: steps + 1 }, () => Math.floor(random() * V));
  const h0 = Float32Array.from({ length: H }, () => random() * 2 - 1);
  const c0 = Float32Array.from({ length: H }, () => random() * 2 - 1);
  const trainer = createSequenceTrainer(model, steps);
  const grads = new Float32Array(PARAMS);
  trainer.run(text, 0, h0, c0, grads);
  const loss = () => trainer.run(text, 0, h0, c0, new Float32Array(PARAMS));

  let worst = 0;
  for (const { name, shape, offset } of tensorLayout(V, H)) {
    const count = shape.reduce((a, b) => a * b, 1);
    const direction = Float64Array.from({ length: count }, () => random() * 2 - 1);
    const length = Math.hypot(...direction);
    for (let i = 0; i < count; i++) direction[i] /= length;
    const original = params.slice(offset, offset + count);
    const epsilon = 1e-2;
    const at = sign => {
      for (let i = 0; i < count; i++) params[offset + i] = original[i] + sign * epsilon * direction[i];
      return loss();
    };
    const numeric = (at(1) - at(-1)) / (2 * epsilon);
    params.set(original, offset);
    let analytic = 0;
    for (let i = 0; i < count; i++) analytic += grads[offset + i] * direction[i];
    const error = Math.abs(numeric - analytic) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic));
    worst = Math.max(worst, error);
    console.log(`${name.padEnd(17)} analytic ${analytic.toExponential(5).padStart(13)}  numeric ${numeric.toExponential(5).padStart(13)}  relative error ${error.toExponential(2)}`);
  }
  if (worst > 1e-3) throw new Error(`Gradient check failed (worst relative error ${worst})`);
  console.log('Gradient check passed.');
}

if (isMainThread) {
  try {
    const options = parseArguments(process.argv.slice(2));
    if (options.checkGradients) checkGradients();
    else if (!options.corpusDir || !options.outDir) {
      throw new Error('Usage: node scripts/train-char-lstm.mjs <corpus-dir> <output-dir> [--epochs N] [--seed N] [--workers N] | --check-gradients');
    } else await train(options);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
} else {
  workerMain(workerData);
}
