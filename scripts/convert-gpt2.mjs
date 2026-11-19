#!/usr/bin/env node
/**
 * Converts the official GPT-2 small checkpoint into Neural Network Visualizer's browser format.
 *
 * Usage:
 *   node scripts/convert-gpt2.mjs <model.safetensors> <vocab.json> <merges.txt> <output-dir>
 *
 * Source: https://huggingface.co/openai-community/gpt2 (OpenAI, MIT license).
 * Every weight matrix is stored transposed as [outputs][inputs] and quantized to
 * int8 in blocks of 64 inputs, each with a float16 scale (symmetric,
 * scale = max|block| / 127). Embeddings are quantized the same way. Biases and
 * layer-norm parameters stay float32.
 *
 * GPT-2's final hidden state has a few dimensions far larger than the rest; they
 * magnify rounding in the token-embedding matrix when it scores next tokens. The
 * script finds them by running the original model on sample text and stores those
 * embedding columns exactly (float16), which keeps next-token probabilities within
 * a fraction of a percentage point of the original model.
 *
 * The result (~130 MB) is split into files under 25 MB with a manifest.
 */
import { readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const engine = await import(fileURLToPath(new URL('../src/transformer/gpt2.ts', import.meta.url)));
const { Gpt2Tokenizer } = await import(fileURLToPath(new URL('../src/transformer/tokenizer.ts', import.meta.url)));

const [modelPath, vocabPath, mergesPath, outDir] = process.argv.slice(2);
if (!modelPath || !vocabPath || !mergesPath || !outDir) {
  console.error('Usage: node scripts/convert-gpt2.mjs <model.safetensors> <vocab.json> <merges.txt> <output-dir>');
  process.exit(1);
}

const CONFIG = { nLayer: 12, nHead: 12, nEmbd: 768, vocabSize: 50257, nCtx: 1024, layerNormEpsilon: 1e-5 };
const FILE_COUNT = 7;
const GROUP_SIZE = 64;
const OUTLIER_COLUMNS = 8;

const file = readFileSync(modelPath);
const headerLength = Number(file.readBigUInt64LE(0));
const header = JSON.parse(file.subarray(8, 8 + headerLength).toString('utf8'));
const dataStart = 8 + headerLength;

/** Reads a float32 tensor; accepts names with or without a `transformer.` prefix. */
function tensor(name) {
  const entry = header[name] ?? header[`transformer.${name}`];
  if (!entry) throw new Error(`Missing tensor ${name}`);
  if (entry.dtype !== 'F32') throw new Error(`${name} has dtype ${entry.dtype}; expected F32`);
  const [start, end] = entry.data_offsets;
  const bytes = file.subarray(dataStart + start, dataStart + end);
  // Copy into an aligned buffer.
  const values = new Float32Array(new ArrayBuffer(end - start));
  new Uint8Array(values.buffer).set(bytes);
  return { values, shape: entry.shape };
}

/** Transposes a [rows][cols] float32 matrix. */
function transpose(values, rows, cols) {
  const out = new Float32Array(values.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = values[r * cols + c];
  return out;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
/** float32 → IEEE half precision (round to nearest; values here are normal halves). */
function toHalf(value) {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exponent = ((x >>> 23) & 0xff) - 112;
  const mantissa = x & 0x7fffff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  const half = sign | (exponent << 10) | (mantissa >> 13);
  return mantissa & 0x1000 ? half + 1 : half;
}

/** Symmetric int8 quantization in blocks of GROUP_SIZE inputs with float16 scales. */
function quantize(values, rows, cols) {
  const blocks = cols / GROUP_SIZE;
  const data = new Int8Array(rows * cols);
  const halves = new Uint16Array(rows * blocks);
  let squaredError = 0;
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < blocks; b++) {
      const start = r * cols + b * GROUP_SIZE;
      let max = 0;
      for (let c = 0; c < GROUP_SIZE; c++) max = Math.max(max, Math.abs(values[start + c]));
      const half = toHalf(max > 0 ? max / 127 : 1);
      halves[r * blocks + b] = half;
      const scale = engine.halfToFloat(new Uint16Array([half]))[0];
      for (let c = 0; c < GROUP_SIZE; c++) {
        const q = Math.max(-127, Math.min(127, Math.round(values[start + c] / scale)));
        data[start + c] = q;
        const error = q * scale - values[start + c];
        squaredError += error * error;
      }
    }
  }
  return { data, scales: halves, rmsError: Math.sqrt(squaredError / values.length) };
}

const parts = [];
const tensors = {};
let offset = 0;
const align = () => { const pad = (16 - (offset % 16)) % 16; if (pad) { parts.push(new Uint8Array(pad)); offset += pad; } };
function addBytes(bytes) { align(); const at = offset; parts.push(bytes); offset += bytes.byteLength; return at; }
function addFloat(name, values) {
  tensors[name] = { dtype: 'float32', shape: [values.length], offset: addBytes(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)) };
}
function addMatrix(name, values, rows, cols) {
  const { data, scales, rmsError } = quantize(values, rows, cols);
  const dataOffset = addBytes(new Uint8Array(data.buffer));
  const scalesOffset = addBytes(new Uint8Array(scales.buffer));
  tensors[name] = { dtype: 'int8', shape: [rows, cols], offset: dataOffset, scales: scalesOffset, groupSize: GROUP_SIZE };
  return rmsError;
}

/** Runs the original float32 model on sample text to find the outsized final-hidden dimensions. */
function outlierDimensions() {
  const { nEmbd: E } = CONFIG;
  const matrix = (data, rows, cols) => ({ rows, cols, data, scales: null, groupSize: cols });
  const lin = (p, inputs, outputs) => ({ weight: matrix(transpose(tensor(`${p}.weight`).values, inputs, outputs), outputs, inputs), bias: tensor(`${p}.bias`).values });
  const norm = p => ({ gamma: tensor(`${p}.weight`).values, beta: tensor(`${p}.bias`).values });
  const weights = {
    config: CONFIG,
    tokenEmbedding: matrix(tensor('wte.weight').values, CONFIG.vocabSize, E),
    positionEmbedding: matrix(tensor('wpe.weight').values, CONFIG.nCtx, E),
    finalNorm: norm('ln_f'),
    layers: Array.from({ length: CONFIG.nLayer }, (_, i) => ({
      ln1: norm(`h.${i}.ln_1`), attention: lin(`h.${i}.attn.c_attn`, E, 3 * E), attentionOutput: lin(`h.${i}.attn.c_proj`, E, E),
      ln2: norm(`h.${i}.ln_2`), mlpUp: lin(`h.${i}.mlp.c_fc`, E, 4 * E), mlpDown: lin(`h.${i}.mlp.c_proj`, 4 * E, E),
    })),
  };
  const tokenizer = new Gpt2Tokenizer(JSON.parse(readFileSync(vocabPath, 'utf8')), readFileSync(mergesPath, 'utf8'));
  const samples = [
    'The quick brown fox jumps over the lazy dog.',
    'In 1969, astronauts landed on the Moon and returned safely to Earth.',
    'def add(a, b):\n    return a + b',
    'She opened the letter, read it twice, and smiled.',
    'Prices rose 3.2% in March, according to the report.',
    'Neural networks learn patterns from examples',
  ];
  const magnitude = new Float32Array(E);
  for (const text of samples) {
    const session = new engine.Gpt2Session(weights);
    session.setTokens(tokenizer.encode(text));
    // Every position's final hidden state, not just the last.
    for (const position of session.positions) {
      const hidden = engine.layerNorm(position.layers[CONFIG.nLayer - 1].output, weights.finalNorm, CONFIG.layerNormEpsilon);
      for (let d = 0; d < E; d++) magnitude[d] = Math.max(magnitude[d], Math.abs(hidden[d]));
    }
  }
  return Array.from(magnitude.keys()).sort((a, b) => magnitude[b] - magnitude[a]).slice(0, OUTLIER_COLUMNS).sort((a, b) => a - b);
}

const { nEmbd, nLayer } = CONFIG;
const errors = [];
const outliers = outlierDimensions();
errors.push(['wte', addMatrix('wte', tensor('wte.weight').values, CONFIG.vocabSize, nEmbd)]);
{
  // Exact (float16) token-embedding values for the outsized dimensions, used when scoring next tokens.
  const wte = tensor('wte.weight').values;
  const exact = new Uint16Array(CONFIG.vocabSize * outliers.length);
  for (let t = 0; t < CONFIG.vocabSize; t++) outliers.forEach((d, j) => { exact[t * outliers.length + j] = toHalf(wte[t * nEmbd + d]); });
  tensors['wte.outliers'] = { dtype: 'float16', shape: [CONFIG.vocabSize, outliers.length], offset: addBytes(new Uint8Array(exact.buffer)), dims: outliers };
}
errors.push(['wpe', addMatrix('wpe', tensor('wpe.weight').values, CONFIG.nCtx, nEmbd)]);
for (let i = 0; i < nLayer; i++) {
  const p = `h.${i}`;
  addFloat(`${p}.ln_1.weight`, tensor(`${p}.ln_1.weight`).values);
  addFloat(`${p}.ln_1.bias`, tensor(`${p}.ln_1.bias`).values);
  // Conv1D stores [inputs][outputs]; transpose so each output's weights are contiguous.
  errors.push([`${p}.attn.c_attn`, addMatrix(`${p}.attn.c_attn.weight`, transpose(tensor(`${p}.attn.c_attn.weight`).values, nEmbd, 3 * nEmbd), 3 * nEmbd, nEmbd)]);
  addFloat(`${p}.attn.c_attn.bias`, tensor(`${p}.attn.c_attn.bias`).values);
  errors.push([`${p}.attn.c_proj`, addMatrix(`${p}.attn.c_proj.weight`, transpose(tensor(`${p}.attn.c_proj.weight`).values, nEmbd, nEmbd), nEmbd, nEmbd)]);
  addFloat(`${p}.attn.c_proj.bias`, tensor(`${p}.attn.c_proj.bias`).values);
  addFloat(`${p}.ln_2.weight`, tensor(`${p}.ln_2.weight`).values);
  addFloat(`${p}.ln_2.bias`, tensor(`${p}.ln_2.bias`).values);
  errors.push([`${p}.mlp.c_fc`, addMatrix(`${p}.mlp.c_fc.weight`, transpose(tensor(`${p}.mlp.c_fc.weight`).values, nEmbd, 4 * nEmbd), 4 * nEmbd, nEmbd)]);
  addFloat(`${p}.mlp.c_fc.bias`, tensor(`${p}.mlp.c_fc.bias`).values);
  errors.push([`${p}.mlp.c_proj`, addMatrix(`${p}.mlp.c_proj.weight`, transpose(tensor(`${p}.mlp.c_proj.weight`).values, 4 * nEmbd, nEmbd), nEmbd, 4 * nEmbd)]);
  addFloat(`${p}.mlp.c_proj.bias`, tensor(`${p}.mlp.c_proj.bias`).values);
}
addFloat('ln_f.weight', tensor('ln_f.weight').values);
addFloat('ln_f.bias', tensor('ln_f.bias').values);
align();

const total = new Uint8Array(offset);
let cursor = 0;
for (const part of parts) { total.set(part, cursor); cursor += part.byteLength; }

mkdirSync(outDir, { recursive: true });
const chunkSize = Math.ceil(total.byteLength / FILE_COUNT / 16) * 16;
const files = [];
for (let i = 0; i < FILE_COUNT; i++) {
  const chunk = total.subarray(i * chunkSize, Math.min(total.byteLength, (i + 1) * chunkSize));
  const name = `weights-${String(i).padStart(2, '0')}.bin`;
  writeFileSync(join(outDir, name), chunk);
  files.push({ name, bytes: chunk.byteLength, sha256: createHash('sha256').update(chunk).digest('hex') });
}
const manifest = { format: 'nnvisualizer-gpt2-int8', version: 2, groupSize: GROUP_SIZE, source: 'openai-community/gpt2 (MIT)', config: CONFIG, totalBytes: total.byteLength, files, tensors };
writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest));
copyFileSync(vocabPath, join(outDir, 'vocab.json'));
copyFileSync(mergesPath, join(outDir, 'merges.txt'));

const worst = errors.reduce((a, b) => (b[1] > a[1] ? b : a));
console.log(`Wrote ${files.length} files, ${(total.byteLength / 1e6).toFixed(1)} MB, chunk ≤ ${(chunkSize / 1e6).toFixed(1)} MB`);
console.log(`Quantization RMS error: worst ${worst[0]} ${worst[1].toExponential(2)}`);
console.log(`Exact embedding columns for outsized hidden dimensions: ${outliers.join(', ')}`);
