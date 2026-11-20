/**
 * GPT-2 small on the GPU, through WebGPU.
 *
 * Gpt2GpuSession mirrors Gpt2Session in gpt2.ts: the same int8 block weights,
 * the same steps in the same order, and the same PositionTrace for every token,
 * so the page draws exactly what the GPU computed. The weights are uploaded
 * once: int8 values packed four to a u32 in one buffer per matrix, and every
 * scale, bias and norm in one float buffer. Keys and values stay on the GPU for
 * every position. Each batch of new tokens is one command buffer (embedding,
 * twelve blocks, the final norm, the vocabulary and the surprise statistics),
 * and every intermediate comes back in a single copy.
 *
 * Kernels (WGSL):
 * - embed: token and position rows, dequantized, and their sum.
 * - norm: layer norm, one workgroup per vector (two-pass mean and variance).
 * - matmul: y = W x + b for int8 blocks of 64 inputs with one scale each,
 *   64 outputs by 16 tokens per workgroup; optional GELU, residual add and
 *   key/value cache writes.
 * - matvec: the same for up to four tokens (generating one token at a time),
 *   with sixteen lanes sharing each row so every thread has work.
 * - vocab: the tiled product against the token embedding, plus the exact
 *   correction for the outsized hidden dimensions. It is always tiled, so the
 *   lens's last row and the trace's logits come out identical.
 * - attention: one workgroup per token and head: causal scores, softmax
 *   weights and the weighted sum of values, or zeros for a switched-off head
 *   (whose scores and weights are still traced).
 * - stats: per row of logits, the maximum, the softmax denominator, how many
 *   tokens beat a given one (its rank) and the three most likely tokens.
 */
import { checkHeadMask, Gpt2Session, topTokens, type Gpt2Config, type Gpt2Weights, type HeadMask, type Matrix, type PositionTrace, type Surprise } from './gpt2';

/** Why the GPU isn't running the model. */
export type GpuReason = 'unsupported' | 'no-adapter' | 'device' | 'memory' | 'mismatch' | 'lost' | 'error';

export class GpuError extends Error {
  readonly reason: GpuReason;
  constructor(reason: GpuReason, detail?: string) {
    super(detail ? `${reason}: ${detail}` : reason);
    this.reason = reason;
  }
}

/** How likely a token is after a probe's last token, its rank (1 = likeliest), and the likeliest token. */
export interface Probe {
  probability: number;
  rank: number;
  top: number;
}

/** The likeliest next tokens at one stage of the logit lens, most likely first. */
export interface TopTokens {
  ids: number[];
  probabilities: number[];
}

/** How many tokens the lens reports per stage. */
export const LENS_TOP = 3;

// WebGPU flags; the DOM typings declare the interfaces but not these constants.
const USAGE_MAP_READ = 0x1;
const USAGE_COPY_SRC = 0x4;
const USAGE_COPY_DST = 0x8;
const USAGE_UNIFORM = 0x40;
const USAGE_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;
const STAGE_COMPUTE = 0x4;

/** Inputs per quantization block, which is also the matmul's tile depth. */
const BLOCK = 64;
const TILE_ROWS = 64;
const TILE_TOKENS = 16;
/** Most new tokens in one command buffer; longer inputs run in chunks. */
const MAX_BATCH = 64;
/** Floats of intermediates one batch may read back (32 MB). */
const TRACE_BUDGET = 8 << 20;
/** Each dispatch's parameters sit at their own uniform offset. */
const SLOT_BYTES = 256;
const SLOT = SLOT_BYTES / 4;
/** Enough for a run (about 90 dispatches) or twelve probes from the first block. */
const MAX_SLOTS = 1152;
/** u32 values per row of logit statistics. */
const STAT = 16;
const NONE = 0xffffffff;
const GELU = 1;
const RESIDUAL = 2;
const CACHE = 4;
/** The first prompt token ("The") the start-up check runs on both processors. */
const CHECK_TOKEN = 464;
/** The check passes when every logit is this close to the CPU's (they agree to about 1e-4). */
const CHECK_TOLERANCE = 0.02;

/**
 * The browser's GPU adapter, or a GpuError saying why there isn't one.
 * Software adapters (such as SwiftShader) are only used when allowed.
 */
export async function findAdapter(allowSoftware: boolean): Promise<GPUAdapter> {
  const gpu = typeof navigator === 'undefined' ? undefined : navigator.gpu;
  if (!gpu) throw new GpuError('unsupported');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter) throw new GpuError('no-adapter');
  const software = (adapter.info as GPUAdapterInfo & { isFallbackAdapter?: boolean } | undefined)?.isFallbackAdapter
    ?? (adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter;
  if (software && !allowSoftware) throw new GpuError('no-adapter', 'only a software adapter');
  return adapter;
}

export async function requestGpuDevice(allowSoftware: boolean): Promise<GPUDevice> {
  const adapter = await findAdapter(allowSoftware);
  try {
    return await adapter.requestDevice({ label: 'GPT-2' });
  } catch (error) {
    throw new GpuError('device', error instanceof Error ? error.message : undefined);
  }
}

/* ---------- WGSL ---------- */

function shaders(config: Gpt2Config, dims: number[] | null) {
  const { nEmbd: D, nHead: H, vocabSize: V, nCtx } = config;
  const common = /* wgsl */ `
const D = ${D}u;
const H = ${H}u;
const V = ${V}u;
const NONE = 0xffffffffu;

/** Four int8 values packed little-endian in a u32. */
fn unpack(word: u32) -> vec4<f32> {
  let w = bitcast<i32>(word);
  return vec4<f32>(f32(extractBits(w, 0u, 8u)), f32(extractBits(w, 8u, 8u)), f32(extractBits(w, 16u, 8u)), f32(extractBits(w, 24u, 8u)));
}
`;

  const embed = common + /* wgsl */ `
struct P { tokens: u32, start: u32, tokenOut: u32, positionOut: u32, sumOut: u32, tokenScale: u32, positionScale: u32, ids: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> wte: array<u32>;
@group(0) @binding(2) var<storage, read> wpe: array<u32>;
@group(0) @binding(3) var<storage, read> fp: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<u32>;
@group(0) @binding(5) var<storage, read_write> work: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let word = id.x;
  let i = id.y;
  if (word >= D / 4u || i >= p.tokens) { return; }
  let token = aux[p.ids + i];
  let position = p.start + i;
  let block = word / 16u;
  let blocks = D / 64u;
  let a = unpack(wte[token * (D / 4u) + word]) * fp[p.tokenScale + token * blocks + block];
  let b = unpack(wpe[position * (D / 4u) + word]) * fp[p.positionScale + position * blocks + block];
  let s = a + b;
  let o = i * D + word * 4u;
  for (var k = 0u; k < 4u; k++) {
    work[p.tokenOut + o + k] = a[k];
    work[p.positionOut + o + k] = b[k];
    work[p.sumOut + o + k] = s[k];
  }
}
`;

  const norm = common + /* wgsl */ `
const EPS: f32 = ${config.layerNormEpsilon.toExponential()};
struct P { rows: u32, inOffset: u32, inStride: u32, outOffset: u32, outStride: u32, gamma: u32, beta: u32, table: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fp: array<f32>;
@group(0) @binding(2) var<storage, read> aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> work: array<f32>;
var<workgroup> red: array<f32, 256>;

/** Sums red[] over the workgroup. */
fn total(lid: u32) -> f32 {
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid < k) { red[lid] += red[lid + k]; }
    workgroupBarrier();
  }
  let sum = red[0];
  workgroupBarrier();
  return sum;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wg.x;
  // Rows are evenly spaced, or listed one by one in a table.
  var base = p.inOffset + row * p.inStride;
  if (p.table != NONE) { base = aux[p.table + row]; }
  var s = 0.0;
  for (var c = lid; c < D; c += 256u) { s += work[base + c]; }
  red[lid] = s;
  workgroupBarrier();
  let mean = total(lid) / f32(D);
  var v = 0.0;
  for (var c = lid; c < D; c += 256u) {
    let d = work[base + c] - mean;
    v += d * d;
  }
  red[lid] = v;
  workgroupBarrier();
  let inv = 1.0 / sqrt(total(lid) / f32(D) + EPS);
  let dest = p.outOffset + row * p.outStride;
  for (var c = lid; c < D; c += 256u) {
    work[dest + c] = (work[base + c] - mean) * inv * fp[p.gamma + c] + fp[p.beta + c];
  }
}
`;

  /**
   * y = W x + b for int8 weights in blocks of 64 inputs, each block with its own
   * scale. The layer version finishes each output with the bias and, as asked,
   * GELU, the residual add and a copy into the key/value cache; the vocabulary
   * version adds the exact correction for the outsized hidden dimensions.
   */
  const product = (vocab: boolean, vector: boolean) => common + /* wgsl */ `
struct P {
  rows: u32, cols: u32, tokens: u32, flags: u32,
  inOffset: u32, inStride: u32, outOffset: u32, outStride: u32,
  resOffset: u32, resStride: u32, scale: u32, bias: u32,
  cache: u32, err: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> fp: array<f32>;
${vocab ? `@group(0) @binding(3) var<storage, read> work: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;

fn finish(row: u32, t: u32, value: f32) {
  let x = p.inOffset + t * p.inStride;
  let e = p.err + row * ${dims?.length ?? 0}u;
  logits[p.outOffset + t * V + row] = value${dims?.length ? ` + (${dims.map((d, k) => `fp[e + ${k}u] * work[x + ${d}u]`).join(' + ')})` : ''};
}` : `@group(0) @binding(3) var<storage, read_write> work: array<f32>;
@group(0) @binding(4) var<storage, read_write> kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> vc: array<f32>;

/** GPT-2's tanh GELU, with tanh written out so large inputs can't overflow. */
fn gelu(x: f32) -> f32 {
  let z = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  let e = exp(2.0 * min(abs(z), 15.0));
  return 0.5 * x * (1.0 + sign(z) * (1.0 - 2.0 / (e + 1.0)));
}

fn finish(row: u32, t: u32, value: f32) {
  var y = value + fp[p.bias + row];
  if ((p.flags & ${GELU}u) != 0u) { y = gelu(y); }
  if ((p.flags & ${RESIDUAL}u) != 0u) { y += work[p.resOffset + t * p.resStride + row]; }
  work[p.outOffset + t * p.outStride + row] = y;
  if ((p.flags & ${CACHE}u) != 0u && row >= D) {
    if (row < 2u * D) { kc[p.cache + t * D + row - D] = y; } else { vc[p.cache + t * D + row - 2u * D] = y; }
  }
}`}
${vector ? `
var<workgroup> red: array<vec4<f32>, 256>;

/**
 * Up to four tokens: sixteen lanes share each row, each reading every
 * sixteenth word, so neighbouring lanes read neighbouring weights. Word w lies
 * in block w / 16; a weight times its scale is exact in f32.
 */
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 15u;
  let row = wg.x * 16u + (lid >> 4u);
  let words = p.cols >> 2u;
  let blocks = p.cols >> 6u;
  var acc = vec4<f32>(0.0);
  if (row < p.rows) {
    for (var w = lane; w < words; w += 16u) {
      let q = unpack(W[row * words + w]) * fp[p.scale + row * blocks + (w >> 4u)];
      for (var t = 0u; t < p.tokens; t++) {
        let x = p.inOffset + t * p.inStride + w * 4u;
        acc[t] += dot(q, vec4<f32>(work[x], work[x + 1u], work[x + 2u], work[x + 3u]));
      }
    }
  }
  red[lid] = acc;
  workgroupBarrier();
  for (var k = 8u; k > 0u; k >>= 1u) {
    if (lane < k) { red[lid] += red[lid + k]; }
    workgroupBarrier();
  }
  if (lane != 0u || row >= p.rows) { return; }
  let sums = red[lid];
  for (var t = 0u; t < p.tokens; t++) { finish(row, t, sums[t]); }
}` : `
var<workgroup> ws: array<u32, 1088>;
var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ss: array<f32, 64>;

/**
 * 64 output rows for 16 tokens per workgroup. Thread (tl, rl) owns token tl
 * and rows rl, rl + 16, rl + 32 and rl + 48, so a 32-wide SIMD group shares two
 * tokens and the weight rows it reads sit in different banks. Each k tile is
 * one quantization block: sum the block, then apply its scale, as the CPU
 * does. Threads without a token still load tiles but skip the math.
 */
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let rowBase = wg.x * 64u;
  let tokenBase = wg.y * 16u;
  let rl = lid & 15u;
  let tl = lid >> 4u;
  let t = tokenBase + tl;
  let live = t < p.tokens;
  let words = p.cols >> 2u;
  let blocks = p.cols >> 6u;
  var acc = vec4<f32>(0.0);
  for (var b = 0u; b < blocks; b++) {
    for (var j = 0u; j < 4u; j++) {
      let f = lid + j * 256u;
      let r = f >> 4u;
      let w = f & 15u;
      var word = 0u;
      if (rowBase + r < p.rows) { word = W[(rowBase + r) * words + b * 16u + w]; }
      ws[r * 17u + w] = word;
    }
    var x = vec4<f32>(0.0);
    if (live) {
      let o = p.inOffset + t * p.inStride + b * 64u + rl * 4u;
      x = vec4<f32>(work[o], work[o + 1u], work[o + 2u], work[o + 3u]);
    }
    xs[lid] = x;
    if (lid < 64u) {
      var s = 0.0;
      if (rowBase + lid < p.rows) { s = fp[p.scale + (rowBase + lid) * blocks + b]; }
      ss[lid] = s;
    }
    workgroupBarrier();
    if (live) {
      var part = vec4<f32>(0.0);
      let xb = tl * 16u;
      for (var w = 0u; w < 16u; w++) {
        let xv = xs[xb + w];
        part.x += dot(unpack(ws[rl * 17u + w]), xv);
        part.y += dot(unpack(ws[(rl + 16u) * 17u + w]), xv);
        part.z += dot(unpack(ws[(rl + 32u) * 17u + w]), xv);
        part.w += dot(unpack(ws[(rl + 48u) * 17u + w]), xv);
      }
      acc += part * vec4<f32>(ss[rl], ss[rl + 16u], ss[rl + 32u], ss[rl + 48u]);
    }
    workgroupBarrier();
  }
  if (!live) { return; }
  for (var j = 0u; j < 4u; j++) {
    let row = rowBase + rl + 16u * j;
    if (row < p.rows) { finish(row, t, acc[j]); }
  }
}`}
`;

  const attention = common + /* wgsl */ `
struct P { tokens: u32, start: u32, q: u32, stride: u32, attended: u32, kv: u32, scores: u32, weights: u32, rowLength: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> work: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> row: array<f32, ${nCtx}>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.x;
  let h = wg.y;
  let t = p.start + i;
  let head = h * 64u;
  qs[lid] = work[p.q + i * p.stride + head + lid];
  workgroupBarrier();
  let traced = (i * H + h) * p.rowLength;
  var m = -3.0e38;
  for (var s = lid; s <= t; s += 64u) {
    let key = p.kv + s * D + head;
    var sum = 0.0;
    for (var d = 0u; d < 64u; d++) { sum += qs[d] * kc[key + d]; }
    let score = sum * 0.125;
    row[s] = score;
    work[p.scores + traced + s] = score;
    m = max(m, score);
  }
  red[lid] = m;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (lid < k) { red[lid] = max(red[lid], red[lid + k]); }
    workgroupBarrier();
  }
  let top = red[0];
  workgroupBarrier();
  var partial = 0.0;
  for (var s = lid; s <= t; s += 64u) {
    let e = exp(row[s] - top);
    row[s] = e;
    partial += e;
  }
  red[lid] = partial;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (lid < k) { red[lid] += red[lid + k]; }
    workgroupBarrier();
  }
  let denominator = red[0];
  for (var s = lid; s <= t; s += 64u) {
    let w = row[s] / denominator;
    row[s] = w;
    work[p.weights + traced + s] = w;
  }
  workgroupBarrier();
  // A switched-off head (bit h of p.off) keeps its scores and weights for the trace, but passes on zeros.
  if (((p.off >> h) & 1u) != 0u) {
    work[p.attended + i * p.stride + head + lid] = 0.0;
    return;
  }
  var acc = 0.0;
  for (var s = 0u; s <= t; s++) { acc += row[s] * vc[p.kv + s * D + head + lid]; }
  work[p.attended + i * p.stride + head + lid] = acc;
}
`;

  const stats = common + /* wgsl */ `
struct P { rows: u32, logits: u32, wanted: u32, dest: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
var<workgroup> rf: array<f32, 256>;
var<workgroup> ru: array<u32, 256>;
var<workgroup> tv: array<f32, 768>;
var<workgroup> ti: array<u32, 768>;

/** Higher logits first; equal logits keep the lower token id first, as topTokens does. */
fn ahead(av: f32, ai: u32, bv: f32, bi: u32) -> bool {
  return av > bv || (av == bv && ai < bi);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let r = wg.x;
  let base = p.logits + r * V;
  let scored = p.wanted != NONE;
  var wanted = 0u;
  if (scored) { wanted = aux[p.wanted + r]; }
  let wantedLogit = logits[base + wanted];
  var m = -3.0e38;
  var v0 = -3.0e38;
  var v1 = -3.0e38;
  var v2 = -3.0e38;
  var i0 = NONE;
  var i1 = NONE;
  var i2 = NONE;
  for (var k = lid; k < V; k += 256u) {
    let x = logits[base + k];
    m = max(m, x);
    if (x > v2) {
      if (x > v1) {
        v2 = v1; i2 = i1;
        if (x > v0) { v1 = v0; i1 = i0; v0 = x; i0 = k; } else { v1 = x; i1 = k; }
      } else { v2 = x; i2 = k; }
    }
  }
  rf[lid] = m;
  tv[lid * 3u] = v0; tv[lid * 3u + 1u] = v1; tv[lid * 3u + 2u] = v2;
  ti[lid * 3u] = i0; ti[lid * 3u + 1u] = i1; ti[lid * 3u + 2u] = i2;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid < k) {
      rf[lid] = max(rf[lid], rf[lid + k]);
      let a = lid * 3u;
      let b = (lid + k) * 3u;
      var av = array<f32, 3>(tv[a], tv[a + 1u], tv[a + 2u]);
      var ai = array<u32, 3>(ti[a], ti[a + 1u], ti[a + 2u]);
      var bv = array<f32, 3>(tv[b], tv[b + 1u], tv[b + 2u]);
      var bi = array<u32, 3>(ti[b], ti[b + 1u], ti[b + 2u]);
      var x = 0u;
      var y = 0u;
      for (var n = 0u; n < 3u; n++) {
        if (y >= 3u || (x < 3u && ahead(av[x], ai[x], bv[y], bi[y]))) {
          tv[a + n] = av[x]; ti[a + n] = ai[x]; x++;
        } else {
          tv[a + n] = bv[y]; ti[a + n] = bi[y]; y++;
        }
      }
    }
    workgroupBarrier();
  }
  let top = rf[0];
  workgroupBarrier();
  var s = 0.0;
  var c = 0u;
  for (var k = lid; k < V; k += 256u) {
    let x = logits[base + k];
    s += exp(x - top);
    if (x > wantedLogit) { c++; }
  }
  rf[lid] = s;
  ru[lid] = c;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (lid < k) {
      rf[lid] += rf[lid + k];
      ru[lid] += ru[lid + k];
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let o = p.dest + r * ${STAT}u;
    stats[o] = bitcast<u32>(top);
    stats[o + 1u] = bitcast<u32>(rf[0]);
    stats[o + 2u] = bitcast<u32>(wantedLogit);
    stats[o + 3u] = ru[0];
    for (var n = 0u; n < 3u; n++) {
      stats[o + 4u + n] = bitcast<u32>(tv[n]);
      stats[o + 7u + n] = ti[n];
    }
  }
}
`;

  return { embed, norm, matmul: product(false, false), matvec: product(false, true), vocab: product(true, false), attention, stats };
}

/* ---------- Session ---------- */

type Kernel = 'embed' | 'norm' | 'matmul' | 'matvec' | 'vocab' | 'attention' | 'stats';
type Binding = 'uniform' | 'read-only-storage' | 'storage';
/** Each kernel's bindings; the matrix-vector kernel shares the matmul's, so they share bind groups too. */
type Shape = Exclude<Kernel, 'matvec'>;
const BINDINGS: Record<Shape, Binding[]> = {
  embed: ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage'],
  norm: ['uniform', 'read-only-storage', 'read-only-storage', 'storage'],
  matmul: ['uniform', 'read-only-storage', 'read-only-storage', 'storage', 'storage', 'storage'],
  vocab: ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage'],
  attention: ['uniform', 'storage', 'read-only-storage', 'read-only-storage'],
  stats: ['uniform', 'read-only-storage', 'read-only-storage', 'storage'],
};
const KERNELS: Record<Kernel, Shape> = { embed: 'embed', norm: 'norm', matmul: 'matmul', matvec: 'matmul', vocab: 'vocab', attention: 'attention', stats: 'stats' };
/** Batches up to this many tokens use the matrix-vector kernel for the blocks' products. */
const VECTOR_TOKENS = 4;

type Projection = 'qkv' | 'proj' | 'up' | 'down';
/** Float offsets of a linear layer's block scales and bias, or a norm's gamma and beta. */
type Pair = [number, number];

/** A position as the GPU session keeps it: the CPU's trace plus its final layer-normed vector once known. */
interface GpuPosition extends PositionTrace {
  hidden?: Float32Array;
}

/** Where one batch keeps its values in the work buffer, in floats. */
interface Layout {
  embeddings: number;
  records: number;
  scores: number;
  weights: number;
  /** Everything before this comes back to the CPU. */
  traceEnd: number;
  normed2: number;
  previous: number;
  lensInput: number;
  final: number;
  size: number;
}

interface Groups {
  embed: GPUBindGroup;
  norm: GPUBindGroup;
  layers: Record<Projection, GPUBindGroup>[];
  vocab: GPUBindGroup;
  attention: GPUBindGroup;
  stats: GPUBindGroup;
}

export class Gpt2GpuSession {
  readonly backend = 'gpu' as const;
  readonly weights: Gpt2Weights;
  readonly positions: GpuPosition[] = [];
  private readonly device: GPUDevice;
  private readonly config: Gpt2Config;
  /** Floats per token per block: normed, q, k, v, attended, afterAttention, output (768 each), then the MLP (3,072). */
  private readonly record: number;
  private lost = false;
  private destroyed = false;
  private pipelines: Record<Kernel, GPUComputePipeline>;
  private layouts: Record<Shape, GPUBindGroupLayout>;
  private tensors: { wte: GPUBuffer; wpe: GPUBuffer; layers: Record<Projection, GPUBuffer>[] };
  private floats: GPUBuffer;
  private offsets: { wte: number; wpe: number; err: number; final: Pair; layers: ({ ln1: Pair; ln2: Pair } & Record<Projection, Pair>)[] };
  private uniforms: GPUBuffer;
  private aux: GPUBuffer;
  private statBuffer: GPUBuffer;
  private logitBuffer: GPUBuffer;
  private work: GPUBuffer | null = null;
  private keys: GPUBuffer | null = null;
  private values: GPUBuffer | null = null;
  private capacity = 0;
  private staging: GPUBuffer | null = null;
  private groups: Groups | null = null;
  /** Switched-off heads, one bitmask per block (see HeadMask in gpt2.ts). */
  private mask: number[];

  private constructor(device: GPUDevice, weights: Gpt2Weights, onLost?: (session: Gpt2GpuSession) => void) {
    this.device = device;
    this.weights = weights;
    this.config = weights.config;
    this.mask = checkHeadMask(null, this.config);
    this.record = 7 * this.config.nEmbd + 4 * this.config.nEmbd;
    device.lost.then(info => {
      this.lost = true;
      if (!this.destroyed && info.reason !== 'destroyed') onLost?.(this);
    });
  }

  /**
   * Uploads the weights and compiles the kernels; `onLost` runs if the GPU later goes away.
   * The upload happens before this returns its promise, so CPU work can overlap the compiling.
   */
  static async create(device: GPUDevice, weights: Gpt2Weights, onLost?: (session: Gpt2GpuSession) => void): Promise<Gpt2GpuSession> {
    const { nEmbd, nHead } = weights.config;
    const matrices = [weights.tokenEmbedding, weights.positionEmbedding, ...weights.layers.flatMap(layer =>
      [layer.attention.weight, layer.attentionOutput.weight, layer.mlpUp.weight, layer.mlpDown.weight])];
    if (nEmbd / nHead !== 64 || nEmbd % BLOCK || !matrices.every(m => m.data instanceof Int8Array && m.scales && m.groupSize === BLOCK)) {
      throw new GpuError('error', 'these weights need the CPU');
    }
    const session = new Gpt2GpuSession(device, weights, onLost);
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    try {
      session.upload();
      await session.compile();
    } finally {
      const [validation, memory] = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
      if (memory) throw new GpuError('memory', memory.message);
      if (validation) throw new GpuError('error', validation.message);
    }
    return session;
  }

  /**
   * Another session on the same device, sharing the uploaded weights and the
   * kernels, with its own tokens, head mask and key/value cache: the worker
   * keeps one with every head on beside one with heads switched off.
   * Destroying either session frees the device for both.
   */
  sibling(): Gpt2GpuSession {
    this.usable();
    const other = new Gpt2GpuSession(this.device, this.weights);
    other.pipelines = this.pipelines;
    other.layouts = this.layouts;
    other.tensors = this.tensors;
    other.floats = this.floats;
    other.offsets = this.offsets;
    other.uniforms = this.uniforms;
    other.aux = this.aux;
    other.statBuffer = this.statBuffer;
    other.logitBuffer = this.logitBuffer;
    return other;
  }

  get tokens(): number[] {
    return this.positions.map(position => position.token);
  }

  /** The switched-off heads, one bitmask per block (a copy). */
  get headMask(): number[] {
    return this.mask.slice();
  }

  /**
   * Same contract as Gpt2Session.setTokens: keeps the shared prefix, computes the rest, returns how many were reused.
   * A new head mask changes every block from the first one it touches, so the GPU computes every position again.
   */
  async setTokens(tokens: number[], mask?: HeadMask | null): Promise<number> {
    this.usable();
    const { nCtx, vocabSize } = this.config;
    if (tokens.length > nCtx) throw new Error(`GPT-2 reads at most ${nCtx} tokens.`);
    for (const token of tokens) if (!Number.isInteger(token) || token < 0 || token >= vocabSize) throw new Error(`Unknown token ${token}.`);
    const next = mask === undefined ? this.mask : checkHeadMask(mask, this.config);
    if (next.some((bits, l) => bits !== this.mask[l])) this.positions.length = 0;
    this.mask = next;
    let shared = 0;
    while (shared < tokens.length && shared < this.positions.length && tokens[shared] === this.positions[shared].token) shared++;
    this.positions.length = shared;
    let at = shared;
    while (at < tokens.length) {
      let count = Math.min(tokens.length - at, MAX_BATCH);
      while (count > 1 && this.traceSize(count, at + count) > TRACE_BUDGET) count = Math.ceil(count / 2);
      await this.batch(tokens.slice(at, at + count));
      at += count;
    }
    // A shortened prompt ends on a position whose prediction was never read back.
    const last = this.positions.at(-1);
    if (last && (!last.logits || !last.hidden)) await this.batch([]);
    return shared;
  }

  /** Forgets every position; the next run computes from scratch. */
  reset(): void {
    this.positions.length = 0;
  }

  surprisal(): (Surprise | null)[] {
    return this.positions.map((position, t) => (t === 0 ? null : position.surprise ?? null));
  }

  /** Next-token logits after the last position (a copy the caller may keep or transfer). */
  logits(): Float32Array | null {
    return this.positions.at(-1)?.logits?.slice() ?? null;
  }

  /** The final layer-normed vector at the last position. */
  finalHidden(): Float32Array | null {
    return this.positions.at(-1)?.hidden?.slice() ?? null;
  }

  /**
   * The logit lens at `position`: the likeliest next tokens after the embedding
   * and after every block, read through the final norm and the unembedding
   * exactly like the real output.
   */
  async lens(position: number): Promise<TopTokens[]> {
    this.usable();
    const trace = this.positions[position];
    if (!trace) throw new RangeError(`There is no token at position ${position}.`);
    const { nEmbd: D, nLayer: L } = this.config;
    const rows = L + 1;
    const layout = this.layout(0, 0, rows);
    const streams = new Float32Array(rows * D);
    streams.set(trace.input);
    trace.layers.forEach((step, l) => streams.set(step.output, (l + 1) * D));
    const bytes = rows * STAT * 4;
    const data = await this.submit(layout, bytes, (encode, groups) => {
      this.device.queue.writeBuffer(this.work, layout.lensInput * 4, streams);
      this.unembed(encode, groups, layout, NONE, NONE, rows, layout.lensInput);
    }, encoder => encoder.copyBufferToBuffer(this.statBuffer, 0, this.staging, 0, bytes));
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    return Array.from({ length: rows }, (_, r) => {
      const o = r * STAT;
      const max = floats[o];
      const total = floats[o + 1];
      const ids = Array.from(words.subarray(o + 7, o + 7 + LENS_TOP));
      return { ids, probabilities: ids.map((_id, k) => Math.exp(floats[o + 4 + k] - max) / total) };
    });
  }

  /**
   * Runs `tokens` once per mask, with that mask's heads switched off, and reads back only how likely `target` is
   * after the last token (or the likeliest token, when `target` is null): no trace, so a probe costs little more than
   * its computing, and all of them share one command buffer. Every mask must leave the blocks before `from` whole;
   * from a later block, `stream` holds that block's input for every token (as with every head on, [token][768]).
   * Probes overwrite the key/value cache, so the session forgets its positions.
   */
  async probe(tokens: number[], masks: HeadMask[], target: number | null, from = 0, stream: Float32Array | null = null): Promise<Probe[]> {
    this.usable();
    const { nEmbd: D, nLayer: L, nHead: H, vocabSize } = this.config;
    const bits = masks.map(mask => checkHeadMask(mask, this.config));
    const T = tokens.length;
    if (T < 1 || T > MAX_BATCH) throw new Error(`A probe reads 1 to ${MAX_BATCH} tokens.`);
    for (const token of [...tokens, target ?? 0]) if (!Number.isInteger(token) || token < 0 || token >= vocabSize) throw new Error(`Unknown token ${token}.`);
    // Dispatches per probe: seven per block, the embedding from the first block, then the norm, vocabulary and statistics.
    const dispatches = 7 * (L - from) + (from === 0 ? 1 : 0) + 3;
    if (bits.length < 1 || bits.length * dispatches > MAX_SLOTS || bits.length * STAT > (MAX_BATCH + 1) * STAT) throw new Error('Too many probes at once.');
    if (from > 0 && (stream?.length !== T * D || bits.some(mask => mask.slice(0, from).some(Boolean)))) throw new Error(`Probes from block ${from + 1} need its input and no heads off before it.`);
    this.positions.length = 0;
    const layout = this.layout(T, T, 1);
    const { field, record } = this.fields(layout, T);
    // aux: token ids, where the last position's residual stream lives, and the token to rank.
    const aux = Uint32Array.of(...tokens, record(L - 1, T - 1) + field.output, target ?? 0);
    const bytes = bits.length * STAT * 4;
    const data = await this.submit(layout, bytes, (encode, groups) => {
      this.device.queue.writeBuffer(this.aux, 0, aux);
      if (from > 0) this.device.queue.writeBuffer(this.work, (layout.embeddings + 2 * T * D) * 4, stream);
      bits.forEach((mask, i) => {
        this.encodeBlocks(encode, groups, layout, T, 0, mask, from);
        // The vocabulary for the last position, then this probe's own row of statistics.
        this.unembed(encode, groups, layout, T, T + 1, 1, 0, 0);
        encode('stats', groups.stats, [1, 0, T + 1, i * STAT], 1);
      });
    }, encoder => encoder.copyBufferToBuffer(this.statBuffer, 0, this.staging, 0, bytes), T, 0);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    return bits.map((_mask, i) => {
      const o = i * STAT;
      const [max, total] = [floats[o], floats[o + 1]];
      return target === null
        ? { probability: Math.exp(floats[o + 4] - max) / total, rank: 1, top: words[o + 7] }
        : { probability: Math.exp(floats[o + 2] - max) / total, rank: words[o + 3] + 1, top: words[o + 7] };
    });
  }

  /** The CPU's logits for the start-up check's token. */
  static reference(weights: Gpt2Weights): Float32Array {
    const session = new Gpt2Session(weights);
    session.setTokens([CHECK_TOKEN]);
    return session.logits();
  }

  /**
   * Runs the check token on the GPU and compares its logits with the CPU's
   * (from `reference`), so a GPU that computes wrongly is never used. Leaves the session empty.
   */
  async check(expected: Float32Array): Promise<void> {
    await this.setTokens([CHECK_TOKEN]);
    const actual = this.logits();
    this.reset();
    let worst = 0;
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
    const sameTop = topTokens(expected, 5).ids.join() === topTokens(actual, 5).ids.join();
    if (!(worst <= CHECK_TOLERANCE) || !sameTop) throw new GpuError('mismatch', `largest logit difference ${worst}`);
  }

  /** Frees the GPU memory; the session can't be used afterwards. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.device.destroy();
  }

  private usable(): void {
    if (this.destroyed || this.lost) throw new GpuError('lost');
  }

  /** Whether the device is gone. A failed readback can arrive just before the lost notice, so wait a moment for it. */
  private gone(): Promise<boolean> {
    if (this.destroyed || this.lost) return Promise.resolve(true);
    return Promise.race([this.device.lost.then(() => true), new Promise<boolean>(resolve => setTimeout(() => resolve(false), 100))]);
  }

  /* ---------- Setup ---------- */

  private upload(): void {
    const { device, weights } = this;
    const matrix = (m: Matrix, label: string) => {
      const buffer = device.createBuffer({ label, size: m.data.byteLength, usage: USAGE_STORAGE, mappedAtCreation: true });
      new Int8Array(buffer.getMappedRange()).set(m.data as Int8Array);
      buffer.unmap();
      return buffer;
    };
    this.tensors = {
      wte: matrix(weights.tokenEmbedding, 'wte'),
      wpe: matrix(weights.positionEmbedding, 'wpe'),
      layers: weights.layers.map((layer, l) => ({
        qkv: matrix(layer.attention.weight, `h.${l}.attn.c_attn`),
        proj: matrix(layer.attentionOutput.weight, `h.${l}.attn.c_proj`),
        up: matrix(layer.mlpUp.weight, `h.${l}.mlp.c_fc`),
        down: matrix(layer.mlpDown.weight, `h.${l}.mlp.c_proj`),
      })),
    };

    // Every scale, bias and norm in one buffer, addressed by offset.
    const parts: Float32Array[] = [];
    let size = 0;
    const put = (values: Float32Array) => {
      const at = size;
      parts.push(values);
      size += values.length;
      return at;
    };
    const linear = (weight: Matrix, bias: Float32Array): Pair => [put(weight.scales), put(bias)];
    this.offsets = {
      wte: put(weights.tokenEmbedding.scales),
      wpe: put(weights.positionEmbedding.scales),
      err: put(this.correction()),
      final: [put(weights.finalNorm.gamma), put(weights.finalNorm.beta)],
      layers: weights.layers.map(layer => ({
        ln1: [put(layer.ln1.gamma), put(layer.ln1.beta)] as Pair,
        ln2: [put(layer.ln2.gamma), put(layer.ln2.beta)] as Pair,
        qkv: linear(layer.attention.weight, layer.attention.bias),
        proj: linear(layer.attentionOutput.weight, layer.attentionOutput.bias),
        up: linear(layer.mlpUp.weight, layer.mlpUp.bias),
        down: linear(layer.mlpDown.weight, layer.mlpDown.bias),
      })),
    };
    this.floats = device.createBuffer({ label: 'scales, biases and norms', size: Math.max(4, size * 4), usage: USAGE_STORAGE, mappedAtCreation: true });
    const mapped = new Float32Array(this.floats.getMappedRange());
    let at = 0;
    for (const part of parts) {
      mapped.set(part, at);
      at += part.length;
    }
    this.floats.unmap();

    const { vocabSize: V, nLayer: L } = this.config;
    const logitRows = Math.max(MAX_BATCH + 1, L + 1);
    this.uniforms = device.createBuffer({ label: 'parameters', size: MAX_SLOTS * SLOT_BYTES, usage: USAGE_UNIFORM | USAGE_COPY_DST });
    this.aux = device.createBuffer({ label: 'token ids and row tables', size: 4 * 4 * (MAX_BATCH + 2), usage: USAGE_STORAGE | USAGE_COPY_DST });
    this.statBuffer = device.createBuffer({ label: 'logit statistics', size: logitRows * STAT * 4, usage: USAGE_STORAGE | USAGE_COPY_SRC });
    this.logitBuffer = device.createBuffer({ label: 'logits', size: logitRows * V * 4, usage: USAGE_STORAGE | USAGE_COPY_SRC });
  }

  /**
   * The exact-minus-rounded token embedding on the outsized hidden dimensions,
   * which Gpt2Session.logitsFor adds when scoring the vocabulary.
   */
  private correction(): Float32Array {
    const { tokenEmbedding, outputCorrection } = this.weights;
    if (!outputCorrection) return new Float32Array(0);
    const { dims, values } = outputCorrection;
    const { rows, cols, data, scales, groupSize } = tokenEmbedding;
    const blocks = cols / groupSize;
    const k = dims.length;
    const error = new Float32Array(rows * k);
    for (let t = 0; t < rows; t++) {
      for (let j = 0; j < k; j++) {
        const d = dims[j];
        error[t * k + j] = values[t * k + j] - data[t * cols + d] * scales[t * blocks + Math.floor(d / groupSize)];
      }
    }
    return error;
  }

  private async compile(): Promise<void> {
    const dims = this.weights.outputCorrection?.dims ?? null;
    const sources = shaders(this.config, dims);
    const shapes = Object.keys(BINDINGS) as Shape[];
    this.layouts = Object.fromEntries(shapes.map(shape => [shape, this.device.createBindGroupLayout({
      label: shape,
      entries: BINDINGS[shape].map((type, binding) => ({
        binding,
        visibility: STAGE_COMPUTE,
        buffer: type === 'uniform' ? { type, hasDynamicOffset: true } : { type },
      })),
    })])) as Record<Shape, GPUBindGroupLayout>;
    const kernels = Object.keys(KERNELS) as Kernel[];
    const pipelines = await Promise.all(kernels.map(kernel => this.device.createComputePipelineAsync({
      label: kernel,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layouts[KERNELS[kernel]]] }),
      compute: { module: this.device.createShaderModule({ label: kernel, code: sources[kernel] }), entryPoint: 'main' },
    }).catch(error => {
      throw new GpuError('error', `${kernel}: ${error instanceof Error ? error.message : error}`);
    })));
    this.pipelines = Object.fromEntries(kernels.map((kernel, i) => [kernel, pipelines[i]])) as Record<Kernel, GPUComputePipeline>;
  }

  /* ---------- Buffers ---------- */

  private traceSize(tokens: number, n: number): number {
    const { nEmbd: D, nLayer: L, nHead: H } = this.config;
    return 3 * tokens * D + L * tokens * this.record + 2 * L * tokens * H * n;
  }

  private layout(tokens: number, n: number, rows: number): Layout {
    const { nEmbd: D, nLayer: L, nHead: H } = this.config;
    let size = 0;
    const take = (floats: number) => {
      const at = size;
      size += floats;
      return at;
    };
    const embeddings = take(3 * tokens * D);
    const records = take(L * tokens * this.record);
    const scores = take(L * tokens * H * n);
    const weights = take(L * tokens * H * n);
    const traceEnd = size;
    return {
      embeddings, records, scores, weights, traceEnd,
      normed2: take(tokens * D),
      previous: take(D),
      lensInput: take((L + 1) * D),
      final: take(Math.max(rows, L + 1) * D),
      size,
    };
  }

  /** Grows the work buffer, the key/value cache and the staging buffer as needed. */
  private reserve(floats: number, positions: number, encoder: GPUCommandEncoder, keep: number): void {
    const { nEmbd: D, nLayer: L, nCtx } = this.config;
    if (!this.work || this.work.size < floats * 4) {
      this.work?.destroy();
      this.work = this.device.createBuffer({ label: 'work', size: Math.max(floats * 4, 16 << 20), usage: USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST });
      this.groups = null;
    }
    if (positions > this.capacity) {
      const capacity = Math.min(nCtx, Math.max(64, Math.ceil(Math.max(positions, this.capacity * 2) / 64) * 64));
      const make = (label: string) => this.device.createBuffer({ label, size: L * capacity * D * 4, usage: USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST });
      const keys = make('keys');
      const values = make('values');
      if (this.keys && keep > 0) {
        for (let l = 0; l < L; l++) {
          encoder.copyBufferToBuffer(this.keys, l * this.capacity * D * 4, keys, l * capacity * D * 4, keep * D * 4);
          encoder.copyBufferToBuffer(this.values, l * this.capacity * D * 4, values, l * capacity * D * 4, keep * D * 4);
        }
      }
      // The old buffers go in a microtask, which runs after submit() has submitted this command buffer (it does so
      // without awaiting); a buffer destroyed after submission stays alive until the GPU finishes these copies.
      const old = [this.keys, this.values];
      this.keys = keys;
      this.values = values;
      this.capacity = capacity;
      this.groups = null;
      queueMicrotask(() => old.forEach(buffer => buffer?.destroy()));
    }
  }

  private stage(bytes: number): void {
    if (this.staging && this.staging.size >= bytes) return;
    this.staging?.destroy();
    this.staging = this.device.createBuffer({ label: 'readback', size: Math.max(bytes, 1 << 20), usage: USAGE_MAP_READ | USAGE_COPY_DST });
  }

  private bindGroups(): Groups {
    if (this.groups) return this.groups;
    const { device, layouts, tensors, floats, work, keys, values, aux, uniforms } = this;
    const group = (kernel: Shape, buffers: GPUBuffer[]) => device.createBindGroup({
      label: kernel,
      layout: layouts[kernel],
      entries: [{ binding: 0, resource: { buffer: uniforms, offset: 0, size: SLOT_BYTES } },
        ...buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))],
    });
    this.groups = {
      embed: group('embed', [tensors.wte, tensors.wpe, floats, aux, work]),
      norm: group('norm', [floats, aux, work]),
      layers: tensors.layers.map(layer => ({
        qkv: group('matmul', [layer.qkv, floats, work, keys, values]),
        proj: group('matmul', [layer.proj, floats, work, keys, values]),
        up: group('matmul', [layer.up, floats, work, keys, values]),
        down: group('matmul', [layer.down, floats, work, keys, values]),
      })),
      vocab: group('vocab', [tensors.wte, floats, work, this.logitBuffer]),
      attention: group('attention', [work, keys, values]),
      stats: group('stats', [this.logitBuffer, aux, this.statBuffer]),
    };
    return this.groups;
  }

  /* ---------- Running ---------- */

  /**
   * Records one command buffer, runs it, and reads back `bytes` from the
   * staging buffer. `record` adds the dispatches; `copy` fills the staging buffer.
   */
  private async submit(layout: Layout, bytes: number, record: (encode: Encode, groups: Groups) => void,
      copy: (encoder: GPUCommandEncoder) => void, positions = this.capacity, keep = 0): Promise<ArrayBuffer> {
    const { device } = this;
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    this.reserve(layout.size, positions, encoder, keep);
    this.stage(bytes);
    const groups = this.bindGroups();
    // Sized for a run; a command buffer of probes grows it.
    let params = new Uint32Array(128 * SLOT);
    let slot = 0;
    const pass = encoder.beginComputePass();
    let current: GPUComputePipeline | null = null;
    const encode: Encode = (kernel, group, values, x, y = 1) => {
      if (slot === MAX_SLOTS) throw new GpuError('error', 'too many dispatches');
      if ((slot + 1) * SLOT > params.length) {
        const grown = new Uint32Array(Math.min(MAX_SLOTS, 2 * params.length / SLOT) * SLOT);
        grown.set(params);
        params = grown;
      }
      params.set(values, slot * SLOT);
      if (this.pipelines[kernel] !== current) pass.setPipeline(current = this.pipelines[kernel]);
      pass.setBindGroup(0, group, [slot * SLOT_BYTES]);
      pass.dispatchWorkgroups(x, y);
      slot++;
    };
    try {
      record(encode, groups);
    } finally {
      pass.end();
    }
    copy(encoder);
    device.queue.writeBuffer(this.uniforms, 0, params, 0, slot * SLOT);
    device.queue.submit([encoder.finish()]);
    const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(([validation, memory]) => {
      if (memory) throw new GpuError('memory', memory.message);
      if (validation) throw new GpuError('error', validation.message);
    });
    try {
      await Promise.all([this.staging.mapAsync(MAP_MODE_READ, 0, bytes), errors]);
    } catch (error) {
      if (await this.gone()) throw new GpuError('lost');
      throw error instanceof GpuError ? error : new GpuError('error', error instanceof Error ? error.message : undefined);
    }
    const data = this.staging.getMappedRange(0, bytes).slice(0);
    this.staging.unmap();
    return data;
  }

  /**
   * The final norm, the vocabulary and the statistics for `rows` vectors. A run ranks each row's next token, so by
   * default the last row, whose next token isn't known yet, gets no statistics.
   */
  private unembed(encode: Encode, groups: Groups, layout: Layout, table: number, wanted: number, rows: number, start = 0,
      statRows = wanted === NONE ? rows : rows - 1): void {
    const { nEmbd: D, vocabSize: V } = this.config;
    const [gamma, beta] = this.offsets.final;
    encode('norm', groups.norm, [rows, start, D, layout.final, D, gamma, beta, table], rows);
    encode('vocab', groups.vocab, [V, D, rows, 0, layout.final, D, 0, V, 0, 0, this.offsets.wte, 0, 0, this.offsets.err],
      Math.ceil(V / TILE_ROWS), Math.ceil(rows / TILE_TOKENS));
    if (statRows > 0) encode('stats', groups.stats, [statRows, 0, wanted, 0], statRows);
  }

  /** Where each value sits in a token's record for one block, and where token i's record for block l starts. */
  private fields(layout: Layout, tokens: number) {
    const D = this.config.nEmbd;
    return {
      field: { normed: 0, q: D, attended: 4 * D, after: 5 * D, output: 6 * D, mlp: 7 * D },
      record: (l: number, i = 0) => layout.records + (l * tokens + i) * this.record,
    };
  }

  /**
   * Encodes the embedding and every block for `tokens` new tokens after `start` positions, with `mask`'s heads
   * switched off. Keys and values go to the cache; every intermediate stays in the work buffer as `layout` places it.
   * From a later block `from`, the embedding is skipped: the caller has put that block's input where the embedding
   * sum would be.
   */
  private encodeBlocks(encode: Encode, groups: Groups, layout: Layout, tokens: number, start: number, mask: number[], from = 0): void {
    const { nEmbd: D, nLayer: L, nHead: H } = this.config;
    const REC = this.record;
    const T = tokens;
    const n = start + T;
    const { field, record } = this.fields(layout, T);
    const kv = (l: number) => l * this.capacity * D;
    if (from === 0) {
      encode('embed', groups.embed, [T, start, layout.embeddings, layout.embeddings + T * D, layout.embeddings + 2 * T * D,
        this.offsets.wte, this.offsets.wpe, 0], Math.ceil(D / 4 / 64), T);
    }
    // y = W x + b for this batch: one row per 16 lanes for a few tokens, 64 rows by 16 tokens otherwise.
    const product = (group: GPUBindGroup, values: number[], rows: number) => (T <= VECTOR_TOKENS
      ? encode('matvec', group, values, Math.ceil(rows / 16))
      : encode('matmul', group, values, Math.ceil(rows / TILE_ROWS), Math.ceil(T / TILE_TOKENS)));
    for (let l = from; l < L; l++) {
      const at = this.offsets.layers[l];
      const base = record(l);
      // The block's input: the embedding sum (or what stands in for it), or the previous block's output.
      const [input, inputStride] = l === from ? [layout.embeddings + 2 * T * D, D] : [record(l - 1) + field.output, REC];
      const layer = groups.layers[l];
      encode('norm', groups.norm, [T, input, inputStride, base + field.normed, REC, at.ln1[0], at.ln1[1], NONE], T);
      product(layer.qkv, [3 * D, D, T, CACHE, base + field.normed, REC, base + field.q, REC, 0, 0,
        at.qkv[0], at.qkv[1], kv(l) + start * D, 0], 3 * D);
      encode('attention', groups.attention, [T, start, base + field.q, REC, base + field.attended, kv(l),
        layout.scores + l * T * H * n, layout.weights + l * T * H * n, n, mask[l]], T, H);
      product(layer.proj, [D, D, T, RESIDUAL, base + field.attended, REC, base + field.after, REC, input, inputStride,
        at.proj[0], at.proj[1], 0, 0], D);
      encode('norm', groups.norm, [T, base + field.after, REC, layout.normed2, D, at.ln2[0], at.ln2[1], NONE], T);
      product(layer.up, [4 * D, D, T, GELU, layout.normed2, D, base + field.mlp, REC, 0, 0,
        at.up[0], at.up[1], 0, 0], 4 * D);
      product(layer.down, [D, 4 * D, T, RESIDUAL, base + field.mlp, REC, base + field.output, REC, base + field.after, REC,
        at.down[0], at.down[1], 0, 0], D);
    }
  }

  /**
   * Computes `fresh` tokens after the current positions in one command buffer
   * and reads every new intermediate back. With no tokens, it only scores the
   * vocabulary after the last position.
   */
  private async batch(fresh: number[]): Promise<void> {
    const { nEmbd: D, nLayer: L, nHead: H, vocabSize: V } = this.config;
    const T = fresh.length;
    const start = this.positions.length;
    const n = start + T;
    // Logits are scored for the position before the new ones (it predicts the first) through the last.
    const first = Math.max(0, start - 1);
    const rows = n - first;
    const layout = this.layout(T, n, rows);
    const { field, record } = this.fields(layout, T);

    // aux: token ids, where each logit row's residual stream lives, and the token each row should rank.
    const table = T;
    const wanted = T + rows;
    const aux = new Uint32Array(T + 2 * rows);
    aux.set(fresh);
    for (let j = 0; j < rows; j++) {
      const position = first + j;
      aux[table + j] = position < start ? layout.previous : record(L - 1, position - start) + field.output;
      if (j < rows - 1) aux[wanted + j] = fresh[position + 1 - start];
    }

    const traceBytes = layout.traceEnd * 4;
    const hiddenAt = traceBytes;
    const logitsAt = hiddenAt + D * 4;
    const statsAt = logitsAt + V * 4;
    const bytes = statsAt + (rows - 1) * STAT * 4;

    const data = await this.submit(layout, bytes, (encode, groups) => {
      const { queue } = this.device;
      queue.writeBuffer(this.aux, 0, aux);
      if (start > 0) queue.writeBuffer(this.work, layout.previous * 4, this.positions[start - 1].layers[L - 1].output);
      if (T > 0) this.encodeBlocks(encode, groups, layout, T, start, this.mask);
      this.unembed(encode, groups, layout, table, wanted, rows);
    }, encoder => {
      if (traceBytes) encoder.copyBufferToBuffer(this.work, 0, this.staging, 0, traceBytes);
      encoder.copyBufferToBuffer(this.work, (layout.final + (rows - 1) * D) * 4, this.staging, hiddenAt, D * 4);
      encoder.copyBufferToBuffer(this.logitBuffer, (rows - 1) * V * 4, this.staging, logitsAt, V * 4);
      if (rows > 1) encoder.copyBufferToBuffer(this.statBuffer, 0, this.staging, statsAt, (rows - 1) * STAT * 4);
    }, n, start);

    // Views into one copy: the session keeps them as each position's trace.
    const view = (offset: number, length: number) => new Float32Array(data, offset * 4, length);
    fresh.forEach((token, i) => {
      const t = start + i;
      const position: GpuPosition = {
        token,
        tokenEmbedding: view(layout.embeddings + i * D, D),
        positionEmbedding: view(layout.embeddings + (T + i) * D, D),
        input: view(layout.embeddings + (2 * T + i) * D, D),
        layers: [],
      };
      for (let l = 0; l < L; l++) {
        const base = record(l, i);
        const rowAt = (h: number) => ((l * T + i) * H + h) * n;
        position.layers.push({
          normed: view(base + field.normed, D),
          q: view(base + field.q, D),
          k: view(base + field.q + D, D),
          v: view(base + field.q + 2 * D, D),
          scores: Array.from({ length: H }, (_, h) => view(layout.scores + rowAt(h), t + 1)),
          weights: Array.from({ length: H }, (_, h) => view(layout.weights + rowAt(h), t + 1)),
          attended: view(base + field.attended, D),
          afterAttention: view(base + field.after, D),
          mlp: view(base + field.mlp, 4 * D),
          output: view(base + field.output, D),
        });
      }
      this.positions.push(position);
    });
    const floats = new Float32Array(data, statsAt, (rows - 1) * STAT);
    const words = new Uint32Array(data, statsAt, (rows - 1) * STAT);
    for (let j = 0; j < rows - 1; j++) {
      const o = j * STAT;
      // −log2 p(token) = (log-sum-exp − logit) / ln 2, clamped at p ≈ 1 like surpriseOf.
      const bits = Math.max(0, (floats[o] + Math.log(floats[o + 1]) - floats[o + 2]) / Math.LN2);
      this.positions[first + j + 1].surprise = { bits, rank: words[o + 3] + 1 };
    }
    const last = this.positions[n - 1];
    last.logits = view(logitsAt / 4, V);
    last.hidden = view(hiddenAt / 4, D);
  }
}

type Encode = (kernel: Kernel, group: GPUBindGroup, values: number[], x: number, y?: number) => void;
