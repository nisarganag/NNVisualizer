import { checkHeadMask, Gpt2Session, headsOff, matrixRow, surpriseOf, topTokens, weightsFromBuffer, type Gpt2Manifest, type Gpt2Weights,
  type HeadMask, type PositionTrace, type Surprise } from './gpt2';
import { findAdapter, Gpt2GpuSession, GpuError, LENS_TOP, requestGpuDevice, type GpuReason, type Probe } from './gpu';
import { MODEL_CACHE } from './config';

/**
 * Downloads, caches and runs GPT-2 off the main thread, on the GPU (WebGPU) when it can and on the CPU otherwise.
 * Each processor keeps a session with every head on. Once heads are switched off, a second session runs them,
 * and the first one's prediction comes along for comparison; both reuse the tokens they already hold.
 */

export type Backend = 'gpu' | 'cpu';
export type { GpuReason };

export type TransformerRequest =
  /**
   * `backend` is the processor to start on; 'gpu' falls back to the CPU when no GPU can run the model.
   * `softwareGpu` also accepts a software GPU such as SwiftShader, which is usually slower than the CPU path.
   */
  | { type: 'load'; base: string; backend: Backend; softwareGpu: boolean }
  /**
   * Runs on `backend` (switching first if needed, which recomputes every token there) with `mask`'s heads switched
   * off: one bitmask per block, bit h for head h (see HeadMask in gpt2.ts). Leaving it out switches every head on.
   */
  | { type: 'run'; id: number; tokens: number[]; backend: Backend; mask?: number[] }
  /** The logit lens at one position of the tokens from the latest run, with its heads switched off. */
  | { type: 'lens'; id: number; position: number }
  | { type: 'weights'; layer: number; head: number }
  /**
   * Which heads matter: switches off each head of `blocks` on its own, with every other head on, and measures the
   * likeliest token after the last of `tokens`. Replies come a head at a time; requests in between still run.
   */
  | { type: 'importance'; id: number; tokens: number[]; blocks: number[] }
  /** Ends the importance test in progress. */
  | { type: 'stop-importance' };

/** One stage of the logit lens: the likeliest next tokens if the model stopped there. */
export interface LensRow {
  /** Most likely first. */
  ids: number[];
  probabilities: number[];
}

/** Where the model runs, and whether the page can offer the GPU. */
export interface EngineState {
  backend: Backend;
  /** A GPU can run the model on this page. */
  gpu: boolean;
  /** Why the GPU can't run the model, when it can't. */
  reason: GpuReason | null;
}

/** Every intermediate value for the current tokens, packed into flat arrays. */
export interface Trace {
  n: number;
  /** [3][n][768]: token embedding, position embedding, their sum. */
  embeddings: Float32Array;
  /** [layer][n][768] each. */
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  /** Concatenated head outputs before the output projection. */
  attended: Float32Array;
  /** Residual stream leaving each block. */
  output: Float32Array;
  /** [layer][n][3072] after GELU. */
  mlp: Float32Array;
  /** [layer][head][n][n]; masked (future) entries are NaN. */
  scores: Float32Array;
  weights: Float32Array;
  /** Final layer-normed vector at the last position. */
  finalHidden: Float32Array;
  logits: Float32Array;
  /** [n] How unexpected each token was: −log2 p(token | earlier tokens). NaN for the first token, which has no context. */
  surprisal: Float32Array;
  /** [n] Each token's rank among all possible next tokens (1 = the model's top guess); 0 for the first token. */
  ranks: Uint32Array;
}

/** The same tokens with every head on, sent beside a trace that has heads switched off. */
export interface Baseline {
  logits: Float32Array;
  /** [n] As in Trace. */
  surprisal: Float32Array;
  ranks: Uint32Array;
}

export type TransformerResponse =
  | { type: 'progress'; loaded: number; total: number; cached: boolean }
  /** The weights are in; the GPU is being set up. */
  | { type: 'preparing' }
  | { type: 'ready'; nLayer: number; nHead: number; nEmbd: number; engine: EngineState }
  /** Where the model runs changed (a switch, or the GPU stopped working). */
  | { type: 'engine'; engine: EngineState }
  /** `mask` is the trace's switched-off heads; `baseline` is null when every head is on. */
  | { type: 'result'; id: number; tokens: number[]; mask: number[]; trace: Trace; baseline: Baseline | null; reused: number; ms: number; backend: Backend }
  /** Rows: the embedding, then each block. Empty when the position is not in the latest tokens. */
  | { type: 'lens'; id: number; tokens: number[]; mask: number[]; position: number; rows: LensRow[]; ms: number; backend: Backend }
  | { type: 'weights'; layer: number; head: number; q: Float32Array; k: Float32Array; v: Float32Array }
  /**
   * The importance test so far. `target` is the likeliest token after the last of `tokens` with every head on, with
   * probability `baseline`. With head h of block l off alone, it has probability values[l * nHead + h] (NaN until
   * tested) and rank ranks[l * nHead + h]. `stopped` means the test ended before every head was tested.
   */
  | { type: 'importance'; id: number; tokens: number[]; target: number; baseline: number; values: Float32Array; ranks: Uint32Array;
    done: number; total: number; ms: number; backend: Backend; stopped: boolean }
  | { type: 'error'; message: string };

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<TransformerRequest>) => void;
  postMessage: (message: TransformerResponse, transfer?: Transferable[]) => void;
};

/** What a run, the trace and the lens need from one session; the CPU and GPU sessions both provide it. */
interface Session {
  readonly positions: PositionTrace[];
  readonly tokens: number[];
  readonly headMask: number[];
  surprisal(): (Surprise | null)[];
  finalHidden(): Float32Array | null;
  logits(): Float32Array | null;
  lens(position: number): LensRow[] | Promise<LensRow[]>;
}

/** One processor's sessions. */
interface Engine {
  readonly backend: Backend;
  readonly weights: Gpt2Weights;
  /**
   * Runs `tokens` with `mask`'s heads switched off. Returns the session to show, how many of its positions were
   * reused whole, and, when any head is off, the session with every head on for the same tokens.
   */
  run(tokens: number[], mask: number[]): Promise<{ shown: Session; reused: number; baseline: Session | null }>;
  /** For each mask, how likely `target` (or the likeliest token, when null) is after `tokens` with its heads off, leaving the runs as they are. */
  probe(tokens: number[], masks: number[][], target: number | null): Promise<Probe[]>;
  /** Forgets every position, so the next run computes from scratch. */
  reset(): void;
}

function cpuSession(session: Gpt2Session): Session {
  return {
    get positions() { return session.positions; },
    get tokens() { return session.tokens; },
    get headMask() { return session.headMask; },
    surprisal: () => session.surprisal(),
    finalHidden: () => session.finalHidden(),
    logits: () => session.logits(),
    lens: position => session.lens(position).map(logits => topTokens(logits, LENS_TOP)),
  };
}

/** Roughly how many block passes `session` needs to hold `tokens` with `mask`: new positions take every block, kept ones those from the first changed block. */
function cost(session: Gpt2Session, tokens: number[], mask: number[]): number {
  const { nLayer } = session.weights.config;
  const current = session.headMask;
  let from = 0;
  while (from < nLayer && current[from] === mask[from]) from++;
  let shared = 0;
  while (shared < tokens.length && shared < session.positions.length && tokens[shared] === session.positions[shared].token) shared++;
  return (tokens.length - shared) * nLayer + shared * (nLayer - from);
}

function cpuEngine(weights: Gpt2Weights): Engine {
  const on = new Gpt2Session(weights);
  let off: Gpt2Session | null = null;
  return {
    backend: 'cpu',
    weights,
    async run(tokens, mask) {
      const reused = on.setTokens(tokens);
      if (!headsOff(mask)) return { shown: cpuSession(on), reused, baseline: null };
      // Start from whichever needs less work: the last session with heads off, or a copy of the one with every
      // head on, which only recomputes from the first block with a head off.
      if (!off || cost(on, tokens, mask) < cost(off, tokens, mask)) off = on.fork();
      return { shown: cpuSession(off), reused: off.setTokens(tokens, mask), baseline: cpuSession(on) };
    },
    async probe(tokens, masks, target) {
      return masks.map(mask => {
        // A copy of the session with every head on, recomputed from the first block with a head off.
        const session = on.fork();
        session.setTokens(tokens, mask);
        const logits = session.logits();
        const top = topTokens(logits, 1).ids[0];
        const { bits, rank } = surpriseOf(logits, target ?? top);
        return { probability: 2 ** -bits, rank, top };
      });
    },
    reset() {
      on.setTokens([]);
      off = null;
    },
  };
}

function gpuEngineFor(on: Gpt2GpuSession): Engine {
  let off: Gpt2GpuSession | null = null;
  /** A third session whose cache the probes may overwrite. */
  let probing: Gpt2GpuSession | null = null;
  return {
    backend: 'gpu',
    weights: on.weights,
    async run(tokens, mask) {
      const reused = await on.setTokens(tokens);
      if (!headsOff(mask)) return { shown: on, reused, baseline: null };
      off ??= on.sibling();
      return { shown: off, reused: await off.setTokens(tokens, mask), baseline: on };
    },
    probe(tokens, masks, target) {
      probing ??= on.sibling();
      // Blocks before the first switched-off head match the run with every head on: when that run holds these
      // tokens, the probes start from its stream there.
      const { nEmbd } = on.weights.config;
      const from = Math.min(...masks.map(mask => Math.max(0, mask.findIndex(Boolean))));
      const held = on.positions;
      if (from > 0 && held.length >= tokens.length && tokens.every((token, t) => held[t].token === token)) {
        const stream = new Float32Array(tokens.length * nEmbd);
        tokens.forEach((_token, t) => stream.set(held[t].layers[from - 1].output, t * nEmbd));
        return probing.probe(tokens, masks, target, from, stream);
      }
      return probing.probe(tokens, masks, target);
    },
    reset() {
      on.reset();
      off?.reset();
    },
  };
}

let weights: Gpt2Weights | null = null;
let engine: Engine | null = null;
let gpu: Gpt2GpuSession | null = null;
/** The engine around `gpu`, while there is one. */
let gpuEngine: Engine | null = null;
/** The session the latest run showed, which the lens reads. */
let shown: Session | null = null;
/** Why the GPU can't be used; null while it can (or hasn't been tried). */
let gpuReason: GpuReason | null = null;
/** Whether a GPU adapter exists, before the GPU has been set up. */
let gpuSeen = false;
/** The latest run, so a lost GPU can be replaced without the page asking again. */
let latest: { id: number; tokens: number[]; mask: number[] } | null = null;

const engineState = (): EngineState => ({
  backend: engine?.backend ?? 'cpu',
  gpu: gpu !== null || (gpuSeen && gpuReason === null),
  reason: gpuReason,
});
const announce = () => scope.postMessage({ type: 'engine', engine: engineState() });

// Requests run one at a time: a GPU run awaits its results, and nothing else may touch the session meanwhile.
let queue: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void> | void): void {
  queue = queue.then(task).catch(error => {
    scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Something went wrong.' });
  });
}

/** Whether a software GPU (such as SwiftShader) may run the model. */
let softwareGpu = false;

/**
 * Sets up the GPU once: a device, the uploaded weights, compiled kernels, and
 * a one-token check against the CPU. Any failure leaves the CPU in charge.
 */
let gpuSetup: Promise<void> | null = null;
function startGpu(): Promise<void> {
  gpuSetup ??= (async () => {
    let device: GPUDevice | null = null;
    try {
      device = await requestGpuDevice(softwareGpu);
      const creating = Gpt2GpuSession.create(device, weights, lost => enqueue(() => replaceLostGpu(lost)));
      creating.catch(() => undefined);
      // The CPU's answer for the check, computed while the GPU compiles its kernels.
      const expected = Gpt2GpuSession.reference(weights);
      const session = await creating;
      await session.check(expected);
      gpu = session;
      gpuEngine = gpuEngineFor(session);
      gpuReason = null;
    } catch (error) {
      device?.destroy();
      gpu = null;
      gpuEngine = null;
      gpuReason = error instanceof GpuError ? error.reason : 'device';
    }
  })();
  return gpuSetup;
}

/** Looks for a GPU without setting it up, so a page that starts on the CPU can still offer it. */
async function probeGpu(): Promise<void> {
  try {
    await findAdapter(softwareGpu);
    gpuSeen = true;
  } catch (error) {
    gpuReason = error instanceof GpuError ? error.reason : 'no-adapter';
  }
}

/** Makes `backend` the engine, starting from no tokens there (a fair timing for comparisons). */
async function select(backend: Backend): Promise<void> {
  if (backend === 'gpu' && !gpu && gpuReason === null) await startGpu();
  const next = backend === 'gpu' && gpuEngine ? gpuEngine : engine?.backend === 'cpu' ? engine : cpuEngine(weights);
  if (next === engine) return;
  gpuEngine?.reset();
  engine = next;
  announce();
}

/** Gives up on the GPU and continues on the CPU. */
function dropGpu(reason: GpuReason): void {
  gpu?.destroy();
  gpu = null;
  gpuEngine = null;
  gpuReason = reason;
  if (engine?.backend !== 'cpu') engine = cpuEngine(weights);
  announce();
}

/** When the GPU goes away between requests, recompute the latest tokens on the CPU so the page stays current. */
async function replaceLostGpu(lost: Gpt2GpuSession): Promise<void> {
  if (gpu !== lost) return;
  const wasRunning = engine === gpuEngine;
  dropGpu('lost');
  if (wasRunning && latest) await run(latest.id, latest.tokens, latest.mask, 'cpu');
}

async function load(base: string, backend: Backend, software: boolean): Promise<void> {
  softwareGpu = software;
  const manifest: Gpt2Manifest = await (await fetch(`${base}manifest.json`)).json();
  let cache: Cache | null = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    // Older model versions are no longer needed.
    for (const name of await caches.keys()) if (name.startsWith('nnv-gpt2-') && name !== MODEL_CACHE) await caches.delete(name);
  } catch {
    cache = null; // Cache Storage needs a secure context; downloads still work without it.
  }
  const bytes = new Uint8Array(manifest.totalBytes);
  let loaded = 0;
  let reported = 0;
  for (const file of manifest.files) {
    const url = `${base}${file.name}`;
    let response = cache ? await cache.match(url) : undefined;
    const fromCache = Boolean(response);
    if (!response) {
      response = await fetch(url);
      if (!response.ok) throw new Error(`Could not download ${file.name} (${response.status}).`);
    }
    const reader = response.body!.getReader();
    const start = loaded;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, loaded);
      loaded += value.byteLength;
      // Report about every megabyte rather than every network chunk.
      if (loaded - reported > 1 << 20 || loaded === manifest.totalBytes) {
        reported = loaded;
        scope.postMessage({ type: 'progress', loaded, total: manifest.totalBytes, cached: fromCache });
      }
    }
    if (loaded - start !== file.bytes) throw new Error(`${file.name} is incomplete. Reload to try again.`);
    if (cache && !fromCache) await cache.put(url, new Response(bytes.slice(start, loaded))).catch(() => undefined);
  }
  weights = weightsFromBuffer(manifest, bytes.buffer);
  engine = null;
  if (backend === 'gpu') {
    scope.postMessage({ type: 'preparing' });
    await startGpu();
    gpuSeen = gpu !== null;
  } else {
    await probeGpu();
  }
  await select(backend);
  const { nLayer, nHead, nEmbd } = manifest.config;
  scope.postMessage({ type: 'ready', nLayer, nHead, nEmbd, engine: engineState() });
}

/** Bits of surprise and ranks per token, kept by the session so only new tokens cost anything. */
function surprises(session: Session): { surprisal: Float32Array; ranks: Uint32Array } {
  const n = session.positions.length;
  const surprisal = new Float32Array(n).fill(Number.NaN);
  const ranks = new Uint32Array(n);
  session.surprisal().forEach((surprise, t) => {
    if (!surprise) return;
    surprisal[t] = surprise.bits;
    ranks[t] = surprise.rank;
  });
  return { surprisal, ranks };
}

function pack(current: Session): Trace {
  const { nLayer, nHead, nEmbd, vocabSize } = weights.config;
  const hidden = 4 * nEmbd;
  const positions = current.positions;
  const n = positions.length;
  const embeddings = new Float32Array(3 * n * nEmbd);
  const make = () => new Float32Array(nLayer * n * nEmbd);
  const q = make(), k = make(), v = make(), attended = make(), output = make();
  const mlp = new Float32Array(nLayer * n * hidden);
  const scores = new Float32Array(nLayer * nHead * n * n).fill(Number.NaN);
  const attention = new Float32Array(nLayer * nHead * n * n);
  positions.forEach((position, t) => {
    embeddings.set(position.tokenEmbedding, t * nEmbd);
    embeddings.set(position.positionEmbedding, (n + t) * nEmbd);
    embeddings.set(position.input, (2 * n + t) * nEmbd);
    position.layers.forEach((step, l) => {
      const at = (l * n + t) * nEmbd;
      q.set(step.q, at);
      k.set(step.k, at);
      v.set(step.v, at);
      attended.set(step.attended, at);
      output.set(step.output, at);
      mlp.set(step.mlp, (l * n + t) * hidden);
      for (let h = 0; h < nHead; h++) {
        const row = ((l * nHead + h) * n + t) * n;
        scores.set(step.scores[h], row);
        attention.set(step.weights[h], row);
      }
    });
  });
  return {
    n, embeddings, q, k, v, attended, output, mlp, scores, weights: attention,
    finalHidden: current.finalHidden() ?? new Float32Array(nEmbd),
    logits: current.logits() ?? new Float32Array(vocabSize),
    ...surprises(current),
  };
}

async function run(id: number, tokens: number[], mask: HeadMask | null, backend: Backend): Promise<void> {
  if (!weights) throw new Error('Load the model first.');
  const bits = checkHeadMask(mask, weights.config);
  await select(backend);
  const started = performance.now();
  let outcome: Awaited<ReturnType<Engine['run']>>;
  try {
    outcome = await engine.run(tokens, bits);
  } catch (error) {
    if (!(error instanceof GpuError)) throw error;
    // The GPU failed partway: finish this run on the CPU.
    dropGpu(error.reason);
    outcome = await engine.run(tokens, bits);
  }
  latest = { id, tokens, mask: bits };
  shown = outcome.shown;
  const trace = pack(shown);
  const baseline: Baseline | null = outcome.baseline && { logits: outcome.baseline.logits(), ...surprises(outcome.baseline) };
  const arrays = [trace.embeddings, trace.q, trace.k, trace.v, trace.attended, trace.output, trace.mlp, trace.scores, trace.weights, trace.finalHidden,
    trace.logits, trace.surprisal, trace.ranks, ...(baseline ? [baseline.logits, baseline.surprisal, baseline.ranks] : [])];
  scope.postMessage({ type: 'result', id, tokens, mask: bits, trace, baseline, reused: outcome.reused, ms: performance.now() - started, backend: engine.backend },
    arrays.map(array => array.buffer as ArrayBuffer));
}

/** The top three next tokens after the embedding and after each block. */
async function lens(id: number, position: number): Promise<void> {
  const started = performance.now();
  const known = shown !== null && position >= 0 && position < shown.positions.length;
  let rows: LensRow[] = [];
  if (known) {
    try {
      rows = await shown.lens(position);
    } catch (error) {
      if (!(error instanceof GpuError) || !latest) throw error;
      // Continue on the CPU, which first recomputes the tokens the page shows.
      dropGpu(error.reason);
      await run(latest.id, latest.tokens, latest.mask, 'cpu');
      rows = await shown.lens(position);
    }
  }
  scope.postMessage({ type: 'lens', id, tokens: shown?.tokens ?? [], mask: shown?.headMask ?? checkHeadMask(null, weights.config), position, rows,
    ms: performance.now() - started, backend: engine?.backend ?? 'cpu' });
}

/** The importance test in progress; a stop request or a newer test ends it. */
let test: { stopped: boolean } | null = null;

/**
 * Switches off each head of `blocks` alone and measures the likeliest token after `tokens` (with every head on)
 * each time. Each head is its own task in the queue, so runs and lenses asked for meanwhile wait for one head at most.
 */
function importance(id: number, tokens: number[], blocks: number[]): void {
  if (test) test.stopped = true;
  const job: { stopped: boolean } = { stopped: false };
  test = job;
  const { nLayer, nHead } = weights.config;
  const none = checkHeadMask(null, weights.config);
  const heads = blocks.flatMap(l => Array.from({ length: nHead }, (_, h) => [l, h]));
  const values = new Float32Array(nLayer * nHead).fill(Number.NaN);
  const ranks = new Uint32Array(nLayer * nHead);
  const started = performance.now();
  let target = -1;
  let baseline = 0;
  let done = 0;
  const report = (stopped: boolean) => scope.postMessage({ type: 'importance', id, tokens, target, baseline, values: values.slice(),
    ranks: ranks.slice(), done, total: heads.length, ms: performance.now() - started, backend: engine?.backend ?? 'cpu', stopped });
  const step = async () => {
    if (job.stopped) return;
    if (!engine || !tokens.length) {
      report(true);
      return;
    }
    try {
      if (target < 0) {
        ({ top: target, probability: baseline } = (await engine.probe(tokens, [none], null))[0]);
      } else {
        // The GPU tests a block's heads in one go (half a block for long prompts); the CPU one head at a time, so other
        // requests never wait long.
        const size = engine.backend === 'cpu' ? 1 : tokens.length > 24 ? nHead / 2 : nHead;
        const chunk = heads.slice(done, done + Math.min(size, nHead - heads[done][1]));
        const masks = chunk.map(([l, h]) => {
          const mask = none.slice();
          mask[l] = 1 << h;
          return mask;
        });
        const results = await engine.probe(tokens, masks, target);
        chunk.forEach(([l, h], i) => {
          values[l * nHead + h] = results[i].probability;
          ranks[l * nHead + h] = results[i].rank;
        });
        done += chunk.length;
      }
    } catch (error) {
      if (!(error instanceof GpuError)) throw error;
      // The test ends; the page continues on the CPU, which first recomputes the tokens it shows.
      job.stopped = true;
      report(true);
      dropGpu(error.reason);
      if (latest) await run(latest.id, latest.tokens, latest.mask, 'cpu');
      return;
    }
    if (job.stopped) return;
    report(false);
    if (done < heads.length) enqueue(step);
    else if (test === job) test = null;
  };
  enqueue(step);
}

function headWeights(layer: number, head: number) {
  if (!weights) return;
  const { nEmbd, nHead } = weights.config;
  const size = nEmbd / nHead;
  const matrix = weights.layers[layer].attention.weight;
  // Rows of the stacked projection: Q, then K, then V, each split into heads.
  const slice = (part: number) => {
    const out = new Float32Array(size * nEmbd);
    const row = new Float32Array(nEmbd);
    for (let d = 0; d < size; d++) out.set(matrixRow(matrix, part * nEmbd + head * size + d, row), d * nEmbd);
    return out;
  };
  const q = slice(0), k = slice(1), v = slice(2);
  scope.postMessage({ type: 'weights', layer, head, q, k, v }, [q.buffer, k.buffer, v.buffer]);
}

scope.onmessage = ({ data }) => {
  if (data.type === 'load') {
    enqueue(() => load(data.base, data.backend, data.softwareGpu));
  } else if (data.type === 'run') {
    enqueue(() => run(data.id, data.tokens, data.mask ?? null, data.backend));
  } else if (data.type === 'lens') {
    enqueue(() => lens(data.id, data.position));
  } else if (data.type === 'importance') {
    importance(data.id, data.tokens, data.blocks);
  } else if (data.type === 'stop-importance') {
    if (test) test.stopped = true;
  } else if (data.type === 'weights') {
    // Reads only the weights, so it doesn't wait for a run.
    try {
      headWeights(data.layer, data.head);
    } catch (error) {
      scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Something went wrong.' });
    }
  }
};
