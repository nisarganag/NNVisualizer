import { Gpt2Session, matrixRow, topTokens, weightsFromBuffer, type Gpt2Manifest, type Gpt2Weights, type PositionTrace, type Surprise } from './gpt2';
import { findAdapter, Gpt2GpuSession, GpuError, LENS_TOP, requestGpuDevice, type GpuReason } from './gpu';
import { MODEL_CACHE } from './config';

/** Downloads, caches and runs GPT-2 off the main thread, on the GPU (WebGPU) when it can and on the CPU otherwise. */

export type Backend = 'gpu' | 'cpu';
export type { GpuReason };

export type TransformerRequest =
  /**
   * `backend` is the processor to start on; 'gpu' falls back to the CPU when no GPU can run the model.
   * `softwareGpu` also accepts a software GPU such as SwiftShader, which is usually slower than the CPU path.
   */
  | { type: 'load'; base: string; backend: Backend; softwareGpu: boolean }
  /** Runs on `backend` (switching first if needed, which recomputes every token there). */
  | { type: 'run'; id: number; tokens: number[]; backend: Backend }
  /** The logit lens at one position of the tokens from the latest run. */
  | { type: 'lens'; id: number; position: number }
  | { type: 'weights'; layer: number; head: number };

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

export type TransformerResponse =
  | { type: 'progress'; loaded: number; total: number; cached: boolean }
  /** The weights are in; the GPU is being set up. */
  | { type: 'preparing' }
  | { type: 'ready'; nLayer: number; nHead: number; nEmbd: number; engine: EngineState }
  /** Where the model runs changed (a switch, or the GPU stopped working). */
  | { type: 'engine'; engine: EngineState }
  | { type: 'result'; id: number; tokens: number[]; trace: Trace; reused: number; ms: number; backend: Backend }
  /** Rows: the embedding, then each block. Empty when the position is not in the latest tokens. */
  | { type: 'lens'; id: number; tokens: number[]; position: number; rows: LensRow[]; ms: number; backend: Backend }
  | { type: 'weights'; layer: number; head: number; q: Float32Array; k: Float32Array; v: Float32Array }
  | { type: 'error'; message: string };

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<TransformerRequest>) => void;
  postMessage: (message: TransformerResponse, transfer?: Transferable[]) => void;
};

/** What a run, the trace and the lens need from a session; the CPU and GPU sessions both provide it. */
interface Engine {
  readonly backend: Backend;
  readonly weights: Gpt2Weights;
  readonly positions: PositionTrace[];
  readonly tokens: number[];
  setTokens(tokens: number[]): number | Promise<number>;
  surprisal(): (Surprise | null)[];
  finalHidden(): Float32Array | null;
  logits(): Float32Array | null;
  lens(position: number): LensRow[] | Promise<LensRow[]>;
}

function cpuEngine(weights: Gpt2Weights): Engine {
  const session = new Gpt2Session(weights);
  return {
    backend: 'cpu',
    weights,
    get positions() { return session.positions; },
    get tokens() { return session.tokens; },
    setTokens: tokens => session.setTokens(tokens),
    surprisal: () => session.surprisal(),
    finalHidden: () => session.finalHidden(),
    logits: () => session.logits(),
    lens: position => session.lens(position).map(logits => topTokens(logits, LENS_TOP)),
  };
}

let weights: Gpt2Weights | null = null;
let engine: Engine | null = null;
let gpu: Gpt2GpuSession | null = null;
/** Why the GPU can't be used; null while it can (or hasn't been tried). */
let gpuReason: GpuReason | null = null;
/** Whether a GPU adapter exists, before the GPU has been set up. */
let gpuSeen = false;
/** The latest run, so a lost GPU can be replaced without the page asking again. */
let latest: { id: number; tokens: number[] } | null = null;

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
      gpuReason = null;
    } catch (error) {
      device?.destroy();
      gpu = null;
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
  const next = backend === 'gpu' && gpu ? gpu : engine?.backend === 'cpu' ? engine : cpuEngine(weights);
  if (next === engine) return;
  if (gpu && next !== gpu) gpu.reset();
  if (next === gpu) gpu.reset();
  engine = next;
  announce();
}

/** Gives up on the GPU and continues on the CPU. */
function dropGpu(reason: GpuReason): void {
  gpu?.destroy();
  gpu = null;
  gpuReason = reason;
  if (engine?.backend !== 'cpu') engine = cpuEngine(weights);
  announce();
}

/** When the GPU goes away between requests, recompute the latest tokens on the CPU so the page stays current. */
async function replaceLostGpu(lost: Gpt2GpuSession): Promise<void> {
  if (gpu !== lost) return;
  const wasRunning = engine === lost;
  dropGpu('lost');
  if (wasRunning && latest) await run(latest.id, latest.tokens, 'cpu');
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

function pack(current: Engine): Trace {
  const { nLayer, nHead, nEmbd } = current.weights.config;
  const hidden = 4 * nEmbd;
  const positions = current.positions;
  const n = positions.length;
  const embeddings = new Float32Array(3 * n * nEmbd);
  const make = () => new Float32Array(nLayer * n * nEmbd);
  const q = make(), k = make(), v = make(), attended = make(), output = make();
  const mlp = new Float32Array(nLayer * n * hidden);
  const scores = new Float32Array(nLayer * nHead * n * n).fill(Number.NaN);
  const weights = new Float32Array(nLayer * nHead * n * n);
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
        weights.set(step.weights[h], row);
      }
    });
  });
  // Scored once per position and kept by the session, so only new tokens cost anything.
  const surprisal = new Float32Array(n).fill(Number.NaN);
  const ranks = new Uint32Array(n);
  current.surprisal().forEach((surprise, t) => {
    if (!surprise) return;
    surprisal[t] = surprise.bits;
    ranks[t] = surprise.rank;
  });
  return {
    n, embeddings, q, k, v, attended, output, mlp, scores, weights,
    finalHidden: current.finalHidden() ?? new Float32Array(nEmbd),
    logits: current.logits() ?? new Float32Array(current.weights.config.vocabSize),
    surprisal, ranks,
  };
}

async function run(id: number, tokens: number[], backend: Backend): Promise<void> {
  if (!weights) throw new Error('Load the model first.');
  await select(backend);
  const started = performance.now();
  let reused: number;
  try {
    reused = await engine.setTokens(tokens);
  } catch (error) {
    if (!(error instanceof GpuError)) throw error;
    // The GPU failed partway: finish this run on the CPU.
    dropGpu(error.reason);
    reused = await engine.setTokens(tokens);
  }
  latest = { id, tokens };
  const trace = pack(engine);
  scope.postMessage({ type: 'result', id, tokens, trace, reused, ms: performance.now() - started, backend: engine.backend },
    [trace.embeddings, trace.q, trace.k, trace.v, trace.attended, trace.output, trace.mlp, trace.scores, trace.weights, trace.finalHidden, trace.logits,
      trace.surprisal, trace.ranks].map(array => array.buffer as ArrayBuffer));
}

/** The top three next tokens after the embedding and after each block. */
async function lens(id: number, position: number): Promise<void> {
  const started = performance.now();
  const known = engine !== null && position >= 0 && position < engine.positions.length;
  let rows: LensRow[] = [];
  if (known) {
    try {
      rows = await engine.lens(position);
    } catch (error) {
      if (!(error instanceof GpuError) || !latest) throw error;
      // Continue on the CPU, which first recomputes the tokens the page shows.
      dropGpu(error.reason);
      await run(latest.id, latest.tokens, 'cpu');
      rows = await engine.lens(position);
    }
  }
  scope.postMessage({ type: 'lens', id, tokens: engine?.tokens ?? [], position, rows, ms: performance.now() - started, backend: engine?.backend ?? 'cpu' });
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
    enqueue(() => run(data.id, data.tokens, data.backend));
  } else if (data.type === 'lens') {
    enqueue(() => lens(data.id, data.position));
  } else if (data.type === 'weights') {
    // Reads only the weights, so it doesn't wait for a run.
    try {
      headWeights(data.layer, data.head);
    } catch (error) {
      scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Something went wrong.' });
    }
  }
};
