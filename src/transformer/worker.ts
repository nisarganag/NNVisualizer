import { Gpt2Session, matrixRow, topTokens, weightsFromBuffer, type Gpt2Manifest } from './gpt2';
import { MODEL_CACHE } from './config';

/** Downloads, caches and runs GPT-2 off the main thread. */

export type TransformerRequest =
  | { type: 'load'; base: string }
  | { type: 'run'; id: number; tokens: number[] }
  /** The logit lens at one position of the tokens from the latest run. */
  | { type: 'lens'; id: number; position: number }
  | { type: 'weights'; layer: number; head: number };

/** One stage of the logit lens: the likeliest next tokens if the model stopped there. */
export interface LensRow {
  /** Most likely first. */
  ids: number[];
  probabilities: number[];
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
  | { type: 'ready'; nLayer: number; nHead: number; nEmbd: number }
  | { type: 'result'; id: number; tokens: number[]; trace: Trace; reused: number; ms: number }
  /** Rows: the embedding, then each block. Empty when the position is not in the latest tokens. */
  | { type: 'lens'; id: number; tokens: number[]; position: number; rows: LensRow[]; ms: number }
  | { type: 'weights'; layer: number; head: number; q: Float32Array; k: Float32Array; v: Float32Array }
  | { type: 'error'; message: string };

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<TransformerRequest>) => void;
  postMessage: (message: TransformerResponse, transfer?: Transferable[]) => void;
};
let session: Gpt2Session | null = null;

async function load(base: string): Promise<void> {
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
  session = new Gpt2Session(weightsFromBuffer(manifest, bytes.buffer));
  const { nLayer, nHead, nEmbd } = manifest.config;
  scope.postMessage({ type: 'ready', nLayer, nHead, nEmbd });
}

function pack(current: Gpt2Session): Trace {
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

/** The top three next tokens after the embedding and after each block. */
function lens(id: number, position: number) {
  const started = performance.now();
  const known = session !== null && position >= 0 && position < session.positions.length;
  const rows = known ? session.lens(position).map(logits => topTokens(logits, 3)) : [];
  scope.postMessage({ type: 'lens', id, tokens: session?.tokens ?? [], position, rows, ms: performance.now() - started });
}

function headWeights(layer: number, head: number) {
  if (!session) return;
  const { nEmbd, nHead } = session.weights.config;
  const size = nEmbd / nHead;
  const matrix = session.weights.layers[layer].attention.weight;
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
  try {
    if (data.type === 'load') {
      load(data.base).catch(error => scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'The model could not load.' }));
    } else if (data.type === 'run') {
      if (!session) throw new Error('Load the model first.');
      const started = performance.now();
      const reused = session.setTokens(data.tokens);
      const trace = pack(session);
      scope.postMessage({ type: 'result', id: data.id, tokens: data.tokens, trace, reused, ms: performance.now() - started },
        [trace.embeddings, trace.q, trace.k, trace.v, trace.attended, trace.output, trace.mlp, trace.scores, trace.weights, trace.finalHidden, trace.logits,
          trace.surprisal, trace.ranks].map(array => array.buffer as ArrayBuffer));
    } else if (data.type === 'lens') {
      lens(data.id, data.position);
    } else if (data.type === 'weights') {
      headWeights(data.layer, data.head);
    }
  } catch (error) {
    scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Something went wrong.' });
  }
};
