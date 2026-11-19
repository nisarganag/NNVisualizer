/** Turns next-token logits into the probabilities a sampler actually draws from. */

export interface SamplingSettings {
  /** Divides the logits: below 1 sharpens the distribution, above 1 flattens it. */
  temperature: number;
  mode: 'top-k' | 'top-p';
  /** Keep the k most likely tokens. */
  k: number;
  /** Keep the smallest set of most likely tokens whose probabilities reach p. */
  p: number;
}

export interface SamplingResult {
  /** softmax(logits / temperature) over the whole vocabulary. */
  scaled: Float32Array;
  /** The kept tokens renormalized to sum to 1; every other token is 0. */
  probabilities: Float32Array;
  /** Kept token ids, most likely first. */
  kept: number[];
  /** All token ids ordered by probability (most likely first). */
  ranking: Uint32Array;
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  const t = Math.max(temperature, 1e-3);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t);
  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] / t - max); total += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= total;
  return out;
}

export function applySampling(logits: Float32Array, settings: SamplingSettings): SamplingResult {
  const scaled = softmax(logits, settings.temperature);
  const ranking = new Uint32Array(scaled.length);
  for (let i = 0; i < ranking.length; i++) ranking[i] = i;
  ranking.sort((a, b) => scaled[b] - scaled[a]);
  const kept: number[] = [];
  if (settings.mode === 'top-k') {
    const k = Math.max(1, Math.min(Math.round(settings.k), ranking.length));
    for (let i = 0; i < k; i++) kept.push(ranking[i]);
  } else {
    let cumulative = 0;
    for (let i = 0; i < ranking.length && (cumulative < settings.p || kept.length === 0); i++) {
      kept.push(ranking[i]);
      cumulative += scaled[ranking[i]];
    }
  }
  const probabilities = new Float32Array(scaled.length);
  let total = 0;
  for (const id of kept) total += scaled[id];
  for (const id of kept) probabilities[id] = scaled[id] / total;
  return { scaled, probabilities, kept, ranking };
}

/** Draws a token id from the kept probabilities. */
export function sample(result: SamplingResult, random = Math.random): number {
  let r = random();
  for (const id of result.kept) {
    r -= result.probabilities[id];
    if (r <= 0) return id;
  }
  return result.kept[result.kept.length - 1];
}
