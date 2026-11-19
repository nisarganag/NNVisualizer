import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { Gpt2Session, topTokens, weightsFromBuffer, type Gpt2Manifest } from '../src/transformer/gpt2';
import { Gpt2Tokenizer } from '../src/transformer/tokenizer';
import { applySampling, softmax } from '../src/transformer/sampling';

const directory = new URL('../public/models/gpt2-v2/', import.meta.url);
const manifest: Gpt2Manifest = JSON.parse(readFileSync(new URL('manifest.json', directory), 'utf8'));
const tokenizer = new Gpt2Tokenizer(JSON.parse(readFileSync(new URL('vocab.json', directory), 'utf8')),
  readFileSync(new URL('merges.txt', directory), 'utf8'));
const bytes = new Uint8Array(manifest.totalBytes);
let offset = 0;
for (const file of manifest.files) {
  const chunk = readFileSync(new URL(file.name, directory));
  bytes.set(chunk, offset);
  offset += chunk.byteLength;
}
const weights = weightsFromBuffer(manifest, bytes.buffer);
const argmax = (values: Float32Array) => values.reduce((best, value, index) => (value > values[best] ? index : best), 0);

describe('GPT-2 tokenizer', () => {
  it('matches the reference byte-level BPE ids and round-trips any text', () => {
    expect(tokenizer.encode('Hello world')).toEqual([15496, 995]);
    expect(tokenizer.encode('The dog')).toEqual([464, 3290]);
    const text = "Data visualization empowers users to explore, isn't it? 😊 2026!";
    expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
    expect(tokenizer.tokenText(3290)).toBe(' dog');
  });
});

describe('GPT-2 small with 8-bit weights', () => {
  it('reproduces the original model’s greedy continuation token for token', { timeout: 60_000 }, () => {
    // The float32 checkpoint's greedy output for "The dog" (also the Hugging Face reference test).
    const expected = [464, 3290, 373, 1043, 287, 257, 2214, 1474, 262, 16246, 286, 2688, 290, 2688, 27262, 13, 198, 198, 464, 3290];
    const session = new Gpt2Session(weights);
    const ids = [464, 3290];
    session.setTokens(ids);
    while (ids.length < expected.length) {
      ids.push(argmax(session.logits()));
      expect(session.setTokens(ids)).toBe(ids.length - 1);
    }
    expect(ids).toEqual(expected);
    expect(tokenizer.decode(ids)).toBe('The dog was found in a field near the intersection of West and West Streets.\n\nThe dog');
  });

  it('records every intermediate value with causal attention', { timeout: 60_000 }, () => {
    const session = new Gpt2Session(weights);
    session.setTokens(tokenizer.encode('Data visualization empowers users to'));
    const { nLayer, nHead, nEmbd } = weights.config;
    expect(session.positions).toHaveLength(6);
    session.positions.forEach((position, t) => {
      expect(position.layers).toHaveLength(nLayer);
      for (const step of position.layers) {
        expect(step.q).toHaveLength(nEmbd);
        expect(step.mlp).toHaveLength(4 * nEmbd);
        expect(step.weights).toHaveLength(nHead);
        for (const row of step.weights) {
          expect(row).toHaveLength(t + 1);
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
        }
      }
    });
    const top = argmax(session.logits());
    expect(tokenizer.tokenText(top)).toBe(' visualize');
  });

  it('reuses the shared prefix when the prompt changes', { timeout: 60_000 }, () => {
    const session = new Gpt2Session(weights);
    session.setTokens(tokenizer.encode('The capital of France'));
    const before = session.positions[2].layers[5].k;
    expect(session.setTokens(tokenizer.encode('The capital of Spain'))).toBe(3);
    expect(session.positions[2].layers[5].k).toBe(before);
  });
});

describe('reading the prediction as it forms', () => {
  it('ends the logit lens at the model’s own prediction', { timeout: 60_000 }, () => {
    const session = new Gpt2Session(weights);
    const ids = tokenizer.encode('The capital of France is');
    session.setTokens(ids);
    const rows = session.lens(ids.length - 1);
    expect(rows).toHaveLength(weights.config.nLayer + 1);
    const logits = session.logits();
    expect(rows.at(-1)).toEqual(logits);
    expect(topTokens(rows.at(-1), 3)).toEqual(topTokens(logits, 3));
    expect(tokenizer.tokenText(topTokens(rows.at(-1), 1).ids[0])).toBe(' the');

    // An earlier position reads exactly like a session that stops there.
    const prefix = new Gpt2Session(weights);
    prefix.setTokens(ids.slice(0, 3));
    expect(session.lens(2).at(-1)).toEqual(prefix.logits());
    expect(() => session.lens(ids.length)).toThrow(RangeError);
  });

  it('ranks the top tokens like a full sort of the softmax', { timeout: 60_000 }, () => {
    const session = new Gpt2Session(weights);
    session.setTokens(tokenizer.encode('A neural network learns by'));
    for (const row of session.lens(4)) {
      const { ranking, scaled } = applySampling(row, { temperature: 1, mode: 'top-k', k: 3, p: 1 });
      const top = topTokens(row, 3);
      expect(top.ids).toEqual(Array.from(ranking.subarray(0, 3)));
      top.probabilities.forEach((p, i) => expect(p).toBeCloseTo(scaled[top.ids[i]], 6));
    }
  });

  it('finds a likely word far less surprising than an unlikely one', { timeout: 60_000 }, () => {
    const lastSurprise = (text: string) => {
      const session = new Gpt2Session(weights);
      session.setTokens(tokenizer.encode(text));
      return session.surprisal().at(-1);
    };
    const paris = lastSurprise('The capital of France is Paris');
    const banana = lastSurprise('The capital of France is banana');
    expect(banana.bits - paris.bits).toBeGreaterThan(10);
    expect(paris.rank).toBeLessThanOrEqual(5);
    expect(banana.rank).toBeGreaterThan(1000);

    // Bits are −log2 of the probability the model gave the token before seeing it.
    const session = new Gpt2Session(weights);
    session.setTokens(tokenizer.encode('The capital of France is'));
    const probability = softmax(session.logits())[tokenizer.encode(' Paris')[0]];
    expect(paris.bits).toBeCloseTo(-Math.log2(probability), 4);
  });

  it('scores only new tokens, matching a computation from scratch', { timeout: 60_000 }, () => {
    const ids = tokenizer.encode('Once upon a time, there was a little robto who');
    const whole = new Gpt2Session(weights);
    whole.setTokens(ids);
    const expected = whole.surprisal();
    expect(expected[0]).toBeNull();
    for (const surprise of expected.slice(1)) {
      expect(Number.isFinite(surprise.bits)).toBe(true);
      expect(surprise.bits).toBeGreaterThanOrEqual(0);
      expect(surprise.rank).toBeGreaterThanOrEqual(1);
      expect(surprise.rank).toBeLessThanOrEqual(weights.config.vocabSize);
    }

    // The worker's pattern: score the tokens, then predict the next one, after every new token.
    const growing = new Gpt2Session(weights);
    const projections = vi.spyOn(growing, 'logitsFor');
    let scores = [];
    for (let n = 1; n <= ids.length; n++) {
      expect(growing.setTokens(ids.slice(0, n))).toBe(n - 1);
      projections.mockClear();
      scores = growing.surprisal();
      growing.logits();
      // One full-vocabulary readout for the new last position; the rest is kept.
      expect(projections.mock.calls.flatMap(([hidden]) => hidden)).toHaveLength(1);
    }
    expect(scores).toEqual(expected);

    // Changing the last token keeps every earlier score and reuses the logits before it.
    const edited = [...ids.slice(0, -1), tokenizer.encode(' that')[0]];
    growing.setTokens(edited);
    projections.mockClear();
    const after = growing.surprisal();
    expect(projections).not.toHaveBeenCalled();
    expect(after.slice(0, -1)).toEqual(expected.slice(0, -1));
    const fresh = new Gpt2Session(weights);
    fresh.setTokens(edited);
    expect(after).toEqual(fresh.surprisal());
  });
});

describe('sampling', () => {
  const logits = new Float32Array([2, 1, 0, -1, -5]);
  it('applies temperature, top-k and top-p before renormalizing', () => {
    const plain = applySampling(logits, { temperature: 1, mode: 'top-k', k: 5, p: 1 });
    expect(plain.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    const cold = applySampling(logits, { temperature: .2, mode: 'top-k', k: 5, p: 1 });
    expect(cold.probabilities[0]).toBeGreaterThan(plain.probabilities[0]);
    const topK = applySampling(logits, { temperature: 1, mode: 'top-k', k: 2, p: 1 });
    expect(topK.kept).toEqual([0, 1]);
    expect(topK.probabilities[2]).toBe(0);
    const topP = applySampling(logits, { temperature: 1, mode: 'top-p', k: 5, p: .7 });
    expect(topP.kept).toEqual([0, 1]);
    expect(topP.probabilities[0] + topP.probabilities[1]).toBeCloseTo(1, 6);
  });
});
