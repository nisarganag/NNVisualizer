import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Gpt2Session, weightsFromBuffer, type Gpt2Manifest } from '../src/transformer/gpt2';
import { Gpt2Tokenizer } from '../src/transformer/tokenizer';
import { applySampling } from '../src/transformer/sampling';

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
