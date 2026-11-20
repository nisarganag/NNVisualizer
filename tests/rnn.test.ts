// @vitest-environment node
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  CharLstmRun, GATE_ORDER, bitsPerCharacter, encodeText, lstmCell, parseCharLstm, readout, softmaxInto, type CharLstmMeta,
} from '../src/rnn/model';

function readModelFile(name: string): ArrayBuffer {
  const bytes = readFileSync(new URL(`../public/models/${name}`, import.meta.url));
  const buffer = new ArrayBuffer(bytes.length);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

const meta: CharLstmMeta = JSON.parse(new TextDecoder().decode(readModelFile('char-lstm-v1.json')));
const weights = readModelFile('char-lstm-v1.bin');
const model = parseCharLstm(meta, weights);
const H = model.hiddenSize;
const V = model.inputSize;

/**
 * Held-out text: from the last 5% of The Wonderful Wizard of Oz, which training never
 * saw (scripts/train-char-lstm.mjs holds out the end of every book), normalized the
 * same way as the training text.
 */
const HELD_OUT = 'Dorothy ran down and helped the Scarecrow to his feet, and the Lion came up to her, feeling rather bruised and sore, and said, "It is useless to fight people with shooting heads; no one can withstand them."\n'
  + '"What can we do, then?" she asked.\n'
  + '"Call the Winged Monkeys," suggested the Tin Woodman. "You have still the right to command them once more."\n'
  + '"Very well," she answered, and putting on the Golden Cap she uttered the magic words. The Monkeys were as prompt as ever, and in a few moments the entire band stood before her.\n'
  + '"What are your commands?" inquired the King of the Monkeys, bowing low.\n'
  + '"Carry us over the hill to the country of the Quadlings," answered the girl.\n'
  + '"It shall be done," said the King, and at once the Winged Monkeys caught the four travelers and Toto up in their arms and flew away with them. As they passed over the hill the Hammer-Heads yelled with vexation, and shot their heads high in the air, but they could not reach the Winged Monkeys, which carried Dorothy and her comrades safely over the hill and set them down in the beautiful country of the Quadlings.\n'
  + '"This is the last time you can summon us," said the leader to Dorothy; "so good-bye and good luck to you."\n'
  + '"Good-bye, and thank you very much," returned the girl; and the Monkeys rose into the air and were out of sight in a twinkling.';

const encode = (text: string) => {
  const encoded = encodeText(model, text);
  expect(encoded.skipped).toEqual([]);
  return encoded.indices;
};

/**
 * log p_t(target) when the one-hot input at step s is scaled by (1 + epsilon): the
 * continuous relaxation in which "how much of character x_s is present" can vary.
 */
function relaxedLogProbability(indices: number[], t: number, target: number, s: number, epsilon: number): number {
  const G = 4 * H;
  let h = new Float32Array(H);
  let c = new Float32Array(H);
  let hNext = new Float32Array(H);
  let cNext = new Float32Array(H);
  const gates = new Float32Array(G);
  const input = new Float32Array(G);
  for (let step = 0; step <= t; step++) {
    const row = indices[step] * G;
    const amount = step === s ? 1 + epsilon : 1;
    for (let j = 0; j < G; j++) input[j] = model.inputWeights[row + j] * amount;
    lstmCell(model, input, 0, h, 0, c, 0, gates, 0, cNext, 0, hNext, 0);
    [h, hNext] = [hNext, h];
    [c, cNext] = [cNext, c];
  }
  const logits = new Float32Array(V);
  readout(model, h, 0, logits, 0);
  return logits[target] - softmaxInto(logits, 0, V, new Float32Array(V), 0);
}

describe('character LSTM weights', () => {
  it('parse with the documented layout and reject the wrong sizes', () => {
    expect(model.hiddenSize).toBe(256);
    expect(model.vocabulary.join('')).toBe(meta.vocabulary);
    expect(V).toBeGreaterThanOrEqual(60);
    expect(V).toBeLessThanOrEqual(80);
    expect(meta.gateOrder).toEqual([...GATE_ORDER]);
    expect(meta.parameters).toBe(4 * H * (V + H + 1) + V * (H + 1));
    expect(weights.byteLength).toBe(meta.parameters * 4);
    expect(model.inputWeights).toHaveLength(V * 4 * H);
    expect(model.recurrentWeights).toHaveLength(4 * H * H);
    // The forget gate's bias started at 1; training keeps it mostly positive, so cells remember by default.
    const forgetBias = model.bias.subarray(H, 2 * H);
    expect(forgetBias.reduce((a, b) => a + b, 0) / H).toBeGreaterThan(0);
    for (const title of ['Alice’s Adventures in Wonderland', 'The Wonderful Wizard of Oz', 'Peter Pan']) {
      expect(meta.corpus.map(book => book.title)).toContain(title);
    }

    expect(() => parseCharLstm(meta, weights.slice(0, weights.byteLength - 4))).toThrow(/expected \d+ bytes/);
    expect(() => parseCharLstm(meta, new ArrayBuffer(weights.byteLength + 4))).toThrow(/expected \d+ bytes/);
    expect(() => parseCharLstm({ ...meta, hiddenSize: 64 }, weights)).toThrow(/Invalid LSTM metadata/);
    expect(() => parseCharLstm({ ...meta, vocabulary: meta.vocabulary.slice(1) }, weights)).toThrow(/input size/);
    expect(() => parseCharLstm({ ...meta, gateOrder: ['forget', 'input', 'cell', 'output'] }, weights)).toThrow(/gate order/);
    const corrupt = weights.slice(0);
    new DataView(corrupt).setFloat32(400, Number.NaN, true);
    expect(() => parseCharLstm(meta, corrupt)).toThrow(/NaN/);
  });
});

describe('a run of the LSTM', () => {
  const indices = encode(HELD_OUT.slice(0, 400));
  const run = new CharLstmRun(model);
  run.setIndices(indices);

  it('keeps gates within their ranges and next-character probabilities summing to 1', () => {
    expect(run.length).toBe(indices.length);
    let gateMin = Infinity;
    let gateMax = -Infinity;
    let candidateMax = 0;
    let hiddenMax = 0;
    let cellError = 0;
    let hiddenError = 0;
    let probabilityMin = Infinity;
    let sumError = 0;
    for (let t = 0; t < run.length; t++) {
      const step = run.step(t);
      expect(step.index).toBe(indices[t]);
      expect(step.character).toBe(HELD_OUT[t]);
      for (const gate of [step.input, step.forget, step.output]) {
        for (const value of gate) { gateMin = Math.min(gateMin, value); gateMax = Math.max(gateMax, value); }
      }
      let total = 0;
      for (const p of step.probabilities) { probabilityMin = Math.min(probabilityMin, p); total += p; }
      sumError = Math.max(sumError, Math.abs(total - 1));
      for (let k = 0; k < H; k++) {
        candidateMax = Math.max(candidateMax, Math.abs(step.candidate[k]));
        hiddenMax = Math.max(hiddenMax, Math.abs(step.hidden[k]));
        // c = f ⊙ c_prev + i ⊙ g and h = o ⊙ tanh(c), as stored.
        const previous = t > 0 ? run.cells[(t - 1) * H + k] : 0;
        cellError = Math.max(cellError, Math.abs(step.cell[k] - (step.forget[k] * previous + step.input[k] * step.candidate[k])));
        hiddenError = Math.max(hiddenError, Math.abs(step.hidden[k] - step.output[k] * Math.tanh(step.cell[k])));
      }
    }
    expect(gateMin).toBeGreaterThanOrEqual(0);
    expect(gateMax).toBeLessThanOrEqual(1);
    // Gates that never move would make a dull page.
    expect(gateMin).toBeLessThan(.05);
    expect(gateMax).toBeGreaterThan(.95);
    expect(candidateMax).toBeLessThanOrEqual(1);
    expect(hiddenMax).toBeLessThanOrEqual(1);
    expect(cellError).toBeLessThan(1e-5);
    expect(hiddenError).toBeLessThan(1e-6);
    expect(probabilityMin).toBeGreaterThanOrEqual(0);
    expect(sumError).toBeLessThan(1e-5);
    expect(Number.isNaN(run.surprisal(0))).toBe(true);
    expect(run.surprisal(1)).toBeCloseTo(-Math.log2(run.step(0).probabilities[indices[1]]), 6);
  });

  it('extends a prompt by computing only the new steps, exactly like a fresh run', () => {
    const growing = new CharLstmRun(model);
    expect(growing.setIndices(indices.slice(0, 120))).toBe(0);
    expect(growing.setIndices(indices.slice(0, 121))).toBe(120);
    expect(growing.setIndices(indices)).toBe(121);
    // An edit in the middle keeps everything before it.
    const edited = [...indices];
    edited[200] = model.lookup.get('X');
    const editedRun = new CharLstmRun(model);
    editedRun.setIndices(edited);
    expect(growing.setIndices(edited)).toBe(200);
    expect(growing.setIndices(indices)).toBe(200);
    // Shrinking reuses the prefix without computing anything.
    const shorter = new CharLstmRun(model);
    expect(shorter.setIndices(indices)).toBe(0);
    expect(shorter.setIndices(indices.slice(0, 50))).toBe(50);
    expect(shorter.length).toBe(50);

    const n = indices.length;
    for (const [incremental, fresh, length] of [[growing, run, n], [shorter, run, 50]] as const) {
      expect(incremental.length).toBe(length);
      expect(Array.from(incremental.indices.subarray(0, length))).toEqual(Array.from(fresh.indices.subarray(0, length)));
      expect(incremental.gates.subarray(0, length * 4 * H)).toEqual(fresh.gates.subarray(0, length * 4 * H));
      expect(incremental.cells.subarray(0, length * H)).toEqual(fresh.cells.subarray(0, length * H));
      expect(incremental.hidden.subarray(0, length * H)).toEqual(fresh.hidden.subarray(0, length * H));
      expect(incremental.probabilities.subarray(0, length * V)).toEqual(fresh.probabilities.subarray(0, length * V));
    }
    // One character at a time, as typing and Generate do.
    const typed = new CharLstmRun(model);
    indices.slice(0, 80).forEach((index, t) => expect(typed.setIndices(indices.slice(0, t + 1))).toBe(t));
    expect(typed.cells.subarray(0, 80 * H)).toEqual(run.cells.subarray(0, 80 * H));
    expect(editedRun.cells.subarray(201 * H, 202 * H)).not.toEqual(run.cells.subarray(201 * H, 202 * H));
  });

  it('measures influence with the exact gradient, matching finite differences', () => {
    const passage = indices.slice(0, 160);
    const short = new CharLstmRun(model);
    short.setIndices(passage);
    for (const t of [159, 97, 12]) {
      for (const target of [short.topCharacter(t), passage[Math.min(t + 1, passage.length - 1)]]) {
        const influence = short.influence(t, target);
        expect(influence).toHaveLength(t + 1);
        // The log-probability the gradient explains, as the run computed it.
        const logits = short.step(t).logits;
        let max = -Infinity;
        for (const value of logits) max = Math.max(max, value);
        let sum = 0;
        for (const value of logits) sum += Math.exp(value - max);
        expect(relaxedLogProbability(passage, t, target, -1, 0)).toBeCloseTo(logits[target] - max - Math.log(sum), 5);
        for (const s of [t, t - 1, t - 2, t - 5, t - 11, Math.floor(t / 2), 0]) {
          // Central differences with Richardson extrapolation: the error shrinks as ε⁴, so
          // the step can stay large enough for the float32 activations not to matter.
          const central = (epsilon: number) => (relaxedLogProbability(passage, t, target, s, epsilon)
            - relaxedLogProbability(passage, t, target, s, -epsilon)) / (2 * epsilon);
          const numeric = (4 * central(5e-3) - central(1e-2)) / 3;
          expect(Math.abs(numeric - influence[s])).toBeLessThan(1e-4 + 1e-3 * Math.abs(numeric));
        }
      }
    }
    // The default target is the most likely next character.
    expect(short.influence(159)).toEqual(short.influence(159, short.topCharacter(159)));
  });

  it('can trace the memory span over the latest characters only', () => {
    const t = 150;
    const reach = 40;
    const full = run.influence(t);
    const near = run.influence(t, undefined, reach);
    expect(near).toHaveLength(t + 1);
    expect(Array.from(near.subarray(t + 1 - reach))).toEqual(Array.from(full.subarray(t + 1 - reach)));
    expect(near.subarray(0, t + 1 - reach).every(value => value === 0)).toBe(true);
  });

  it('reads ahead the word it would write, without changing the run', () => {
    const before = run.cells.slice();
    const letter = (index: number) => /[A-Za-z']/.test(model.vocabulary[index]);
    for (const t of [3, 40, 121, 260]) {
      for (const first of [run.topCharacter(t), model.lookup.get('t'), model.lookup.get('s'), model.lookup.get('W')]) {
        // A copy of the run that reads `first` and then keeps taking its likeliest letter.
        const copy = new CharLstmRun(model);
        copy.setIndices([...indices.slice(0, t + 1), first]);
        const expected = [first];
        while (letter(first) && expected.length < 14) {
          const next = copy.topCharacter(copy.length - 1);
          if (!letter(next)) break;
          expected.push(next);
          copy.push(next);
        }
        while (expected.length > 1 && model.vocabulary[expected.at(-1)] === "'") expected.pop();
        expect(run.completeWord(t, first)).toEqual(expected);
      }
    }
    // Anything that isn't a letter stands alone.
    const space = model.lookup.get(' ');
    expect(run.completeWord(10, space)).toEqual([space]);
    expect(run.cells).toEqual(before);
  });
});

describe('held-out text', () => {
  /*
   * Thresholds. The trainer measured the exported model on all of the held-out text of
   * its nineteen books (meta.validationBitsPerChar: 1.77 bits per character). This
   * passage costs it 1.50 bits, and passages of the same length across the held-out text
   * range from 1.53 to 2.59 (median 1.75). The earlier 128-cell model, trained on four
   * books, scored 1.91 overall and 1.68 here, so 1.85 and 1.65 leave room for a retrained
   * model but fail that smaller one, and anything that has lost its context: a bigram
   * model needs 3.3 bits per character on this passage and character frequencies alone 4.4.
   */
  it('costs the model under 1.65 bits per character', () => {
    expect(meta.validationBitsPerChar).toBeLessThan(1.85);
    const { bits, count } = bitsPerCharacter(model, encode(HELD_OUT));
    expect(count).toBe(HELD_OUT.length - 1);
    expect(bits).toBeLessThan(1.65);
  });
});

describe('encoding text', () => {
  it('maps curly quotes, dashes and accents to the vocabulary and reports the rest', () => {
    const encoded = encodeText(model, '“Café”—it’s naïve… cost 5 € – fine');
    expect(encoded.text).toBe('"Cafe"--it\'s naive... cost   - fine');
    expect(encoded.skipped).toEqual(['5', '€']);
    expect(encoded.indices.map(index => model.vocabulary[index]).join('')).toBe(encoded.text);
    expect(encodeText(model, 'Œuvre\r\nÆsop').text).toBe('Oeuvre\nAesop');
    // Characters the network knows map to themselves.
    expect(encodeText(model, meta.vocabulary).indices).toEqual(Array.from(meta.vocabulary, (_c, i) => i));
  });
});
