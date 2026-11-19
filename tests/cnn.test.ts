// @vitest-environment node
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ATTACK_PIXEL_CAP, ATTACK_STEP_PIXELS, runAttack, startAttack, stepAttack, topDigit, type Attack } from '../src/cnn/attack';
import { CNN_LAYOUT, inputGradient, parseCnnWeights, parseSamples, preprocessDrawing, runCnn, type CnnActivations } from '../src/cnn/model';

function readModelFile(name: string): ArrayBuffer {
  const bytes = readFileSync(new URL(`../public/models/${name}`, import.meta.url));
  const buffer = new ArrayBuffer(bytes.length);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

const weights = parseCnnWeights(readModelFile('mnist-cnn-v1.bin'));
const samples = parseSamples(readModelFile('mnist-samples-v1.bin'));

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function classify(input: Float32Array) {
  const { probabilities } = runCnn(weights, input);
  const digit = argmax(probabilities);
  return { digit, confidence: probabilities[digit] };
}

/** Index-space center of mass (Σx·v/Σv, Σy·v/Σv) of a 28×28 input. */
function centerOfMass(input: Float32Array): [number, number] {
  let mass = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < 784; i++) {
    mass += input[i];
    x += (i % 28) * input[i];
    y += Math.floor(i / 28) * input[i];
  }
  return [x / mass, y / mass];
}

type Point = [number, number];

/** Renders round-capped, antialiased brush strokes (points in 0..1 canvas units) as alpha. */
function drawStrokes(size: number, strokes: Point[][], brushWidth: number): Uint8ClampedArray {
  const alpha = new Uint8ClampedArray(size * size);
  const half = brushWidth / 2;
  for (const stroke of strokes) {
    for (let s = 0; s + 1 < stroke.length; s++) {
      const [ax, ay] = [stroke[s][0] * size, stroke[s][1] * size];
      const [bx, by] = [stroke[s + 1][0] * size, stroke[s + 1][1] * size];
      const dx = bx - ax;
      const dy = by - ay;
      const lengthSquared = dx * dx + dy * dy || 1;
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - half - 1));
      const x1 = Math.min(size - 1, Math.ceil(Math.max(ax, bx) + half + 1));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by) - half - 1));
      const y1 = Math.min(size - 1, Math.ceil(Math.max(ay, by) + half + 1));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5 - ax;
          const py = y + 0.5 - ay;
          const t = Math.max(0, Math.min(1, (px * dx + py * dy) / lengthSquared));
          const distance = Math.hypot(px - t * dx, py - t * dy);
          const coverage = Math.max(0, Math.min(1, half + 0.5 - distance));
          alpha[y * size + x] = Math.max(alpha[y * size + x], coverage * 255);
        }
      }
    }
  }
  return alpha;
}

function ellipse(cx: number, cy: number, rx: number, ry: number, segments = 72): Point[] {
  return Array.from({ length: segments + 1 }, (_, i): Point => {
    const angle = (2 * Math.PI * i) / segments;
    return [cx + rx * Math.cos(angle), cy + ry * Math.sin(angle)];
  });
}

const VERTICAL_STROKE: Point[][] = [[[0.5, 0.15], [0.5, 0.85]]];
const RING: Point[][] = [ellipse(0.5, 0.5, 0.25, 0.33)];

describe('MNIST CNN weights and samples', () => {
  it('parses the shipped weights into tensors of the documented sizes', () => {
    expect(readModelFile('mnist-cnn-v1.bin').byteLength).toBe(52138 * 4);
    expect(weights.conv1.weights).toHaveLength(8 * 1 * 3 * 3);
    expect(weights.conv1.biases).toHaveLength(8);
    expect(weights.conv2.weights).toHaveLength(16 * 8 * 3 * 3);
    expect(weights.conv2.biases).toHaveLength(16);
    expect(weights.dense.weights).toHaveLength(64 * 784);
    expect(weights.dense.biases).toHaveLength(64);
    expect(weights.output.weights).toHaveLength(10 * 64);
    expect(weights.output.biases).toHaveLength(10);
    const total = Object.values(weights).reduce((sum, layer) => sum + layer.weights.length + layer.biases.length, 0);
    expect(total).toBe(52138);
  });

  it('rejects buffers of the wrong size', () => {
    expect(() => parseCnnWeights(new ArrayBuffer(52138 * 4 - 4))).toThrow(/expected 208552 bytes/);
    expect(() => parseCnnWeights(new ArrayBuffer(52138 * 4 + 4))).toThrow(/expected 208552 bytes/);
    expect(() => parseCnnWeights(new ArrayBuffer(0))).toThrow(/Invalid CNN weights/);
    expect(() => parseSamples(new ArrayBuffer(7839))).toThrow(/expected 7840 bytes/);
  });

  it('ships ten 28×28 example digits in 0..1 that the model classifies correctly', () => {
    expect(samples).toHaveLength(10);
    samples.forEach((sample, digit) => {
      expect(sample).toHaveLength(784);
      expect(Math.min(...sample)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...sample)).toBeLessThanOrEqual(1);
      const result = classify(sample);
      expect(result.digit, `sample ${digit}`).toBe(digit);
      expect(result.confidence).toBeGreaterThan(0.99);
    });
  });
});

describe('runCnn', () => {
  const activations = runCnn(weights, samples[7]);

  it('returns activations with the documented lengths', () => {
    const { input, conv1, pool1, conv2, pool2, dense, output } = CNN_LAYOUT;
    expect(activations.input).toHaveLength(input.channels * input.size ** 2);
    expect(activations.conv1).toHaveLength(conv1.filters * conv1.size ** 2);
    expect(activations.pool1).toHaveLength(pool1.channels * pool1.size ** 2);
    expect(activations.conv2).toHaveLength(conv2.filters * conv2.size ** 2);
    expect(activations.pool2).toHaveLength(pool2.channels * pool2.size ** 2);
    expect(activations.dense).toHaveLength(dense.units);
    expect(activations.logits).toHaveLength(output.units);
    expect(activations.probabilities).toHaveLength(output.units);
    expect([6272, 1568, 3136, 784]).toEqual([activations.conv1.length, activations.pool1.length,
      activations.conv2.length, activations.pool2.length]);
  });

  it('keeps ReLU layers non-negative and every layer active', () => {
    for (const name of ['conv1', 'pool1', 'conv2', 'pool2', 'dense'] as const) {
      expect(Math.min(...activations[name]), name).toBeGreaterThanOrEqual(0);
      expect(Math.max(...activations[name]), name).toBeGreaterThan(0);
    }
  });

  it('returns probabilities that sum to 1', () => {
    for (const sample of samples) {
      const { probabilities, logits } = runCnn(weights, sample);
      expect(probabilities.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 5);
      expect(Math.min(...probabilities)).toBeGreaterThanOrEqual(0);
      expect(argmax(probabilities)).toBe(argmax(logits));
    }
  });

  it('follows the documented channel-major layouts', () => {
    const a = activations;
    const relu = (v: number) => Math.max(0, v);
    // conv1[f][y][x] with zero padding: check filter 3 at (y, x) = (10, 0).
    let sum = weights.conv1.biases[3];
    for (let ky = 0; ky < 3; ky++) {
      for (let kx = 0; kx < 3; kx++) {
        const y = 10 + ky - 1;
        const x = kx - 1;
        if (x >= 0) sum += weights.conv1.weights[3 * 9 + ky * 3 + kx] * a.input[y * 28 + x];
      }
    }
    expect(a.conv1[3 * 784 + 10 * 28]).toBeCloseTo(relu(sum), 5);
    // pool1 is the 2×2 max of conv1; pool2 of conv2.
    expect(a.pool1[5 * 196 + 6 * 14 + 4]).toBe(Math.max(a.conv1[5 * 784 + 12 * 28 + 8], a.conv1[5 * 784 + 12 * 28 + 9],
      a.conv1[5 * 784 + 13 * 28 + 8], a.conv1[5 * 784 + 13 * 28 + 9]));
    expect(a.pool2[9 * 49 + 3 * 7 + 2]).toBe(Math.max(a.conv2[9 * 196 + 6 * 14 + 4], a.conv2[9 * 196 + 6 * 14 + 5],
      a.conv2[9 * 196 + 7 * 14 + 4], a.conv2[9 * 196 + 7 * 14 + 5]));
    // conv2[f][y][x] sums over all 8 pool1 channels: check filter 11 at (6, 7).
    sum = weights.conv2.biases[11];
    for (let c = 0; c < 8; c++) {
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          sum += weights.conv2.weights[11 * 72 + c * 9 + ky * 3 + kx] * a.pool1[c * 196 + (5 + ky) * 14 + 6 + kx];
        }
      }
    }
    expect(a.conv2[11 * 196 + 6 * 14 + 7]).toBeCloseTo(relu(sum), 4);
    // dense row j reads the flattened pool2 (c, y, x); output row k reads dense.
    for (let j = 0; j < 64; j++) {
      let total = weights.dense.biases[j];
      for (let i = 0; i < 784; i++) total += weights.dense.weights[j * 784 + i] * a.pool2[i];
      expect(a.dense[j]).toBeCloseTo(relu(total), 4);
    }
    for (let k = 0; k < 10; k++) {
      let total = weights.output.biases[k];
      for (let j = 0; j < 64; j++) total += weights.output.weights[k * 64 + j] * a.dense[j];
      expect(a.logits[k]).toBeCloseTo(total, 4);
    }
  });

  it('rejects inputs of the wrong length', () => {
    expect(() => runCnn(weights, new Float32Array(783))).toThrow(/784/);
  });
});

describe('preprocessDrawing', () => {
  it('returns null for a blank or near-blank canvas', () => {
    expect(preprocessDrawing(new Uint8ClampedArray(280 * 280), 280, 280)).toBeNull();
    expect(preprocessDrawing(new Uint8ClampedArray(280 * 280).fill(12), 280, 280)).toBeNull(); // faint haze
    expect(preprocessDrawing(drawStrokes(280, [[[0.3, 0.4], [0.3, 0.4]]], 4), 280, 280)).toBeNull(); // a speck
  });

  it('classifies a thick vertical stroke as 1', () => {
    const input = preprocessDrawing(drawStrokes(280, VERTICAL_STROKE, 22), 280, 280);
    expect(input).not.toBeNull();
    expect(classify(input!).digit).toBe(1);
  });

  it('classifies a thick ring as 0', () => {
    const input = preprocessDrawing(drawStrokes(280, RING, 22), 280, 280);
    expect(input).not.toBeNull();
    expect(classify(input!).digit).toBe(0);
  });

  it('fits the ink into 20×20 and centers its mass at (14, 14)', () => {
    const drawings: Point[][][] = [
      VERTICAL_STROKE,
      RING,
      [[[0.05, 0.05], [0.25, 0.05], [0.12, 0.3]]], // small 7 in the top-left corner
      [[[0.2, 0.1], [0.2, 0.9], [0.9, 0.9]]], // L-shape with off-center mass
    ];
    for (const strokes of drawings) {
      const input = preprocessDrawing(drawStrokes(280, strokes, 20), 280, 280)!;
      expect(input).toHaveLength(784);
      expect(Math.min(...input)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...input)).toBe(1);
      const [x, y] = centerOfMass(input);
      expect(Math.abs(x - 14)).toBeLessThanOrEqual(1);
      expect(Math.abs(y - 14)).toBeLessThanOrEqual(1);
      let top = 28;
      let bottom = -1;
      let left = 28;
      let right = -1;
      for (let i = 0; i < 784; i++) {
        if (input[i] < 0.05) continue;
        top = Math.min(top, Math.floor(i / 28));
        bottom = Math.max(bottom, Math.floor(i / 28));
        left = Math.min(left, i % 28);
        right = Math.max(right, i % 28);
      }
      expect(Math.max(bottom - top + 1, right - left + 1)).toBeGreaterThanOrEqual(19);
      expect(Math.max(bottom - top + 1, right - left + 1)).toBeLessThanOrEqual(21);
    }
  });

  it('gives the same input at devicePixelRatio 1 and 2', () => {
    const small = preprocessDrawing(drawStrokes(280, RING, 22), 280, 280)!;
    const large = preprocessDrawing(drawStrokes(560, RING, 44), 560, 560)!;
    let worst = 0;
    for (let i = 0; i < 784; i++) worst = Math.max(worst, Math.abs(small[i] - large[i]));
    expect(worst).toBeLessThan(0.1);
  });

  it('turns MNIST digits drawn large on a canvas back into inputs the model recognizes', () => {
    samples.forEach((sample, digit) => {
      const size = 280;
      const alpha = new Uint8ClampedArray(size * size);
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) alpha[y * size + x] = sample[Math.floor(y / 10) * 28 + Math.floor(x / 10)] * 255;
      }
      const input = preprocessDrawing(alpha, size, size)!;
      expect(classify(input).digit, `digit ${digit}`).toBe(digit);
    });
  });

  it('rejects alpha arrays that do not match the dimensions', () => {
    expect(() => preprocessDrawing(new Uint8ClampedArray(280 * 280 * 4), 280, 280)).toThrow(/one alpha byte per pixel/);
    expect(() => preprocessDrawing(new Uint8ClampedArray(0), 0, 0)).toThrow(RangeError);
  });
});

/** Deterministic pseudo-random numbers in [0, 1). */
function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Which linear piece of the network an input falls in: every ReLU's on/off state and
 * every max-pool window's winning position (first on a tie, as inputGradient routes it).
 * Two inputs with the same piece are joined by a segment on which every logit is affine.
 */
function linearPiece(activations: CnnActivations): string {
  const parts: number[] = [];
  for (const name of ['conv1', 'conv2', 'dense'] as const) for (const value of activations[name]) parts.push(value > 0 ? 1 : 0);
  for (const [name, planes, size] of [['conv1', 8, 28], ['conv2', 16, 14]] as const) {
    const values = activations[name];
    for (let c = 0; c < planes; c++) {
      for (let y = 0; y < size; y += 2) {
        for (let x = 0; x < size; x += 2) {
          const i = c * size * size + y * size + x;
          const max = Math.max(values[i], values[i + 1], values[i + size], values[i + size + 1]);
          parts.push(values[i] === max ? 0 : values[i + 1] === max ? 1 : values[i + size] === max ? 2 : 3);
        }
      }
    }
  }
  return parts.join('');
}

describe('inputGradient', () => {
  it('matches central finite differences on the shipped weights', () => {
    // The network is piecewise linear, so wherever x − ε and x + ε fall in the same
    // linear piece as x, the central difference is the exact derivative up to rounding.
    // The logits are stored as float32: below 32 their rounding step is 2^-19 ≈ 1.9e-6,
    // so each carries an error of about 1e-6 (measured worst: 0.95e-6), and the central
    // difference about 1e-6 / ε. The tolerance, 4e-6 / ε, is four times that.
    const random = mulberry32(5);
    const steps = [1e-2, 3e-3, 1e-3, 3e-4];
    const checked = { noisy: 0, clean: 0 };
    const attempts = { noisy: 0, clean: 0 };
    const cases: [sample: number, digits: number[], noisy: boolean][] = [
      [2, [2, 7, 3], true], [5, [5, 6, 8], true], [7, [7, 1, 9], true], [8, [8, 0, 3], true],
      // Clean digits have exact ties in blank background, where the derivative does not
      // exist (any change breaks the tie), so many of their pixels are skipped.
      [3, [3, 8], false], [9, [9, 4], false],
    ];
    for (const [sample, digits, noisy] of cases) {
      const kind = noisy ? 'noisy' : 'clean';
      // A little noise everywhere breaks the ties of blank background between pool positions.
      const input = noisy ? samples[sample].map(value => value + 0.02 + 0.03 * random()) : samples[sample].slice();
      const activations = runCnn(weights, input);
      const piece = linearPiece(activations);
      for (const digit of digits) {
        const gradient = inputGradient(weights, activations, digit);
        expect(gradient).toHaveLength(784);
        // The four most sensitive pixels, then random ones (stroke, edge and background).
        const strongest = Array.from(gradient.keys()).sort((a, b) => Math.abs(gradient[b]) - Math.abs(gradient[a])).slice(0, 4);
        const pixels = [...strongest, ...Array.from({ length: 8 }, () => Math.floor(random() * 784))];
        for (const pixel of pixels) {
          attempts[kind]++;
          for (const epsilon of steps) {
            const plus = input.slice();
            const minus = input.slice();
            plus[pixel] += epsilon;
            minus[pixel] -= epsilon;
            const above = runCnn(weights, plus);
            const below = runCnn(weights, minus);
            if (linearPiece(above) !== piece || linearPiece(below) !== piece) continue;
            const numeric = (above.logits[digit] - below.logits[digit]) / (2 * epsilon);
            expect(Math.abs(numeric - gradient[pixel]), `sample ${sample}, digit ${digit}, pixel ${pixel}, ε ${epsilon}`)
              .toBeLessThanOrEqual(4e-6 / epsilon + 1e-4 * Math.abs(gradient[pixel]));
            checked[kind]++;
            break;
          }
        }
      }
    }
    // 144 noisy and 48 clean pixel/digit pairs were tried; most must have been compared.
    expect(checked.noisy).toBeGreaterThanOrEqual(0.75 * attempts.noisy);
    expect(checked.clean).toBeGreaterThanOrEqual(0.25 * attempts.clean);
  });

  it('is nonzero on and around the strokes and supports the predicted digit there', () => {
    for (const [digit, sample] of samples.entries()) {
      const activations = runCnn(weights, sample);
      const gradient = inputGradient(weights, activations, digit);
      let support = 0;
      let active = 0;
      for (let i = 0; i < 784; i++) {
        support += gradient[i] * sample[i];
        if (gradient[i] !== 0) active++;
      }
      // Gradient × input: on balance the ink argues for the digit the network reads.
      expect(support, `digit ${digit}`).toBeGreaterThan(0);
      expect(active).toBeGreaterThan(100);
    }
  });

  it('differentiates a difference of two scores in the same pass', () => {
    const activations = runCnn(weights, samples[4]);
    const four = inputGradient(weights, activations, 4);
    const nine = inputGradient(weights, activations, 9);
    const margin = inputGradient(weights, activations, 4, 9);
    for (let i = 0; i < 784; i++) expect(margin[i]).toBeCloseTo(four[i] - nine[i], 5);
    expect(inputGradient(weights, activations, 4, 4).every(value => value === 0)).toBe(true);
  });

  it('rejects bad digits and activations', () => {
    const activations = runCnn(weights, samples[0]);
    expect(() => inputGradient(weights, activations, 10)).toThrow(RangeError);
    expect(() => inputGradient(weights, activations, 1.5)).toThrow(/0 to 9/);
    expect(() => inputGradient(weights, activations, 1, -1)).toThrow(/against/);
    expect(() => inputGradient(weights, { ...activations, conv2: new Float32Array(10) }, 1)).toThrow(/conv2/);
  });
});

describe('fooling the network', () => {
  const drawn = [
    preprocessDrawing(drawStrokes(280, VERTICAL_STROKE, 22), 280, 280)!,
    preprocessDrawing(drawStrokes(280, RING, 22), 280, 280)!,
  ];
  const runs: { original: Float32Array; digit: number; target: number; attack: Attack }[] = [];
  for (const original of [...samples, ...drawn]) {
    const digit = classify(original).digit;
    for (let target = 0; target < 10; target++) {
      if (target !== digit) runs.push({ original, digit, target, attack: runAttack(weights, original, target) });
    }
  }

  it('reaches the target for most target choices within the pixel cap', () => {
    const fooled = runs.filter(run => run.attack.status === 'fooled');
    // All 108 succeed with the shipped weights; allow a little slack for a retrained model.
    expect(fooled.length).toBeGreaterThanOrEqual(0.9 * runs.length);
    for (const { attack, target } of fooled) {
      expect(attack.order.length).toBeGreaterThan(0);
      expect(attack.order.length).toBeLessThanOrEqual(ATTACK_PIXEL_CAP);
      expect(topDigit(attack.activations.probabilities)).toBe(target);
      expect(classify(attack.input).digit).toBe(target);
    }
    // A few pixels usually suffice: the median attack changes under 5% of the image.
    const counts = fooled.map(run => run.attack.order.length).sort((a, b) => a - b);
    expect(counts[Math.floor(counts.length / 2)]).toBeLessThan(0.05 * 784);
    // A failure is only reported when the network really does not read the target.
    for (const { attack, target } of runs.filter(run => run.attack.status === 'failed')) {
      expect(attack.order.length).toBeLessThanOrEqual(ATTACK_PIXEL_CAP);
      expect(topDigit(attack.activations.probabilities)).not.toBe(target);
    }
  });

  it('never revisits a pixel and keeps every value within [0, 1]', () => {
    for (const { original, attack } of runs) {
      expect(new Set(attack.order).size).toBe(attack.order.length);
      expect(attack.changes.reduce((count, change) => count + (change !== 0 ? 1 : 0), 0)).toBe(attack.order.length);
      for (let i = 0; i < 784; i++) {
        expect(attack.input[i]).toBeGreaterThanOrEqual(0);
        expect(attack.input[i]).toBeLessThanOrEqual(1);
        if (attack.changes[i] === 0) expect(attack.input[i]).toBe(original[i]);
        else {
          expect(attack.input[i]).toBe(attack.changes[i] > 0 ? 1 : 0);
          expect(attack.input[i]).not.toBe(original[i]);
        }
      }
    }
  });

  it('changes 1–3 pixels a step, each the way the gradient says helps, and stops once it reads the target', () => {
    for (const [digit, target] of [[3, 8], [7, 1], [2, 4], [0, 6]]) {
      const attack = startAttack(weights, samples[digit], target);
      const seen = new Set<number>();
      while (attack.status === 'running') {
        expect(topDigit(attack.activations.probabilities)).not.toBe(target);
        const logits = attack.activations.logits;
        let rival = target === 0 ? 1 : 0;
        for (let k = 0; k < 10; k++) if (k !== target && logits[k] > logits[rival]) rival = k;
        const gradient = inputGradient(weights, attack.activations, target, rival);
        const before = attack.input.slice();
        const changed = stepAttack(weights, attack);
        expect(changed.length).toBeGreaterThanOrEqual(1);
        expect(changed.length).toBeLessThanOrEqual(ATTACK_STEP_PIXELS);
        for (const pixel of changed) {
          expect(seen.has(pixel)).toBe(false);
          seen.add(pixel);
          expect(attack.input[pixel]).toBe(gradient[pixel] > 0 ? 1 : 0);
          expect(attack.input[pixel]).not.toBe(before[pixel]);
        }
      }
      expect(attack.status, `${digit} → ${target}`).toBe('fooled');
      expect(stepAttack(weights, attack)).toEqual([]);
      expect(attack.steps).toBeLessThanOrEqual(attack.order.length);
    }
  });

  it('reports failure when the cap is reached', () => {
    const attack = runAttack(weights, samples[7], 4, 5);
    expect(attack.status).toBe('failed');
    expect(attack.order).toHaveLength(5);
    expect(topDigit(attack.activations.probabilities)).not.toBe(4);
  });

  it('has nothing to do when the network already reads the target', () => {
    const attack = startAttack(weights, samples[3], 3);
    expect(attack.status).toBe('fooled');
    expect(stepAttack(weights, attack)).toEqual([]);
    expect(attack.order).toHaveLength(0);
    expect(() => startAttack(weights, samples[3], 10)).toThrow(RangeError);
    expect(() => startAttack(weights, new Float32Array(10), 1)).toThrow(/784/);
  });
});
