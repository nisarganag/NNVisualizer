// @vitest-environment node
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseCnnWeights, preprocessDrawing, runCnn } from '../src/cnn/model';
import {
  CLASSES, PIXELS, createDecoding, decode, encode, klDivergence, majorityLabel, nearestLabels, parseCodes, parseExamples, parseVae,
  reconstructionLoss, vaeLayout, vaeParameterCount, type DigitCodes, type VaeMeta,
} from '../src/vae/model';

function readModelFile(name: string): ArrayBuffer {
  const bytes = readFileSync(new URL(`../public/models/${name}`, import.meta.url));
  const buffer = new ArrayBuffer(bytes.length);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

const meta = JSON.parse(readFileSync(new URL('../public/models/mnist-vae-v1.json', import.meta.url), 'utf8')) as VaeMeta;
const weightsBuffer = readModelFile('mnist-vae-v1.bin');
const model = parseVae(meta, weightsBuffer);
const codes = parseCodes(readModelFile(meta.codes.file), meta.codes.count);
const examples = parseExamples(readModelFile(meta.examples.file), meta.examples.count);
const reader = parseCnnWeights(readModelFile('mnist-cnn-v1.bin'));
const centers = meta.classCenters as [number, number][];
/** The map holds every n-th test digit, so test digit i is code i / stride. */
const stride = Math.round(10000 / codes.count);

const K = 5;
// Thresholds, each set between the shipped model and what it should be told apart from
// (figures from scripts/train-mnist-vae.mjs's report and from the checks below):
//
// Reconstruction, in nats per digit decoded from its mean code, averaged over the 100
// example digits (random test digits; one standard error of that mean is about 5.5 nats).
// Their own mean image, the best any decoder that ignored its two numbers could do on
// them, scores 204; a model trained for one epoch scores 167; the shipped model 136.5
// here and 130.7 on all 10,000 test digits. 150 is 2.5 standard errors above the shipped
// model and far below both failures.
const RECONSTRUCTION_LIMIT = 150;
// Separation: 5-nearest-neighbor accuracy among the map's codes, leaving each one out.
// Always guessing the most common digit scores 11.8%, the one-epoch model 44%, the shipped
// model 81%. (Against all 60,000 training codes, 4, 5 and 9 overlap most: 65-67% each.)
const SEPARATION_LIMIT = 0.7;
// Smoothness along the straight path between two class centers, in 100 equal steps. The
// largest change of one pixel in one step is 0.23 for the shipped model, where a decoder
// that cut from one end's digit to the other's would change some pixel by at least 0.76
// (for every pair here); and no step carries more than 3.2% of a path's whole change,
// where even steps would carry 1% each and a cut 100% in one.
const PIXEL_STEP_LIMIT = 0.35;
const STEP_SHARE_LIMIT = 0.08;

/** Share of codes whose K nearest other codes vote for their own digit. */
function leaveOneOutAccuracy(map: DigitCodes): number {
  let correct = 0;
  for (let n = 0; n < map.count; n++) {
    const votes = nearestLabels(map, map.codes[2 * n], map.codes[2 * n + 1], K, n);
    if (majorityLabel(votes) === map.labels[n]) correct++;
  }
  return correct / map.count;
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
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
      for (let y = Math.max(0, Math.floor(Math.min(ay, by) - half - 1)); y <= Math.min(size - 1, Math.ceil(Math.max(ay, by) + half + 1)); y++) {
        for (let x = Math.max(0, Math.floor(Math.min(ax, bx) - half - 1)); x <= Math.min(size - 1, Math.ceil(Math.max(ax, bx) + half + 1)); x++) {
          const px = x + 0.5 - ax;
          const py = y + 0.5 - ay;
          const t = Math.max(0, Math.min(1, (px * dx + py * dy) / lengthSquared));
          const coverage = Math.max(0, Math.min(1, half + 0.5 - Math.hypot(px - t * dx, py - t * dy)));
          alpha[y * size + x] = Math.max(alpha[y * size + x], coverage * 255);
        }
      }
    }
  }
  return alpha;
}

/** Where a drawing lands: the digit most of its 15 nearest test digits are. */
function landsAmong(strokes: Point[][]): number {
  const input = preprocessDrawing(drawStrokes(280, strokes, 21), 280, 280);
  const { mean } = encode(model, input);
  return majorityLabel(nearestLabels(codes, mean[0], mean[1], 15));
}

describe('VAE weights and metadata', () => {
  it('parses the shipped weights into layers of the documented sizes', () => {
    expect(meta.latentSize).toBe(2);
    expect(meta.hiddenSizes).toEqual([256, 128]);
    expect(model.parameters).toHaveLength(469268);
    expect(vaeParameterCount(2, [256, 128])).toBe(469268);
    expect(weightsBuffer.byteLength).toBe(469268 * 4);
    const shapes = (layer: { inputs: number; outputs: number; weights: Float32Array; biases: Float32Array }) =>
      [layer.inputs, layer.outputs, layer.weights.length, layer.biases.length];
    expect(shapes(model.encoder.hidden1)).toEqual([784, 256, 784 * 256, 256]);
    expect(shapes(model.encoder.hidden2)).toEqual([256, 128, 256 * 128, 128]);
    expect(shapes(model.encoder.mean)).toEqual([128, 2, 256, 2]);
    expect(shapes(model.encoder.logVariance)).toEqual([128, 2, 256, 2]);
    expect(shapes(model.decoder.hidden1)).toEqual([2, 128, 256, 128]);
    expect(shapes(model.decoder.hidden2)).toEqual([128, 256, 128 * 256, 256]);
    expect(shapes(model.decoder.output)).toEqual([256, 784, 256 * 784, 784]);
    // The layout tiles the file with no gaps.
    const layout = vaeLayout(2, [256, 128]);
    expect(meta.tensors).toEqual(layout);
    layout.forEach((tensor, i) => {
      const end = tensor.offset + tensor.shape.reduce((a, b) => a * b, 1);
      expect(end).toBe(i + 1 < layout.length ? layout[i + 1].offset : 469268);
    });
  });

  it('rejects files of the wrong size and inconsistent metadata', () => {
    const resized = (bytes: number) => {
      const buffer = new ArrayBuffer(bytes);
      new Uint8Array(buffer).set(new Uint8Array(weightsBuffer, 0, Math.min(bytes, weightsBuffer.byteLength)));
      return buffer;
    };
    expect(() => parseVae(meta, resized(weightsBuffer.byteLength - 4))).toThrow(/expected 1877072 bytes/);
    expect(() => parseVae(meta, resized(weightsBuffer.byteLength + 4))).toThrow(/expected 1877072 bytes/);
    expect(() => parseVae(meta, new ArrayBuffer(0))).toThrow(/Invalid VAE weights/);
    expect(() => parseVae(meta, new Float32Array(469268) as unknown as ArrayBuffer)).toThrow(TypeError);
    expect(() => parseVae({ ...meta, parameters: 469267 }, weightsBuffer)).toThrow(/parameters/);
    expect(() => parseVae({ ...meta, hiddenSizes: [512, 256] }, weightsBuffer)).toThrow(/layout/);
    expect(() => parseVae({ ...meta, format: 'mnist-vae-v0' } as unknown as VaeMeta, weightsBuffer)).toThrow(/format/);
    const shuffled = meta.tensors.map(tensor => ({ ...tensor }));
    [shuffled[0], shuffled[2]] = [shuffled[2], shuffled[0]];
    expect(() => parseVae({ ...meta, tensors: shuffled }, weightsBuffer)).toThrow(/layout/);
    const poisoned = resized(weightsBuffer.byteLength);
    new DataView(poisoned).setFloat32(400, Number.NaN, true);
    expect(() => parseVae(meta, poisoned)).toThrow(/is NaN/);

    expect(() => parseCodes(new ArrayBuffer(codes.count * 9 - 1), codes.count)).toThrow(/expected 45000 bytes/);
    const badLabel = new ArrayBuffer(9);
    new Uint8Array(badLabel)[8] = 10;
    expect(() => parseCodes(badLabel, 1)).toThrow(/not a digit/);
    expect(() => parseExamples(new ArrayBuffer(examples.length * PIXELS + 1), examples.length)).toThrow(/expected 78400 bytes/);
  });

  it('keeps everything the page downloads for the model to about 2 MB', () => {
    const json = readFileSync(new URL('../public/models/mnist-vae-v1.json', import.meta.url)).length;
    const total = json + weightsBuffer.byteLength + codes.count * 9 + examples.length * PIXELS;
    expect(total).toBeLessThan(2.1e6);
  });
});

describe('decoder', () => {
  it('draws ink probabilities between 0 and 1 anywhere on the plane, even far off the map', () => {
    const decoding = createDecoding(model);
    const reach = 3 * meta.view.max;
    let outside = 0;
    let sigmoidError = 0;
    for (let row = 0; row <= 40; row++) {
      for (let column = 0; column <= 40; column++) {
        const { image, logits } = decode(model, [-reach + (2 * reach * column) / 40, -reach + (2 * reach * row) / 40], decoding);
        for (let i = 0; i < PIXELS; i++) {
          // Written so that NaN counts as outside.
          if (!(image[i] >= 0 && image[i] <= 1)) outside++;
          sigmoidError = Math.max(sigmoidError, Math.abs(image[i] - 1 / (1 + Math.exp(-logits[i]))));
        }
      }
    }
    expect(outside).toBe(0);
    expect(sigmoidError).toBeLessThan(1e-6);
    for (const far of [[1e4, -1e4], [-1e6, 3], [0, 1e8]] as Point[]) {
      const { image } = decode(model, far);
      expect(image.every(value => value >= 0 && value <= 1)).toBe(true);
    }
  });

  it('is deterministic and rejects malformed points', () => {
    const a = decode(model, [0.3, -1.2]).image;
    const b = decode(model, [0.3, -1.2]).image;
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(() => decode(model, [0.3])).toThrow(/expects 2 numbers/);
    expect(() => decode(model, [Number.NaN, 0])).toThrow(/finite/);
    expect(() => encode(model, new Float32Array(783))).toThrow(/784-value image/);
  });
});

describe('encoder', () => {
  it('encodes the example digits to the codes stored for them on the map', () => {
    const { labels, testIndices } = meta.examples;
    expect(examples).toHaveLength(100);
    expect(new Set(testIndices).size).toBe(100);
    for (let digit = 0; digit < CLASSES; digit++) expect(labels.filter(label => label === digit)).toHaveLength(10);
    examples.forEach((image, e) => {
      expect(testIndices[e] % stride).toBe(0);
      const slot = testIndices[e] / stride;
      expect(codes.labels[slot], `example ${e}`).toBe(labels[e]);
      const { mean } = encode(model, image);
      expect(Math.abs(mean[0] - codes.codes[2 * slot]), `example ${e}`).toBeLessThan(1e-5);
      expect(Math.abs(mean[1] - codes.codes[2 * slot + 1]), `example ${e}`).toBeLessThan(1e-5);
    });
  });

  it('places drawn strokes among the digits they look like', () => {
    expect(landsAmong([[[0.5, 0.15], [0.5, 0.85]]])).toBe(1);
    const ring = Array.from({ length: 73 }, (_, i): Point => {
      const angle = (2 * Math.PI * i) / 72;
      return [0.5 + 0.25 * Math.cos(angle), 0.5 + 0.33 * Math.sin(angle)];
    });
    expect(landsAmong([ring])).toBe(0);
  });
});

describe('reconstruction', () => {
  // The examples are 100 test digits drawn at random, ten per class, so they are a fair
  // (if small) sample of the test set.
  const lossAtCode = examples.map(image => reconstructionLoss(decode(model, encode(model, image).mean).logits, image));
  const average = lossAtCode.reduce((a, b) => a + b, 0) / lossAtCode.length;

  it('redraws test digits from two numbers far better than the average digit does', () => {
    // Predicting every example with the examples' own mean image, the best any decoder
    // that ignored its two numbers could do on them.
    const meanImage = new Float64Array(PIXELS);
    for (const image of examples) for (let i = 0; i < PIXELS; i++) meanImage[i] += image[i] / examples.length;
    const meanLogits = Array.from(meanImage, p => Math.log(Math.max(p, 1e-6) / Math.max(1 - p, 1e-6)));
    const baseline = examples.reduce((sum, image) => sum + reconstructionLoss(meanLogits, image), 0) / examples.length;
    expect(baseline).toBeGreaterThan(190);
    expect(average).toBeLessThan(RECONSTRUCTION_LIMIT);
    expect(average).toBeLessThan(baseline - 50);
  });

  it('agrees with the test-set scores stored with the model', () => {
    const scores = meta.scores as { reconstructionAtMean: number; elbo: number; kl: number; reconstruction: number };
    expect(scores.elbo).toBeCloseTo(-(scores.reconstruction + scores.kl), 1);
    // Within three standard errors of the 100-example mean (about 5.5 nats each).
    expect(Math.abs(average - scores.reconstructionAtMean)).toBeLessThan(16.5);
    expect(scores.reconstructionAtMean).toBeLessThan(RECONSTRUCTION_LIMIT);
  });

  it('spends a few nats per digit on its two numbers', () => {
    const kl = examples.map(image => {
      const { mean, logVariance } = encode(model, image);
      return klDivergence(mean, logVariance);
    });
    const averageKl = kl.reduce((a, b) => a + b, 0) / kl.length;
    expect(averageKl).toBeGreaterThan(3);
    expect(averageKl).toBeLessThan(12);
  });
});

describe('the map', () => {
  it('keeps most test digits next to digits of their own class', () => {
    const accuracy = leaveOneOutAccuracy(codes);
    expect(accuracy).toBeGreaterThan(SEPARATION_LIMIT);
    expect(accuracy).toBeCloseTo((meta.scores as { knn: { leaveOneOutOnMap: number } }).knn.leaveOneOutOnMap, 4);
    // The examples' fresh codes, matched against the other digits on the map.
    let correct = 0;
    examples.forEach((image, e) => {
      const { mean } = encode(model, image);
      const votes = nearestLabels(codes, mean[0], mean[1], K, meta.examples.testIndices[e] / stride);
      if (majorityLabel(votes) === meta.examples.labels[e]) correct++;
    });
    expect(correct / examples.length).toBeGreaterThan(SEPARATION_LIMIT - 0.1);
  });

  it('labels each digit at a spot that decodes to that digit, among its own test digits', () => {
    expect(centers).toHaveLength(CLASSES);
    centers.forEach(([x, y], digit) => {
      expect(x).toBeGreaterThan(meta.view.min);
      expect(x).toBeLessThan(meta.view.max);
      expect(y).toBeGreaterThan(meta.view.min);
      expect(y).toBeLessThan(meta.view.max);
      const { probabilities } = runCnn(reader, decode(model, [x, y]).image);
      expect(argmax(probabilities), `center of ${digit}`).toBe(digit);
      expect(probabilities[digit]).toBeGreaterThan(0.9);
      expect(majorityLabel(nearestLabels(codes, x, y, 15)), `center of ${digit}`).toBe(digit);
    });
  });

  it('changes smoothly along the path between any two class centers', () => {
    const steps = 100;
    const decoding = createDecoding(model);
    for (let a = 0; a < CLASSES; a++) {
      for (let b = a + 1; b < CLASSES; b++) {
        const [ax, ay] = centers[a];
        const [bx, by] = centers[b];
        const start = decode(model, [ax, ay], decoding).image.slice();
        let previous = start;
        let largestPixelStep = 0;
        let largestStep = 0;
        let pathChange = 0;
        for (let t = 1; t <= steps; t++) {
          const image = decode(model, [ax + ((bx - ax) * t) / steps, ay + ((by - ay) * t) / steps], decoding).image.slice();
          let change = 0;
          for (let i = 0; i < PIXELS; i++) {
            const pixel = Math.abs(image[i] - previous[i]);
            change += pixel;
            if (pixel > largestPixelStep) largestPixelStep = pixel;
          }
          largestStep = Math.max(largestStep, change);
          pathChange += change;
          previous = image;
        }
        // What cutting straight from one end's digit to the other's would change at once.
        let cut = 0;
        let ends = 0;
        for (let i = 0; i < PIXELS; i++) {
          const difference = Math.abs(previous[i] - start[i]);
          cut = Math.max(cut, difference);
          ends += difference;
        }
        const pair = `${a} to ${b}`;
        // The two ends really are different digits, so the path has somewhere to go.
        expect(ends / PIXELS, pair).toBeGreaterThan(0.05);
        expect(largestPixelStep, pair).toBeLessThan(PIXEL_STEP_LIMIT);
        expect(largestPixelStep, pair).toBeLessThan(cut / 2);
        expect(largestStep / pathChange, pair).toBeLessThan(STEP_SHARE_LIMIT);
      }
    }
  });
});

describe('nearestLabels and majorityLabel', () => {
  const tiny: DigitCodes = { count: 5, codes: Float32Array.from([0, 0, 1, 0, 0, 2, 3, 3, -1, 0]), labels: Uint8Array.from([4, 7, 7, 9, 4]) };

  it('lists labels nearest first, skipping an excluded code', () => {
    expect(nearestLabels(tiny, 0, 0, 3)).toEqual([4, 7, 4]);
    expect(nearestLabels(tiny, 0, 0, 3, 0)).toEqual([7, 4, 7]);
    expect(nearestLabels(tiny, 10, 10, 1)).toEqual([9]);
  });

  it('breaks a tie in favor of the nearest code', () => {
    expect(majorityLabel([7, 4, 4, 7])).toBe(7);
    expect(majorityLabel([4, 7, 7])).toBe(7);
    expect(majorityLabel([])).toBe(-1);
  });
});
