import { describe, expect, it } from 'vitest';
import * as nn from '../src/nn';
import { State, activations, decodePoints, encodePoints, MAX_DRAWN_POINTS, optimizers } from '../src/state';
import { TrainingEngine } from '../src/simulation';
import { captureModel } from '../src/model-state';

const numeric = (f: (x: number) => number, x: number) => (f(x + 1e-5) - f(x - 1e-5)) / 2e-5;

describe('new activations', () => {
  it('Leaky ReLU keeps a small slope below zero', () => {
    const { output, der } = nn.Activations.LEAKY_RELU;
    expect(output(2)).toBe(2);
    expect(output(-2)).toBeCloseTo(-.02, 12);
    expect(der(3)).toBe(1);
    expect(der(-3)).toBe(.01);
  });

  it('GELU matches its tanh approximation and its derivative is exact', () => {
    const { output, der } = nn.Activations.GELU;
    expect(output(0)).toBe(0);
    expect(output(1)).toBeCloseTo(.8411919906, 8);
    expect(output(-1)).toBeCloseTo(-.1588080094, 8);
    for (const x of [-3, -1.2, -.3, 0, .4, 1.7, 3]) expect(der(x)).toBeCloseTo(numeric(output, x), 6);
  });
});

/** One input, one linear output, no bias movement worth tracking: easy to check by hand. */
function tinyNetwork() {
  const network = nn.buildNetwork([1, 1], nn.Activations.LINEAR, nn.Activations.LINEAR, null, ['x']);
  const link = network[1][0].inputLinks[0];
  link.weight = .5;
  network[1][0].bias = 0;
  return { network, link };
}
function accumulate(network: nn.Node[][], x: number, target: number) {
  nn.forwardProp(network, [x]);
  nn.backProp(network, target, nn.Errors.SQUARE);
}

describe('optimizers', () => {
  it('momentum adds 0.9 of the previous velocity to each step', () => {
    const { network, link } = tinyNetwork();
    accumulate(network, 1, 0); // output .5, error derivative .5, dE/dw = .5
    nn.updateWeights(network, .1, 0, 'momentum', 1);
    expect(link.lastGradient).toBeCloseTo(.5, 12);
    expect(link.weight).toBeCloseTo(.5 - .1 * .5, 12);
    const before = link.weight;
    // The bias moved too, so the next output (and dE/dw at x = 1) is w + b = .45 - .05.
    const output = link.weight + network[1][0].bias;
    expect(output).toBeCloseTo(.4, 12);
    accumulate(network, 1, 0);
    nn.updateWeights(network, .1, 0, 'momentum', 2);
    expect(link.weight).toBeCloseTo(before - .1 * (.9 * .5 + output), 12);
  });

  it('Adam’s first step moves each weight by the learning rate, against the gradient', () => {
    const { network, link } = tinyNetwork();
    accumulate(network, 2, 0); // dE/dw = output * x = 1 * 2
    nn.updateWeights(network, .01, 0, 'adam', 1);
    expect(link.weight).toBeCloseTo(.5 - .01, 6);
  });

  it('SGD keeps the original update and records the gradient it applied', () => {
    const { network, link } = tinyNetwork();
    accumulate(network, 1, 0);
    accumulate(network, 2, 0);
    nn.updateWeights(network, .1, 0);
    // Mean of dE/dw over the batch: (.5 * 1 + 1 * 2) / 2.
    expect(link.lastGradient).toBeCloseTo(1.25, 12);
    expect(link.weight).toBeCloseTo(.5 - .1 * 1.25, 12);
  });

  it('switching optimizer clears momentum and Adam state', () => {
    const { network, link } = tinyNetwork();
    accumulate(network, 1, 0);
    nn.updateWeights(network, .1, 0, 'adam', 1);
    expect(link.firstMoment).not.toBe(0);
    nn.resetOptimizerState(network);
    expect([link.velocity, link.firstMoment, link.secondMoment, network[1][0].firstMoment]).toEqual([0, 0, 0, 0]);
  });
});

describe('settings in the URL', () => {
  it('round-trips the optimizer, new activations and the backprop view', () => {
    const state = State.deserializeState('#optimizer=adam&activation=gelu&showBackprop=true');
    expect(state.optimizer).toBe(optimizers.adam);
    expect(state.activation).toBe(activations.gelu);
    expect(state.showBackprop).toBe(true);
    const restored = State.deserializeState('#' + state.toHash());
    expect([restored.optimizer, restored.activation, restored.showBackprop]).toEqual(['adam', activations.gelu, true]);
    expect(State.deserializeState('#activation=leakyRelu').activation).toBe(activations.leakyRelu);
    expect(State.deserializeState('#optimizer=rmsprop').optimizer).toBe('sgd');
  });

  it('stores hand-placed points compactly and drops anything malformed', () => {
    const points = [{ x: 1.23, y: -4.56, label: 1 }, { x: -6, y: 6, label: -1 }, { x: 0, y: 0, label: 1 }];
    const decoded = decodePoints(encodePoints(points));
    expect(decoded).toEqual([{ x: 1.25, y: -4.55, label: 1 }, { x: -6, y: 6, label: -1 }, { x: 0, y: 0, label: 1 }]);
    expect(encodePoints(points)).toMatch(/^[A-Za-z0-9_-]+$/);
    const many = Array.from({ length: 500 }, (_, i) => ({ x: (i % 11) - 5, y: 0, label: 1 }));
    expect(decodePoints(encodePoints(many))).toHaveLength(MAX_DRAWN_POINTS);
    expect(decodePoints('not base64!')).toEqual([]);
    expect(decodePoints(null)).toEqual([]);
    // Bytes outside ±6 (here 0x7f = 6.35) are rejected.
    expect(decodePoints(btoa(String.fromCharCode(127, 0, 1)))).toEqual([]);
    const state = State.deserializeState(`#dataset=custom&points=${encodePoints(points)}`);
    expect(decodePoints(state.points)).toHaveLength(3);
  });
});

describe('training on a hand-drawn dataset', () => {
  it('an empty dataset has no loss but is not reported as divergence', () => {
    const network = nn.buildNetwork([2, 2, 1], nn.Activations.TANH, nn.Activations.TANH, null, ['x', 'y']);
    const engine = new TrainingEngine({ shape: [2, 2, 1], inputIds: ['x', 'y'], visibleInputIds: ['x', 'y'],
      activation: 'tanh', regularization: 'none', problem: 'classification', initZero: false },
      captureModel(network), [], []);
    engine.step({ learningRate: .03, regularizationRate: 0, batchSize: 10, optimizer: 'adam' });
    const snapshot = engine.snapshot(10);
    expect(Number.isNaN(snapshot.lossTrain)).toBe(true);
    expect(snapshot.diverged).toBe(false);
  });

  it('snapshots carry each weight’s latest gradient', () => {
    const network = nn.buildNetwork([2, 2, 1], nn.Activations.TANH, nn.Activations.TANH, null, ['x', 'y']);
    const points = [{ x: 1, y: 1, label: 1 }, { x: -1, y: -1, label: -1 }];
    const engine = new TrainingEngine({ shape: [2, 2, 1], inputIds: ['x', 'y'], visibleInputIds: ['x', 'y'],
      activation: 'gelu', regularization: 'none', problem: 'classification', initZero: false },
      captureModel(network), points, points);
    engine.step({ learningRate: .03, regularizationRate: 0, batchSize: 1, optimizer: 'momentum' });
    const snapshot = engine.snapshot(10);
    expect(snapshot.diverged).toBe(false);
    expect(snapshot.model.links.some(link => Math.abs(link.gradient) > 0)).toBe(true);
  });
});
