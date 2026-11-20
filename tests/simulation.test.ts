import { describe, expect, it } from 'vitest';
import { TrainingEngine } from '../src/simulation';

// Hand-derived one-node linear network: y = 2x + 1, x=3, target=5.
// Loss = (7-5)^2/2 = 2. At lr=.1, w=1.4, b=.8, prediction=5.
const config = {
  shape: [1, 1], inputIds: ['x'], visibleInputIds: ['x', 'y'],
  activation: 'linear' as const, regularization: 'none' as const,
  problem: 'regression' as const, initZero: false,
};
const model = {
  nodes: [{ id: '1', bias: 1 }],
  links: [{ id: 'x-1', weight: 2, isDead: false }],
};
const points = [{ x: 3, y: 0, label: 5 }];

describe('TrainingEngine', () => {
  it('computes the reference half-squared loss and a real gradient update', () => {
    const engine = new TrainingEngine(config, model, points, points);
    expect(engine.snapshot(10)).toMatchObject({ epoch: 0, lossTrain: 2, lossTest: 2 });
    engine.step({ learningRate: .1, regularizationRate: 0, batchSize: 1 });
    expect(engine.snapshot(10)).toMatchObject({
      epoch: 1,
      lossTrain: expect.closeTo(0, 12),
      model: {
        nodes: [{ id: '1', bias: expect.closeTo(.8, 12) }],
        links: [{ id: 'x-1', weight: expect.closeTo(1.4, 12), isDead: false }],
      },
    });
  });

  it('carries a partial batch across epochs as the reference does', () => {
    // Batches of 2 over 3 points: the third point's gradient waits for the next epoch's batch.
    const three = [...points, ...points, { x: 0, y: 0, label: 0 }];
    const engine = new TrainingEngine(config, model, three, three);
    const settings = { learningRate: .1, regularizationRate: 0, batchSize: 2 };
    engine.step(settings);
    // Only the first two points were applied: the same step as above. Flushing the third
    // (error .8 at x = 0) would have moved the bias on to .72.
    expect(engine.snapshot(10).model).toMatchObject({
      nodes: [{ bias: expect.closeTo(.8, 12) }],
      links: [{ weight: expect.closeTo(1.4, 12) }],
    });
    engine.step(settings);
    // The carried gradient joins two points the model now fits exactly: .8 averaged over 3.
    expect(engine.snapshot(10).model).toMatchObject({
      nodes: [{ bias: expect.closeTo(.8 - .1 * .8 / 3, 12) }],
      links: [{ weight: expect.closeTo(1.4, 12) }],
    });
  });

  it('trains a hand-drawn set smaller than one batch as a single batch', () => {
    const engine = new TrainingEngine(config, model, points, points);
    engine.step({ learningRate: .1, regularizationRate: 0, batchSize: 10 });
    // The same update as a batch of 1: one point, one step.
    expect(engine.snapshot(10)).toMatchObject({
      epoch: 1,
      lossTrain: expect.closeTo(0, 12),
      model: { nodes: [{ bias: expect.closeTo(.8, 12) }], links: [{ weight: expect.closeTo(1.4, 12) }] },
    });
  });

  it('edits a bias or weight without resetting the epoch', () => {
    const engine = new TrainingEngine(config, model, points, points);
    engine.step({ learningRate: .1, regularizationRate: 0, batchSize: 1 });
    engine.editParameter('weight', 'x-1', 0);
    engine.editParameter('bias', '1', 5);
    expect(engine.snapshot(10)).toMatchObject({ epoch: 1, lossTrain: 0 });
    expect(() => engine.editParameter('bias', '1', NaN)).toThrow();
  });

  it('supports a model with no inputs and no hidden layers', () => {
    const engine = new TrainingEngine(
      { ...config, shape: [0, 1], inputIds: [] },
      { nodes: [{ id: '1', bias: 0 }], links: [] },
      [{ x: 2, y: -2, label: 1 }], [{ x: 0, y: 0, label: 1 }],
    );
    engine.step({ learningRate: .5, regularizationRate: 0, batchSize: 1 });
    expect(engine.snapshot(10)).toMatchObject({ epoch: 1, lossTrain: .125, lossTest: .125 });
  });

  it('replaces examples while retaining the learned model and epoch', () => {
    const engine = new TrainingEngine(config, model, points, points);
    engine.step({ learningRate: .1, regularizationRate: 0, batchSize: 1 });
    engine.replaceData([{ x: 0, y: 0, label: .8 }], [{ x: 0, y: 0, label: 1.8 }]);
    expect(engine.snapshot(10)).toMatchObject({ epoch: 1, lossTrain: expect.closeTo(0, 12), lossTest: expect.closeTo(.5, 12) });
  });

  it('reports nonfinite training instead of returning an unusable heatmap', () => {
    const engine = new TrainingEngine(config,
      { nodes: [{ id: '1', bias: 1 }], links: [{ id: 'x-1', weight: 1e308, isDead: false }] },
      points, points);
    expect(engine.snapshot(10)).toMatchObject({ diverged: true });
  });
});
