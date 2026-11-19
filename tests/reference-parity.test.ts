import { expect, it } from 'vitest';
import seedrandom from 'seedrandom';
import * as nn from '../src/nn';
import { classifyTwoGaussData, shuffle } from '../src/dataset';
import { captureModel } from '../src/model-state';
import { TrainingEngine } from '../src/simulation';

// Independent checkpoints captured from the LIVE reference on 2026-11-19.
// Reference URL: Gaussian, seed=.68936, 7 features, [8,8,8,8,8], Tanh,
// L1=.01, lr=.001, batch=13, 50% training. The 250+250 points are all in bounds.
const checkpoints = [
  { epoch: 0, train: .4829652146360863, test: .47877586970798247, weight: -.24720687243673473, bias: .1 },
  { epoch: 1, train: .4774043828900691, test: .4732137149337683, weight: -.24596917717075642, bias: .10049920608672473 },
  { epoch: 10, train: .42816492540842566, test: .42371247151434577, weight: -.23491775539933718, bias: .10510650129150328 },
];

it('matches the original website at epochs 0, 1 and 10 for the user-supplied configuration', () => {
  seedrandom('0.68936', { global: true });
  const points = classifyTwoGaussData(500, 0);
  shuffle(points);
  const ids = ['x','y','xSquared','ySquared','xTimesY','sinX','sinY'];
  const shape = [7,8,8,8,8,8,1];
  const network = nn.buildNetwork(shape, nn.Activations.TANH, nn.Activations.TANH,
    nn.RegularizationFunction.L1, ids, false);
  const engine = new TrainingEngine({ shape, inputIds: ids, visibleInputIds: ids,
    activation: 'tanh', regularization: 'L1', problem: 'classification', initZero: false },
    captureModel(network), points.slice(0,250), points.slice(250));
  for (let epoch = 0; epoch <= 10; epoch++) {
    if (epoch) engine.step({ learningRate: .001, regularizationRate: .01, batchSize: 13 });
    const expected = checkpoints.find(point => point.epoch === epoch);
    if (!expected) continue;
    const actual = engine.snapshot(10);
    expect(actual.lossTrain).toBeCloseTo(expected.train, 12);
    expect(actual.lossTest).toBeCloseTo(expected.test, 12);
    expect(actual.model.links.find(link => link.id === 'x-1')!.weight).toBeCloseTo(expected.weight, 12);
    expect(actual.model.nodes.find(node => node.id === '1')!.bias).toBeCloseTo(expected.bias, 12);
  }
});
