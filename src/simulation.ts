// Epoch/loss/grid algorithms adapted from TensorFlow Playground (Google, 2016),
// Apache-2.0. Neural Network Visualizer modifications: pure engine, typed snapshots, safe edits.
import * as nn from './nn';
import type { Example2D } from './dataset';
import { constructFeatures, INPUT_FEATURES } from './features';
import { applyModel, captureModel, type ModelState } from './model-state';
import type { NetworkConfig, ParameterKind, SimulationSnapshot, TrainingSettings } from './worker-protocol';

const activations = {
  relu: nn.Activations.RELU, tanh: nn.Activations.TANH,
  sigmoid: nn.Activations.SIGMOID, linear: nn.Activations.LINEAR,
  leakyRelu: nn.Activations.LEAKY_RELU, gelu: nn.Activations.GELU,
};
const regularizations = {
  none: null, L1: nn.RegularizationFunction.L1, L2: nn.RegularizationFunction.L2,
};

export class TrainingEngine {
  private readonly network: nn.Node[][];
  private epoch = 0;
  /** Weight updates so far with the current optimizer (Adam's bias correction counts them). */
  private updates = 0;
  private optimizer: nn.Optimizer = 'sgd';

  constructor(
    private readonly config: NetworkConfig,
    model: ModelState,
    private trainData: Example2D[],
    private testData: Example2D[],
  ) {
    this.network = nn.buildNetwork(
      config.shape, activations[config.activation],
      config.problem === 'regression' ? nn.Activations.LINEAR : nn.Activations.TANH,
      regularizations[config.regularization], config.inputIds, config.initZero,
    );
    applyModel(this.network, model);
  }

  step(settings: TrainingSettings): void {
    this.epoch++;
    const optimizer = settings.optimizer ?? 'sgd';
    if (optimizer !== this.optimizer) {
      // Momentum and moment estimates belong to the previous optimizer.
      nn.resetOptimizerState(this.network);
      this.optimizer = optimizer;
      this.updates = 0;
    }
    // A hand-drawn set can have fewer training points than one batch, which would never reach a
    // batch boundary and never train; it trains as one batch instead. The reference's datasets
    // always fill a batch, so this changes nothing there.
    const batchSize = Math.max(1, Math.min(settings.batchSize, this.trainData.length));
    this.trainData.forEach((point, i) => {
      nn.forwardProp(this.network, constructFeatures(this.config.inputIds, point.x, point.y));
      nn.backProp(this.network, point.label, nn.Errors.SQUARE);
      // Deliberately match the reference: incomplete batches carry their
      // accumulated gradients into the next epoch rather than being flushed.
      if ((i + 1) % batchSize === 0) {
        nn.updateWeights(this.network, settings.learningRate, settings.regularizationRate, optimizer, ++this.updates);
      }
    });
  }

  replaceData(train: Example2D[], test: Example2D[]): void {
    this.trainData = train;
    this.testData = test;
  }

  editParameter(kind: ParameterKind, id: string, value: number): void {
    if (!Number.isFinite(value)) throw new Error('Enter a finite number.');
    let found = false;
    nn.forEachNode(this.network, true, node => {
      if (kind === 'bias' && node.id === id) {
        node.bias = value;
        found = true;
      }
      if (kind === 'weight') {
        node.inputLinks.forEach(link => {
          if (link.id === id) {
            link.weight = value;
            found = true;
          }
        });
      }
    });
    if (!found) throw new Error('That parameter is no longer in the network.');
  }

  private loss(points: Example2D[]): number {
    let sum = 0;
    for (const point of points) {
      const output = nn.forwardProp(this.network, constructFeatures(this.config.inputIds, point.x, point.y));
      sum += nn.Errors.SQUARE.error(output, point.label);
    }
    return sum / points.length;
  }

  snapshot(density = 100): SimulationSnapshot {
    const lossTrain = this.loss(this.trainData);
    const lossTest = this.loss(this.testData);
    const model = captureModel(this.network);
    // An empty set (possible with a hand-drawn dataset) has no loss rather than a broken one.
    let diverged = (this.trainData.length > 0 && !Number.isFinite(lossTrain))
      || (this.testData.length > 0 && !Number.isFinite(lossTest))
      || model.nodes.some(node => !Number.isFinite(node.bias))
      || model.links.some(link => !Number.isFinite(link.weight));
    const boundaries: Record<string, Float64Array> = {};
    const nodes: nn.Node[] = [];
    nn.forEachNode(this.network, true, node => {
      nodes.push(node);
      boundaries[node.id] = new Float64Array(density * density);
    });
    for (const id of this.config.visibleInputIds) {
      boundaries[id] = new Float64Array(density * density);
    }
    if (!diverged) {
      for (let i = 0; i < density; i++) {
        const tx = i / (density - 1);
        const x = -6 * (1 - tx) + 6 * tx;
        for (let j = 0; j < density; j++) {
          const ty = (j - (density - 1)) / -(density - 1);
          const y = -6 * (1 - ty) + 6 * ty;
          const offset = i * density + j;
          nn.forwardProp(this.network, constructFeatures(this.config.inputIds, x, y));
          for (const node of nodes) {
            if (!Number.isFinite(node.output)) diverged = true;
            boundaries[node.id][offset] = Number.isFinite(node.output) ? node.output : 0;
          }
          for (const id of this.config.visibleInputIds) {
            boundaries[id][offset] = INPUT_FEATURES[id].f(x, y);
          }
        }
      }
    }
    return { epoch: this.epoch, lossTrain, lossTest, model, density, boundaries, diverged };
  }
}
