import type { Example2D } from './dataset';
import type { ModelState } from './model-state';

export type ActivationName = 'relu' | 'tanh' | 'sigmoid' | 'linear';
export type RegularizationName = 'none' | 'L1' | 'L2';
export type ParameterKind = 'weight' | 'bias';

export interface NetworkConfig {
  shape: number[];
  inputIds: string[];
  visibleInputIds: string[];
  activation: ActivationName;
  regularization: RegularizationName;
  problem: 'classification' | 'regression';
  initZero: boolean;
}

export interface TrainingSettings {
  learningRate: number;
  regularizationRate: number;
  batchSize: number;
}

export interface SimulationSnapshot {
  epoch: number;
  lossTrain: number;
  lossTest: number;
  model: ModelState;
  density: number;
  boundaries: Record<string, Float64Array>;
  diverged: boolean;
}

export type WorkerCommand =
  | { type: 'init'; config: NetworkConfig; model: ModelState; trainData: Example2D[]; testData: Example2D[] }
  | { type: 'step'; settings: TrainingSettings }
  | { type: 'data'; trainData: Example2D[]; testData: Example2D[] }
  | { type: 'edit'; kind: ParameterKind; id: string; value: number };

export type WorkerRequest = WorkerCommand & { generation: number };
export type WorkerResponse =
  | { type: 'snapshot'; generation: number; snapshot: SimulationSnapshot }
  | { type: 'error'; generation: number; message: string };

export type Matrix = Array<ArrayLike<number>>;
