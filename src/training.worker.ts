import { TrainingEngine } from './simulation';
import type { WorkerRequest, WorkerResponse } from './worker-protocol';

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<WorkerRequest>) => void;
  postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void;
};
let engine: TrainingEngine | undefined;

scope.onmessage = ({ data }) => {
  try {
    if (data.type === 'init') {
      engine = new TrainingEngine(data.config, data.model, data.trainData, data.testData);
    } else {
      if (!engine) throw new Error('Reset the network to start training.');
      if (data.type === 'step') engine.step(data.settings);
      if (data.type === 'data') engine.replaceData(data.trainData, data.testData);
      if (data.type === 'edit') engine.editParameter(data.kind, data.id, data.value);
    }
    const snapshot = engine.snapshot();
    scope.postMessage({ type: 'snapshot', generation: data.generation, snapshot },
      Object.values(snapshot.boundaries).map(grid => grid.buffer as ArrayBuffer));
  } catch (error) {
    scope.postMessage({ type: 'error', generation: data.generation,
      message: error instanceof Error ? error.message : 'Training could not continue. Reset the network to try again.' });
  }
};
