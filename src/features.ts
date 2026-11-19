// Feature definitions adapted from TensorFlow Playground, Copyright 2016 Google Inc.
// Licensed under Apache-2.0. Extracted into a shared UI/worker module for Neural Network Visualizer.
export interface InputFeature {
  f: (x: number, y: number) => number;
  label: string;
}

export const INPUT_FEATURES: Record<string, InputFeature> = {
  x: { f: x => x, label: 'X_1' },
  y: { f: (_x, y) => y, label: 'X_2' },
  xSquared: { f: x => x * x, label: 'X_1^2' },
  ySquared: { f: (_x, y) => y * y, label: 'X_2^2' },
  xTimesY: { f: (x, y) => x * y, label: 'X_1X_2' },
  sinX: { f: x => Math.sin(x), label: 'sin(X_1)' },
  sinY: { f: (_x, y) => Math.sin(y), label: 'sin(X_2)' },
};

export function constructFeatures(ids: string[], x: number, y: number): number[] {
  return ids.map(id => INPUT_FEATURES[id].f(x, y));
}
