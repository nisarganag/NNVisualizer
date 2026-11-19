import { describe, expect, it } from 'vitest';
import { State, Problem, datasets, activations, regularizations } from '../src/state';

describe('shareable experiment state', () => {
  it('restores the supplied experiment including all seven active features', () => {
    const state = State.deserializeState('#activation=tanh&regularization=L1&batchSize=13&dataset=gauss&learningRate=0.001&regularizationRate=0.01&networkShape=8,8,8,8,8&seed=0.68936&percTrainData=50&x=true&y=true&xSquared=true&ySquared=true&xTimesY=true&sinX=true&sinY=true');
    expect(state).toMatchObject({ batchSize: 13, networkShape: [8,8,8,8,8], seed: '0.68936',
      learningRate: .001, regularizationRate: .01, percTrainData: 50, numHiddenLayers: 5,
      x: true, y: true, xSquared: true, ySquared: true, xTimesY: true, sinX: true, sinY: true });
    expect(state.dataset).toBe(datasets.gauss);
    expect(state.activation).toBe(activations.tanh);
    expect(state.regularization).toBe(regularizations.L1);
  });

  it('round-trips escaped seeds, zero hidden layers, display flags and teaching controls', () => {
    const state = State.deserializeState('#problem=regression&networkShape=&seed=lesson%26one%3D2&showTestData=true&discretize=true&initZero=true&hideText=true&learningRate_hide=true');
    const restored = State.deserializeState('#' + state.toHash());
    expect(restored).toMatchObject({ problem: Problem.REGRESSION, networkShape: [], numHiddenLayers: 0,
      seed: 'lesson&one=2', showTestData: true, discretize: true, initZero: true, hideText: true });
    expect(restored.getHiddenProps()).toContain('learningRate');
  });

  it('bounds expensive or invalid settings before a network is constructed', () => {
    const state = State.deserializeState('#networkShape=9999,8,8,8,8,8,8&batchSize=0&percTrainData=100&noise=-4&learningRate=Infinity&regularizationRate=NaN&activation=__proto__&dataset=__proto__');
    expect(state.networkShape).toEqual([8,8,8,8,8,8]);
    expect(state.batchSize).toBe(1);
    expect(state.percTrainData).toBe(90);
    expect(state.noise).toBe(0);
    expect(state.learningRate).toBe(.001);
    expect(state.regularizationRate).toBe(.01);
    expect(state.activation).toBe(activations.tanh);
    expect(state.dataset).toBe(datasets.gauss);
  });

  it('retains all-off features and ignores malformed lessons and unknown hiding selectors', () => {
    const state = State.deserializeState('#x=false&y=false&xSquared=false&ySquared=false&xTimesY=false&sinX=false&sinY=false&tutorial=../../index&%23main-part_hide=true');
    expect([state.x,state.y,state.xSquared,state.ySquared,state.xTimesY,state.sinX,state.sinY]).toEqual(Array(7).fill(false));
    expect(state.tutorial).toBeNull();
    expect(state.getHiddenProps()).toEqual([]);
  });
});
