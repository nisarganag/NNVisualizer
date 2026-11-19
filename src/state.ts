/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import * as dataset from "./dataset";
import seedrandom from 'seedrandom';

// Modified for Neural Network Visualizer: validated, encoded URL state; supplied initial
// experiment; pure parsing for tests; replaceState avoids slider-history spam.

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";
export const STATE_CHANGE_EVENT = 'nnvisualizer:state-change';
const HIDABLE_PROPERTIES = new Set([
  'showTestData', 'discretize', 'playButton', 'stepButton', 'resetButton',
  'learningRate', 'activation', 'regularization', 'regularizationRate',
  'problem', 'dataset', 'percTrainData', 'noise', 'batchSize', 'numHiddenLayers',
  'optimizer', 'showBackprop',
  'x', 'y', 'xSquared', 'ySquared', 'xTimesY', 'sinX', 'sinY',
  'numNodes1', 'numNodes2', 'numNodes3', 'numNodes4', 'numNodes5', 'numNodes6',
]);

/** A map between names and activation functions. */
export let activations: {[key: string]: nn.ActivationFunction} = {
  "relu": nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  "sigmoid": nn.Activations.SIGMOID,
  "linear": nn.Activations.LINEAR,
  "leakyRelu": nn.Activations.LEAKY_RELU,
  "gelu": nn.Activations.GELU
};

/** How gradients become weight changes; SGD is the original playground's update. */
export let optimizers: {[key: string]: nn.Optimizer} = {
  "sgd": "sgd",
  "momentum": "momentum",
  "adam": "adam"
};

/** A map between names and regularization functions. */
export let regularizations: {[key: string]: nn.RegularizationFunction} = {
  "none": null,
  "L1": nn.RegularizationFunction.L1,
  "L2": nn.RegularizationFunction.L2
};

/** A map between dataset names and functions that generate classification data. */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "circle": dataset.classifyCircleData,
  "xor": dataset.classifyXORData,
  "gauss": dataset.classifyTwoGaussData,
  "spiral": dataset.classifySpiralData,
  // Points placed by hand live in the `points` setting, not in a generator.
  "custom": () => [],
};

/** A point placed by hand on the decision-boundary plot. */
export interface DrawnPoint { x: number; y: number; label: number; }
export const MAX_DRAWN_POINTS = 400;

/** Three bytes per point: x and y in steps of 0.05 within ±6, then the class. URL-safe base64. */
export function encodePoints(points: DrawnPoint[]): string {
  const bytes = new Uint8Array(Math.min(points.length, MAX_DRAWN_POINTS) * 3);
  points.slice(0, MAX_DRAWN_POINTS).forEach((point, i) => {
    const step = (v: number) => (Math.max(-120, Math.min(120, Math.round(v * 20))) + 256) % 256;
    bytes.set([step(point.x), step(point.y), point.label > 0 ? 1 : 0], i * 3);
  });
  let binary = '';
  bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** The inverse of encodePoints; anything malformed is dropped rather than trusted. */
export function decodePoints(text: string | null): DrawnPoint[] {
  if (!text || !/^[A-Za-z0-9_-]*$/.test(text)) return [];
  let binary: string;
  try {
    binary = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  } catch {
    return [];
  }
  const points: DrawnPoint[] = [];
  for (let i = 0; i + 2 < binary.length && points.length < MAX_DRAWN_POINTS; i += 3) {
    const signed = (code: number) => (code > 127 ? code - 256 : code);
    const x = signed(binary.charCodeAt(i)) / 20;
    const y = signed(binary.charCodeAt(i + 1)) / 20;
    if (Math.abs(x) > 6 || Math.abs(y) > 6) continue;
    points.push({ x, y, label: binary.charCodeAt(i + 2) === 1 ? 1 : -1 });
  }
  return points;
}

/** A map between dataset names and functions that generate regression data. */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX) && HIDABLE_PROPERTIES.has(prop.slice(0, -HIDE_STATE_SUFFIX.length))) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// Add the GUI state.
export class State {

  private static PROPS: Property[] = [
    {name: "activation", type: Type.OBJECT, keyMap: activations},
    {name: "regularization", type: Type.OBJECT, keyMap: regularizations},
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "regularizationRate", type: Type.NUMBER},
    {name: "noise", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "showTestData", type: Type.BOOLEAN},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "xTimesY", type: Type.BOOLEAN},
    {name: "xSquared", type: Type.BOOLEAN},
    {name: "ySquared", type: Type.BOOLEAN},
    {name: "cosX", type: Type.BOOLEAN},
    {name: "sinX", type: Type.BOOLEAN},
    {name: "cosY", type: Type.BOOLEAN},
    {name: "sinY", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN},
    {name: "optimizer", type: Type.OBJECT, keyMap: optimizers},
    {name: "showBackprop", type: Type.BOOLEAN},
    {name: "points", type: Type.STRING}
  ];

  [key: string]: any;
  // Default experiment: a small, fast-learning network on the circle dataset.
  learningRate = 0.03;
  regularizationRate = 0;
  showTestData = false;
  noise = 0;
  batchSize = 10;
  discretize = false;
  tutorial: string = null;
  percTrainData = 50;
  activation = nn.Activations.TANH;
  regularization: nn.RegularizationFunction = null;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 2;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [4, 2];
  x = true;
  y = true;
  xTimesY = false;
  xSquared = false;
  ySquared = false;
  cosX = false;
  sinX = false;
  cosY = false;
  sinY = false;
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed = '0.68936';
  optimizer: nn.Optimizer = 'sgd';
  showBackprop = false;
  /** Hand-placed points (see encodePoints); only used by the custom dataset. */
  points: string = null;

  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(hash = window.location.hash): State {
    const map: Record<string, string> = Object.fromEntries(
      new URLSearchParams(hash.replace(/^#/, '')),
    );
    let state = new State();
    const defaults = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && Object.hasOwn(keyMap, map[name])) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name) && Number.isFinite(+map[name])) {
            // The + operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name) && ['true', 'false'].includes(map[name])) {
            state[name] = map[name] === 'true';
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            const values = parseArray(map[name]).map(Number).filter(Number.isFinite);
            if (values.length || map[name].trim() === '') {
              state[name] = values.slice(0, 6).map(value => Math.max(1, Math.min(8, Math.round(value))));
            }
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });
    state.batchSize = Math.max(1, Math.min(30, Math.round(state.batchSize)));
    state.percTrainData = Math.max(10, Math.min(90, Math.round(state.percTrainData)));
    state.noise = Math.max(0, Math.min(80, state.noise));
    if (![.00001, .0001, .001, .003, .01, .03, .1, .3, 1, 3, 10].includes(state.learningRate)) {
      state.learningRate = defaults.learningRate;
    }
    if (![0, .001, .003, .01, .03, .1, .3, 1, 3, 10].includes(state.regularizationRate)) {
      state.regularizationRate = defaults.regularizationRate;
    }
    if (state.tutorial && !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(state.tutorial)) {
      state.tutorial = null;
    }
    state.seed = state.seed.slice(0, 128);
    if (state.points != null) state.points = encodePoints(decodePoints(state.points));
    state.numHiddenLayers = state.networkShape.length;
    seedrandom(state.seed, { global: true });
    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  toHash(): string {
    // Serialize regular properties.
    let props: string[] = [];
    State.PROPS.forEach(({name, type, keyMap}) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER ||
          type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${encodeURIComponent(String(value)).replace(/%2C/gi, ',')}`);
    });
    // Serialize properties that correspond to hiding UI controls.
    getHideProps(this).forEach(prop => {
      props.push(`${prop}=${this[prop]}`);
    });
    return props.join('&');
  }

  serialize(): void {
    const url = new URL(window.location.href);
    url.hash = this.toHash();
    window.history.replaceState(null, '', url);
    window.dispatchEvent(new Event(STATE_CHANGE_EVENT));
  }

  /** Returns all the hidden properties. */
  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    if (HIDABLE_PROPERTIES.has(name)) this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}
