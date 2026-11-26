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
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem,
  optimizers,
  decodePoints,
  encodePoints,
  decodeValuedPoints,
  encodeValuedPoints,
  MAX_DRAWN_POINTS,
  type DrawnPoint
} from "./state";
import {Example2D, shuffle} from "./dataset";
import { CsvFormatError, MAX_CSV_BYTES, parseDatasetCsv, sampleDatasetCsv } from './csv-data';
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';
import seedrandom from 'seedrandom';
import { INPUT_FEATURES } from './features';
import { applyModel, captureModel } from './model-state';
import type { Matrix, NetworkConfig, WorkerCommand, WorkerResponse } from './worker-protocol';
import { signalRange, THEME_CHANGE_EVENT } from './palette';
import { icon } from './studio';

// Neural Network Visualizer modifications: worker-driven training; original studio layout,
// centered circular graph, themed colors, typeset feature labels, graph motion,
// accessible controls and safe editing.

let mainWidth;

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const EASE_OUT = 'cubic-bezier(.22, 1, .36, 1)';
const EASE_SPRING = 'cubic-bezier(.34, 1.4, .64, 1)';

const RECT_SIZE = 36;
const BIAS_SIZE = 8;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
/** Neuron previews average the 100×100 grid into 25×25 cells for smooth, small heatmaps. */
const NODE_DENSITY = 25;

enum HoverType {
  BIAS, WEIGHT
}

const INPUTS = { ...INPUT_FEATURES };

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
  ["Optimizer", "optimizer"],
  ["Backpropagation view", "showBackprop"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();
// Classroom choices take effect on reload and must survive every graph redraw.
const initiallyHiddenProps = state.getHiddenProps();

// Filter out inputs that are hidden.
initiallyHiddenProps.forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: Record<string, Matrix> = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([0.65, 4])
  .clamp(true);
/** Maps signed values to the active theme's colors, interpolated perceptually. */
function buildColorScale() {
  return d3.scale.linear<string, string>()
    .domain([-1, 0, 1])
    .range(signalRange())
    .interpolate(d3.interpolateLab)
    .clamp(true);
}
let colorScale = buildColorScale();
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"), ['train', 'test']);

let lastChartEpoch = -1;
let generation = 0;
let workerBusy = false;
let workerFailed = false;
let queuedSteps = 0;
let editingParameter = false;
/** Mean gradient of each weight at its latest update, for the backprop view. */
let gradients = new Map<string, number>();
/** Play the forward-then-backward wave when the next manual step arrives. */
let waveAfterStep = false;
/** What a click on the plot places when drawing a dataset: 1 violet, -1 amber, 0 erase. */
let paintLabel = 1;
/** Drawing regression data: the value a click places, or erasing instead. */
let paintValue = .5;
let paintErase = false;
const trainingWorker = new Worker(new URL('./training.worker.ts', import.meta.url), { type: 'module' });

/** Info messages appear only if work takes noticeable time; errors appear at once. */
function setStatus(message: string, kind: 'info' | 'error' = 'error') {
  const element = document.querySelector<HTMLElement>('#simulation-status');
  element.textContent = message;
  element.dataset.kind = kind;
  element.hidden = !message;
}

function postCommand(command: WorkerCommand, invalidate = false) {
  if (invalidate) generation++;
  workerBusy = true;
  d3.select('#main-part').attr('aria-busy', 'true');
  trainingWorker.postMessage({ ...command, generation });
}

trainingWorker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
  // Resets, data changes and edits supersede older in-flight snapshots.
  if (data.generation !== generation) return;
  workerBusy = false;
  d3.select('#main-part').attr('aria-busy', 'false');
  if (data.type === 'error') {
    player.pause();
    queuedSteps = 0;
    setStatus(data.message);
    return;
  }
  const snapshot = data.snapshot;
  applyModel(network, snapshot.model);
  gradients = new Map(snapshot.model.links.map(link => [link.id, link.gradient ?? 0]));
  boundary = {};
  for (const [id, grid] of Object.entries(snapshot.boundaries)) {
    boundary[id] = Array.from({ length: snapshot.density }, (_, row) =>
      grid.subarray(row * snapshot.density, (row + 1) * snapshot.density));
  }
  iter = snapshot.epoch;
  lossTrain = snapshot.lossTrain;
  lossTest = snapshot.lossTest;
  if (snapshot.diverged) {
    player.pause();
    queuedSteps = 0;
    setStatus('Training diverged. Try a lower learning rate, a different activation, or reset the network.');
  } else {
    setStatus('');
  }
  updateUI();
  if (waveAfterStep) {
    waveAfterStep = false;
    playBackpropWave();
  }
  if (queuedSteps > 0) {
    queuedSteps--;
    oneStep();
  }
};

trainingWorker.onerror = () => {
  workerFailed = true;
  workerBusy = false;
  queuedSteps = 0;
  player.pause();
  d3.select('#main-part').attr('aria-busy', 'false');
  setStatus('The training worker could not load. Refresh the page to try again.');
};

/**
 * The event being handled. Under Vite's dev pre-bundling, d3 v3's live `d3.event`
 * binding can read as null, so fall back to the browser's current event.
 */
const currentEvent = <T extends Event>() => ((d3.event as Event | null) ?? window.event) as T;

function syncAccessibility() {
  d3.select('#add-layers').property('disabled', state.numHiddenLayers >= 6);
  d3.select('#remove-layers').property('disabled', state.numHiddenLayers <= 0);
  d3.selectAll('canvas[data-dataset], canvas[data-regdataset]')
    .attr('role', 'button').attr('tabindex', 0)
    .attr('aria-label', function() { return this.parentElement.getAttribute('title') + ' dataset'; })
    .attr('aria-pressed', function() { return String(this.classList.contains('selected')); })
    .on('keydown', function() {
      const event = currentEvent<KeyboardEvent>();
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.click(); }
    });
  for (const id of Object.keys(INPUTS)) {
    d3.select(`#canvas-${id}`).attr('role', 'button').attr('tabindex', 0)
      .attr('aria-label', `Toggle input ${id}`).attr('aria-pressed', String(Boolean(state[id])))
      .on('keydown', function() {
        const event = currentEvent<KeyboardEvent>();
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.click(); }
      });
  }
  state.networkShape.forEach((count, index) => {
    // Each stepper reads remove, then add, matching its visual order.
    const buttons = d3.select(`.ui-numNodes${index + 1}`).selectAll('button');
    buttons.attr('aria-label', (_d, i) => i === 0 ? `Remove a neuron from layer ${index + 1}` : `Add a neuron to layer ${index + 1}`)
      .property('disabled', (_d, i) => i === 0 ? count <= 1 : count >= 8);
  });
  initiallyHiddenProps.forEach(prop => d3.selectAll(`.ui-${prop}`).style('display', 'none'));
}

document.querySelector('#refresh-controls')?.addEventListener('click', () => window.location.reload());
window.addEventListener('hashchange', () => window.location.reload());
document.addEventListener('pointerdown', event => {
  if (editingParameter && !(event.target as Element).closest('#hovercard, .link-hover, rect[id^="bias-"]')) {
    editingParameter = false;
    updateHoverCard(null);
  }
});

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();

    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.

    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying)
      .attr("aria-pressed", String(isPlaying))
      .attr("aria-label", isPlaying ? "Pause training" : "Run training");
    d3.select('#play-label').text(isPlaying ? 'Pause training' : 'Run training');
    d3.select('body').classed('is-training', isPlaying);
    d3.select('#training-state').text(isPlaying ? 'Training' : iter > 0 ? 'Paused' : 'Ready to train');
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    oneStep(true);
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    postCommand({ type: 'data', trainData, testData }, true);

  });

  // Drawing a dataset: choose what a click places, then click the plot.
  d3.selectAll('#paint-controls [data-paint]').on('click', function() {
    paintLabel = Number(this.dataset.paint);
    d3.selectAll('#paint-controls [data-paint]').attr('aria-checked', function() { return String(Number(this.dataset.paint) === paintLabel); });
  });
  // Drawing regression data: the value new points get, or erase.
  const valueInput = document.querySelector<HTMLInputElement>('#paint-value');
  const syncPaintValue = () => {
    paintValue = Number(valueInput.value);
    d3.select('#paint-value-text').text(paintValue.toFixed(1));
    d3.select('#paint-value-swatch').style('background', colorScale(paintValue));
  };
  valueInput.addEventListener('input', () => {
    syncPaintValue();
    // Choosing a value means placing points with it.
    setValuePaint(false);
  });
  syncPaintValue();
  const setValuePaint = (erase: boolean) => {
    paintErase = erase;
    d3.selectAll('[data-value-paint]').attr('aria-checked', function() { return String((this.dataset.valuePaint === 'erase') === erase); });
  };
  d3.selectAll('[data-value-paint]').on('click', function() { setValuePaint(this.dataset.valuePaint === 'erase'); });
  d3.select('#clear-points').on('click', () => {
    storeDrawnPoints([]);
    updateDrawnData();
    showCsvMessage('');
  });
  document.querySelector('#heatmap').addEventListener('click', event => placePoint(event as MouseEvent));

  // Your own data from a CSV file, and a sample file that shows the format.
  const csvFile = document.querySelector<HTMLInputElement>('#csv-file');
  d3.select('#csv-upload').on('click', () => csvFile.click());
  csvFile.addEventListener('change', () => {
    const file = csvFile.files?.[0];
    // Cleared, so choosing the same file again (after fixing it) loads it again.
    csvFile.value = '';
    if (file) void loadCsvFile(file);
  });
  d3.select('#csv-sample').on('click', () => {
    const regression = state.problem === Problem.REGRESSION;
    const url = URL.createObjectURL(new Blob([sampleDatasetCsv({ regression })], { type: 'text/csv' }));
    const link = Object.assign(document.createElement('a'), { href: url, download: regression ? 'playground-regression-sample.csv' : 'playground-sample.csv' });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  let optimizerSelect = d3.select('#optimizer').on('change', function() {
    state.optimizer = optimizers[this.value];
    state.serialize();
  });
  optimizerSelect.property('value', state.optimizer);

  let backpropToggle = d3.select('#show-backprop').on('change', function() {
    state.showBackprop = this.checked;
    state.serialize();
    d3.select('.network-panel').classed('show-backprop', state.showBackprop);
  });
  backpropToggle.property('checked', state.showBackprop);
  d3.select('.network-panel').classed('show-backprop', state.showBackprop);

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    showCsvMessage('');

    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    showCsvMessage('');

    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;

    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);

    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();

    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();

    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();

    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();

    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);

    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];

    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();


  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];

    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;

    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();

    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // The graph can change width independently of the document at breakpoints.
  const graphResize = new ResizeObserver(() => {
    let newWidth = document.querySelector("#network").clientWidth;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI();
    }
  });
  graphResize.observe(document.querySelector('#network'));

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select('.workspace-footer').style('display', 'none');
    d3.select("header").style("display", "none");
    d3.select('.workspace-intro').style('display', 'none');
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  // While training, CSS animates the signal flow continuously; weight changes
  // ease in through stroke transitions instead of jumping each epoch.
  // Gradients span orders of magnitude, so scale them by a high percentile.
  const magnitudes = Array.from(gradients.values(), Math.abs).sort((a, b) => a - b);
  const gradientScale = magnitudes.length ? magnitudes[Math.floor(.9 * (magnitudes.length - 1))] || magnitudes.at(-1) : 0;
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
        // The update moves a weight against its gradient: violet up, amber down.
        const gradient = gradients.get(link.id) ?? 0;
        const strength = gradientScale > 0 ? Math.min(1, Math.abs(gradient) / gradientScale) : 0;
        container.select(`#gradient${link.source.id}-${link.dest.id}`)
            .style({
              "stroke": colorScale(gradientScale > 0 ? Math.max(-1, Math.min(1, -gradient / gradientScale)) : 0),
              "stroke-width": .8 + 3.2 * strength,
              "--strength": (.2 + .8 * strength).toFixed(3)
            });
      }
    }
  }
}

const SUBSCRIPT_DIGITS = '₀₁₂₃₄₅₆₇₈₉';
const SUPERSCRIPT_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹';

/** A feature label such as "sin(X_1)" as plain text: "sin(x₁)". */
function plainFeatureLabel(label: string): string {
  return label.toLowerCase()
    .replace(/_(\d)/g, (_match, digit) => SUBSCRIPT_DIGITS[+digit])
    .replace(/\^(\d)/g, (_match, digit) => SUPERSCRIPT_DIGITS[+digit]);
}

/** Typesets a feature label as math: italic variables, upright functions, true scripts. */
function appendMathLabel(text, label: string): void {
  const tokens = label.toLowerCase().match(/sin|cos|[a-z]|[_^]\d|[^a-z_^]+/g) ?? [label];
  let shift = 0;
  for (const token of tokens) {
    const script = token[0] === '_' || token[0] === '^';
    // Offsets in user units, so they stay consistent across font sizes.
    const target = token[0] === '_' ? 4 : token[0] === '^' ? -6 : 0;
    const tspan = text.append('tspan').text(script ? token.slice(1) : token);
    if (target !== shift) {
      tspan.attr('dy', target - shift);
      shift = target;
    }
    if (script) tspan.attr('class', 'math-script');
    else if (/^[a-z]$/.test(token)) tspan.attr('class', 'math-var');
  }
}

/** Highlights the connections into and out of a neuron, dimming the rest. */
function highlightConnections(nodeId: string | null): void {
  d3.select('#network').classed('inspecting', nodeId != null);
  d3.selectAll('#network path.link').classed('related', (link: nn.Link) =>
    nodeId != null && link != null && (link.source.id === nodeId || link.dest.id === nodeId));
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, column: number, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "data-column": column,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
      rx: RECT_SIZE / 2,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -12,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    appendMathLabel(text, label);
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: RECT_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 2,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas",
      "data-column": column
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      highlightConnections(nodeId);
      if (boundary[nodeId]) heatMap.updateBackground(boundary[nodeId], state.discretize);
      d3.select('#inspection-label').text(isInput ? `Input ${plainFeatureLabel(INPUTS[nodeId].label)}` : nodeId === nn.getOutputNode(network).id ? 'Output' : `Neuron ${nodeId}`);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      highlightConnections(null);
      const outputGrid = boundary[nn.getOutputNode(network).id];
      if (outputGrid) heatMap.updateBackground(outputGrid, state.discretize);
      d3.select('#inspection-label').text('Output');
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];

      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  } else {
    div.attr('tabindex', 0).attr('role', 'img')
      .attr('aria-label', nodeId === nn.getOutputNode(network).id ? 'Output neuron activation' : `Neuron ${nodeId} activation`);
  }
  div.on('focus', function() { div.on('mouseenter').call(this); })
    .on('blur', function() { div.on('mouseleave').call(this); });
  let nodeHeatMap = new HeatMap(RECT_SIZE, NODE_DENSITY, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

/** The drawn structure: node ids per layer plus which inputs are switched on. */
let drawnTopology = '';
function topologyOf(network: nn.Node[][]): string {
  return network.map(layer => layer.map(node => node.id).join(',')).join('|') +
    '#' + Object.keys(INPUTS).map(id => state[id] ? 1 : 0).join('');
}

type Placement = Map<string, DOMRect>;
/** Where each node and layer control currently appears, including any running animation. */
function measurePlacement(root: HTMLElement): Placement {
  const placement: Placement = new Map();
  root.querySelectorAll<HTMLElement>('div.canvas, div.plus-minus-neurons, div.layer-label')
    .forEach(element => placement.set(element.id || element.dataset.layer, element.getBoundingClientRect()));
  return placement;
}

/**
 * The first draw wires the network up column by column. Later structural edits
 * glide surviving neurons to their new places and fade the rewired connections in.
 * Redraws that keep the structure (resizes, sliders, theme changes) stay still.
 */
function animateNetwork(motion: 'enter' | 'rewire', root: HTMLElement, before: Placement): void {
  const elements = Array.from(root.querySelectorAll<HTMLElement>('div.canvas, div.plus-minus-neurons, div.layer-label'));
  if (motion === 'enter') {
    elements.forEach(element => {
      const column = Number(element.dataset.column ?? 0);
      if (element.classList.contains('canvas')) {
        element.animate([{ opacity: 0, transform: 'scale(.3)' }, { opacity: 1, transform: 'none' }],
          { duration: 620, delay: 80 + column * 90, easing: EASE_SPRING, fill: 'backwards' });
      } else {
        element.animate([{ opacity: 0 }, { opacity: 1 }],
          { duration: 500, delay: 160 + column * 90, easing: EASE_OUT, fill: 'backwards' });
      }
    });
    root.querySelectorAll<SVGPathElement>('path.link').forEach(path => {
      const length = Math.ceil(path.getTotalLength()) + 1;
      const column = Number(path.dataset.column ?? 0);
      path.animate([
        { strokeDasharray: `${length} ${length}`, strokeDashoffset: length },
        { strokeDasharray: `${length} ${length}`, strokeDashoffset: 0 },
      ], { duration: 760, delay: 180 + column * 90, easing: EASE_OUT, fill: 'backwards' });
    });
    // Labels and bias squares arrive with their neurons rather than ahead of them.
    root.querySelectorAll<SVGElement>('#svg g.node, #svg .output-label').forEach(element => {
      element.animate([{ opacity: 0 }, { opacity: 1 }],
        { duration: 420, delay: 200 + Number(element.dataset.column ?? 0) * 90, easing: EASE_OUT, fill: 'backwards' });
    });
    return;
  }
  // Read every new position before starting animations to avoid repeated layout.
  const after = elements.map(element => [element, element.getBoundingClientRect()] as const);
  for (const [element, bounds] of after) {
    const previous = before.get(element.id || element.dataset.layer);
    const centered = element.classList.contains('canvas') ? '' : 'translateX(-50%) ';
    if (!previous) {
      element.animate(element.classList.contains('canvas')
        ? [{ opacity: 0, transform: 'scale(.3)' }, { opacity: 1, transform: 'none' }]
        : [{ opacity: 0 }, { opacity: 1 }],
        { duration: 460, delay: 60, easing: EASE_SPRING, fill: 'backwards' });
      continue;
    }
    const dx = previous.left - bounds.left;
    const dy = previous.top - bounds.top;
    if (Math.abs(dx) < .5 && Math.abs(dy) < .5) continue;
    element.animate([
      { transform: `${centered}translate(${dx}px, ${dy}px)` },
      { transform: centered || 'none' },
    ], { duration: 520, easing: EASE_OUT });
  }
  // Connections, bias squares and the output label settle in once neurons are nearly in place.
  // Input labels never move, so they stay put.
  root.querySelectorAll<SVGElement>('#svg g.links, #svg g.node:not([data-column="0"]), #svg .output-label')
    .forEach(element => element.animate([{ opacity: 0 }, { opacity: 1 }],
      { duration: 360, delay: 160, easing: 'ease-out', fill: 'backwards' }));
}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  if (!network) return;
  const root = document.querySelector<HTMLElement>('#network');
  const topology = topologyOf(network);
  const motion = reducedMotion.matches || topology === drawnTopology ? null
    : drawnTopology === '' ? 'enter' : 'rewire';
  const before = motion === 'rewire' ? measurePlacement(root) : new Map();
  drawnTopology = topology;
  const width = root.clientWidth;
  mainWidth = width;
  const svg = d3.select('#svg');
  svg.select('g.core').remove();
  d3.select('#network').classed('inspecting', false)
    .selectAll('div.canvas, div.plus-minus-neurons, div.layer-label').remove();

  const inputIds = Object.keys(INPUTS);
  const rows = Math.max(inputIds.length, ...network.map(layer => layer.length), 1);
  const rowGap = 44;
  const top = 96;
  const height = top + (rows - 1) * rowGap + RECT_SIZE + 26;
  root.style.height = `${height}px`;
  svg.attr({ width, height });
  const container = svg.append('g').attr('class', 'core').attr('transform', 'translate(3,3)');
  // Connections sit beneath their hit targets and the nodes.
  const linkLayer = container.append('g').attr('class', 'links');
  // How each weight is being pushed, shown in the backprop view.
  const gradientLayer = container.append('g').attr('class', 'gradients');
  const coordinates: Record<string, { cx: number; cy: number }> = {};
  // Leaves room for the widest typeset label, sin(x₂), plus a margin.
  const inputX = 100;
  const outputX = width - 44;
  const layerX = (index: number) => inputX + (outputX - inputX) * index / (network.length - 1);
  const nodeY = (count: number, index: number) => top + ((rows - count) / 2 + index) * rowGap + RECT_SIZE / 2;
  const label = (text: string, x: number, key: string, column: number) => d3.select('#network').append('div')
    .attr({ class: 'layer-label', 'data-layer': key, 'data-column': column })
    .style('left', `${x + 3}px`).text(text);

  label('Inputs', inputX, 'inputs', 0);
  inputIds.forEach((id, index) => {
    const cy = nodeY(inputIds.length, index);
    coordinates[id] = { cx: inputX, cy };
    drawNode(inputX, cy, id, true, container, 0);
  });

  for (let layerIndex = 1; layerIndex < network.length - 1; layerIndex++) {
    const cx = layerX(layerIndex);
    addPlusMinusControl(cx + 3, layerIndex);
    network[layerIndex].forEach((node, index) => {
      const cy = nodeY(network[layerIndex].length, index);
      coordinates[node.id] = { cx, cy };
      drawNode(cx, cy, node.id, false, container, layerIndex, node);
    });
  }

  const outputColumn = network.length - 1;
  label('Output', outputX, 'output', outputColumn);
  const output = nn.getOutputNode(network);
  const outputY = nodeY(1, 0);
  coordinates[output.id] = { cx: outputX, cy: outputY };
  drawNode(outputX, outputY, output.id, false, container, outputColumn, output);
  d3.select(`#canvas-${output.id}`).classed('output-node', true);
  container.append('text').attr({ x: outputX, y: outputY + RECT_SIZE / 2 + 20,
    'text-anchor': 'middle', class: 'output-label', 'data-column': outputColumn }).text('ŷ');

  for (let layerIndex = 1; layerIndex < network.length; layerIndex++) {
    network[layerIndex].forEach(node => node.inputLinks.forEach((link, index) => {
      drawLink(link, coordinates, linkLayer, gradientLayer, container, layerIndex - 1, index, node.inputLinks.length);
    }));
  }
  // Bias hit targets remain above connection hit targets; canvases have their own layer.
  container.selectAll('g.node').each(function() { this.parentNode.appendChild(this); });
  syncAccessibility();
  // Style weights and biases before anything measures the graph, so new connections
  // start at their final width instead of transitioning from the default.
  updateWeightsUI(network, container);
  updateBiasesUI(network);
  drawCallouts(network, container, coordinates, width, height, inputX, motion);
  if (motion) animateNetwork(motion, root, before);
}

interface Box { x: number; y: number; w: number; h: number; }
const overlaps = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Where a note sits relative to what it points at; the arrow leaves from that side. */
interface CalloutSpot { box: Box; to: [number, number]; side: 'above' | 'below' | 'right'; bend: number; }

/**
 * Margin notes that point out what can be explored, placed in free space next to
 * the graph. Each note tries a few spots and is left out when none is clear of the
 * neurons and layer controls, so large networks stay uncluttered.
 */
function drawCallouts(network: nn.Node[][], container, coordinates: Record<string, { cx: number; cy: number }>,
    width: number, height: number, inputX: number, motion: 'enter' | 'rewire' | null): void {
  const root = d3.select('#network');
  root.selectAll('div.graph-callout').remove();
  const arrows = container.append('g').attr('class', 'callout-arrows');
  // Layer names and neuron steppers occupy the top band; neurons get a margin.
  const obstacles: Box[] = [{ x: 0, y: 0, w: width, h: 88 }];
  Object.values(coordinates).forEach(({ cx, cy }) => obstacles.push({ x: cx - 30, y: cy - 30, w: 60, h: 60 }));
  const delay = motion === 'enter' ? 1.1 : .35;

  const place = (html: string, spots: ((w: number, h: number) => CalloutSpot | null)[]) => {
    const note = root.append('div').attr('class', 'graph-callout').html(html);
    const element = note.node() as HTMLElement;
    const w = element.offsetWidth;
    const h = element.offsetHeight;
    for (const spotFor of spots) {
      const spot = spotFor(w, h);
      if (!spot) continue;
      // Slide sideways to stay inside the stage; the arrow still finds its target.
      const box = { ...spot.box, x: Math.max(8, Math.min(spot.box.x, width - w - 8)) };
      if (box.y < 6 || box.y + h > height - 6) continue;
      if (obstacles.some(obstacle => overlaps(obstacle, box))) continue;
      obstacles.push(box);
      // Coordinates are in graph space, which the SVG offsets by 3px.
      note.style({ left: `${box.x + 3}px`, top: `${box.y + 3}px` });
      const [x2, y2] = spot.to;
      const edgeX = Math.max(box.x + 14, Math.min(x2 - 12, box.x + w - 14));
      const [x1, y1] = spot.side === 'right' ? [box.x - 6, Math.min(y2 - 4, box.y + 14)]
        : spot.side === 'below' ? [edgeX, box.y - 6] : [edgeX, box.y + h + 6];
      const mx = (x1 + x2) / 2 + spot.bend;
      const my = (y1 + y2) / 2;
      const angle = Math.atan2(y2 - my, x2 - mx);
      const head = (turn: number) => `${x2 - 7 * Math.cos(angle + turn)},${y2 - 7 * Math.sin(angle + turn)}`;
      const arrow = arrows.append('g').attr('class', 'callout-arrow');
      arrow.append('path').attr('d', `M${x1},${y1}Q${mx},${my} ${x2},${y2}`);
      arrow.append('path').attr('d', `M${head(.5)}L${x2},${y2}L${head(-.5)}`);
      if (motion) {
        note.classed('is-entering', true).style('--callout-delay', `${delay}s`);
        arrow.classed('is-entering', true).style('--callout-delay', `${delay}s`);
      }
      return;
    }
    note.remove();
  };

  // A neuron in the last hidden layer (or the output when there are none).
  const neuronLayer = network.length > 2 ? network.length - 2 : network.length - 1;
  const neurons = network[neuronLayer].map(node => coordinates[node.id]);
  const lowest = neurons.reduce((a, b) => (b.cy > a.cy ? b : a));
  const highest = neurons.reduce((a, b) => (b.cy < a.cy ? b : a));
  place('Each circle is one <strong>neuron</strong>: its output across all the data. Hover to see it larger.', [
    (w, h) => ({ box: { x: lowest.cx - 34, y: lowest.cy + 62, w, h }, to: [lowest.cx - 9, lowest.cy + 23], side: 'below', bend: -12 }),
    (w, h) => ({ box: { x: highest.cx - 34, y: highest.cy - 62 - h, w, h }, to: [highest.cx - 9, highest.cy - 23], side: 'above', bend: -12 }),
  ]);

  // A connection into the output neuron.
  const output = nn.getOutputNode(network);
  const pointOnLink = (sourceId: string) => {
    const path = document.getElementById(`link${sourceId}-${output.id}`) as unknown as SVGPathElement | null;
    if (!path) return null;
    const point = path.getPointAtLength(path.getTotalLength() * .55);
    return { x: point.x, y: point.y };
  };
  const feeding = network[network.length - 2].filter(node => node.outputs.length);
  const ordered = feeding.map(node => ({ node, ...coordinates[node.id] })).sort((a, b) => a.cy - b.cy);
  if (ordered.length) {
    const upper = pointOnLink(ordered[0].node.id);
    const lower = pointOnLink(ordered[ordered.length - 1].node.id);
    place('Outputs are mixed by <strong>weights</strong>. Thicker lines are stronger. Click one to change it.', [
      (w, h) => upper && ({ box: { x: upper.x - 22, y: upper.y - 52 - h, w, h }, to: [upper.x, upper.y - 7], side: 'above', bend: 12 }),
      (w, h) => lower && ({ box: { x: lower.x - 22, y: lower.y + 52, w, h }, to: [lower.x, lower.y + 7], side: 'below', bend: 12 }),
    ]);
  }

  // Inputs that are switched off.
  const inactive = Object.keys(INPUTS).filter(id => !state[id]).map(id => coordinates[id]).reverse();
  if (inactive.length) {
    place('Faded <strong>inputs</strong> are switched off. Click one to feed it to the network.', inactive.map(({ cx, cy }) =>
      (w: number, h: number): CalloutSpot => ({ box: { x: cx + 58, y: cy - h / 2, w, h }, to: [cx + 25, cy], side: 'right', bend: -6 })));
  }
}

function addPlusMinusControl(x: number, layerIndex: number): void {
  const index = layerIndex - 1;
  const count = state.networkShape[index];
  const control = d3.select('#network').append('div')
    .attr({ class: 'plus-minus-neurons', 'data-layer': `layer-${layerIndex}`, 'data-column': layerIndex })
    .style('left', `${x}px`);
  control.append('div').attr('class', 'layer-name').text(`Layer ${layerIndex}`);
  control.append('div').attr('class', 'neuron-count').text(`${count} neuron${count === 1 ? '' : 's'}`);
  const buttons = control.append('div').attr('class', `neuron-controls ui-numNodes${layerIndex}`);
  buttons.append('button').attr('class', 'node-button').html(icon('minus')).on('click', () => {
    if (state.networkShape[index] <= 1) return;
    state.networkShape[index]--;
    reset();
  });
  buttons.append('button').attr('class', 'node-button').html(icon('plus')).on('click', () => {
    if (state.networkShape[index] >= 8) return;
    state.networkShape[index]++;
    reset();
  });
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (editingParameter) return;
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    player.pause();
    queuedSteps = 0;
    editingParameter = true;
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "" && Number.isFinite(+this.value)) {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        postCommand({ type: 'edit', kind: type === HoverType.WEIGHT ? 'weight' : 'bias',
          id: nodeOrLink.id, value: +this.value }, true);
      }
    });
    input.on("keypress", () => {
      if (currentEvent<KeyboardEvent>().key === 'Enter') {
        editingParameter = false;
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    input.on('keydown', () => {
      if (currentEvent<KeyboardEvent>().key === 'Escape') {
        editingParameter = false;
        updateHoverCard(null);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${Math.max(document.querySelector('.playground-scroll').scrollLeft + 8,
      Math.min(coordinates[0] + 20, document.querySelector('.playground-scroll').scrollLeft + document.querySelector('.playground-scroll').clientWidth - 200))}px`,
    "top": `${Math.max(5, Math.min(coordinates[1], document.querySelector('#network').clientHeight - 90))}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    linkLayer, gradientLayer, container, column: number, index: number, length: number) {
  let line = linkLayer.append("path");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    "data-column": column,
    d: diagonal(datum, 0)
  });
  gradientLayer.append("path").attr({
    class: "gradient-link",
    id: "gradient" + input.source.id + "-" + input.dest.id,
    "data-column": column,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .attr('data-weight-id', input.id)
    .on("mouseenter", function() {
      d3.select(`#link${input.id}`).classed('inspected', true);
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      d3.select(`#link${input.id}`).classed('inspected', false);
      updateHoverCard(null);
    });
  return line;
}

function updateUI() {
  if (!network) return;
  updateWeightsUI(network, d3.select("#svg g.core"));
  updateBiasesUI(network);
  const selectedId = selectedNodeId != null && boundary[selectedNodeId]
    ? selectedNodeId : nn.getOutputNode(network).id;
  if (boundary[selectedId]) heatMap.updateBackground(boundary[selectedId], state.discretize);
  d3.select("#network").selectAll("div.canvas")
    .each(function(data: {heatmap: HeatMap, id: string}) {
      if (boundary[data.id]) {
        data.heatmap.updateBackground(reduceMatrix(boundary[data.id], DENSITY / NODE_DENSITY), state.discretize);
      }
    });
  d3.select("#loss-train").text(Number.isFinite(lossTrain) ? lossTrain.toFixed(3) : "—");
  d3.select("#loss-test").text(Number.isFinite(lossTest) ? lossTest.toFixed(3) : "—");
  d3.select("#iter-number").text(String(iter).padStart(6, '0').replace(/\B(?=(\d{3})+(?!\d))/g, ','));
  d3.select('#chart-empty').property('hidden', iter > 0);
  d3.select('#chart-epochs').text(`${iter.toLocaleString()} epoch${iter === 1 ? '' : 's'}`);
  d3.select('#output-title').text(state.problem === Problem.REGRESSION ? 'Prediction surface' : 'Decision boundary');
  d3.select('#train-count').text(trainData.length);
  d3.select('#test-count').text(testData.length);
  d3.select('#sample-count').text(trainData.length + testData.length);
  d3.select('#seed-value').text(`seed ${state.seed}`);
  if (!d3.select('body').classed('is-training')) d3.select('#training-state').text(iter > 0 ? 'Paused' : 'Ready to train');
  if (lastChartEpoch !== iter && Number.isFinite(lossTrain) && Number.isFinite(lossTest)) {
    lineChart.addDataPoint([lossTrain, lossTest]);
    lastChartEpoch = iter;
  }
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

/**
 * One training step, slowed down: the forward pass lights the connections from
 * inputs to output, then the error signal travels back from the output.
 */
function playBackpropWave(): void {
  if (reducedMotion.matches || !network) return;
  const columns = network.length - 1;
  const forward = 110;
  document.querySelectorAll<SVGPathElement>('#network path.link').forEach(path => {
    const column = Number(path.dataset.column);
    path.animate([{ strokeOpacity: .3 }, { strokeOpacity: 1, offset: .35 }, { strokeOpacity: .3 }],
      { duration: 480, delay: column * forward, easing: 'ease-in-out' });
  });
  const backwardStart = columns * forward + 260;
  document.querySelectorAll<SVGPathElement>('#network path.gradient-link').forEach(path => {
    const column = Number(path.dataset.column);
    const length = Math.ceil(path.getTotalLength()) + 1;
    // A negative offset draws the stroke from its end, so the error runs output to input.
    path.animate([
      { strokeDasharray: `${length} ${length}`, strokeDashoffset: -length },
      { strokeDasharray: `${length} ${length}`, strokeDashoffset: 0 },
    ], { duration: 560, delay: backwardStart + (columns - 1 - column) * 150, easing: EASE_OUT, fill: 'backwards' });
  });
}

function oneStep(manual = false): void {
  if (manual && state.showBackprop) waveAfterStep = true;
  if (workerBusy) {
    if (manual) queuedSteps++;
    return;
  }
  if (workerFailed) return;
  postCommand({ type: 'step', settings: {
    learningRate: +state.learningRate,
    regularizationRate: +state.regularizationRate,
    batchSize: +state.batchSize,
    optimizer: state.optimizer,
  } });
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup = false) {
  lineChart.reset();
  lastChartEpoch = -1;
  state.serialize();
  player.pause();
  queuedSteps = 0;
  editingParameter = false;
  updateHoverCard(null);
  selectedNodeId = null;
  boundary = {};
  const suffix = state.numHiddenLayers !== 1 ? 's' : '';
  d3.select('#layers-label').text('hidden layer' + suffix);
  d3.select('#num-layers').text(state.numHiddenLayers);
  iter = 0;
  lossTrain = lossTest = NaN;
  const inputIds = constructInputIds();
  const shape = [inputIds.length, ...state.networkShape, 1];
  const outputActivation = state.problem === Problem.REGRESSION ? nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation, state.regularization, inputIds, state.initZero);
  let parameterCount = 0;
  nn.forEachNode(network, true, node => { parameterCount += 1 + node.inputLinks.length; });
  d3.select('#parameter-count').text(parameterCount.toLocaleString());
  d3.select('#feature-count').text(inputIds.length);
  d3.select('#inspection-label').text('Output');
  drawNetwork(network);
  updateUI();
  const config: NetworkConfig = {
    shape, inputIds, visibleInputIds: Object.keys(INPUTS),
    activation: getKeyFromValue(activations, state.activation) as NetworkConfig['activation'],
    regularization: getKeyFromValue(regularizations, state.regularization) as NetworkConfig['regularization'],
    problem: state.problem === Problem.REGRESSION ? 'regression' : 'classification',
    initZero: state.initZero,
  };
  setStatus('Preparing the network…', 'info');
  postCommand({ type: 'init', config, model: captureModel(network), trainData, testData }, true);
}

function initTutorial() {
  if (!state.tutorial || state.hideText) return;
  const lessons = import.meta.glob<string>('./tutorials/*.html', { query: '?raw', import: 'default' });
  const load = lessons[`./tutorials/${state.tutorial}.html`];
  // Unknown lesson names keep the complete default article and controls.
  if (!load) return;
  load().then(html => {
    const lesson = new DOMParser().parseFromString(html, 'text/html');
    if (lesson.title) {
      d3.select('#workspace-heading').text(lesson.title);
      document.title = `${lesson.title} — Neural Network Visualizer`;
    }
    const article = document.querySelector('#article-text');
    const body = document.createElement('div');
    body.className = 'l--body';
    body.append(...Array.from(lesson.body.childNodes));
    article.prepend(body);
  }).catch(() => { /* The default article remains available if a lesson fails. */ });
}

/** Thumbnail samples, kept so a theme change can repaint without drawing new random data. */
const thumbnailSamples = new Map<HTMLCanvasElement, Example2D[]>();

const isDrawing = () => (state.problem === Problem.CLASSIFICATION
  ? state.dataset === datasets.custom : state.regDataset === regDatasets['reg-custom']);
/** The current problem's hand-drawn points: classes for classification, values for regression. */
const drawnPoints = () => (state.problem === Problem.REGRESSION ? decodeValuedPoints(state.regPoints) : decodePoints(state.points));
function storeDrawnPoints(points: DrawnPoint[]): void {
  if (state.problem === Problem.REGRESSION) state.regPoints = encodeValuedPoints(points);
  else state.points = encodePoints(points);
}

/** Says how an uploaded file went: what was loaded, or what to fix. An empty text hides it. */
function showCsvMessage(text: string, kind: 'info' | 'error' = 'info'): void {
  const element = document.querySelector<HTMLElement>('#csv-message');
  element.textContent = text;
  element.dataset.kind = kind;
  element.hidden = !text;
}

/** A CSV file becomes the hand-drawn dataset, on a fresh network; a bad file leaves everything as it was. */
async function loadCsvFile(file: File): Promise<void> {
  if (file.size > MAX_CSV_BYTES) {
    showCsvMessage(`${file.name} is ${(file.size / 1e6).toFixed(1)} MB. Files up to 1 MB can be read.`, 'error');
    return;
  }
  const regression = state.problem === Problem.REGRESSION;
  let points: DrawnPoint[];
  let notes: string[];
  try {
    ({ points, notes } = parseDatasetCsv(await file.text(), { regression }));
  } catch (error) {
    if (!(error instanceof CsvFormatError)) throw error;
    showCsvMessage(`${file.name} can’t be used. ${error.message}`, 'error');
    return;
  }
  // The file becomes this problem's hand-drawn dataset.
  if (regression) state.regDataset = regDatasets['reg-custom'];
  else state.dataset = datasets.custom;
  storeDrawnPoints(points);
  const attribute = regression ? 'data-regDataset' : 'data-dataset';
  d3.selectAll(`canvas[${attribute}]`).classed('selected', false);
  d3.select(`canvas[${attribute}=${regression ? 'reg-custom' : 'custom'}]`).classed('selected', true);
  generateData();
  reset();
  renderDrawnThumbnail();
  const count = `${points.length} ${points.length === 1 ? 'point' : 'points'}`;
  showCsvMessage([`Loaded ${count} from ${file.name}. Press Run training, or click the plot to add more.`, ...notes].join(' '));
}

/** Shows the paint tools while the hand-drawn dataset is selected. */
function syncDrawMode(): void {
  const drawing = isDrawing();
  const regression = state.problem === Problem.REGRESSION;
  d3.select('#csv-hint').text(regression ? 'Columns x1, x2, value: from −1 (amber) to 1 (violet).' : 'Columns x1, x2, label: 1 is violet, −1 amber.');
  // Classification paints classes; regression paints values.
  d3.select('#paint-hint').text(regression ? 'Click the prediction surface to place points' : 'Click the decision boundary plot to place points');
  d3.select('.paint-classes').property('hidden', regression);
  d3.select('.paint-values').property('hidden', !regression);
  d3.select('#paint-controls').property('hidden', !drawing);
  d3.select('#plot-hint').property('hidden', !drawing);
  d3.select('#heatmap').classed('is-drawing', drawing);
  // Noise and regeneration only apply to generated patterns.
  d3.select('#noise').property('disabled', drawing);
  d3.select('#data-regen-button').property('disabled', drawing);
}

function placePoint(event: MouseEvent): void {
  if (!isDrawing()) return;
  const canvas = document.querySelector<HTMLCanvasElement>('#heatmap canvas');
  const bounds = canvas.getBoundingClientRect();
  const fx = (event.clientX - bounds.left) / bounds.width;
  const fy = (event.clientY - bounds.top) / bounds.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
  const x = -6 + 12 * fx;
  const y = 6 - 12 * fy;
  const points = drawnPoints();
  const regression = state.problem === Problem.REGRESSION;
  if (regression ? paintErase : paintLabel === 0) {
    let nearest = -1;
    let best = .8 * .8;
    points.forEach((point, i) => {
      const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
      if (distance < best) { best = distance; nearest = i; }
    });
    if (nearest < 0) return;
    points.splice(nearest, 1);
  } else {
    if (points.length >= MAX_DRAWN_POINTS) {
      setStatus(`You can place up to ${MAX_DRAWN_POINTS} points.`, 'info');
      return;
    }
    points.push({ x, y, label: regression ? paintValue : paintLabel });
  }
  storeDrawnPoints(points);
  updateDrawnData();
}

/** New points take effect immediately; the network keeps its weights and keeps learning. */
function updateDrawnData(): void {
  state.serialize();
  generateData(true);
  postCommand({ type: 'data', trainData, testData }, true);
  renderDrawnThumbnail();
}

/** The hand-drawn datasets' tiles: the points placed so far, or a plus inviting the first. */
function renderDrawnThumbnail(): void {
  const tiles = [
    ['canvas[data-dataset="custom"]', decodePoints(state.points)],
    ['canvas[data-regDataset="reg-custom"]', decodeValuedPoints(state.regPoints)],
  ] as const;
  for (const [selector, points] of tiles) {
    const canvas = document.querySelector<HTMLCanvasElement>(selector);
    if (!canvas) continue;
    const size = 112;
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, size, size);
    if (!points.length) {
      context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink-3').trim() || '#888';
      context.lineWidth = 6;
      context.lineCap = 'round';
      context.beginPath();
      context.moveTo(size / 2, size * .3);
      context.lineTo(size / 2, size * .7);
      context.moveTo(size * .3, size / 2);
      context.lineTo(size * .7, size / 2);
      context.stroke();
    }
    points.forEach(point => {
      context.fillStyle = colorScale(point.label);
      context.beginPath();
      context.arc(size * (point.x + 6) / 12, size * (6 - point.y) / 12, 5, 0, 2 * Math.PI);
      context.fill();
    });
  }
}

function drawDatasetThumbnails(reuseSamples = false) {
  function renderThumbnail(canvas: HTMLCanvasElement, dataGenerator: (count: number, noise: number) => Example2D[]) {
    // Drawn at twice the displayed size for sharp dots on high-density screens.
    const size = 112;
    canvas.setAttribute("width", String(size));
    canvas.setAttribute("height", String(size));
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, size, size);
    let data = reuseSamples ? thumbnailSamples.get(canvas) : undefined;
    if (!data) {
      data = dataGenerator(200, 0);
      thumbnailSamples.set(canvas, data);
    }
    data.forEach(d => {
      context.fillStyle = colorScale(d.label);
      context.beginPath();
      // y increases upward, matching the decision boundary plot.
      context.arc(size * (d.x + 6) / 12, size * (6 - d.y) / 12, 3.4, 0, 2 * Math.PI);
      context.fill();
    });
    d3.select(canvas.parentNode as Element).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      if (dataset === 'custom') {
        renderDrawnThumbnail();
        d3.select(canvas.parentNode as Element).style("display", null);
        continue;
      }
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      if (regDataset === 'reg-custom') {
        renderDrawnThumbnail();
        d3.select(canvas.parentNode as Element).style("display", null);
        continue;
      }
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "classroom-checkbox");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "classroom-checkbox-input",
      });
    if (initiallyHiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
    });
    label.append("span")
      .attr("class", "label")
      .text(text);
  });
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();

  }
  seedrandom(state.seed, { global: true });
  syncDrawMode();
  if (isDrawing()) {
    const points = drawnPoints();
    // A fixed, evenly spread assignment keeps each point in the same set as more are added.
    const isTest = (i: number) => (i * 0.6180339887498949) % 1 >= state.percTrainData / 100;
    trainData = points.filter((_point, i) => !isTest(i));
    testData = points.filter((_point, i) => isTest(i));
    heatMap.updatePoints(trainData);
    heatMap.updateTestPoints(state.showTestData ? testData : []);
    return;
  }
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();

// A theme change repaints every canvas and colored SVG mark with the new palette.
// The network keeps its structure, so this redraw does not animate.
window.addEventListener(THEME_CHANGE_EVENT, () => {
  colorScale = buildColorScale();
  heatMap.setPalette();
  drawDatasetThumbnails(true);
  drawNetwork(network);
  updateUI();
});
