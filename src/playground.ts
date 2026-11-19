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
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';
import seedrandom from 'seedrandom';
import { INPUT_FEATURES } from './features';
import { applyModel, captureModel } from './model-state';
import type { Matrix, NetworkConfig, WorkerCommand, WorkerResponse } from './worker-protocol';
import { SIGNAL_COLORS, LOSS_COLORS } from './palette';
import { icon } from './studio';

// NNVisualizer modifications: worker-driven training; original studio layout,
// centered circular graph, shared colors, accessible controls and safe editing.

let mainWidth;

const RECT_SIZE = 36;
const BIAS_SIZE = 8;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

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
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(SIGNAL_COLORS)
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"), LOSS_COLORS);

let lastChartEpoch = -1;
let generation = 0;
let workerBusy = false;
let workerFailed = false;
let queuedSteps = 0;
let editingParameter = false;
const trainingWorker = new Worker(new URL('./training.worker.ts', import.meta.url), { type: 'module' });

function setStatus(message: string) {
  const element = document.querySelector<HTMLElement>('#simulation-status');
  element.textContent = message;
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

function syncAccessibility() {
  d3.select('#add-layers').property('disabled', state.numHiddenLayers >= 6);
  d3.select('#remove-layers').property('disabled', state.numHiddenLayers <= 0);
  d3.selectAll('canvas[data-dataset], canvas[data-regdataset]')
    .attr('role', 'button').attr('tabindex', 0)
    .attr('aria-label', function() { return this.parentElement.getAttribute('title') + ' dataset'; })
    .attr('aria-pressed', function() { return String(this.classList.contains('selected')); })
    .on('keydown', function() {
      const event = d3.event as KeyboardEvent;
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.click(); }
    });
  for (const id of Object.keys(INPUTS)) {
    d3.select(`#canvas-${id}`).attr('role', 'button').attr('tabindex', 0)
      .attr('aria-label', `Toggle input ${id}`).attr('aria-pressed', String(Boolean(state[id])))
      .on('keydown', function() {
        const event = d3.event as KeyboardEvent;
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); this.click(); }
      });
  }
  state.networkShape.forEach((count, index) => {
    const buttons = d3.select(`.ui-numNodes${index + 1}`).selectAll('button');
    buttons.attr('aria-label', (_d, i) => `${i === 0 ? 'Add' : 'Remove'} a neuron ${i === 0 ? 'to' : 'from'} layer ${index + 1}`)
      .property('disabled', (_d, i) => i === 0 ? count >= 8 : count <= 1);
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
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
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
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
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
      "class": "canvas"
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
      if (boundary[nodeId]) heatMap.updateBackground(boundary[nodeId], state.discretize);
      d3.select('#inspection-label').text(isInput ? INPUTS[nodeId].label.replaceAll('_', '') : nodeId === nn.getOutputNode(network).id ? 'Output' : `Neuron ${nodeId}`);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
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
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  if (!network) return;
  const root = document.querySelector<HTMLElement>('#network');
  const width = root.clientWidth;
  mainWidth = width;
  const svg = d3.select('#svg');
  svg.select('g.core').remove();
  d3.select('#network').selectAll('div.canvas, div.plus-minus-neurons, div.layer-label').remove();

  const inputIds = Object.keys(INPUTS);
  const rows = Math.max(inputIds.length, ...network.map(layer => layer.length), 1);
  const rowGap = 44;
  const top = 84;
  const height = top + (rows - 1) * rowGap + RECT_SIZE + 22;
  root.style.height = `${height}px`;
  svg.attr({ width, height });
  const container = svg.append('g').attr('class', 'core').attr('transform', 'translate(3,3)');
  const coordinates: Record<string, { cx: number; cy: number }> = {};
  const inputX = 67;
  const outputX = width - 40;
  const layerX = (index: number) => inputX + (outputX - inputX) * index / (network.length - 1);
  const nodeY = (count: number, index: number) => top + ((rows - count) / 2 + index) * rowGap + RECT_SIZE / 2;
  const label = (text: string, x: number) => d3.select('#network').append('div')
    .attr('class', 'layer-label').style('left', `${x + 3}px`).text(text);

  label('INPUTS', inputX);
  inputIds.forEach((id, index) => {
    const cy = nodeY(inputIds.length, index);
    coordinates[id] = { cx: inputX, cy };
    drawNode(inputX, cy, id, true, container);
  });

  for (let layerIndex = 1; layerIndex < network.length - 1; layerIndex++) {
    const cx = layerX(layerIndex);
    addPlusMinusControl(cx + 3, layerIndex);
    network[layerIndex].forEach((node, index) => {
      const cy = nodeY(network[layerIndex].length, index);
      coordinates[node.id] = { cx, cy };
      drawNode(cx, cy, node.id, false, container, node);
    });
  }

  label('OUTPUT', outputX);
  const output = nn.getOutputNode(network);
  const outputY = nodeY(1, 0);
  coordinates[output.id] = { cx: outputX, cy: outputY };
  drawNode(outputX, outputY, output.id, false, container, output);
  d3.select(`#canvas-${output.id}`).classed('output-node', true);
  container.append('text').attr({ x: outputX, y: outputY + RECT_SIZE / 2 + 18,
    'text-anchor': 'middle', class: 'output-label' }).text('ŷ');

  for (let layerIndex = 1; layerIndex < network.length; layerIndex++) {
    network[layerIndex].forEach(node => node.inputLinks.forEach((link, index) => {
      drawLink(link, coordinates, network, container, index === 0, index, node.inputLinks.length);
    }));
  }
  // Bias hit targets remain above connection hit targets; canvases have their own layer.
  container.selectAll('g.node').each(function() { this.parentNode.appendChild(this); });
  syncAccessibility();
}

function addPlusMinusControl(x: number, layerIndex: number): void {
  const index = layerIndex - 1;
  const control = d3.select('#network').append('div').attr('class', 'plus-minus-neurons')
    .style('left', `${x}px`);
  control.append('div').attr('class', 'layer-name').text(`LAYER ${layerIndex}`);
  const buttons = control.append('div').attr('class', `neuron-controls ui-numNodes${layerIndex}`);
  buttons.append('button').attr('class', 'node-button').html(icon('plus')).on('click', () => {
    if (state.networkShape[index] >= 8) return;
    state.networkShape[index]++;
    reset();
  });
  buttons.append('button').attr('class', 'node-button').html(icon('minus')).on('click', () => {
    if (state.networkShape[index] <= 1) return;
    state.networkShape[index]--;
    reset();
  });
  control.append('div').attr('class', 'neuron-count')
    .text(`${state.networkShape[index]} neuron${state.networkShape[index] === 1 ? '' : 's'}`);
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
      if ((d3.event as any).keyCode === 13) {
        editingParameter = false;
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    input.on('keydown', () => {
      if ((d3.event as KeyboardEvent).key === 'Escape') {
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
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
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
        data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10), state.discretize);
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

function oneStep(manual = false): void {
  if (workerBusy) {
    if (manual) queuedSteps++;
    return;
  }
  if (workerFailed) return;
  postCommand({ type: 'step', settings: {
    learningRate: +state.learningRate,
    regularizationRate: +state.regularizationRate,
    batchSize: +state.batchSize,
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
  setStatus('Preparing the network…');
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
      document.title = `${lesson.title} — NNVisualizer`;
    }
    const article = document.querySelector('#article-text');
    const body = document.createElement('div');
    body.className = 'l--body';
    body.append(...Array.from(lesson.body.childNodes));
    article.prepend(body);
  }).catch(() => { /* The default article remains available if a lesson fails. */ });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
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
