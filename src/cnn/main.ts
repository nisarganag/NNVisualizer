import '@fontsource-variable/mona-sans/wdth.css';
import '../../styles.css';
import './cnn.css';
import * as d3 from 'd3';
import { initializeSite } from '../site';
import { signalRange, THEME_CHANGE_EVENT } from '../palette';
import { CNN_LAYOUT, parseCnnWeights, parseSamples, preprocessDrawing, runCnn, type CnnActivations, type CnnWeights } from './model';

initializeSite();

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const pad = $<HTMLCanvasElement>('draw-pad');
const padContext = pad.getContext('2d', { willReadFrequently: true });
const workbench = $('cnn-workbench');

let weights: CnnWeights | null = null;
let samples: Float32Array[] = [];
let activations: CnnActivations | null = null;
/** Strokes in pad-relative coordinates (0..1), so they redraw at any size or theme. */
let strokes: { x: number; y: number }[][] = [];
let example: Float32Array | null = null;
let exampleOrder: number[] = [];
let inspected: { layer: LayerName; index: number } | null = null;

type LayerName = 'conv1' | 'pool1' | 'conv2' | 'pool2';
const LAYERS: Record<LayerName, { count: number; size: number; title: string }> = {
  conv1: { count: CNN_LAYOUT.conv1.filters, size: CNN_LAYOUT.conv1.size, title: 'Convolution 1' },
  pool1: { count: CNN_LAYOUT.pool1.channels, size: CNN_LAYOUT.pool1.size, title: 'Max pooling 1' },
  conv2: { count: CNN_LAYOUT.conv2.filters, size: CNN_LAYOUT.conv2.size, title: 'Convolution 2' },
  pool2: { count: CNN_LAYOUT.pool2.channels, size: CNN_LAYOUT.pool2.size, title: 'Max pooling 2' },
};

/* ---------- Colors ---------- */

let sequential = new Uint8ClampedArray(256 * 3);
let diverging: (t: number) => string = () => '#888';
let inkRgb: [number, number, number] = [28, 30, 45];
function buildColors() {
  const [negative, neutral, positive] = signalRange();
  const scale = d3.scale.linear<string, string>().domain([0, 1]).range([neutral, positive]).interpolate(d3.interpolateLab);
  for (let i = 0; i < 256; i++) {
    const color = d3.rgb(scale(i / 255));
    sequential.set([color.r, color.g, color.b], i * 3);
  }
  const divergingScale = d3.scale.linear<string, string>().domain([-1, 0, 1]).range([negative, neutral, positive]).interpolate(d3.interpolateLab).clamp(true);
  diverging = t => divergingScale(t);
  const ink = d3.rgb(getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#1c1e2d');
  inkRgb = [ink.r, ink.g, ink.b];
}
buildColors();

function paintMap(canvas: HTMLCanvasElement, values: Float32Array, offset: number, size: number, scale: number) {
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const image = context.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const t = Math.max(0, Math.min(1, values[offset + i] / scale));
    const c = Math.round(t * 255) * 3;
    image.data.set([sequential[c], sequential[c + 1], sequential[c + 2], 255], i * 4);
  }
  context.putImageData(image, 0, 0);
}

/** Ink on a transparent ground: what the network sees. */
function paintInput(canvas: HTMLCanvasElement, input: Float32Array | null) {
  canvas.width = 28;
  canvas.height = 28;
  const context = canvas.getContext('2d');
  const image = context.createImageData(28, 28);
  if (input) for (let i = 0; i < 784; i++) image.data.set([...inkRgb, Math.round(input[i] * 255)], i * 4);
  context.putImageData(image, 0, 0);
}

function layerScale(values: Float32Array): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
  return max || 1;
}

/* ---------- Drawing pad ---------- */

function padSize() {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  return Math.round(pad.getBoundingClientRect().width * ratio);
}

function redrawPad() {
  const size = padSize();
  // Before layout the pad has no size yet; the resize observer redraws it once it does.
  if (!size) return;
  if (pad.width !== size) { pad.width = size; pad.height = size; }
  padContext.clearRect(0, 0, size, size);
  if (example) {
    const tile = document.createElement('canvas');
    paintInput(tile, example);
    padContext.imageSmoothingEnabled = true;
    padContext.drawImage(tile, 0, 0, size, size);
  }
  padContext.strokeStyle = `rgb(${inkRgb.join(' ')})`;
  padContext.lineCap = 'round';
  padContext.lineJoin = 'round';
  padContext.lineWidth = size * .075;
  for (const stroke of strokes) drawStroke(stroke);
}

function drawStroke(stroke: { x: number; y: number }[], from = 0) {
  const size = pad.width;
  padContext.beginPath();
  const start = stroke[Math.max(0, from - 1)];
  padContext.moveTo(start.x * size, start.y * size);
  for (let i = Math.max(1, from); i < stroke.length; i++) padContext.lineTo(stroke[i].x * size, stroke[i].y * size);
  // A tap leaves a dot.
  if (stroke.length === 1) padContext.lineTo(start.x * size + .01, start.y * size);
  padContext.stroke();
}

let pendingFrame = 0;
const scheduleClassify = () => { if (!pendingFrame) pendingFrame = requestAnimationFrame(() => { pendingFrame = 0; classify(); }); };

let drawing: { x: number; y: number }[] | null = null;
const point = (event: PointerEvent) => {
  const bounds = pad.getBoundingClientRect();
  return { x: (event.clientX - bounds.left) / bounds.width, y: (event.clientY - bounds.top) / bounds.height };
};
pad.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  pad.setPointerCapture(event.pointerId);
  drawing = [point(event)];
  strokes.push(drawing);
  padContext.lineWidth = pad.width * .075;
  drawStroke(drawing);
  $('pad-hint').hidden = true;
  scheduleClassify();
});
pad.addEventListener('pointermove', event => {
  if (!drawing) return;
  const from = drawing.length;
  // Coalesced events keep fast strokes smooth.
  for (const coalesced of event.getCoalescedEvents?.() ?? [event]) drawing.push(point(coalesced));
  drawStroke(drawing, from);
  scheduleClassify();
});
const endStroke = () => { if (drawing) { drawing = null; scheduleClassify(); } };
pad.addEventListener('pointerup', endStroke);
pad.addEventListener('pointercancel', endStroke);

$('clear-pad').addEventListener('click', () => {
  strokes = [];
  example = null;
  redrawPad();
  $('pad-hint').hidden = false;
  classify();
});

$('example-digit').addEventListener('click', () => {
  if (!samples.length) return;
  if (!exampleOrder.length) exampleOrder = samples.map((_sample, i) => i).sort(() => Math.random() - .5);
  example = samples[exampleOrder.pop()];
  strokes = [];
  $('pad-hint').hidden = true;
  redrawPad();
  classify();
});

/* ---------- Network views ---------- */

function buildMapTiles() {
  (Object.keys(LAYERS) as LayerName[]).forEach(layer => {
    const grid = $(`${layer}-maps`);
    for (let i = 0; i < LAYERS[layer].count; i++) {
      const tile = document.createElement('button');
      tile.type = 'button';
      tile.className = 'map-tile';
      tile.setAttribute('aria-label', `${LAYERS[layer].title}, map ${i + 1}`);
      tile.innerHTML = '<canvas aria-hidden="true"></canvas>';
      const show = () => inspect(layer, i);
      tile.addEventListener('mouseenter', show);
      tile.addEventListener('focus', show);
      tile.addEventListener('click', show);
      grid.append(tile);
    }
  });
}

function renderMaps() {
  (Object.keys(LAYERS) as LayerName[]).forEach(layer => {
    const { size } = LAYERS[layer];
    const values = activations?.[layer];
    const scale = values ? layerScale(values) : 1;
    $(`${layer}-maps`).querySelectorAll('canvas').forEach((canvas, i) => {
      if (values) paintMap(canvas, values, i * size * size, size, scale);
      else { canvas.width = 1; canvas.height = 1; canvas.getContext('2d').clearRect(0, 0, 1, 1); }
    });
  });
  const dense = $<HTMLCanvasElement>('dense-view');
  const values = activations?.dense;
  if (values) {
    const scale = layerScale(values);
    dense.width = values.length;
    dense.height = 1;
    const context = dense.getContext('2d');
    const image = context.createImageData(values.length, 1);
    values.forEach((value, i) => {
      const c = Math.round(Math.max(0, Math.min(1, value / scale)) * 255) * 3;
      image.data.set([sequential[c], sequential[c + 1], sequential[c + 2], 255], i * 4);
    });
    context.putImageData(image, 0, 0);
  } else {
    dense.getContext('2d').clearRect(0, 0, dense.width, dense.height);
  }
}

function renderPrediction() {
  const list = $('digit-bars');
  if (!list.children.length) {
    for (let digit = 0; digit < 10; digit++) {
      const item = document.createElement('li');
      item.className = 'digit-bar';
      item.innerHTML = `<span class="digit-label readout">${digit}</span><span class="bar-track"><i></i></span><span class="digit-value readout"></span>`;
      list.append(item);
    }
  }
  const probabilities = activations?.probabilities;
  let best = -1;
  if (probabilities) best = probabilities.reduce((top, p, i) => (p > probabilities[top] ? i : top), 0);
  Array.from(list.children).forEach((item, digit) => {
    const p = probabilities ? probabilities[digit] : 0;
    (item.querySelector('.bar-track i') as HTMLElement).style.width = `${p * 100}%`;
    item.querySelector('.digit-value').textContent = probabilities ? `${(p * 100).toFixed(p >= .995 || p < .001 ? 0 : 1)}%` : '';
    item.classList.toggle('is-best', digit === best);
  });
  const digit = $('predicted-digit');
  const previous = digit.textContent;
  digit.textContent = best >= 0 ? String(best) : '?';
  if (best >= 0 && previous !== digit.textContent) {
    digit.classList.remove('is-changing');
    void digit.offsetWidth;
    digit.classList.add('is-changing');
  }
  $('predicted-confidence').textContent = best >= 0 ? `${(probabilities[best] * 100).toFixed(probabilities[best] >= .995 ? 0 : 1)}% sure` : 'Draw to begin';
}

function kernelGrid(values: Float32Array, offset: number, scale: number, label: string) {
  const grid = document.createElement('div');
  grid.className = 'kernel';
  grid.setAttribute('role', 'img');
  grid.setAttribute('aria-label', label);
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('span');
    const value = values[offset + i];
    cell.style.background = diverging(value / scale);
    cell.title = value.toFixed(3);
    grid.append(cell);
  }
  return grid;
}

function inspect(layer: LayerName, index: number) {
  inspected = { layer, index };
  renderInspector();
}

function renderInspector() {
  document.querySelectorAll('.map-tile.is-inspected').forEach(tile => tile.classList.remove('is-inspected'));
  if (!inspected || !weights) return;
  const { layer, index } = inspected;
  $(`${layer}-maps`).children[index]?.classList.add('is-inspected');
  const { size, title } = LAYERS[layer];
  const map = $<HTMLCanvasElement>('inspector-map');
  if (activations) paintMap(map, activations[layer], index * size * size, size, layerScale(activations[layer]));
  else paintMap(map, new Float32Array(size * size), 0, size, 1);
  $('inspector-title').textContent = `${title}, map ${index + 1}`;
  const kernels = $('inspector-kernels');
  kernels.innerHTML = '';
  const pooled = layer === 'pool1' || layer === 'pool2';
  const conv = layer === 'conv1' || layer === 'pool1' ? weights.conv1 : weights.conv2;
  const channels = layer === 'conv1' || layer === 'pool1' ? 1 : CNN_LAYOUT.conv1.filters;
  const offset = index * channels * 9;
  let scale = 0;
  for (let i = 0; i < channels * 9; i++) scale = Math.max(scale, Math.abs(conv.weights[offset + i]));
  for (let c = 0; c < channels; c++) {
    kernels.append(kernelGrid(conv.weights, offset + c * 9, scale || 1,
      channels > 1 ? `Weights applied to map ${c + 1} of the first pooling layer` : '3 by 3 filter weights'));
  }
  kernels.classList.toggle('is-many', channels > 1);
  const bias = conv.biases[index];
  $('inspector-text').textContent = pooled
    ? `Pooling has no weights of its own. It keeps the strongest value in each 2×2 patch of the map made by this filter (bias ${bias.toFixed(3)}).`
    : channels > 1
      ? `This filter reads all 8 pooled maps, with one 3×3 grid of weights for each, plus a bias of ${bias.toFixed(3)}. Violet adds to its response; amber subtracts.`
      : `This 3×3 filter slides over the drawing. Violet weights add to its response and amber weights subtract; bias ${bias.toFixed(3)}.`;
}

function classify() {
  if (!weights) return;
  const { width, height } = pad;
  let input: Float32Array | null = null;
  if (width && height) {
    const rgba = padContext.getImageData(0, 0, width, height).data;
    const alpha = new Uint8ClampedArray(width * height);
    for (let i = 0; i < alpha.length; i++) alpha[i] = rgba[i * 4 + 3];
    input = preprocessDrawing(alpha, width, height);
  }
  activations = input ? runCnn(weights, input) : null;
  workbench.classList.toggle('is-empty', !activations);
  paintInput($<HTMLCanvasElement>('input-view'), input);
  renderMaps();
  renderPrediction();
  renderInspector();
}

function renderAll() {
  redrawPad();
  classify();
}

window.addEventListener(THEME_CHANGE_EVENT, () => { buildColors(); renderAll(); });
new ResizeObserver(() => { if (padSize() !== pad.width) renderAll(); }).observe(pad);

async function initialize() {
  buildMapTiles();
  renderPrediction();
  try {
    const [model, examples] = await Promise.all([
      fetch('/models/mnist-cnn-v1.bin').then(response => response.arrayBuffer()),
      fetch('/models/mnist-samples-v1.bin').then(response => response.arrayBuffer()),
    ]);
    weights = parseCnnWeights(model);
    samples = parseSamples(examples);
    workbench.setAttribute('aria-busy', 'false');
    inspect('conv1', 0);
    renderAll();
  } catch {
    $('predicted-confidence').textContent = 'The model could not load. Reload to try again.';
  }
}
initialize();
