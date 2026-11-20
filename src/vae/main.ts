import '@fontsource-variable/mona-sans/wdth.css';
import '../../styles.css';
import './vae.css';
import { initializeSite, reducedMotion } from '../site';
import { currentTheme, palette, THEME_CHANGE_EVENT } from '../palette';
import { parseCnnWeights, preprocessDrawing, runCnn, type CnnWeights } from '../cnn/model';
import {
  CLASSES, IMAGE_SIZE, PIXELS, createDecoding, createEncoding, decode, encode, nearestLabels, parseCodes, parseExamples, parseVae,
  type DigitCodes, type Vae, type VaeMeta,
} from './model';

initializeSite();

const MODEL_FILES = {
  meta: '/models/mnist-vae-v1.json',
  weights: '/models/mnist-vae-v1.bin',
  codes: '/models/mnist-vae-codes-v1.bin',
  examples: '/models/mnist-vae-examples-v1.bin',
  reader: '/models/mnist-cnn-v1.bin',
};
/** The morph the page opens with; the point starts on the first digit. */
const DEFAULT_MORPH = { from: '1', to: '0' };
/** Pictures in the morph strip, both ends included. */
const STRIP_STEPS = 7;
/** Arrow keys move the point this far on the map; with Shift, ten times as far. */
const KEY_STEP = 0.05;
/** A dot within this many CSS pixels of the pointer highlights its digit. */
const HOVER_RADIUS = 14;
/** How long the point takes to glide to a digit chosen below the map. */
const GLIDE_MS = 380;
/** Test digits counted to say where a drawing landed. */
const NEIGHBORS = 15;

type Point = [number, number];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const workbench = $('vae-workbench');
const mapFrame = $('map-frame');
const mapCanvas = $<HTMLCanvasElement>('map-canvas');
const probe = $('probe');
const decodedView = $<HTMLCanvasElement>('decoded-view');
const fromSelect = $<HTMLSelectElement>('morph-from');
const toSelect = $<HTMLSelectElement>('morph-to');
const morphButton = $<HTMLButtonElement>('morph-start');
const strip = $('morph-strip');
const pad = $<HTMLCanvasElement>('draw-pad');
const padContext = pad.getContext('2d', { willReadFrequently: true });

let model: Vae | null = null;
let reader: CnnWeights | null = null;
let codes: DigitCodes | null = null;
let examples: Float32Array[] = [];
let exampleOrder: number[] = [];
let decoding = null as ReturnType<typeof createDecoding> | null;
let encoding = null as ReturnType<typeof createEncoding> | null;
/** The square of the plane on show. */
let view = { min: -4, max: 4 };
/** Where each digit's label sits, and where morphs start and end. */
let centers: Point[] = [];
/** The point: the two numbers being decoded. */
let point: Point = [0, 0];
/** The drawing or example on the pad, as the encoder sees it, and where it landed (unclamped). */
let drawing: { input: Float32Array; code: Point } | null = null;
/** Strokes in pad-relative coordinates (0..1), so they redraw at any size or theme. */
let strokes: { x: number; y: number }[][] = [];
let example: Float32Array | null = null;
let mapView: 'codes' | 'grid' = 'codes';
/** The digit highlighted on the map: from a digit button, else from the pointer. */
let buttonDigit: number | null = null;
let pointerDigit: number | null = null;
/** Positions of the morph strip's pictures; the map shows their path once a morph has run. */
let stripPoints: Point[] = [];
let showPath = false;
/** A glide or morph in progress. */
let motion: { from: Point; to: Point; start: number; duration: number; frame: number; morph: boolean } | null = null;

/* ---------- Colors ---------- */

type Rgb = [number, number, number];
const colorProbe = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
/** Any CSS color as RGB, resolved by the canvas itself. */
function toRgb(color: string): Rgb {
  colorProbe.clearRect(0, 0, 1, 1);
  colorProbe.fillStyle = '#000';
  colorProbe.fillStyle = color;
  colorProbe.fillRect(0, 0, 1, 1);
  const [r, g, b] = colorProbe.getImageData(0, 0, 1, 1).data;
  return [r, g, b];
}

let colors = { ink: '#1c1e2d', ink2: '#4f5161', ink3: '#6c6e7b', stage: '#f9f9fc', line: 'rgb(28 30 45 / .09)', lineStrong: 'rgb(28 30 45 / .16)', positive: '#744ed7' };
let inkRgb: Rgb = [28, 30, 45];
let fontFamily = 'system-ui, sans-serif';
let mathFamily = 'Georgia, serif';
function readColors() {
  const root = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => root.getPropertyValue(name).trim() || fallback;
  colors = {
    ink: read('--ink', colors.ink),
    ink2: read('--ink-2', colors.ink2),
    ink3: read('--ink-3', colors.ink3),
    stage: read('--stage', colors.stage),
    line: read('--line', colors.line),
    lineStrong: read('--line-strong', colors.lineStrong),
    positive: palette().positive,
  };
  inkRgb = toRgb(colors.ink);
  fontFamily = read('--font-ui', fontFamily);
  mathFamily = read('--font-math', mathFamily);
}
readColors();

/** Ink on a transparent ground: alpha is the decoder's chance of ink. */
function paintDigit(canvas: HTMLCanvasElement, image: Float32Array | null) {
  if (canvas.width !== IMAGE_SIZE) {
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
  }
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  if (!image) return;
  const data = context.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let i = 0; i < PIXELS; i++) {
    data.data[i * 4] = inkRgb[0];
    data.data[i * 4 + 1] = inkRgb[1];
    data.data[i * 4 + 2] = inkRgb[2];
    data.data[i * 4 + 3] = Math.round(image[i] * 255);
  }
  context.putImageData(data, 0, 0);
}

/* ---------- Geometry ---------- */

const span = () => view.max - view.min;
const clamp = (value: number, min: number, max: number) => (value < min ? min : value > max ? max : value);
const clampToView = ([x, y]: Point): Point => [clamp(x, view.min, view.max), clamp(y, view.min, view.max)];
const inView = ([x, y]: Point) => x >= view.min && x <= view.max && y >= view.min && y <= view.max;
/** A point of the plane as a fraction of the map, from the top left. */
const toFraction = ([x, y]: Point): Point => [(x - view.min) / span(), (view.max - y) / span()];
const fromFraction = (fx: number, fy: number): Point => [view.min + fx * span(), view.max - fy * span()];
const lerp = (a: Point, b: Point, t: number): Point => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
/** Two decimals with a true minus sign, and never "−0.00". */
const format = (value: number) => (value <= -0.005 ? '−' : '') + Math.abs(value).toFixed(2);
/** Two decimals for attributes, never "-0.00". */
const fixed = (value: number) => (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);
const plural = (digit: number) => `${digit}s`;

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** What the Digits page's network reads in an image, or null until it has loaded. */
function read(image: Float32Array): { digit: number; confidence: number } | null {
  if (!reader) return null;
  const { probabilities } = runCnn(reader, image);
  const digit = argmax(probabilities);
  return { digit, confidence: probabilities[digit] };
}

/* ---------- The map ---------- */

let mapSize = 0;
const layers = {
  dots: { canvas: document.createElement('canvas'), key: '' },
  grid: { canvas: document.createElement('canvas'), key: '' },
};
let gridAlpha: { tiles: number; alpha: Uint8ClampedArray } | null = null;

/** Every test digit as a small dot of ink, cached for the size and theme. Overlaps add up, so dense areas read darker. */
function dotsLayer(pixels: number, scale: number): HTMLCanvasElement {
  const key = `${pixels}:${currentTheme()}`;
  const layer = layers.dots;
  if (layer.key === key || !codes) return layer.canvas;
  const { canvas } = layer;
  canvas.width = pixels;
  canvas.height = pixels;
  const context = canvas.getContext('2d');
  context.fillStyle = colors.ink;
  context.globalAlpha = currentTheme() === 'dark' ? 0.4 : 0.34;
  const radius = (mapSize < 440 ? 1.25 : 1.6) * scale;
  for (let n = 0; n < codes.count; n++) {
    const code: Point = [codes.codes[2 * n], codes.codes[2 * n + 1]];
    if (!inView(code)) continue;
    const [fx, fy] = toFraction(code);
    context.beginPath();
    context.arc(fx * pixels, fy * pixels, radius, 0, 2 * Math.PI);
    context.fill();
  }
  layer.key = key;
  return canvas;
}

/** Tiles across for the decoded grid: about one every 30 CSS pixels. */
const gridTiles = () => clamp(Math.round(mapSize / 30), 10, 24);

/** The decoded grid: the digit the decoder draws at the center of each tile, cached for the tile count and theme. */
function gridLayer(): HTMLCanvasElement {
  const tiles = gridTiles();
  const key = `${tiles}:${currentTheme()}`;
  const layer = layers.grid;
  if (layer.key === key || !model) return layer.canvas;
  if (gridAlpha?.tiles !== tiles) {
    const alpha = new Uint8ClampedArray(tiles * tiles * PIXELS);
    for (let row = 0; row < tiles; row++) {
      for (let column = 0; column < tiles; column++) {
        const { image } = decode(model, fromFraction((column + 0.5) / tiles, (row + 0.5) / tiles), decoding);
        const offset = (row * tiles + column) * PIXELS;
        for (let i = 0; i < PIXELS; i++) alpha[offset + i] = Math.round(image[i] * 255);
      }
    }
    gridAlpha = { tiles, alpha };
  }
  const size = tiles * IMAGE_SIZE;
  const { canvas } = layer;
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const data = context.createImageData(size, size);
  for (let row = 0; row < tiles; row++) {
    for (let column = 0; column < tiles; column++) {
      const offset = (row * tiles + column) * PIXELS;
      for (let y = 0; y < IMAGE_SIZE; y++) {
        for (let x = 0; x < IMAGE_SIZE; x++) {
          const target = ((row * IMAGE_SIZE + y) * size + column * IMAGE_SIZE + x) * 4;
          data.data[target] = inkRgb[0];
          data.data[target + 1] = inkRgb[1];
          data.data[target + 2] = inkRgb[2];
          data.data[target + 3] = gridAlpha.alpha[offset + y * IMAGE_SIZE + x];
        }
      }
    }
  }
  context.putImageData(data, 0, 0);
  layer.key = key;
  return canvas;
}

const highlighted = () => buttonDigit ?? pointerDigit;

function paintMap() {
  const size = mapFrame.clientWidth;
  if (!size || !model) return;
  mapSize = size;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const pixels = Math.round(size * ratio);
  if (mapCanvas.width !== pixels || mapCanvas.height !== pixels) {
    mapCanvas.width = pixels;
    mapCanvas.height = pixels;
  }
  const scale = pixels / size;
  const context = mapCanvas.getContext('2d');
  context.clearRect(0, 0, pixels, pixels);
  const at = (p: Point): Point => {
    const [fx, fy] = toFraction(p);
    return [fx * pixels, fy * pixels];
  };

  if (mapView === 'grid') {
    context.imageSmoothingEnabled = true;
    context.drawImage(gridLayer(), 0, 0, pixels, pixels);
  }

  // Hairline grid at whole numbers, the axes a step stronger, with their values along two edges.
  context.lineWidth = 1;
  const hairline = (value: number) => Math.round(value) + 0.5;
  for (let value = Math.ceil(view.min); value <= Math.floor(view.max); value++) {
    if (value === view.min || value === view.max) continue;
    const [x] = at([value, 0]);
    const [, y] = at([0, value]);
    context.strokeStyle = value === 0 ? colors.lineStrong : colors.line;
    context.beginPath();
    context.moveTo(hairline(x), 0);
    context.lineTo(hairline(x), pixels);
    context.moveTo(0, hairline(y));
    context.lineTo(pixels, hairline(y));
    context.stroke();
  }
  if (mapView === 'codes') {
    context.fillStyle = colors.ink3;
    context.font = `500 ${10.5 * scale}px ${fontFamily}`;
    // Values too close to an edge would collide with the axis names and with each other in the corners.
    const margin = (30 / size) * span();
    for (let value = Math.ceil(view.min + margin); value <= Math.floor(view.max - margin); value++) {
      if (value === 0) continue;
      const label = value < 0 ? `−${-value}` : String(value);
      const [x] = at([value, 0]);
      const [, y] = at([0, value]);
      context.textAlign = 'center';
      context.textBaseline = 'bottom';
      context.fillText(label, x, pixels - 6 * scale);
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      context.fillText(label, 6 * scale, y);
    }
  }

  const digit = highlighted();
  if (mapView === 'codes' && codes) {
    // Every test digit; when one digit is highlighted, the others fade and its dots turn violet.
    context.globalAlpha = digit === null ? 1 : 0.32;
    context.drawImage(dotsLayer(pixels, scale), 0, 0);
    context.globalAlpha = 1;
    if (digit !== null) {
      context.fillStyle = colors.positive;
      context.globalAlpha = 0.85;
      const radius = (mapSize < 440 ? 1.6 : 2) * scale;
      context.beginPath();
      for (let n = 0; n < codes.count; n++) {
        if (codes.labels[n] !== digit) continue;
        const code: Point = [codes.codes[2 * n], codes.codes[2 * n + 1]];
        if (!inView(code)) continue;
        const [x, y] = at(code);
        context.moveTo(x + radius, y);
        context.arc(x, y, radius, 0, 2 * Math.PI);
      }
      context.fill();
      context.globalAlpha = 1;
    }
  }

  // The morph's path, with a tick at each picture of the strip.
  if (showPath && stripPoints.length > 1) {
    const [x0, y0] = at(stripPoints[0]);
    const [x1, y1] = at(stripPoints[stripPoints.length - 1]);
    context.strokeStyle = colors.ink2;
    context.lineWidth = 1.5 * scale;
    context.lineCap = 'round';
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
    for (const step of stripPoints) {
      const [x, y] = at(step);
      context.beginPath();
      context.arc(x, y, 3 * scale, 0, 2 * Math.PI);
      context.fillStyle = colors.stage;
      context.fill();
      context.stroke();
    }
  }

  // Where the drawing landed, once the point has moved off it: a dashed ring, wide enough to
  // circle a digit's label (the key below the map names it).
  if (drawing && !onDrawing()) {
    const [x, y] = at(clampToView(drawing.code));
    context.beginPath();
    context.arc(x, y, 11 * scale, 0, 2 * Math.PI);
    context.lineWidth = 4 * scale;
    context.strokeStyle = colors.stage;
    context.stroke();
    context.setLineDash([3.2 * scale, 2.6 * scale]);
    context.lineWidth = 1.8 * scale;
    context.strokeStyle = colors.ink;
    context.stroke();
    context.setLineDash([]);
  }

  // Each digit's label at its center, last, haloed by the stage so it reads over everything.
  if (mapView === 'codes' && codes) {
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.lineJoin = 'round';
    centers.forEach((center, label) => {
      if (!inView(center)) return;
      const [x, y] = at(center);
      const strong = label === digit;
      context.font = `${strong ? 760 : 680} ${(strong ? 21 : 17) * scale}px ${fontFamily}`;
      context.lineWidth = 5 * scale;
      context.strokeStyle = colors.stage;
      context.strokeText(String(label), x, y);
      context.fillStyle = digit === null || strong ? colors.ink : colors.ink3;
      context.fillText(String(label), x, y);
    });
  }
}

/** The test digit nearest a position on the map (CSS pixels from its top left), within HOVER_RADIUS. */
function digitNear(px: number, py: number): number | null {
  if (!codes || !mapSize) return null;
  const [x, y] = fromFraction(px / mapSize, py / mapSize);
  const reach = (HOVER_RADIUS / mapSize) * span();
  let best = reach * reach;
  let label: number | null = null;
  for (let n = 0; n < codes.count; n++) {
    const dx = codes.codes[2 * n] - x;
    const dy = codes.codes[2 * n + 1] - y;
    const d = dx * dx + dy * dy;
    if (d < best) {
      best = d;
      label = codes.labels[n];
    }
  }
  return label;
}

function setPointerDigit(digit: number | null) {
  if (digit === pointerDigit) return;
  pointerDigit = digit;
  if (mapView === 'codes') paintMap();
}

/* ---------- The point ---------- */

/** Moves the point (kept on the map) and redraws everything that depends on it. */
function setPoint(next: Point) {
  const wasOnDrawing = onDrawing();
  point = clampToView(next);
  renderPoint();
  if (onDrawing() !== wasOnDrawing) paintMap();
}

/** Whether the point covers the drawing's ring, which is then left out. */
function onDrawing(): boolean {
  if (!drawing || !mapSize) return false;
  const [x, y] = toFraction(clampToView(drawing.code));
  const [px, py] = toFraction(point);
  return Math.hypot(px - x, py - y) * mapSize <= 16;
}

function renderPoint() {
  const [fx, fy] = toFraction(point);
  probe.style.left = `${fx * 100}%`;
  probe.style.top = `${fy * 100}%`;
  $('z1').textContent = format(point[0]);
  $('z2').textContent = format(point[1]);
  if (!model) return;
  const { image } = decode(model, point, decoding);
  paintDigit(decodedView, image);
  const reading = read(image);
  if (reading) {
    $('looks-digit').textContent = String(reading.digit);
    $('looks-sure').textContent = `${Math.round(reading.confidence * 100)}% sure`;
  }
  probe.setAttribute('aria-valuenow', fixed(point[0]));
  probe.setAttribute('aria-valuetext', `z1 ${format(point[0])}, z2 ${format(point[1])}${reading ? `, looks like ${reading.digit}` : ''}`);
  markStrip();
}

/* ---------- Dragging, tapping and keys ---------- */

let dragging: { id: number; offsetX: number; offsetY: number } | null = null;

function pointFromEvent(event: PointerEvent): Point {
  const bounds = mapFrame.getBoundingClientRect();
  const px = event.clientX - bounds.left - (dragging?.offsetX ?? 0);
  const py = event.clientY - bounds.top - (dragging?.offsetY ?? 0);
  return fromFraction(px / bounds.width, py / bounds.height);
}

mapFrame.addEventListener('pointerdown', event => {
  if (event.button !== 0 || !model) return;
  event.preventDefault();
  stopMotion();
  // Grabbing the point keeps it under the finger; anywhere else, the point jumps there.
  const bounds = probe.getBoundingClientRect();
  const onProbe = event.target === probe;
  dragging = {
    id: event.pointerId,
    offsetX: onProbe ? event.clientX - (bounds.left + bounds.width / 2) : 0,
    offsetY: onProbe ? event.clientY - (bounds.top + bounds.height / 2) : 0,
  };
  mapFrame.setPointerCapture(event.pointerId);
  mapFrame.classList.add('is-dragging');
  setBusy(true);
  probe.focus({ preventScroll: true });
  setPoint(pointFromEvent(event));
  const frame = mapFrame.getBoundingClientRect();
  setPointerDigit(digitNear(event.clientX - frame.left, event.clientY - frame.top));
});
mapFrame.addEventListener('pointermove', event => {
  const frame = mapFrame.getBoundingClientRect();
  if (dragging?.id === event.pointerId) {
    setPoint(pointFromEvent(event));
    setPointerDigit(digitNear(event.clientX - frame.left, event.clientY - frame.top));
  } else if (!dragging && event.pointerType === 'mouse') {
    setPointerDigit(digitNear(event.clientX - frame.left, event.clientY - frame.top));
  }
});
const endDrag = (event: PointerEvent) => {
  if (dragging?.id !== event.pointerId) return;
  dragging = null;
  mapFrame.classList.remove('is-dragging');
  setBusy(false);
  // A finger leaves no hover behind.
  if (event.pointerType !== 'mouse') setPointerDigit(null);
};
mapFrame.addEventListener('pointerup', endDrag);
mapFrame.addEventListener('pointercancel', endDrag);
mapFrame.addEventListener('pointerleave', () => { if (!dragging) setPointerDigit(null); });

probe.addEventListener('keydown', event => {
  const step = event.shiftKey ? KEY_STEP * 10 : KEY_STEP;
  const moves: Record<string, Point> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
  const move = moves[event.key];
  if (!move || !model) return;
  event.preventDefault();
  stopMotion();
  setPoint([point[0] + move[0], point[1] + move[1]]);
});

/* ---------- Glides and morphs ---------- */

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** While the point moves on its own or under a finger, screen readers hear the reading once it settles. */
function setBusy(busy: boolean) {
  $('looks-like').setAttribute('aria-busy', String(busy));
}

function animate(from: Point, to: Point, duration: number, morph: boolean) {
  stopMotion();
  setPoint(from);
  setBusy(true);
  motion = { from, to, start: performance.now(), duration, frame: 0, morph };
  const step = (now: number) => {
    if (!motion) return;
    const t = Math.min(1, (now - motion.start) / motion.duration);
    setPoint(lerp(motion.from, motion.to, easeInOut(t)));
    if (t < 1) motion.frame = requestAnimationFrame(step);
    else {
      motion = null;
      setBusy(Boolean(dragging));
      syncMorphButton();
    }
  };
  motion.frame = requestAnimationFrame(step);
  syncMorphButton();
}

function stopMotion() {
  if (!motion) return;
  cancelAnimationFrame(motion.frame);
  motion = null;
  setBusy(Boolean(dragging));
  syncMorphButton();
}

/** Glides to a point, or jumps there when motion is reduced. */
function glideTo(target: Point) {
  if (reducedMotion.matches) {
    stopMotion();
    setPoint(target);
  } else {
    animate([point[0], point[1]], clampToView(target), GLIDE_MS, false);
  }
}

function endpoint(value: string): Point | null {
  if (value === 'point') return [point[0], point[1]];
  if (value === 'drawing') return drawing ? clampToView(drawing.code) : null;
  const center = centers[Number(value)];
  return center ? [center[0], center[1]] : null;
}

/** The strip's positions for the chosen ends. */
function planMorph(): boolean {
  const from = endpoint(fromSelect.value);
  const to = endpoint(toSelect.value);
  if (!from || !to) return false;
  stripPoints = Array.from({ length: STRIP_STEPS }, (_, i) => lerp(from, to, i / (STRIP_STEPS - 1)));
  paintStrip();
  return true;
}

/**
 * Slides the point along the strip's path while the digit morphs. With reduced motion
 * the point goes straight to the end, and the strip's pictures are the steps: each one
 * moves the point there.
 */
function startMorph() {
  if (motion?.morph) {
    stopMotion();
    return;
  }
  if (!model || !planMorph()) return;
  showPath = true;
  paintMap();
  const from = stripPoints[0];
  const to = stripPoints[STRIP_STEPS - 1];
  // With reduced motion, or nowhere to go, the point just goes to the end.
  if (reducedMotion.matches || distance(from, to) < 1e-6) {
    stopMotion();
    setPoint(to);
    return;
  }
  animate(from, to, clamp(650 * distance(from, to), 1400, 3600), true);
}

function syncMorphButton() {
  const running = Boolean(motion?.morph);
  morphButton.classList.toggle('is-running', running);
  morphButton.setAttribute('aria-pressed', String(running));
  $('morph-label').textContent = running ? 'Stop' : 'Morph';
}

function buildStrip() {
  for (let i = 0; i < STRIP_STEPS; i++) {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `Step ${i + 1} of ${STRIP_STEPS}`);
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    button.append(canvas);
    button.addEventListener('click', () => {
      if (!stripPoints[i]) return;
      stopMotion();
      setPoint(stripPoints[i]);
    });
    item.append(button);
    strip.append(item);
  }
}

function paintStrip() {
  strip.querySelectorAll('button').forEach((button, i) => {
    const at = stripPoints[i];
    const canvas = button.querySelector('canvas');
    if (!at || !model) {
      paintDigit(canvas, null);
      return;
    }
    const { image } = decode(model, at, decoding);
    paintDigit(canvas, image);
    const reading = read(image);
    button.setAttribute('aria-label', `Step ${i + 1} of ${STRIP_STEPS}${reading ? `, looks like ${reading.digit}` : ''}`);
  });
  markStrip();
}

/** Marks the strip's picture the point is on, if any. */
function markStrip() {
  const tolerance = span() * 0.004;
  strip.querySelectorAll('button').forEach((button, i) => {
    const on = Boolean(stripPoints[i]) && distance(stripPoints[i], point) <= tolerance;
    if (on) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
}

for (const select of [fromSelect, toSelect]) {
  select.addEventListener('change', () => {
    stopMotion();
    showPath = false;
    planMorph();
    paintMap();
  });
}
morphButton.addEventListener('click', startMorph);

/* ---------- Controls below the map ---------- */

function buildDigitButtons() {
  const row = $('class-row');
  for (let digit = 0; digit < CLASSES; digit++) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'class-button';
    button.textContent = String(digit);
    button.setAttribute('aria-label', `Go to the ${plural(digit)}`);
    const show = () => { buttonDigit = digit; syncDigitButtons(); paintMap(); };
    const hide = () => { if (buttonDigit === digit) { buttonDigit = null; syncDigitButtons(); paintMap(); } };
    button.addEventListener('mouseenter', show);
    button.addEventListener('focus', show);
    button.addEventListener('mouseleave', hide);
    button.addEventListener('blur', hide);
    button.addEventListener('click', () => { if (centers[digit]) glideTo(centers[digit]); });
    row.append(button);
  }
}

function syncDigitButtons() {
  const digit = highlighted();
  document.querySelectorAll('.class-button').forEach((button, i) => button.classList.toggle('is-highlighted', i === digit));
}

document.querySelectorAll<HTMLButtonElement>('.segmented [data-view]').forEach(button => {
  button.addEventListener('click', () => {
    mapView = button.dataset.view === 'grid' ? 'grid' : 'codes';
    document.querySelectorAll<HTMLButtonElement>('.segmented [data-view]').forEach(other => {
      other.setAttribute('aria-checked', String(other === button));
    });
    mapFrame.classList.toggle('is-grid', mapView === 'grid');
    paintMap();
  });
});

/* ---------- Drawing pad ---------- */

function padSize() {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  return Math.round(pad.getBoundingClientRect().width * ratio);
}

function redrawPad() {
  const size = padSize();
  // Before layout the pad has no size yet; the resize observer redraws it once it does.
  if (!size) return;
  if (pad.width !== size) {
    pad.width = size;
    pad.height = size;
  }
  padContext.clearRect(0, 0, size, size);
  if (example) {
    const tile = document.createElement('canvas');
    paintDigit(tile, example);
    padContext.imageSmoothingEnabled = true;
    padContext.drawImage(tile, 0, 0, size, size);
  }
  padContext.strokeStyle = `rgb(${inkRgb.join(' ')})`;
  padContext.lineCap = 'round';
  padContext.lineJoin = 'round';
  padContext.lineWidth = size * 0.075;
  for (const stroke of strokes) drawStroke(stroke);
}

function drawStroke(stroke: { x: number; y: number }[], from = 0) {
  const size = pad.width;
  padContext.beginPath();
  const start = stroke[Math.max(0, from - 1)];
  padContext.moveTo(start.x * size, start.y * size);
  for (let i = Math.max(1, from); i < stroke.length; i++) padContext.lineTo(stroke[i].x * size, stroke[i].y * size);
  // A tap leaves a dot.
  if (stroke.length === 1) padContext.lineTo(start.x * size + 0.01, start.y * size);
  padContext.stroke();
}

let padFrame = 0;
const scheduleRead = () => { if (!padFrame) padFrame = requestAnimationFrame(() => { padFrame = 0; readDrawing(); }); };

let stroke: { x: number; y: number }[] | null = null;
const padPoint = (event: PointerEvent) => {
  const bounds = pad.getBoundingClientRect();
  return { x: (event.clientX - bounds.left) / bounds.width, y: (event.clientY - bounds.top) / bounds.height };
};
pad.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  pad.setPointerCapture(event.pointerId);
  stroke = [padPoint(event)];
  strokes.push(stroke);
  padContext.lineWidth = pad.width * 0.075;
  drawStroke(stroke);
  $('pad-hint').hidden = true;
  scheduleRead();
});
pad.addEventListener('pointermove', event => {
  if (!stroke) return;
  const from = stroke.length;
  // Coalesced events keep fast strokes smooth.
  for (const coalesced of event.getCoalescedEvents?.() ?? [event]) stroke.push(padPoint(coalesced));
  drawStroke(stroke, from);
  scheduleRead();
});
const endStroke = () => { if (stroke) { stroke = null; scheduleRead(); } };
pad.addEventListener('pointerup', endStroke);
pad.addEventListener('pointercancel', endStroke);

/** Encodes what is on the pad and moves the point to where it lands. */
function readDrawing() {
  if (!model) return;
  let input: Float32Array | null = null;
  if (example && !strokes.length) {
    // An untouched example is used exactly as stored.
    input = example;
  } else if (pad.width && pad.height) {
    const rgba = padContext.getImageData(0, 0, pad.width, pad.height).data;
    const alpha = new Uint8ClampedArray(pad.width * pad.height);
    for (let i = 0; i < alpha.length; i++) alpha[i] = rgba[i * 4 + 3];
    input = preprocessDrawing(alpha, pad.width, pad.height);
  }
  if (input) {
    const { mean } = encode(model, input, encoding);
    drawing = { input, code: [mean[0], mean[1]] };
    stopMotion();
    setPoint(drawing.code);
  } else {
    drawing = null;
  }
  syncDrawingOptions();
  paintPadViews();
  paintMap();
}

/** Where the drawing landed, in words: among which test digits, or between which two. */
function describeDrawing(): string {
  if (!drawing || !codes) return 'Draw a digit to see where it lands.';
  const [x, y] = drawing.code;
  const neighbors = nearestLabels(codes, x, y, NEIGHBORS);
  const votes = new Array(CLASSES).fill(0);
  for (const label of neighbors) votes[label]++;
  const ranked = votes.map((count, digit) => ({ count, digit })).sort((a, b) => b.count - a.count);
  const [first, second] = ranked;
  if (!inView(drawing.code)) return `It lands past the edge of the map, beyond the ${plural(first.digit)}. The point stops at the edge.`;
  const where = first.count >= 0.6 * neighbors.length || second.count === 0
    ? `among the ${plural(first.digit)}`
    : `between the ${plural(first.digit)} and the ${plural(second.digit)}`;
  return `It lands ${where}, at ${format(x)}, ${format(y)}.`;
}

function paintPadViews() {
  paintDigit($<HTMLCanvasElement>('input-view'), drawing?.input ?? null);
  paintDigit($<HTMLCanvasElement>('reconstruction-view'), drawing && model ? decode(model, drawing.code, decoding).image : null);
  $('drawing-readout').textContent = describeDrawing();
}

/** "Your drawing" is a morph end only while there is one. */
function syncDrawingOptions() {
  $('drawing-key').hidden = !drawing;
  for (const select of [fromSelect, toSelect]) {
    const option = select.querySelector<HTMLOptionElement>('option[value="drawing"]');
    option.disabled = !drawing;
    if (!drawing && select.value === 'drawing') {
      select.value = select === fromSelect ? 'point' : DEFAULT_MORPH.to;
      showPath = false;
      planMorph();
    }
  }
}

$('clear-pad').addEventListener('click', () => {
  strokes = [];
  example = null;
  drawing = null;
  redrawPad();
  $('pad-hint').hidden = false;
  syncDrawingOptions();
  paintPadViews();
  paintMap();
});

$('example-digit').addEventListener('click', () => {
  if (!examples.length) return;
  if (!exampleOrder.length) exampleOrder = examples.map((_example, i) => i).sort(() => Math.random() - 0.5);
  example = examples[exampleOrder.pop()];
  strokes = [];
  $('pad-hint').hidden = true;
  redrawPad();
  readDrawing();
});

/* ---------- Themes, sizes and loading ---------- */

function renderAll() {
  paintMap();
  renderPoint();
  paintStrip();
  redrawPad();
  paintPadViews();
}

window.addEventListener(THEME_CHANGE_EVENT, () => {
  readColors();
  layers.dots.key = '';
  layers.grid.key = '';
  renderAll();
});
new ResizeObserver(() => paintMap()).observe(mapFrame);
new ResizeObserver(() => { if (padSize() !== pad.width) redrawPad(); }).observe(pad);
document.fonts?.ready.then(() => paintMap());

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.arrayBuffer();
}

async function initialize() {
  buildDigitButtons();
  buildStrip();
  // The Digits page's network says what the point looks like; the page works without it.
  const readerLoad = fetchBuffer(MODEL_FILES.reader).then(parseCnnWeights).catch(() => null);
  try {
    const [meta, weights, codeBytes, exampleBytes] = await Promise.all([
      fetch(MODEL_FILES.meta).then(response => {
        if (!response.ok) throw new Error(`${response.status}`);
        return response.json() as Promise<VaeMeta>;
      }),
      fetchBuffer(MODEL_FILES.weights),
      fetchBuffer(MODEL_FILES.codes),
      fetchBuffer(MODEL_FILES.examples),
    ]);
    const parsed = parseVae(meta, weights);
    if (parsed.latentSize !== 2) throw new Error('The map needs a model with two numbers per digit.');
    const { view: extent, classCenters } = meta;
    if (!(extent?.max > extent?.min) || classCenters?.length !== CLASSES) throw new Error('The model has no map extent or class centers.');
    model = parsed;
    decoding = createDecoding(model);
    encoding = createEncoding(model);
    codes = parseCodes(codeBytes, meta.codes.count);
    examples = parseExamples(exampleBytes, meta.examples.count);
    view = { min: extent.min, max: extent.max };
    centers = classCenters.map(([x, y]) => [x, y] as Point);
    probe.setAttribute('aria-valuemin', String(view.min));
    probe.setAttribute('aria-valuemax', String(view.max));
    const count = codes.count.toLocaleString('en-US');
    $('code-count').textContent = count;
    $('credits-count').textContent = count;
    $('parameter-total').textContent = model.parameters.length.toLocaleString('en-US');
    const knn = (meta.scores?.knn as { leaveOneOutOnMap?: number } | undefined)?.leaveOneOutOnMap;
    if (typeof knn === 'number') $('knn-score').textContent = `${Math.round(knn * 100)}%`;
    fromSelect.value = DEFAULT_MORPH.from;
    toSelect.value = DEFAULT_MORPH.to;
    point = clampToView(centers[Number(DEFAULT_MORPH.from)]);
    planMorph();
    morphButton.disabled = false;
    $<HTMLButtonElement>('example-digit').disabled = false;
    workbench.classList.remove('is-loading');
    workbench.setAttribute('aria-busy', 'false');
    renderAll();
  } catch {
    $('looks-sure').textContent = 'The model could not load. Reload to try again.';
    return;
  }
  reader = await readerLoad;
  $('decoded-heading').closest('.decoded-panel').classList.toggle('no-reader', !reader);
  renderPoint();
  paintStrip();
}
initialize();
