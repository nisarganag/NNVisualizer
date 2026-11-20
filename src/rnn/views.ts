/**
 * Drawing and wording the Recurrent page's views share: the memory panel (main.ts)
 * and the comparison with the gateless network (gates.ts) draw memory spans and name
 * characters the same way.
 */

/** The memory span traces this many characters back; earlier ones have no measurable pull. */
export const SPAN_REACH = 200;
/**
 * Influence (change in log-probability per unit of input) that fills the span bars.
 * A larger maximum rescales them; a smaller one keeps faint pulls looking faint. Most
 * guesses peak well above it (about 1 when unsure, 0.1 when 90–98% sure); guesses
 * that are all but certain peak near 0.02 and so look faint, as they should.
 */
export const SPAN_SCALE = .1;
/** Above this probability the guess hardly depends on any one character, and the span views say so. */
export const NEARLY_CERTAIN = .98;

export interface ChartColors {
  positive: string;
  negative: string;
  ink: string;
  line: string;
  /** A quiet background wash, for bands behind a chart. */
  well: string;
}

export const characterName = (c: string) => (c === ' ' ? 'space' : c === '\n' ? 'new paragraph' : c);
export const glyph = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);
export const percent = (p: number) => (p < .001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
/** A correlation to two decimals, rounding halves away from zero (−0.735 is −0.74), with a true minus sign. */
export const correlationText = (r: number) => `${r < 0 ? '−' : ''}${(Math.round(Math.abs(r) * 100) / 100).toFixed(2)}`;
/** “1 character”, “4 characters”. */
export const characters = (count: number) => (count === 1 ? 'character' : `${count} characters`);
/** Tint steps for surprise: none below 1 bit, then stronger at 3, 5 and 8 bits. */
export const surpriseLevel = (bits: number) => (bits < 1 ? 0 : bits < 3 ? 1 : bits < 5 ? 2 : bits < 8 ? 3 : 4);

/** A canvas sized to its CSS box at the screen's resolution, cleared. */
export function prepareCanvas(canvas: HTMLCanvasElement) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

/**
 * Span bars over the characters up to step t, one column per step of `steps`: violet
 * where a character pushed toward the guess and amber where it pushed away, scaled to
 * the strongest pull (at least SPAN_SCALE).
 */
export function paintSpanBars(canvas: HTMLCanvasElement, values: Float32Array | null, t: number, steps: number, colors: ChartColors) {
  const { context, width, height } = prepareCanvas(canvas);
  if (!values || t < 0 || !steps) return;
  const step = width / steps;
  let max = SPAN_SCALE;
  for (let s = 0; s <= t; s++) max = Math.max(max, Math.abs(values[s]));
  const base = Math.round(height * .72) + .5;
  context.strokeStyle = colors.line;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, base);
  context.lineTo(width, base);
  context.stroke();
  const gap = Math.min(3, step * .2);
  for (let s = 0; s <= t; s++) {
    const v = values[s] / max;
    const room = v >= 0 ? base - 2 : height - base - 1;
    const h = Math.abs(v) * room;
    if (h < .5) continue;
    context.fillStyle = v >= 0 ? colors.positive : colors.negative;
    const x = s * step + gap;
    const w = Math.max(1, step - 2 * gap);
    context.beginPath();
    if (v >= 0) context.roundRect(x, base - h, w, h, [2, 2, 0, 0]);
    else context.roundRect(x, base, w, h, [0, 0, 2, 2]);
    context.fill();
  }
}
