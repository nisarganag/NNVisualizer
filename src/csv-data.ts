// Your own data as a CSV file: one point per row, x1, x2 and then a label (classification)
// or a value (regression), like the sample files.
import { MAX_DRAWN_POINTS, type DrawnPoint } from './state';

/** The largest file read, far more than 400 points need. */
export const MAX_CSV_BYTES = 1_000_000;
/** The plot runs from -6 to 6; rescaled coordinates keep a small margin inside it. */
const PLOT_LIMIT = 6;
const RESCALED_LIMIT = 5.5;
const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** A file that can't become a dataset; the message says what to fix. */
export class CsvFormatError extends Error {}

export interface CsvDataset {
  points: DrawnPoint[];
  /** Things worth knowing about how the file was read, such as rescaling. */
  notes: string[];
}

export interface CsvOptions {
  /** Regression: the third column is a value to predict, from -1 to 1, rather than a class. */
  regression?: boolean;
  maxPoints?: number;
}

const formatCount = (n: number) => n.toLocaleString('en-US');
/** A number as a message quotes it: at most four significant digits. */
const short = (n: number) => +n.toPrecision(4);
/** A value quoted in a message, shortened so a binary file can't flood it. */
const quoted = (text: string) => `“${text.length > 24 ? `${text.slice(0, 24)}…` : text}”`;

function splitRow(line: string, delimiter: string): string[] {
  let fields = line.split(delimiter).map(field => field.trim().replace(/^"(.*)"$/, '$1').trim());
  // Spreadsheets that separate values with semicolons or tabs often write 1,5 for 1.5.
  if (delimiter !== ',') fields = fields.map(field => field.replace(/^([+-]?\d*),(\d+)$/, '$1.$2'));
  // A trailing delimiter, as some spreadsheets write, adds an empty last field.
  while (fields.length > 3 && fields[fields.length - 1] === '') fields.pop();
  return fields;
}

function label(text: string, line: number): number {
  const value = NUMBER.test(text) ? Number(text) : NaN;
  if (value === 1) return 1;
  if (value === -1 || value === 0) return -1;
  throw new CsvFormatError(`Line ${line}: the label is ${quoted(text)}. Use 1 for violet and -1 for amber (0 also means amber).`);
}

/** The smallest and largest of a column, with a loop: Math.min(...values) can overflow the stack for a large file. */
function range(values: number[]): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  return [low, high];
}

/** Stretches or squeezes values from [low, high] onto [-limit, limit], like normalizing a feature. */
const rescale = (value: number, [low, high]: [number, number], limit: number) =>
  high > low ? ((value - (low + high) / 2) / ((high - low) / 2)) * limit : 0;

/**
 * Reads a CSV of x1, x2 and a label (1 or -1) or, for regression, a value. The first row
 * may be a header; commas, semicolons or tabs separate values. Coordinates outside the
 * plot's -6 to 6 are rescaled to fit, as are regression values outside -1 to 1, and a
 * file with more points than a dataset holds keeps an evenly spread subset.
 */
export function parseDatasetCsv(text: string, { regression = false, maxPoints = MAX_DRAWN_POINTS }: CsvOptions = {}): CsvDataset {
  const third = regression ? 'value' : 'label';
  const lines = text.replace(/^﻿/, '').split(/\r\n|\n|\r/);
  const rows: { x: number; y: number; label: number }[] = [];
  let delimiter = '';
  let first = true;
  for (const [index, raw] of lines.entries()) {
    const line = index + 1;
    if (!raw.trim()) continue;
    if (!delimiter) delimiter = raw.includes(';') ? ';' : raw.includes('\t') ? '\t' : ',';
    const fields = splitRow(raw, delimiter);
    const numeric = fields.map(field => NUMBER.test(field));
    if (first && !numeric.some(Boolean)) {
      // A header row, such as "x1,x2,label".
      first = false;
      if (fields.length !== 3) {
        throw new CsvFormatError(`The first row has ${fields.length} ${fields.length === 1 ? 'column' : 'columns'}. A dataset needs 3: x1, x2 and ${third}.`);
      }
      continue;
    }
    first = false;
    if (fields.length !== 3) {
      throw new CsvFormatError(`Line ${line} has ${fields.length} ${fields.length === 1 ? 'value' : 'values'}. Each row needs 3: x1, x2 and ${third}.`);
    }
    const bad = fields.findIndex((_field, i) => i < (regression ? 3 : 2) && !numeric[i]);
    if (bad >= 0) {
      throw new CsvFormatError(`Line ${line}: ${['x1', 'x2', 'the value'][bad]} is ${quoted(fields[bad] || 'empty')}, not a number.`);
    }
    rows.push({ x: Number(fields[0]), y: Number(fields[1]), label: regression ? Number(fields[2]) : label(fields[2], line) });
  }
  if (!rows.length) {
    throw new CsvFormatError(`No points found. Each row needs x1, x2 and ${third}, as in the sample file.`);
  }

  const notes: string[] = [];
  const rescaled: string[] = [];
  for (const axis of ['x', 'y'] as const) {
    const span = range(rows.map(row => row[axis]));
    if (span[0] >= -PLOT_LIMIT && span[1] <= PLOT_LIMIT) continue;
    for (const row of rows) row[axis] = rescale(row[axis], span, RESCALED_LIMIT);
    rescaled.push(`${axis === 'x' ? 'x1' : 'x2'} ran from ${short(span[0])} to ${short(span[1])}`);
  }
  if (rescaled.length) {
    notes.push(`${rescaled.join(' and ').replace(' and x2 ran', ' and x2')}, so ${rescaled.length > 1 ? 'both were' : 'it was'} rescaled to fit the plot.`);
  }
  if (regression) {
    // The output's colors, and a comfortable size for its errors, run from -1 to 1.
    const span = range(rows.map(row => row.label));
    if (span[0] < -1 || span[1] > 1) {
      for (const row of rows) row.label = rescale(row.label, span, 1);
      notes.push(`The values ran from ${short(span[0])} to ${short(span[1])}, so they were rescaled to -1 to 1.`);
    }
  }

  let points: DrawnPoint[] = rows;
  if (rows.length > maxPoints) {
    // Evenly spaced rows keep both classes even when the file is sorted by label.
    points = Array.from({ length: maxPoints }, (_, i) => rows[Math.floor((i * rows.length) / maxPoints)]);
    notes.push(`The file has ${formatCount(rows.length)} points; ${formatCount(maxPoints)} evenly spread ones were kept, the most a dataset holds.`);
  }
  if (regression) {
    if (points.every(point => point.label === points[0].label)) {
      notes.push('Every point has the same value, so there is nothing to fit. Give some points other values.');
    }
  } else {
    const violet = points.filter(point => point.label > 0).length;
    if (violet === 0 || violet === points.length) {
      notes.push(`Every point is ${violet ? 'violet' : 'amber'}, so there is nothing to tell apart. Add rows with label ${violet ? '-1' : '1'}.`);
    }
  }
  return { points, notes };
}

/** mulberry32, so the sample files are the same every time. */
function seeded(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A small example in the expected format. Classification: two overlapping clouds, violet
 * at the top right and amber at the bottom left. Regression: a smooth hill, 1 (violet) in
 * the middle and falling toward -1 (amber) at the edges.
 */
export function sampleDatasetCsv({ regression = false, count = 80 }: { regression?: boolean; count?: number } = {}): string {
  const random = seeded(20261126);
  const clamp = (v: number) => Math.max(-PLOT_LIMIT, Math.min(PLOT_LIMIT, v));
  if (regression) {
    const rows = ['x1,x2,value'];
    for (let i = 0; i < count; i++) {
      const x = random() * 10 - 5;
      const y = random() * 10 - 5;
      const value = 2 * Math.exp(-(x * x + y * y) / 12) - 1;
      rows.push(`${x.toFixed(2)},${y.toFixed(2)},${value.toFixed(2)}`);
    }
    return rows.join('\n') + '\n';
  }
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const rows = ['x1,x2,label'];
  for (let i = 0; i < count; i++) {
    const labelValue = i % 2 === 0 ? 1 : -1;
    const x = clamp(labelValue * 2 + gaussian() * 1.3);
    const y = clamp(labelValue * 2 + gaussian() * 1.3);
    rows.push(`${x.toFixed(2)},${y.toFixed(2)},${labelValue}`);
  }
  return rows.join('\n') + '\n';
}
