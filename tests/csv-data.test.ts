import { describe, expect, it } from 'vitest';
import { CsvFormatError, parseDatasetCsv, sampleDatasetCsv } from '../src/csv-data';

const error = (text: string) => {
  try {
    parseDatasetCsv(text);
  } catch (caught) {
    expect(caught).toBeInstanceOf(CsvFormatError);
    return (caught as Error).message;
  }
  throw new Error('expected a format error');
};

describe('reading a dataset CSV', () => {
  it('reads the sample file back: a header, then both classes within the plot', () => {
    const sample = sampleDatasetCsv();
    expect(sample.split('\n')[0]).toBe('x1,x2,label');
    const { points, notes } = parseDatasetCsv(sample);
    expect(points).toHaveLength(80);
    expect(notes).toEqual([]);
    expect(points.filter(point => point.label === 1)).toHaveLength(40);
    expect(points.every(point => Math.abs(point.x) <= 6 && Math.abs(point.y) <= 6)).toBe(true);
    expect(sampleDatasetCsv()).toBe(sample);
  });

  it('needs no header, and takes 0 as amber', () => {
    expect(parseDatasetCsv('1.5,-2,1\n-3,0.25,0\n2,2,-1.0').points).toEqual([
      { x: 1.5, y: -2, label: 1 }, { x: -3, y: .25, label: -1 }, { x: 2, y: 2, label: -1 },
    ]);
  });

  it('accepts semicolons with decimal commas, tabs, quotes, blank lines, CRLF and a byte-order mark', () => {
    expect(parseDatasetCsv('﻿x1;x2;label\r\n1,5;-2,25;1\r\n\r\n').points).toEqual([{ x: 1.5, y: -2.25, label: 1 }]);
    expect(parseDatasetCsv('x1\tx2\tlabel\n"1"\t"2"\t"-1"\n').points).toEqual([{ x: 1, y: 2, label: -1 }]);
    expect(parseDatasetCsv('1,2,1,\n').points).toEqual([{ x: 1, y: 2, label: 1 }]);
  });

  it('says which line is wrong and how', () => {
    expect(error('x1,x2,label\n1,2,1\n3,4')).toBe('Line 3 has 2 values. Each row needs 3: x1, x2 and label.');
    expect(error('x1,x2,label\nabc,2,1')).toBe('Line 2: x1 is “abc”, not a number.');
    expect(error('1,,1')).toBe('Line 1: x2 is “empty”, not a number.');
    expect(error('1,2,2')).toBe('Line 1: the label is “2”. Use 1 for violet and -1 for amber (0 also means amber).');
    expect(error('1,2,cat')).toBe('Line 1: the label is “cat”. Use 1 for violet and -1 for amber (0 also means amber).');
    expect(error('name,x1,x2,label\n')).toBe('The first row has 4 columns. A dataset needs 3: x1, x2 and label.');
    expect(error('x1,x2,label\n')).toBe('No points found. Each row needs x1, x2 and label, as in the sample file.');
    expect(error('')).toBe('No points found. Each row needs x1, x2 and label, as in the sample file.');
  });

  it('rescales an axis that leaves the plot onto it, and says so', () => {
    const { points, notes } = parseDatasetCsv('0,1,1\n50,2,-1\n100,-1,1');
    expect(points.map(point => point.x)).toEqual([-5.5, 0, 5.5]);
    expect(points.map(point => point.y)).toEqual([1, 2, -1]);
    expect(notes).toEqual(['x1 ran from 0 to 100, so it was rescaled to fit the plot.']);
    expect(parseDatasetCsv('0,-20,1\n118,18,-1').notes).toEqual(['x1 ran from 0 to 118 and x2 from -20 to 18, so both were rescaled to fit the plot.']);
  });

  it('keeps an evenly spread subset of a large file, with both classes even when sorted', () => {
    const rows = Array.from({ length: 1000 }, (_, i) => `${(i % 10) - 5},0,${i < 500 ? 1 : -1}`);
    const { points, notes } = parseDatasetCsv(rows.join('\n'));
    expect(points).toHaveLength(400);
    expect(points.filter(point => point.label === 1)).toHaveLength(200);
    expect(notes).toEqual(['The file has 1,000 points; 400 evenly spread ones were kept, the most a dataset holds.']);
  });

  it('reads regression values, rescaling them onto -1 to 1 when they leave it', () => {
    const sample = sampleDatasetCsv({ regression: true });
    expect(sample.split('\n')[0]).toBe('x1,x2,value');
    const { points, notes } = parseDatasetCsv(sample, { regression: true });
    expect(points).toHaveLength(80);
    expect(notes).toEqual([]);
    expect(points.every(point => point.label >= -1 && point.label <= 1)).toBe(true);
    // A hill: higher in the middle than at the edges.
    const near = points.filter(point => Math.hypot(point.x, point.y) < 2);
    const far = points.filter(point => Math.hypot(point.x, point.y) > 4);
    expect(Math.min(...near.map(point => point.label))).toBeGreaterThan(Math.max(...far.map(point => point.label)));

    expect(parseDatasetCsv('x1,x2,value\n1,2,0.25\n-1,0,-0.5', { regression: true }).points)
      .toEqual([{ x: 1, y: 2, label: .25 }, { x: -1, y: 0, label: -.5 }]);
    const wide = parseDatasetCsv('0,0,100\n1,1,300\n2,2,200', { regression: true });
    expect(wide.points.map(point => point.label)).toEqual([-1, 1, 0]);
    expect(wide.notes).toEqual(['The values ran from 100 to 300, so they were rescaled to -1 to 1.']);
  });

  it('says what is wrong with a regression file', () => {
    const regressionError = (text: string) => {
      try {
        parseDatasetCsv(text, { regression: true });
      } catch (caught) {
        return (caught as Error).message;
      }
      throw new Error('expected a format error');
    };
    expect(regressionError('x1,x2,value\n1,2,high')).toBe('Line 2: the value is “high”, not a number.');
    expect(regressionError('x1,x2,value\n1,2')).toBe('Line 2 has 2 values. Each row needs 3: x1, x2 and value.');
    expect(regressionError('x1,x2,value\n')).toBe('No points found. Each row needs x1, x2 and value, as in the sample file.');
    expect(parseDatasetCsv('1,1,.5\n2,2,.5', { regression: true }).notes).toEqual(['Every point has the same value, so there is nothing to fit. Give some points other values.']);
  });

  it('notes when every point has the same class', () => {
    expect(parseDatasetCsv('1,1,1\n2,2,1').notes).toEqual(['Every point is violet, so there is nothing to tell apart. Add rows with label -1.']);
  });
});
