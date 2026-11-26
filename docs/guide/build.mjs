#!/usr/bin/env node
/**
 * Builds the explanation guide: a landscape PDF that walks through every page of the
 * site with annotated screenshots and concept diagrams.
 *
 *   npm run build                      (the guide photographs the production build)
 *   node docs/guide/build.mjs          captures screenshots, then writes the PDF
 *     --out <file>       PDF path (default docs/Neural-Network-Visualizer-Guide.pdf)
 *     --work <dir>       screenshots, guide.html and previews (default: a temp folder)
 *     --url <url>        use a running server instead of starting `vite preview`
 *     --skip-capture     reuse the screenshots from the last run in --work
 *     --only <path>      capture only that page again (for example /transformer/, or phones)
 *     --previews         also write a PNG of every slide to <work>/previews
 *
 * Screenshots come from a real browser: full Chromium, so the Transformer page can use
 * the GPU, at twice the pixel density. Every number on a screenshot is placed on the
 * element it explains by measuring that element in the page; what each number says is
 * in content.mjs.
 */
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SHOTS, SLIDES } from './content.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../..');
const require = createRequire(join(repo, 'package.json'));
const { chromium } = require('@playwright/test');

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const out = resolve(option('--out', join(repo, 'docs/Neural-Network-Visualizer-Guide.pdf')));
const work = resolve(option('--work', join(tmpdir(), 'nnv-guide')));
const shotsDir = join(work, 'shots');
const recordPath = join(shotsDir, 'shots.json');
mkdirSync(shotsDir, { recursive: true });

const VIEWPORT = { width: 1440, height: 900 };

/* ---------- Server ---------- */

async function startServer() {
  const given = option('--url');
  if (given) return { url: given.replace(/\/$/, ''), stop: () => {} };
  if (!existsSync(join(repo, 'dist/index.html'))) throw new Error('No production build: run `npm run build` first.');
  const port = 4190;
  const server = spawn(join(repo, 'node_modules/.bin/vite'), ['preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], { cwd: repo, stdio: 'ignore' });
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 80; i++) {
    try {
      if ((await fetch(url)).ok) return { url, stop: () => server.kill() };
    } catch {
      // not up yet
    }
    await new Promise(r => setTimeout(r, 250));
  }
  server.kill();
  throw new Error(`The preview server did not start on port ${port}.`);
}

/* ---------- Capture ---------- */

const record = existsSync(recordPath) ? JSON.parse(readFileSync(recordPath, 'utf8')) : {};
const pause = ms => new Promise(r => setTimeout(r, ms));

/** The region's box and every marker's box, in viewport pixels, after the page is laid out. */
function measure(page, spec) {
  return page.evaluate(({ region, markers }) => {
    const box = el => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; };
    const union = boxes => {
      const x = Math.min(...boxes.map(b => b.x));
      const y = Math.min(...boxes.map(b => b.y));
      return { x, y, w: Math.max(...boxes.map(b => b.x + b.w)) - x, h: Math.max(...boxes.map(b => b.y + b.h)) - y };
    };
    const options = typeof region === 'string' || Array.isArray(region) ? { sel: region } : region;
    const selectors = [options.sel].flat();
    const elements = selectors.map(sel => document.querySelector(sel));
    const lost = selectors.filter((_s, i) => !elements[i]);
    if (lost.length) return { error: `region not found: ${lost.join(', ')}` };
    const raw = union(elements.map(box));
    const pad = options.pad ?? 12;
    let reg = { x: raw.x - (options.padLeft ?? pad), y: raw.y - (options.padTop ?? pad), w: raw.w + (options.padLeft ?? pad) + pad, h: raw.h + (options.padTop ?? pad) + (options.padBottom ?? pad) };
    if (options.maxHeight) reg.h = Math.min(reg.h, options.maxHeight);
    if (options.within) {
      // Clip to a container, for content that scrolls sideways inside it.
      const outer = box(document.querySelector(options.within));
      const right = Math.min(reg.x + reg.w, outer.x + outer.w);
      reg.x = Math.max(reg.x, outer.x);
      reg.w = right - reg.x;
    }
    reg.x = Math.max(0, reg.x);
    reg.w = Math.min(innerWidth - reg.x, reg.w);
    const marks = markers.map(({ sel, textBox }) => {
      const el = document.querySelector(sel);
      if (!el) return { sel, missing: true };
      // textBox: the box of the text itself, for headings and words in wider elements.
      const range = document.createRange();
      range.selectNodeContents(el);
      const b = textBox ? box(range) : box(el);
      return { sel, x: b.x - reg.x, y: b.y - reg.y, w: b.w, h: b.h };
    });
    return { reg, marks, scroll: scrollY };
  }, { region: spec.region, markers: spec.markers ?? [] });
}

async function setHeaderHidden(page, hidden) {
  await page.evaluate(hide => {
    let style = document.getElementById('guide-hide-header');
    if (hide && !style) {
      style = document.createElement('style');
      style.id = 'guide-hide-header';
      style.textContent = '.app-header { visibility: hidden !important; } .simulation-status, .status-toast { display: none !important; }';
      document.head.append(style);
    } else if (!hide && style) style.remove();
  }, hidden);
}

async function capture(page, name) {
  const spec = SHOTS[name];
  if (!spec) throw new Error(`No shot named ${name}`);
  const file = join(shotsDir, `${name}.jpg`);
  await setHeaderHidden(page, !spec.header);
  if (spec.region === 'viewport') {
    if (!spec.phone) await page.setViewportSize(VIEWPORT);
    await page.evaluate(() => window.scrollTo(0, 0));
    await pause(500);
    const size = page.viewportSize();
    await page.screenshot({ path: file, type: 'jpeg', quality: 88 });
    record[name] = { file, width: size.width, height: size.height, markers: [] };
    console.log(`  ${name}`);
    return;
  }
  const width = spec.width ?? VIEWPORT.width;
  if (page.viewportSize().width !== width) {
    await page.setViewportSize({ width, height: page.viewportSize().height });
    await pause(700);
  }
  let measured;
  for (let attempt = 0; attempt < 3; attempt++) {
    measured = await measure(page, spec);
    if (measured.error) throw new Error(`${name}: ${measured.error}`);
    const needed = Math.ceil(measured.reg.h + 24);
    const height = Math.min(4000, Math.max(VIEWPORT.height, needed));
    if (page.viewportSize().height !== height) {
      await page.setViewportSize({ width, height });
      await pause(450);
    }
    await page.evaluate(top => window.scrollBy(0, top - 8), measured.reg.y);
    await pause(350);
    measured = await measure(page, spec);
    if (measured.reg.y >= 0 && measured.reg.y + measured.reg.h <= page.viewportSize().height + 1) break;
  }
  const { reg, marks } = measured;
  const clip = { x: reg.x, y: Math.max(0, reg.y), width: reg.w, height: Math.min(reg.h, page.viewportSize().height - Math.max(0, reg.y)) };
  await page.screenshot({ path: file, type: 'jpeg', quality: 88, clip });
  for (const mark of marks) {
    if (mark.missing) console.warn(`    ! ${name}: marker not found: ${mark.sel}`);
    else if (mark.x + mark.w < 0 || mark.y + mark.h < 0 || mark.x > clip.width || mark.y > clip.height) console.warn(`    ! ${name}: marker outside the shot: ${mark.sel}`);
  }
  record[name] = { file, width: clip.width, height: clip.height, markers: marks };
  console.log(`  ${name}${marks.length ? ` (${marks.filter(m => !m.missing).length} markers)` : ''}`);
}

/** A seeded Math.random, so "Try an example" and friends pick the same thing every run. */
const seededRandom = () => {
  let seed = 20261121;
  Math.random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
};

async function setRange(page, id, value) {
  await page.locator(`#${id}`).evaluate((el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function trainTo(page, epochs) {
  const start = await page.evaluate(() => Number(document.querySelector('#iter-number').textContent.replace(/\D/g, '')));
  await page.click('#play-pause-button');
  await page.waitForFunction(target => Number(document.querySelector('#iter-number').textContent.replace(/\D/g, '')) >= target, start + epochs, { timeout: 180_000 });
  await page.click('#play-pause-button');
  await pause(500);
}

/** Draws a digit on a pad from points given as fractions of its size. */
async function drawStrokes(page, selector, strokes) {
  const pad = await page.locator(selector).boundingBox();
  for (const points of strokes) {
    await page.mouse.move(pad.x + points[0][0] * pad.width, pad.y + points[0][1] * pad.height);
    await page.mouse.down();
    for (const [x, y] of points.slice(1)) await page.mouse.move(pad.x + x * pad.width, pad.y + y * pad.height, { steps: 2 });
    await page.mouse.up();
  }
  await pause(600);
}

/** A hand-drawn 3: two bumps. */
const THREE = (() => {
  const points = [];
  for (let i = 0; i <= 20; i++) { const a = -Math.PI * 0.85 + (i / 20) * Math.PI * 1.35; points.push([0.5 + 0.2 * Math.cos(a), 0.33 + 0.15 * Math.sin(a)]); }
  for (let i = 0; i <= 22; i++) { const a = -Math.PI * 0.5 + (i / 22) * Math.PI * 1.35; points.push([0.5 + 0.22 * Math.cos(a), 0.66 + 0.17 * Math.sin(a)]); }
  return [points];
})();
/** A hand-drawn 0: one ring. */
const ZERO = [Array.from({ length: 33 }, (_, i) => { const a = (i / 32) * 2 * Math.PI; return [0.5 + 0.2 * Math.sin(a), 0.5 - 0.3 * Math.cos(a)]; })];

async function transformerSettled(page) {
  await page.locator('#transformer-flow[aria-busy="false"]').waitFor({ timeout: 300_000 });
  await page.waitForFunction(() => !document.querySelector('#lens')?.classList.contains('is-stale'), null, { timeout: 300_000 });
  await pause(600);
}

const PAGES = [
  {
    path: '/',
    async run(page, shot) {
      await page.locator('#main-part[aria-busy="false"]').waitFor();
      await shot('header');
      await trainTo(page, 250);
      await shot('hero-playground');
      await shot('theme-light');
      await page.emulateMedia({ colorScheme: 'dark' });
      await pause(900);
      await shot('theme-dark');
      await page.emulateMedia({ colorScheme: 'light' });
      await pause(900);
      await shot('pg-overview');
      await shot('pg-training');
      await shot('pg-data');
      await shot('pg-features');
      await shot('pg-network');
      await page.click('label[for="show-test-data"]');
      await pause(300);
      await shot('pg-output');
      await shot('pg-noise-0');
      await page.click('label[for="show-test-data"]');
      await page.click('label[for="show-backprop"]');
      await page.click('#next-step-button');
      await pause(900);
      await shot('pg-backprop');
      await page.click('label[for="show-backprop"]');
      await shot('pg-datasets-classification');
      await page.selectOption('#problem', 'regression');
      await pause(500);
      await shot('pg-datasets-regression');
      await page.selectOption('#problem', 'classification');
      await pause(500);
      await page.locator('canvas[data-dataset="circle"]').click();
      await setRange(page, 'noise', 40);
      await pause(400);
      await trainTo(page, 250);
      await shot('pg-noise-40');
      await setRange(page, 'noise', 0);
      // Hand-drawn data: a violet cluster at the top left and an amber one at the bottom right.
      await page.locator('.dataset-custom').click();
      await pause(400);
      const plot = await page.locator('#heatmap').boundingBox();
      const dot = (fx, fy) => page.mouse.click(plot.x + plot.width * fx, plot.y + plot.height * fy);
      const violet = [[0.22, 0.2], [0.3, 0.26], [0.18, 0.32], [0.33, 0.16], [0.26, 0.38], [0.4, 0.24], [0.14, 0.24], [0.36, 0.34]];
      const amber = [[0.7, 0.72], [0.78, 0.66], [0.64, 0.8], [0.82, 0.78], [0.72, 0.6], [0.6, 0.68], [0.86, 0.7], [0.76, 0.84]];
      await page.locator('#paint-controls [data-paint="1"]').click();
      for (const [x, y] of violet) await dot(x, y);
      await page.locator('#paint-controls [data-paint="-1"]').click();
      for (const [x, y] of amber) await dot(x, y);
      await trainTo(page, 1500);
      await shot('pg-draw');
      await page.locator('canvas[data-dataset="circle"]').click();
      await pause(400);
      await shot('pg-guide');
      await page.getByRole('button', { name: 'Classroom settings' }).click();
      await page.locator('#classroom-panel').waitFor({ state: 'visible' });
      await pause(400);
      await shot('pg-classroom');
    },
  },
  {
    path: '/cnn/',
    async run(page, shot) {
      await page.locator('#cnn-workbench[aria-busy="false"]').waitFor();
      await drawStrokes(page, '#draw-pad', THREE);
      await page.locator('#conv2-maps .map-tile').nth(5).focus();
      await pause(400);
      await shot('hero-digits');
      await shot('cnn-overview');
      await shot('cnn-layers');
      await shot('cnn-prediction');
      await shot('cnn-saliency');
      await page.selectOption('#fool-target', '8');
      await page.click('#fool-start');
      await page.waitForFunction(() => document.querySelector('#predicted-digit').textContent.trim() === '8' && !document.querySelector('#fool-undo').hidden, null, { timeout: 60_000 });
      await pause(600);
      await shot('cnn-fooled');
    },
  },
  {
    path: '/vae/',
    async run(page, shot) {
      await page.locator('#example-digit:not([disabled])').waitFor({ timeout: 60_000 });
      await page.selectOption('#morph-from', '4');
      await page.selectOption('#morph-to', '9');
      await page.click('#morph-start');
      await pause(1200);
      await shot('hero-vae');
      await shot('vae-overview');
      await page.click('button[data-view="grid"]');
      await pause(900);
      await shot('vae-grid');
      await page.click('button[data-view="codes"]');
      await pause(500);
      // A drawn digit that lands among its own kind: the 3 from the Digits page, or a 0.
      await drawStrokes(page, '#draw-pad', THREE);
      let readout = await page.locator('#drawing-readout').textContent();
      if (!/among the 3s/.test(readout)) {
        await page.click('#clear-pad');
        await drawStrokes(page, '#draw-pad', ZERO);
        readout = await page.locator('#drawing-readout').textContent();
      }
      console.log(`    the drawing: ${readout}`);
      await shot('vae-example');
      await shot('vae-input');
      await shot('vae-output');
    },
  },
  {
    path: '/rnn/',
    async run(page, shot) {
      await page.locator('#rnn-workbench[aria-busy="false"]').waitFor();
      const text = await page.inputValue('#prompt-input');
      // A step inside a word whose real next letter is the network's 2nd to 4th guess, so the
      // word previews show the part already written and the real next letter stands apart.
      // In the default text that is the "T" of "To": "There" first, the real "To" second.
      const cells = page.locator('#char-cells .char-cell');
      let chosen = text.indexOf('"To') + 1;
      for (let t = 1; t < text.length - 1; t++) {
        if (!/[A-Za-z]/.test(text[t]) || !/[A-Za-z]/.test(text[t + 1])) continue;
        await cells.nth(t).hover();
        await pause(160);
        const fits = await page.evaluate(() => {
          const items = [...document.querySelectorAll('#candidate-list .candidate')];
          const actual = items.findIndex(item => item.classList.contains('is-actual'));
          return actual >= 1 && actual <= 3 && (items[0]?.querySelector('.candidate-word .known')?.textContent ?? '').length >= 1;
        });
        if (fits) { chosen = t; break; }
      }
      await cells.nth(chosen).hover();
      await page.mouse.move(4, 4);
      await pause(500);
      console.log(`    selected step: "${text.slice(Math.max(0, chosen - 12), chosen + 1)}"`);
      await shot('hero-rnn');
      await shot('rnn-overview');
      await shot('rnn-memory-top');
      await shot('rnn-memory-bottom');
      await shot('rnn-next');
      await page.locator('#gates-panel').scrollIntoViewIfNeeded();
      await page.waitForFunction(() => /\d+%/.test(document.querySelector('#score-rnn-first')?.textContent ?? ''), null, { timeout: 60_000 });
      await page.click('#continue-button');
      await page.waitForFunction(() => (document.querySelector('#continue-rnn')?.textContent ?? '').trim().length > 40, null, { timeout: 60_000 });
      await pause(800);
      await shot('rnn-gates-top');
      await shot('rnn-gates-bottom');
    },
  },
  {
    path: '/transformer/',
    async run(page, shot) {
      await page.locator('#load-model').waitFor();
      await shot('tf-loader');
      await page.click('#load-model');
      await page.locator('.candidate').first().waitFor({ timeout: 300_000 });
      await page.selectOption('#example-select', { label: 'When Mary and John went to the store, John gave a drink to' });
      await transformerSettled(page);
      await page.getByRole('tab', { name: 'Block 12', exact: true }).click();
      await page.getByRole('tab', { name: 'Head 11', exact: true }).click();
      await transformerSettled(page);
      await shot('hero-transformer');
      await shot('tf-dock');
      await shot('tf-overview');
      await shot('tf-embedding-tokens');
      await shot('tf-embedding-maps');
      await shot('tf-blocks');
      await shot('tf-head');
      await shot('tf-mlp');
      await shot('tf-next');
      await shot('tf-lens');
      await shot('tf-surprise');
      await page.click('#head-toggle');
      await transformerSettled(page);
      await shot('tf-off');
      await shot('tf-off-next');
      await page.click('#restore-heads');
      await transformerSettled(page);
      await page.click('#importance-start');
      await page.waitForFunction(() => /^Every head tested/.test(document.querySelector('#importance-progress')?.textContent ?? ''), null, { timeout: 600_000 });
      await pause(600);
      await shot('tf-importance');
    },
  },
];

async function captureAll(url) {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chromium' });
  } catch {
    console.warn('Full Chromium is not installed; using the headless shell (GPT-2 runs on the CPU, more slowly).');
    browser = await chromium.launch();
  }
  try {
    const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 2, colorScheme: 'light', reducedMotion: 'reduce', locale: 'en-US' });
    await context.addInitScript(seededRandom);
    const only = option('--only');
    for (const site of PAGES.filter(site => !only || site.path === only)) {
      console.log(site.path);
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize(VIEWPORT);
      await page.goto(url + site.path);
      await site.run(page, name => capture(page, name));
      if (errors.length) console.warn(`    ! page errors on ${site.path}: ${errors.join(' | ')}`);
      await page.close();
    }
    await context.close();
    // Phones: the top of three pages at 390 × 844.
    if (only && only !== 'phones') return;
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: 'light', reducedMotion: 'reduce' });
    await phone.addInitScript(seededRandom);
    for (const [name, path] of [['phone-playground', '/'], ['phone-vae', '/vae/'], ['phone-rnn', '/rnn/']]) {
      const page = await phone.newPage();
      await page.goto(url + path);
      await pause(2500);
      await capture(page, name);
      await page.close();
    }
    await phone.close();
  } finally {
    await browser.close();
    writeFileSync(recordPath, JSON.stringify(record, null, 1));
  }
}

/* ---------- Slides ---------- */

const fontUrl = pathToFileURL(join(repo, 'node_modules/@fontsource-variable/mona-sans/files/mona-sans-latin-wdth-normal.woff2')).href;
const imageUrl = name => {
  const shot = record[name];
  if (!shot) throw new Error(`Missing screenshot ${name}; run without --skip-capture.`);
  return pathToFileURL(shot.file).href;
};

/** A screenshot with numbered markers on the elements it explains, numbered on from `start`. */
function annotated(name, options = {}) {
  const shot = record[name];
  if (!shot) throw new Error(`Missing screenshot ${name}`);
  const spec = SHOTS[name];
  const start = options.start ?? 0;
  const visible = (spec.markers ?? []).map((marker, i) => ({ ...marker, box: shot.markers[i] })).filter(m => m.box && !m.box.missing);
  const badges = visible.map((m, i) => {
    const { x, y, w, h } = m.box;
    const anchor = m.anchor ?? 'tl';
    const outside = anchor.startsWith('o') ? anchor.slice(1) : '';
    const side = outside || anchor;
    const ax = side === 'l' ? x : side === 'r' ? x + w : side.includes('r') ? x + w : side.includes('l') ? x : x + w / 2;
    const ay = ['l', 'r', 'c'].includes(side) ? y + h / 2 : side.includes('b') ? y + h : side === 't' || side.includes('t') ? y : y + h / 2;
    const left = outside ? (ax / shot.width) * 100 : Math.max(1.4, Math.min(98.6, (ax / shot.width) * 100));
    const top = outside ? (ay / shot.height) * 100 : Math.max(2, Math.min(98, (ay / shot.height) * 100));
    return `<span class="badge${outside ? ` out-${outside}` : ''}" style="left:${left}%;top:${top}%">${start + i + 1}</span>`;
  }).join('');
  return { html: `<figure class="shot" data-w="${shot.width}" data-h="${shot.height}" ${options.maxWidth ? `data-max-w="${options.maxWidth}"` : ''}><img src="${imageUrl(name)}" alt="">${badges}</figure>`, notes: visible };
}

const noteList = notes => `<ol class="notes">${notes.map((note, i) => `<li><span class="badge static">${i + 1}</span><div><strong>${note.title}</strong>${note.text ? ` ${note.text}` : ''}</div></li>`).join('')}</ol>`;
const plainNotes = notes => `<ul class="notes plain">${notes.map(([title, text]) => `<li><div><strong>${title}</strong> ${text}</div></li>`).join('')}</ul>`;

function slideHtml(slide, number) {
  const head = slide.type === 'cover' ? '' : `<header class="slide-head"><span class="section">${slide.section}</span><h2>${slide.title}</h2>${slide.lead && slide.type !== 'divider' ? `<p class="lead">${slide.lead}</p>` : ''}</header>`;
  const foot = slide.type === 'cover' ? '' : `<footer class="slide-foot"><span>Neural Network Visualizer</span><span>${number}</span></footer>`;
  let body = '';
  if (slide.type === 'cover') {
    body = `<div class="cover-copy"><p class="kicker">An explanation guide</p><h1>Neural Network Visualizer</h1><p class="subtitle">Five neural networks you can look inside, and what every panel, color and chart on the website shows.</p></div>
      <div class="cover-strip">${['hero-playground', 'hero-digits', 'hero-vae', 'hero-rnn', 'hero-transformer'].map((name, i) => `<figure><img src="${imageUrl(name)}" alt=""><figcaption>${['Playground', 'Digits', 'Autoencoder', 'Recurrent', 'Transformer'][i]}</figcaption></figure>`).join('')}</div>`;
  } else if (slide.type === 'divider') {
    body = `<div class="divider-copy"><p class="lead big">${slide.lead}</p><ul class="points">${slide.points.map(p => `<li>${p}</li>`).join('')}</ul></div>
      <div class="fit divider-image"><figure class="shot plain" data-w="${record[slide.shot].width}" data-h="${record[slide.shot].height}"><img src="${imageUrl(slide.shot)}" alt=""></figure></div>`;
  } else if (slide.type === 'shot' && slide.shots) {
    // Parts of one tall panel side by side, numbered on from one part to the next.
    const parts = [];
    for (const name of slide.shots) parts.push(annotated(name, { start: parts.reduce((n, part) => n + part.notes.length, 0) }));
    const list = `${noteList(parts.flatMap(part => part.notes))}${slide.after ? `<p class="tip">${slide.after}</p>` : ''}`;
    body = `<div class="side"><div class="shot-row">${parts.map(part => part.html).join('')}</div><div class="side-notes">${list}</div></div>`;
  } else if (slide.type === 'shot') {
    const { html, notes } = annotated(slide.shot, { maxWidth: slide.imageWidth });
    const list = `${noteList(notes)}${slide.after ? `<p class="tip">${slide.after}</p>` : ''}`;
    body = slide.layout === 'stack'
      ? `<div class="stack"><div class="fit stack-image">${html}</div><div class="stack-notes">${list}</div></div>`
      : `<div class="side"><div class="fit side-image">${html}</div><div class="side-notes">${list}</div></div>`;
  } else if (slide.type === 'images') {
    const figures = slide.images.map(({ shot, caption }) => `<div class="image-cell"><div class="fit"><figure class="shot plain" data-w="${record[shot].width}" data-h="${record[shot].height}"><img src="${imageUrl(shot)}" alt=""></figure></div>${caption ? `<p class="caption">${caption}</p>` : ''}</div>`).join('');
    const grid = `<div class="image-grid n${slide.images.length}${slide.columns ? ` cols-${slide.columns}` : ''}">${figures}</div>`;
    body = slide.layout === 'stack'
      ? `<div class="stack images-stack">${grid}<div class="stack-notes">${plainNotes(slide.notes)}</div></div>`
      : `<div class="side">${grid}<div class="side-notes">${plainNotes(slide.notes)}</div></div>`;
  } else if (slide.type === 'text') {
    body = `<div class="text-body">${slide.html.replace(/\{\{img:([\w-]+)\}\}/g, (_m, name) => imageUrl(name))}</div>`;
  }
  return `<section class="slide ${slide.type}${slide.layout ? ` layout-${slide.layout}` : ''}">${head}<div class="slide-body">${body}</div>${foot}</section>`;
}

const CSS = `
@font-face { font-family: 'Mona Sans'; src: url('${fontUrl}') format('woff2'); font-weight: 200 900; font-stretch: 75% 125%; }
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #f3f4f9; }
body { font-family: 'Mona Sans', system-ui, sans-serif; color: #1c1e2d; -webkit-print-color-adjust: exact; print-color-adjust: exact; font-variant-numeric: tabular-nums; }
.slide { position: relative; display: flex; flex-direction: column; width: 1280px; height: 720px; padding: 34px 52px 0; overflow: hidden; background: #f3f4f9; break-after: page; page-break-after: always; }
.slide-head { flex: 0 0 auto; }
.section { display: block; margin-bottom: 4px; color: #744ed7; font-size: 13px; font-weight: 650; letter-spacing: .01em; }
h2 { margin: 0; font-size: 30px; font-weight: 720; font-stretch: 112%; letter-spacing: -.02em; line-height: 1.12; }
.lead { max-width: 1080px; margin: 8px 0 0; color: #4f5161; font-size: 15.5px; line-height: 1.45; }
.slide-body { position: relative; flex: 1 1 auto; min-height: 0; margin: 16px 0 0; }
.slide-foot { flex: 0 0 38px; display: flex; align-items: center; justify-content: space-between; color: #8a8c99; font-size: 11.5px; }
.fit { position: relative; width: 100%; height: 100%; min-height: 0; display: flex; align-items: flex-start; justify-content: center; }
.shot { position: relative; margin: 0; border-radius: 14px; overflow: visible; box-shadow: 0 0 0 1px rgba(28,30,45,.10), 0 10px 30px -18px rgba(28,30,45,.45); background: #fff; }
.shot img { display: block; width: 100%; height: 100%; border-radius: 14px; }
.badge { position: absolute; z-index: 2; display: grid; place-items: center; width: 23px; height: 23px; margin: -11.5px 0 0 -11.5px; border-radius: 50%; background: #1c1e2d; color: #fff; font-size: 12px; font-weight: 700; box-shadow: 0 0 0 2.5px #fff, 0 3px 8px rgba(28,30,45,.35); }
.badge.static { position: static; flex: 0 0 auto; margin: 0; box-shadow: none; }
.badge.out-l { margin-left: -27px; }
.badge.out-r { margin-left: 4px; }
.badge.out-t { margin-top: -27px; }
.badge.out-b { margin-top: 4px; }
.side { display: grid; grid-template-columns: minmax(0, 1fr) 372px; gap: 30px; height: 100%; }
.side-image { justify-content: center; }
.shot-row { display: flex; align-items: flex-start; justify-content: center; gap: 28px; min-width: 0; min-height: 0; height: 100%; }
.side-notes { min-height: 0; overflow: hidden; }
.notes { display: grid; gap: 9px; margin: 0; padding: 0; list-style: none; font-size: 13px; line-height: 1.42; color: #4f5161; }
.notes li { display: flex; gap: 10px; align-items: flex-start; }
.notes strong { color: #1c1e2d; font-weight: 680; }
.notes.plain { gap: 14px; font-size: 14px; }
.notes.plain li { display: block; padding-left: 14px; border-left: 3px solid #744ed7; }
.tip { margin: 14px 0 0; padding: 10px 14px; border-radius: 12px; background: #fff; color: #4f5161; font-size: 13px; line-height: 1.45; box-shadow: 0 0 0 1px rgba(28,30,45,.08); }
.stack { display: flex; flex-direction: column; gap: 18px; height: 100%; }
.stack-image { flex: 0 1 auto; height: auto; max-height: 52%; }
.stack-notes { flex: 1 1 auto; min-height: 0; overflow: hidden; }
.stack-notes .notes { grid-template-columns: repeat(3, minmax(0, 1fr)); column-gap: 26px; }
.image-grid { display: grid; gap: 18px; min-height: 0; height: 100%; }
.image-grid.n1 { grid-template-columns: 1fr; }
.image-grid.n2 { grid-template-columns: 1fr 1fr; }
.image-grid.n3 { grid-template-columns: 1fr 1fr 1fr; }
.image-grid.n4 { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; }
.image-grid.cols-1 { grid-template-columns: 1fr; }
.images-stack .image-grid { flex: 1 1 auto; height: auto; }
.images-stack .stack-notes { flex: 0 0 auto; }
.images-stack .notes.plain { grid-template-columns: repeat(3, minmax(0, 1fr)); column-gap: 30px; }
.image-cell { display: flex; flex-direction: column; justify-content: flex-start; min-height: 0; }
.image-cell .fit { flex: 0 0 auto; height: auto; }
.caption { margin: 8px 0 0; color: #6c6e7b; font-size: 12px; text-align: center; }
.text-body { height: 100%; display: flex; flex-direction: column; gap: 16px; }
.text-body svg { display: block; }
.columns { display: grid; gap: 22px; font-size: 13.5px; line-height: 1.5; color: #4f5161; }
.columns p { margin: 0; }
.columns strong { color: #1c1e2d; }
.columns.two { grid-template-columns: 1fr 1fr; }
.columns.three { grid-template-columns: repeat(3, 1fr); }
.columns.four { grid-template-columns: repeat(4, 1fr); }
.columns.roomy { gap: 30px 44px; font-size: 15px; }
.columns h3 { margin: 0 0 6px; color: #1c1e2d; font-size: 17px; font-weight: 700; }
.footnote { margin: 0; color: #8a8c99; font-size: 11.5px; }
.svg-title { font: 650 13px 'Mona Sans'; fill: #1c1e2d; }
.svg-label { font: 600 12px 'Mona Sans'; }
.svg-note { font: 450 11px 'Mona Sans'; fill: #6c6e7b; }
.plot-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 22px; }
.plot-card { margin: 0; display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 12px; align-items: center; }
.plot-card figcaption { font-size: 12.5px; line-height: 1.42; color: #4f5161; }
.plot-card strong { display: inline; color: #1c1e2d; font-size: 14px; }
.plot-card code { margin-left: 4px; padding: 1px 6px; border-radius: 6px; background: #fff; color: #744ed7; font: 600 11.5px 'Mona Sans'; }
.plot-card span { display: block; margin-top: 4px; }
.legend-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 40px; }
.legend-row { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: 16px; align-items: start; font-size: 13.5px; line-height: 1.45; color: #4f5161; }
.legend-row strong { display: block; margin-bottom: 2px; color: #1c1e2d; font-size: 14.5px; }
.swatch { display: block; width: 64px; height: 18px; margin-top: 2px; border-radius: 6px; box-shadow: inset 0 0 0 1px rgba(28,30,45,.12); }
.swatch-pair { display: flex; gap: 6px; }
.swatch-pair .swatch { width: 29px; }
.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.card img { display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; object-position: top; margin-bottom: 8px; border-radius: 10px; box-shadow: 0 0 0 1px rgba(28,30,45,.10); }
.card { display: flex; flex-direction: column; gap: 6px; padding: 16px; border-radius: 18px; background: #fff; box-shadow: 0 0 0 1px rgba(28,30,45,.08); font-size: 13px; line-height: 1.45; color: #4f5161; }
.card strong { color: #1c1e2d; font-size: 20px; font-stretch: 110%; }
.card em { color: #744ed7; font-style: normal; font-weight: 600; font-size: 12.5px; }
.cover { padding: 0; background: #1c1e2d; }
.cover .slide-body { margin: 0; display: flex; flex-direction: column; justify-content: space-between; padding: 70px 64px 46px; color: #fff; }
.cover h1 { margin: 6px 0 0; font-size: 64px; font-weight: 760; font-stretch: 118%; letter-spacing: -.03em; line-height: 1; }
.kicker { margin: 0; color: #a692fb; font-size: 17px; font-weight: 600; }
.subtitle { max-width: 760px; margin: 18px 0 0; color: #c9cad6; font-size: 20px; line-height: 1.45; }
.cover-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
.cover-strip figure { margin: 0; }
.cover-strip img { display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; object-position: top; border-radius: 10px; box-shadow: 0 0 0 1px rgba(255,255,255,.14); }
.cover-strip figcaption { margin-top: 8px; color: #c9cad6; font-size: 13px; font-weight: 600; }
.divider .slide-body { display: grid; grid-template-columns: 430px minmax(0, 1fr); gap: 40px; }
.divider h2 { font-size: 46px; }
.divider .lead.big { margin: 0; color: #1c1e2d; font-size: 18px; line-height: 1.5; }
.points { margin: 20px 0 0; padding: 0; list-style: none; display: grid; gap: 12px; color: #4f5161; font-size: 14.5px; line-height: 1.5; }
.points li { padding-left: 14px; border-left: 3px solid #744ed7; }
.glossary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px 30px; margin: 0; font-size: 12.5px; line-height: 1.4; }
.glossary dt { color: #1c1e2d; font-weight: 700; }
.glossary dd { margin: 1px 0 0; color: #4f5161; }
.credits-list { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 40px; font-size: 14px; line-height: 1.5; color: #4f5161; }
.credits-list p { margin: 0; }
.credits-list strong { color: #1c1e2d; }
.ae-diagram { display: flex; align-items: center; justify-content: center; gap: 22px; padding: 18px; border-radius: 14px; background: #fff; box-shadow: 0 0 0 1px rgba(28,30,45,.08); }
.ae-diagram figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center; }
.ae-diagram figcaption { font-size: 12px; color: #6c6e7b; line-height: 1.3; }
.ae-diagram figcaption strong { display: block; color: #1c1e2d; font-size: 13.5px; }
.ae-image img { width: 112px; height: 112px; border-radius: 10px; image-rendering: pixelated; box-shadow: 0 0 0 1px rgba(28,30,45,.12); }
.ae-numbers { display: flex; gap: 6px; }
.ae-numbers span { display: grid; place-items: center; width: 44px; height: 44px; border-radius: 10px; background: #744ed7; color: #fff; font-weight: 700; font-size: 16px; }
`;

/** Sizes every screenshot to fit its box, and shrinks note lists that would overflow. */
const FIT_SCRIPT = `
for (const row of document.querySelectorAll('.shot-row')) {
  // Each part as tall as the row allows, then all shrunk together if they are too wide.
  const figures = [...row.querySelectorAll('figure.shot')];
  const scales = figures.map(f => row.clientHeight / Number(f.dataset.h));
  const width = figures.reduce((sum, f, i) => sum + Number(f.dataset.w) * scales[i], 0) + 28 * (figures.length - 1);
  const shrink = Math.min(1, row.clientWidth / width);
  figures.forEach((f, i) => {
    f.style.width = Math.floor(Number(f.dataset.w) * scales[i] * shrink) + 'px';
    f.style.height = Math.floor(Number(f.dataset.h) * scales[i] * shrink) + 'px';
  });
}
for (const figure of document.querySelectorAll('figure.shot')) {
  if (figure.closest('.shot-row')) continue;
  const box = figure.parentElement;
  const w = Number(figure.dataset.w), h = Number(figure.dataset.h);
  const maxW = Math.min(box.clientWidth, Number(figure.dataset.maxW || Infinity));
  let availH = box.clientHeight || 400;
  if (box.classList.contains('stack-image')) availH = box.parentElement.clientHeight * 0.52;
  const cell = box.closest('.image-cell');
  if (cell) {
    const caption = cell.querySelector('.caption');
    availH = cell.clientHeight - (caption ? caption.offsetHeight + 8 : 0);
    box.style.height = 'auto';
  }
  const scale = Math.min(maxW / w, availH / h);
  figure.style.width = Math.floor(w * scale) + 'px';
  figure.style.height = Math.floor(h * scale) + 'px';
  if (box.classList.contains('stack-image') || cell) box.style.height = Math.floor(h * scale) + 'px';
}
for (const list of document.querySelectorAll('.side-notes, .stack-notes')) {
  let size = parseFloat(getComputedStyle(list.querySelector('.notes')).fontSize);
  while (list.scrollHeight > list.clientHeight + 1 && size > 10) {
    size -= 0.5;
    list.querySelector('.notes').style.fontSize = size + 'px';
  }
}
document.body.dataset.ready = 'yes';
`;

async function render() {
  const slides = SLIDES.map((slide, i) => slideHtml(slide, i + 1)).join('\n');
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Neural Network Visualizer: explanation guide</title><style>${CSS}</style></head><body>${slides}<script>document.fonts.ready.then(() => { ${FIT_SCRIPT} });</script></body></html>`;
  const htmlPath = join(work, 'guide.html');
  writeFileSync(htmlPath, html);
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto(pathToFileURL(htmlPath).href);
    await page.waitForFunction(() => document.body.dataset.ready === 'yes');
    const overflow = await page.evaluate(() => [...document.querySelectorAll('.slide')].map((s, i) => {
      const body = s.querySelector('.slide-body');
      const over = [...body.querySelectorAll('*')].some(el => { const r = el.getBoundingClientRect(); const b = s.getBoundingClientRect(); return r.bottom > b.bottom - 36 + 1 && r.height > 0; });
      return over ? i + 1 : 0;
    }).filter(Boolean));
    if (overflow.length) console.warn(`  ! content runs into the footer on slides ${overflow.join(', ')}`);
    await page.pdf({ path: out, width: '1280px', height: '720px', printBackground: true, preferCSSPageSize: true });
    console.log(`Wrote ${out} (${SLIDES.length} pages)`);
    if (args.includes('--previews')) {
      const dir = join(work, 'previews');
      mkdirSync(dir, { recursive: true });
      const slidesEls = await page.locator('.slide').all();
      for (const [i, el] of slidesEls.entries()) await el.screenshot({ path: join(dir, `slide-${String(i + 1).padStart(2, '0')}.png`) });
      console.log(`Previews in ${dir}`);
    }
  } finally {
    await browser.close();
  }
}

const server = args.includes('--skip-capture') ? null : await startServer();
try {
  if (server) {
    console.log(`Capturing from ${server.url}`);
    await captureAll(server.url);
  }
  await render();
} finally {
  server?.stop();
}
