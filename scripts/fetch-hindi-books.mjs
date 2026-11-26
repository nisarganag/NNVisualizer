#!/usr/bin/env node
/**
 * Downloads the nineteen public-domain Hindi books the Recurrent page's Hindi networks
 * are trained on, from Hindi Wikisource, as plain text.
 *
 * Usage:
 *   node scripts/fetch-hindi-books.mjs <corpus-dir> [--offline]
 *     --offline      rebuild the text from the wikitext an earlier run kept in <corpus-dir>/raw
 *     --only <slug>  fetch one book again
 *
 * Writes <corpus-dir>/<slug>.txt for every book in HINDI_BOOKS (scripts/hindi-books.mjs),
 * one paragraph per line, and <corpus-dir>/hindi-sources.json with every scanned page
 * read and its revision, so the corpus can be checked or rebuilt exactly.
 *
 * A Wikisource book is a main page and chapters, each showing a range of proofread scan
 * pages (<pages index=... from=... to=... next=...>). This follows the chapters from
 * the main page through each one's "next", skipping prefaces and contents, and reads
 * their scan pages' wikitext fifty at a time through the query API, which is cheap for
 * the servers (rendering every chapter instead runs into their rate limits). Pages not
 * yet proofread (quality 0 or 1) are skipped. Wikitext becomes plain text in
 * wikitextToText: formatting templates keep their text, hyphenation templates join the
 * word split across two pages, and footnotes, verse, tables, comments and the page
 * headers and footers are dropped. Requests go one at a time, three seconds apart, and
 * back off when the server asks; a full run takes about ten minutes.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { HINDI_BOOKS } from './hindi-books.mjs';

const API = 'https://hi.wikisource.org/w/api.php';
const USER_AGENT = 'hindi-corpus-fetch/1.0 (public-domain text research)';
const PAUSE_MS = 3000;
const BATCH = 50;
const PAGE_NAMESPACE = 'पृष्ठ';
/** Chapters that are not the story: prefaces, contents, dedications and publishers' notes. */
const SKIP_CHAPTERS = ['भूमिका', 'प्रस्तावना', 'निवेदन', 'विषय-सूची', 'विषयसूची', 'सूची', 'समर्पण', 'प्रकाशकीय', 'वक्तव्य', 'दो शब्द'];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function query(params) {
  const body = new URLSearchParams({ ...params, format: 'json', formatversion: '2', maxlag: '5' });
  for (let attempt = 1; attempt <= 10; attempt++) {
    // POST, since fifty titles can make a long address.
    const response = await fetch(API, { method: 'POST', body, headers: { 'User-Agent': USER_AGENT } });
    if (response.status === 429 || response.status >= 500) {
      const wait = Number(response.headers.get('retry-after')) || 60 * attempt;
      console.log(`    (${response.status}, waiting ${wait} s)`);
      await pause(wait * 1000);
      continue;
    }
    if (!response.ok) throw new Error(`${response.status} from the query API`);
    const data = await response.json();
    await pause(PAUSE_MS);
    if (data.error?.code === 'maxlag') continue;
    if (data.error) throw new Error(data.error.info);
    return data;
  }
  throw new Error('The query API kept refusing');
}

/** The current wikitext and revision of each title, fifty at a time; missing pages are left out. */
async function wikitexts(titles) {
  const out = new Map();
  for (let i = 0; i < titles.length; i += BATCH) {
    const data = await query({ action: 'query', prop: 'revisions', rvprop: 'content|ids', rvslots: 'main', titles: titles.slice(i, i + BATCH).join('|') });
    const renamed = new Map((data.query.normalized ?? []).map(({ from, to }) => [to, from]));
    for (const page of data.query.pages) {
      const revision = page.revisions?.[0];
      if (!revision) continue;
      const entry = { title: page.title, revid: revision.revid, text: revision.slots.main.content };
      out.set(page.title, entry);
      if (renamed.has(page.title)) out.set(renamed.get(page.title), entry);
    }
  }
  return out;
}

const DEVANAGARI_DIGITS = '०१२३४५६७८९';
const toNumber = text => Number(String(text).replace(/[०-९]/g, d => DEVANAGARI_DIGITS.indexOf(d)));
const toDevanagari = n => String(n).replace(/\d/g, d => DEVANAGARI_DIGITS[d]);

/** The <pages .../> tags of a page's wikitext: which scan, which pages, and the next chapter. */
function pagesTags(text) {
  return Array.from(text.matchAll(/<pages\b([^>]*?)\/?>/g), ([, attributes]) => {
    const attribute = name => new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|([^\\s/>]+))`).exec(attributes)?.slice(1).find(Boolean);
    const next = attribute('next')?.match(/\[\[([^|\]]+)/)?.[1]?.trim();
    return { index: attribute('index'), from: attribute('from'), to: attribute('to'), next };
  });
}

const chapterName = (book, title) => title.slice(book.length + 1);

/** The book's chapters in reading order: from the main page's "next", chapter by chapter. */
async function chapters(book, main, read, known, skipAlso = []) {
  const titles = known ?? (await query({ action: 'query', list: 'allpages', apprefix: `${book}/`, apnamespace: '0', aplimit: '500' })).query.allpages.map(page => page.title);
  const texts = await read(titles);
  const order = [];
  let next = pagesTags(main.text)[0]?.next;
  while (next && texts.has(next) && !order.includes(texts.get(next).title)) {
    const chapter = texts.get(next);
    order.push(chapter.title);
    next = pagesTags(chapter.text)[0]?.next;
  }
  // Chapters the chain missed, by their number and then their name (१, २, … १०), so nothing is silently lost.
  const number = title => toNumber(/^[\d०-९]+/.exec(chapterName(book, title))?.[0] ?? 'NaN');
  // A chapter either shows scan pages or holds its text itself.
  const hasText = title => pagesTags(texts.get(title)?.text ?? '').length > 0 || wikitextToText(texts.get(title)?.text ?? '').trim().length > 500;
  const rest = titles.filter(title => !order.includes(title) && hasText(title))
    .sort((a, b) => (number(a) || Infinity) - (number(b) || Infinity) || a.localeCompare(b));
  order.push(...rest);
  const skip = [...SKIP_CHAPTERS, ...skipAlso];
  return order.map(title => texts.get(title)).filter(chapter => !skip.some(part => chapterName(book, chapter.title).includes(part)));
}

const KEEP = new Set(['c', 'center', 'centre', 'larger', 'x-larger', 'xx-larger', 'xxx-larger', 'smaller', 'small', 'x-smaller', 'bold', 'b', 'i', 'italic', 'sc', 'smallcaps',
  'nowrap', 'dropinitial', 'di', 'big', 'u', 'uc', 'lc', 'block center', 'bc', 'float right', 'fr', 'float left', 'fl', 'right', 'r', 'left', 'l', 'justify', 'font-size',
  'fs', 'color', 'highlight', 'ce', 'cb', 'blockquote', 'quote', 'text', 'ls', 'letter-spacing']);
const DROP = new Set(['gap', 'rule', 'dhr', 'rh', 'page break', 'pb', 'clear', '-', 'ts', 'dotted toc line', 'tocline', 'pd-india', 'reflist',
  'smallrefs', 'img float', 'figure', 'missing image', 'hr', 'blank line', 'brace', 'brace2']);

/** One template's text: formatting keeps its text, a word split across pages is whole once, layout is dropped. */
function template(inner) {
  const parts = inner.split('|');
  const name = parts[0].trim().toLowerCase().replace(/_/g, ' ');
  const positional = parts.slice(1).filter(part => !/^\s*[\w-]+\s*=/.test(part));
  if (name === 'hws' || name === 'hyphenated word start') return positional[1] ?? positional[0] ?? '';
  if (name === 'hwe' || name === 'hyphenated word end') return '';
  // Proofreaders put {{nop}} where a paragraph ends at the foot of a page.
  if (name === 'nop') return '\n\n';
  if (DROP.has(name)) return '';
  if (KEEP.has(name)) return positional.find(part => part.trim()) ?? '';
  // Unknown: keep a last argument that is Hindi text, else nothing.
  const last = positional[positional.length - 1] ?? '';
  return /[ऀ-ॿ]/.test(last) ? last : '';
}

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', hellip: '…' };
const decode = text => text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, code) => {
  if (code[0] === '#') return String.fromCodePoint(code[1] === 'x' || code[1] === 'X' ? parseInt(code.slice(2), 16) : Number(code.slice(1)));
  return ENTITIES[code.toLowerCase()] ?? match;
});

/** A proofread page's wikitext as plain text: paragraphs separated by blank lines. */
export function wikitextToText(wikitext) {
  let text = wikitext
    .replace(/<noinclude>[\s\S]*?<\/noinclude>/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<ref\b[^>]*\/>/g, '')
    .replace(/<ref\b[^>]*>[\s\S]*?<\/ref>/g, '')
    .replace(/<poem\b[^>]*>[\s\S]*?<\/poem>/g, '\n\n')
    .replace(/<(math|score|syntaxhighlight)\b[^>]*>[\s\S]*?<\/\1>/g, '')
    .replace(/\{\|[\s\S]*?\|\}/g, '\n\n');
  // Templates, innermost first.
  for (let guard = 0; guard < 20 && text.includes('{{'); guard++) {
    text = text.replace(/\{\{([^{}]*)\}\}/g, (_match, inner) => template(inner));
  }
  text = text
    .replace(/\[\[(?:File|Image|चित्र|फ़ाइल|फाइल|श्रेणी|Category):[^\]]*\]\]/gi, '')
    .replace(/\[\[[^|\]]*\|([^\]]*)\]\]/g, '$1')
    .replace(/\[\[([^\]]*)\]\]/g, '$1')
    .replace(/\[https?:[^\s\]]+ ?([^\]]*)\]/g, '$1')
    .replace(/'{2,}/g, '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/?(?:p|div|center)\b[^>]*>/gi, '\n\n')
    .replace(/<[^>]+>/g, '')
    .replace(/^[:;*#]+\s*/gm, '')
    .replace(/^=+\s*(.*?)\s*=+\s*$/gm, '\n$1\n');
  return decode(text);
}

/** Paragraphs, one per line: blank lines separate them, and a single line break inside one is a space. */
const paragraphs = text => text.split(/\n[ \t]*\n/).map(part => part.replace(/[\s​­]+/g, ' ').trim()).filter(Boolean);

async function fetchBook(book, dir, offline) {
  const cachePath = join(dir, 'raw', `${book.slug}.json`);
  const cache = offline ? new Map(JSON.parse(readFileSync(cachePath, 'utf8'))) : new Map();
  const read = async titles => {
    if (offline) return new Map(titles.filter(title => cache.has(title)).map(title => [title, cache.get(title)]));
    const found = await wikitexts(titles);
    for (const [title, entry] of found) cache.set(title, entry);
    return found;
  };
  const main = (await read([book.page])).get(book.page);
  if (!main) throw new Error(`${book.page} is missing`);
  const list = book.scan ? [] : await chapters(main.title, main, read, offline ? [...cache.keys()].filter(title => title.startsWith(`${main.title}/`)) : null, book.skip);
  // A book without chapters is shown whole on its main page; one with `scan` is read from its scan pages.
  const sources = book.scan ? [{ text: `<pages index="${book.scan.index}" from=${book.scan.from} to=${book.scan.to} />` }] : list.length ? list : [main];
  const scans = [];
  const inline = [];
  for (const source of sources) {
    const tags = pagesTags(source.text);
    if (!tags.length) inline.push(source.text);
    for (const { index, from, to } of tags) {
      if (!index || !from) continue;
      for (let n = toNumber(from); n <= toNumber(to ?? from); n++) {
        const title = `${PAGE_NAMESPACE}:${index}/${toDevanagari(n)}`;
        if (!scans.includes(title)) scans.push(title);
      }
    }
  }
  const texts = await read(scans);
  if (!offline) {
    mkdirSync(join(dir, 'raw'), { recursive: true });
    writeFileSync(cachePath, JSON.stringify([...cache]));
  }
  const used = [];
  let unproofread = 0;
  const pieces = [];
  for (const title of scans) {
    const page = texts.get(title);
    if (!page) continue;
    const level = Number(/<pagequality\s+level="(\d)"/.exec(page.text)?.[1] ?? 3);
    if (level <= 1) { unproofread++; continue; }
    used.push({ title: page.title, revid: page.revid, quality: level });
    // The line break after the page header is layout; a blank line left after it starts a paragraph.
    const body = wikitextToText(page.text).replace(/^[ \t]*\n/, '').replace(/\s+$/, '');
    pieces.push(body.startsWith('\n') ? `\n${body}` : body);
  }
  for (const text of inline) pieces.push(wikitextToText(text));
  // Pages join as Wikisource joins them: a paragraph runs on from one page to the next unless one is marked to end.
  const text = paragraphs(pieces.join(' ')).join('\n');
  const missing = scans.filter(title => !texts.has(title)).length;
  return { text, chapters: list.map(chapter => ({ title: chapter.title, revid: chapter.revid })), pages: used, unproofread, missing };
}

async function main() {
  const dir = process.argv[2];
  const offline = process.argv.includes('--offline');
  if (!dir) {
    console.error('Usage: node scripts/fetch-hindi-books.mjs <corpus-dir> [--offline] [--only <slug>]');
    process.exit(1);
  }
  mkdirSync(dir, { recursive: true });
  // --only <slug> fetches one book again and keeps the others' records.
  const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
  const sourcesPath = join(dir, 'hindi-sources.json');
  const sources = only && existsSync(sourcesPath) ? JSON.parse(readFileSync(sourcesPath, 'utf8')).filter(entry => entry.slug !== only) : [];
  for (const book of HINDI_BOOKS.filter(entry => !only || entry.slug === only)) {
    console.log(`${book.page} (${book.slug})`);
    const { text, chapters: list, pages, unproofread, missing } = await fetchBook(book, dir, offline);
    writeFileSync(join(dir, `${book.slug}.txt`), text + '\n');
    console.log(`  ${list.length} chapters, ${pages.length} pages (${unproofread} not yet proofread and ${missing} missing, left out), ${[...text].length.toLocaleString('en-US')} characters`);
    sources.push({ slug: book.slug, page: book.page, chapters: list, pages });
    sources.sort((a, b) => HINDI_BOOKS.findIndex(entry => entry.slug === a.slug) - HINDI_BOOKS.findIndex(entry => entry.slug === b.slug));
    writeFileSync(sourcesPath, JSON.stringify(sources, null, 1));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
