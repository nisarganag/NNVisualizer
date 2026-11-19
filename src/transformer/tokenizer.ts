/**
 * GPT-2 byte-level BPE tokenizer (the same algorithm as OpenAI's encoder.py),
 * using the vocabulary and merges shipped with the model.
 */

/** GPT-2 maps every byte to a printable character before applying merges. */
function byteToUnicode(): string[] {
  const bytes: number[] = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.slice();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + extra++);
    }
  }
  const table: string[] = new Array(256);
  bytes.forEach((b, i) => { table[b] = String.fromCodePoint(chars[i]); });
  return table;
}

const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Gpt2Tokenizer {
  private encoder: Map<string, number>;
  private decoder: string[];
  private ranks: Map<string, number>;
  private byteEncoder: string[];
  private byteDecoder: Map<string, number>;
  private cache = new Map<string, string[]>();
  private utf8 = new TextEncoder();

  constructor(vocab: Record<string, number>, merges: string) {
    this.encoder = new Map(Object.entries(vocab));
    this.decoder = [];
    for (const [token, id] of this.encoder) this.decoder[id] = token;
    this.ranks = new Map();
    merges.split('\n').filter(line => line && !line.startsWith('#version')).forEach((line, rank) => {
      this.ranks.set(line, rank);
    });
    this.byteEncoder = byteToUnicode();
    this.byteDecoder = new Map(this.byteEncoder.map((char, byte) => [char, byte]));
  }

  get vocabularySize(): number {
    return this.decoder.length;
  }

  private bpe(word: string): string[] {
    const cached = this.cache.get(word);
    if (cached) return cached;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; best = i; }
      }
      if (best < 0) break;
      const first = parts[best];
      const second = parts[best + 1];
      const merged: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === first && parts[i + 1] === second) {
          merged.push(first + second);
          i++;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    this.cache.set(word, parts);
    return parts;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    for (const match of text.matchAll(PRETOKENIZE)) {
      const mapped = Array.from(this.utf8.encode(match[0]), byte => this.byteEncoder[byte]).join('');
      for (const piece of this.bpe(mapped)) {
        const id = this.encoder.get(piece);
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      for (const char of this.decoder[id] ?? '') bytes.push(this.byteDecoder.get(char) ?? 63);
    }
    return new TextDecoder('utf-8').decode(new Uint8Array(bytes));
  }

  /** The text a single token stands for (a leading space is part of most word tokens). */
  tokenText(id: number): string {
    return this.decode([id]);
  }
}
