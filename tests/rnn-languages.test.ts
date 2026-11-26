import { describe, expect, it } from 'vitest';
import { IN_WORD, LETTER, languageFrom } from '../src/rnn/languages';
import { characterName, glyph } from '../src/rnn/views';
import { simplifyCharacters } from '../src/rnn/model';

describe('the Recurrent page in English or Hindi', () => {
  it('takes the language from the address, English unless it asks for Hindi', () => {
    expect(languageFrom('').code).toBe('en');
    expect(languageFrom('?lang=fr').code).toBe('en');
    const hindi = languageFrom('?lang=hi');
    expect(hindi.code).toBe('hi');
    expect(hindi.lstm).toBe('/models/char-lstm-hi-v1');
    expect(hindi.rnn).toBe('/models/char-rnn-hi-v1');
    expect(hindi.examples).toContain(hindi.text);
  });

  it('counts Devanagari vowel signs, viramas and nuktas as part of a word, and the danda as punctuation', () => {
    for (const c of ['a', 'Z', 'क', 'ि', '्', 'ं', '़']) expect(LETTER.test(c)).toBe(true);
    for (const c of ['।', ' ', '"', '-', '\n']) expect(LETTER.test(c)).toBe(false);
    expect(IN_WORD.test("'")).toBe(true);
  });

  it('shows a lone vowel sign on a dotted circle and names the danda', () => {
    expect(glyph('ि')).toBe('◌ि');
    expect(glyph('क')).toBe('क');
    expect(glyph(' ')).toBe('␣');
    expect(characterName('।')).toBe('danda');
    expect(characterName('ा')).toBe('◌ा');
  });

  it('splits the nukta off precomposed letters, as the Hindi networks were trained', () => {
    // क़ (U+0958) is क + ़ in the training text.
    expect(simplifyCharacters('क़')).toBe('क़');
    expect(simplifyCharacters('मित्रता थी।')).toBe('मित्रता थी।');
  });
});
