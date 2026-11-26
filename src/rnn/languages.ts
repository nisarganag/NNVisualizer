/**
 * The Recurrent page reads English or Hindi: each language has its own pair of networks
 * (the LSTM and the gateless one, trained on that language's nineteen books), its own
 * example texts and its own credits. The page's own words stay in English. The language
 * comes from the address (?lang=hi), so a Hindi link opens in Hindi.
 */
export type Language = 'en' | 'hi';

export interface LanguageContent {
  code: Language;
  /** Model files, without the .json and .bin extensions. */
  lstm: string;
  rnn: string;
  /** Text the page starts with, and the examples offered. */
  text: string;
  /** Written as the networks read them, with the nukta as its own character (as in ज़), one box per character. */
  examples: string[];
  /** The introduction under the heading. */
  intro: string;
  /**
   * The credits paragraph; `{lstm-bits}` and `{rnn-bits}` mark where each network's held-out
   * score goes, as spans the page fills in.
   */
  credits: string;
  /** Names for characters that don't start a word, besides the shared ones. */
  marks: Record<string, string>;
}

const HINDI: LanguageContent = {
  code: 'hi',
  lstm: '/models/char-lstm-hi-v1',
  rnn: '/models/char-rnn-hi-v1',
  text: 'तोते ने कहा-"कंजरों को जगा दे, नहीं तो शेर अभी हम सब के ऊपर आ जायगा"। राजा की लड़की झट उठ खड़ी हुई और जितने ज़ोर से उससे हो सका चिल्लाने लगी-"शेर शेर, जागो जागो!" और कंजर जल्द जाग गये।',
  examples: [
    'तोते ने कहा-"कंजरों को जगा दे, नहीं तो शेर अभी हम सब के ऊपर आ जायगा"। राजा की लड़की झट उठ खड़ी हुई और जितने ज़ोर से उससे हो सका चिल्लाने लगी-"शेर शेर, जागो जागो!" और कंजर जल्द जाग गये।',
    'हरिद्वार से आगे उत्तर की तरफ़ पहाड़ों में दूर तक चले जाइये तो एक ऐसे मुक़ाम पर पहुंचियेगा',
    'जुम्मन शेख और अलगू चौधरी में गाढ़ी मित्रता थी। साझेमें खेती होती थी। कुछ लेन-देनमें भी साझा था।',
    '"क्या चालीस हजार पर भी नहीं?" "चालीस हजार नहीं, चालीस लाखपर भी असम्भव है।"',
    'छोटा रोबोट समुद्र के किनारे गया और बोला, "नमस्ते?"',
  ],
  intro: 'A small recurrent network reads your text one character at a time, carrying a memory of 256 numbers from each character to the next. From nineteen public-domain Hindi books it learned how Hindi is spelled and punctuated, but not what it means, so its own sentences sound like Hindi without making sense. It reads Devanagari one Unicode character at a time, so कि takes two steps: क, then the vowel sign ि.',
  credits: 'This network has the same design as the English one, a long short-term memory (LSTM) with one layer of 256 cells, and was trained for this site on nineteen public-domain Hindi books from <a href="https://hi.wikisource.org/">Hindi Wikisource</a>, first published between 1803 and 1930: folk and children’s tales such as <cite lang="hi">रानी केतकी की कहानी</cite>, Shridhar Pathak’s <cite lang="hi">तिलिस्माती मुँदरी</cite> and Lallu Lal’s <cite lang="hi">प्रेमसागर</cite>, Premchand’s stories and novels, among them <cite lang="hi">सप्तसरोज</cite> and <cite lang="hi">निर्मला</cite>, Jaishankar Prasad’s stories <cite lang="hi">आकाश-दीप</cite>, and four volumes of Devaki Nandan Khatri’s <cite lang="hi">चन्द्रकांता सन्तति</cite>. Paragraphs with caste, communal or gendered slurs were left out. The last 5% of each book was held out from training; on that text it averages {lstm-bits} bits per character. The network it is compared with has no gates: one layer of 512 units, h = tanh(W<sub>x</sub>x + Uh + b), trained the same way on the same books. On the same held-out text it averages {rnn-bits} bits per character. Both run entirely in your browser; nothing you type leaves your device.',
  marks: { '।': 'danda' },
};

const ENGLISH: LanguageContent = {
  code: 'en',
  lstm: '/models/char-lstm-v1',
  rnn: '/models/char-rnn-v1',
  // The English page's text, examples, introduction and credits are in rnn/index.html.
  text: '',
  examples: [],
  intro: '',
  credits: '',
  marks: {},
};

/** The language the address asks for; English unless it says ?lang=hi. */
export function languageFrom(search: string): LanguageContent {
  return new URLSearchParams(search).get('lang') === 'hi' ? HINDI : ENGLISH;
}

/** Letters, including the vowel signs and other marks Devanagari writes with them. */
export const LETTER = /[\p{L}\p{M}]/u;
/** What can sit inside a word: letters, marks and an English apostrophe. */
export const IN_WORD = /[\p{L}\p{M}']/u;
