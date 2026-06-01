/**
 * tokenizer.ts
 *
 * TypeScript port of the BioBERT (BertTokenizerFast) tokenizer used by the V05
 * Python pipeline. It must produce byte-for-byte identical `input_ids`, word
 * alignment (`wordIds`), and character `offsets` to
 * `AutoTokenizer.from_pretrained(artifacts/v05)`, because any divergence
 * silently corrupts every downstream NER prediction and span.
 *
 * Behaviour is driven by the authoritative `tokenizer.json` for these artifacts:
 *   BertNormalizer { clean_text: true, handle_chinese_chars: true,
 *                    strip_accents: null (→ follows lowercase), lowercase: true }
 *   BertPreTokenizer (split on whitespace + isolate punctuation)
 *   WordPiece { continuing_subword_prefix: "##", unk_token: "[UNK]",
 *               max_input_chars_per_word: 100 }
 *
 * OFFSETS: every token carries (start, end) char offsets into the ORIGINAL
 * (un-normalized) text. We track each normalized character back to its source
 * position through every normalization step, so offsets survive accent
 * stripping / lowercasing (e.g. "Café" → token "cafe" still maps to [0,4]).
 * Indexing is by Unicode code point (via Array.from), matching Python str
 * indices; this assumes BMP input (true for clinical English).
 *
 * NOTE: lowercasing is intentional and matches the exported tokenizer, even
 * though the backbone is the *cased* biobert-base-cased-v1.1. See the project
 * memory open-question note before "fixing" this.
 */

/** Special-token surface strings, per special_tokens_map.json. */
const CLS_TOKEN = '[CLS]';
const SEP_TOKEN = '[SEP]';
const UNK_TOKEN = '[UNK]';

/** WordPiece settings from tokenizer.json. */
const SUBWORD_PREFIX = '##';
const MAX_INPUT_CHARS_PER_WORD = 100;

/** A normalized character paired with its origin span in the original text. */
type AlignedChar = {
  /** Single normalized character. */
  ch: string;
  /** Inclusive start code-point index in the original text. */
  start: number;
  /** Exclusive end code-point index in the original text. */
  end: number;
};

/** One encoded sequence, aligned element-by-element. */
export type Encoding = {
  /** Token surface strings, including [CLS] / [SEP]. */
  tokens: string[];
  /** Vocab ids, parallel to `tokens`. */
  ids: number[];
  /**
   * Word index each token belongs to, parallel to `tokens`. Special tokens
   * ([CLS]/[SEP]) are null. Sub-words of the same word share an index.
   */
  wordIds: (number | null)[];
  /**
   * (start, end) char offsets into the ORIGINAL text, parallel to `tokens`.
   * Special tokens get [0, 0], matching HF's offset_mapping.
   */
  offsets: [number, number][];
};

/**
 * Parses a BERT `vocab.txt` (one token per line, line number = id) into a
 * token→id lookup map.
 */
export function loadVocab(vocabText: string): Map<string, number> {
  const vocab = new Map<string, number>();
  const lines = vocabText.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const token = lines[i].replace(/\r$/, '');
    if (token.length === 0 && i === lines.length - 1) {
      continue; // trailing blank line from a final newline
    }
    vocab.set(token, i);
  }
  return vocab;
}

/**
 * Builds a token→id map from a vocab array (index = id), as produced by
 * `bundle_app_assets.py` into `assets/model/vocab.json`. This is the form the
 * app uses at runtime; `loadVocab` (from raw text) is kept for the parity test.
 */
export function vocabFromArray(tokens: string[]): Map<string, number> {
  const vocab = new Map<string, number>();
  for (let i = 0; i < tokens.length; i++) {
    vocab.set(tokens[i], i);
  }
  return vocab;
}

/** True for whitespace per BERT (_is_whitespace): space/tab/newline/CR + Zs. */
function isWhitespace(ch: string): boolean {
  if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
    return true;
  }
  return /\p{Zs}/u.test(ch);
}

/** True for control chars per BERT (_is_control): category C* except tab/nl/cr. */
function isControl(ch: string): boolean {
  if (ch === '\t' || ch === '\n' || ch === '\r') {
    return false;
  }
  return /\p{C}/u.test(ch);
}

/** True for punctuation per BERT: the ASCII punct ranges plus Unicode P*. */
function isPunctuation(ch: string): boolean {
  const cp = ch.codePointAt(0) ?? 0;
  if (
    (cp >= 33 && cp <= 47) ||
    (cp >= 58 && cp <= 64) ||
    (cp >= 91 && cp <= 96) ||
    (cp >= 123 && cp <= 126)
  ) {
    return true;
  }
  return /\p{P}/u.test(ch);
}

/** True for combining marks (Unicode Mn), removed by accent stripping. */
function isCombiningMark(ch: string): boolean {
  return /\p{Mn}/u.test(ch);
}

/** True for CJK codepoints handled specially by BertNormalizer. */
function isChineseChar(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

/**
 * Applies the BertNormalizer pipeline (clean text → CJK spacing → strip accents
 * → lowercase) while preserving, for every output character, the code-point
 * span it originated from in `text`. Returns the aligned character sequence.
 */
function normalizeAligned(text: string): AlignedChar[] {
  // Start from the original code points, each mapped to its own [i, i+1) span.
  const original = Array.from(text);

  // 1. clean_text: drop NUL/replacement/control chars, collapse whitespace.
  let chars: AlignedChar[] = [];
  for (let i = 0; i < original.length; i++) {
    const ch = original[i];
    const cp = ch.codePointAt(0) ?? 0;
    if (cp === 0 || cp === 0xfffd || isControl(ch)) {
      continue;
    }
    chars.push({ ch: isWhitespace(ch) ? ' ' : ch, start: i, end: i + 1 });
  }

  // 2. handle_chinese_chars: surround each CJK char with (zero-width) spaces.
  let spaced: AlignedChar[] = [];
  for (const a of chars) {
    const cp = a.ch.codePointAt(0) ?? 0;
    if (isChineseChar(cp)) {
      spaced.push({ ch: ' ', start: a.start, end: a.start });
      spaced.push(a);
      spaced.push({ ch: ' ', start: a.end, end: a.end });
    } else {
      spaced.push(a);
    }
  }

  // 3. strip_accents: NFD-decompose each char, drop combining marks; surviving
  //    base characters inherit the source char's origin span.
  let deAccented: AlignedChar[] = [];
  for (const a of spaced) {
    for (const d of a.ch.normalize('NFD')) {
      if (isCombiningMark(d)) {
        continue;
      }
      deAccented.push({ ch: d, start: a.start, end: a.end });
    }
  }

  // 4. lowercase: a single char may lowercase to multiple chars; all inherit
  //    the source origin span.
  const lowered: AlignedChar[] = [];
  for (const a of deAccented) {
    for (const d of a.ch.toLowerCase()) {
      lowered.push({ ch: d, start: a.start, end: a.end });
    }
  }

  return lowered;
}

/**
 * Splits the aligned, normalized characters into "words" the way
 * BertPreTokenizer does: break on whitespace, then emit each punctuation char as
 * its own word. Each word is the list of its aligned characters (empty runs
 * dropped).
 */
function preTokenizeAligned(chars: AlignedChar[]): AlignedChar[][] {
  const words: AlignedChar[][] = [];
  let current: AlignedChar[] = [];

  const flush = (): void => {
    if (current.length > 0) {
      words.push(current);
      current = [];
    }
  };

  for (const a of chars) {
    if (isWhitespace(a.ch)) {
      flush();
    } else if (isPunctuation(a.ch)) {
      flush();
      words.push([a]);
    } else {
      current.push(a);
    }
  }
  flush();
  return words;
}

/** A WordPiece sub-token: surface string + its slice range within the word. */
type SubToken = {
  token: string;
  /** Inclusive start index into the word's aligned-char array. */
  startIdx: number;
  /** Exclusive end index into the word's aligned-char array. */
  endIdx: number;
};

/**
 * Greedy longest-match-first WordPiece for one pre-token word (given as its
 * aligned characters). Returns sub-tokens with their char-index ranges so the
 * caller can map them to original-text offsets. Returns a single [UNK] over the
 * whole word if it can't be segmented or exceeds the length cap.
 */
function wordpiece(word: AlignedChar[], vocab: Map<string, number>): SubToken[] {
  if (word.length > MAX_INPUT_CHARS_PER_WORD) {
    return [{ token: UNK_TOKEN, startIdx: 0, endIdx: word.length }];
  }

  const subTokens: SubToken[] = [];
  let start = 0;
  while (start < word.length) {
    let end = word.length;
    let curSubstr: string | null = null;
    while (start < end) {
      let substr = word
        .slice(start, end)
        .map(a => a.ch)
        .join('');
      if (start > 0) {
        substr = SUBWORD_PREFIX + substr;
      }
      if (vocab.has(substr)) {
        curSubstr = substr;
        break;
      }
      end -= 1;
    }
    if (curSubstr === null) {
      // is_bad → whole word maps to a single UNK spanning all chars.
      return [{ token: UNK_TOKEN, startIdx: 0, endIdx: word.length }];
    }
    subTokens.push({ token: curSubstr, startIdx: start, endIdx: end });
    start = end;
  }
  return subTokens;
}

/**
 * Encodes a single text into BERT inputs: [CLS] + word-piece tokens + [SEP],
 * with parallel ids, per-token word indices, and original-text char offsets.
 * Does not truncate (V05 inputs are short; add truncation in the inference layer
 * if ever needed).
 */
export function encode(text: string, vocab: Map<string, number>): Encoding {
  const tokens: string[] = [CLS_TOKEN];
  const wordIds: (number | null)[] = [null];
  const offsets: [number, number][] = [[0, 0]];

  const words = preTokenizeAligned(normalizeAligned(text));
  for (let w = 0; w < words.length; w++) {
    const word = words[w];
    for (const sub of wordpiece(word, vocab)) {
      tokens.push(sub.token);
      wordIds.push(w);
      offsets.push([word[sub.startIdx].start, word[sub.endIdx - 1].end]);
    }
  }

  tokens.push(SEP_TOKEN);
  wordIds.push(null);
  offsets.push([0, 0]);

  const unkId = vocab.get(UNK_TOKEN)!;
  const ids = tokens.map(t => vocab.get(t) ?? unkId);

  return { tokens, ids, wordIds, offsets };
}
