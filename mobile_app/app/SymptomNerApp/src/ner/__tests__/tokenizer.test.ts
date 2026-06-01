/**
 * tokenizer.test.ts
 *
 * Parity gate for the TypeScript BioBERT tokenizer. Each golden vector below was
 * produced by the real Python `AutoTokenizer.from_pretrained('mobile_app/artifacts/v05')`
 * (transformers 4.57.3). The TS `encode()` must reproduce both `input_ids` and
 * the word alignment exactly — any mismatch is a blocking bug, since the on-device
 * app must stay bit-for-bit consistent with the Python evaluation pipeline.
 *
 * Vocab is read from the canonical artifacts copy; it will be bundled into the
 * app's assets in Block C step 3.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { encode, loadVocab } from '../tokenizer';

/** A single Python-derived expectation. wordIds use -1 for special tokens. */
type GoldenVector = {
  text: string;
  ids: number[];
  wordIds: number[];
  offsets: [number, number][];
};

// Generated from AutoTokenizer.from_pretrained('mobile_app/artifacts/v05').
const GOLDEN: GoldenVector[] = [
  {
    text: 'Patient denies Fever.',
    ids: [101, 5351, 26360, 10880, 119, 102],
    wordIds: [-1, 0, 1, 2, 3, -1],
    offsets: [[0, 0], [0, 7], [8, 14], [15, 20], [20, 21], [0, 0]],
  },
  {
    text: 'No Headache but reports NAUSEA',
    ids: [101, 1185, 16320, 1133, 3756, 22882, 102],
    wordIds: [-1, 0, 1, 2, 3, 4, -1],
    offsets: [[0, 0], [0, 2], [3, 11], [12, 15], [16, 23], [24, 30], [0, 0]],
  },
  {
    text: 'Female patient, 38 years old, presents with a 5-day history of allergic reaction.',
    ids: [
      101, 2130, 5351, 117, 3383, 1201, 1385, 117, 8218, 1114, 170, 126, 118,
      1285, 1607, 1104, 1155, 26949, 3943, 119, 102,
    ],
    wordIds: [
      -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, -1,
    ],
    offsets: [
      [0, 0], [0, 6], [7, 14], [14, 15], [16, 18], [19, 24], [25, 28], [28, 29],
      [30, 38], [39, 43], [44, 45], [46, 47], [47, 48], [48, 51], [52, 59],
      [60, 62], [63, 66], [66, 71], [72, 80], [80, 81], [0, 0],
    ],
  },
  {
    text: 'She denies jaundice, and there is no history of similar episodes.',
    ids: [
      101, 1131, 26360, 179, 3984, 12090, 2093, 117, 1105, 1175, 1110, 1185,
      1607, 1104, 1861, 3426, 119, 102,
    ],
    wordIds: [-1, 0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1],
    offsets: [
      [0, 0], [0, 3], [4, 10], [11, 12], [12, 14], [14, 17], [17, 19], [19, 20],
      [21, 24], [25, 30], [31, 33], [34, 36], [37, 44], [45, 47], [48, 55],
      [56, 64], [64, 65], [0, 0],
    ],
  },
  {
    text: 'On direct questioning she endorses maculopapular rash.',
    ids: [
      101, 1113, 2904, 11402, 1131, 1322, 18649, 1116, 23639, 22806, 4163,
      16091, 5815, 187, 10733, 119, 102,
    ],
    wordIds: [-1, 0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, -1],
    offsets: [
      [0, 0], [0, 2], [3, 9], [10, 21], [22, 25], [26, 29], [29, 33], [33, 34],
      [35, 38], [38, 41], [41, 43], [43, 45], [45, 48], [49, 50], [50, 53],
      [53, 54], [0, 0],
    ],
  },
  {
    text: '38.5C fever x3 days, non-productive cough.',
    ids: [
      101, 3383, 119, 126, 1665, 10880, 193, 1495, 1552, 117, 1664, 118, 13909,
      21810, 119, 102,
    ],
    wordIds: [-1, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, -1],
    offsets: [
      [0, 0], [0, 2], [2, 3], [3, 4], [4, 5], [6, 11], [12, 13], [13, 14],
      [15, 19], [19, 20], [21, 24], [24, 25], [25, 35], [36, 41], [41, 42],
      [0, 0],
    ],
  },
  {
    text: 'Café au lait macules noted on exam.',
    ids: [101, 17287, 12686, 2495, 2875, 23639, 11806, 2382, 1113, 12211, 119, 102],
    wordIds: [-1, 0, 1, 2, 2, 3, 3, 4, 5, 6, 7, -1],
    offsets: [
      [0, 0], [0, 4], [5, 7], [8, 10], [10, 12], [13, 16], [16, 20], [21, 26],
      [27, 29], [30, 34], [34, 35], [0, 0],
    ],
  },
  {
    text: 'c/o shortness of breath; denies chest-pain.',
    ids: [
      101, 172, 120, 184, 1603, 1757, 1104, 2184, 132, 26360, 2229, 118, 2489,
      119, 102,
    ],
    wordIds: [-1, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1],
    offsets: [
      [0, 0], [0, 1], [1, 2], [2, 3], [4, 9], [9, 13], [14, 16], [17, 23],
      [23, 24], [25, 31], [32, 37], [37, 38], [38, 42], [42, 43], [0, 0],
    ],
  },
  {
    // Control char (U+0007) is dropped (no word boundary); tab becomes a space
    // (word boundary). Verified against AutoTokenizer.
    text: 'fever and\tchills',
    ids: [101, 10880, 1105, 11824, 1116, 102],
    wordIds: [-1, 0, 1, 2, 2, -1],
    offsets: [[0, 0], [0, 5], [7, 10], [11, 16], [16, 17], [0, 0]],
  },
];

const VOCAB_PATH = resolve(
  __dirname,
  '../../../../../artifacts/v05/vocab.txt',
);

describe('BioBERT tokenizer parity with Python AutoTokenizer', () => {
  const vocab = loadVocab(readFileSync(VOCAB_PATH, 'utf8'));

  it('loads the cased BioBERT vocab (28996 tokens, special ids correct)', () => {
    expect(vocab.size).toBe(28996);
    expect(vocab.get('[PAD]')).toBe(0);
    expect(vocab.get('[UNK]')).toBe(100);
    expect(vocab.get('[CLS]')).toBe(101);
    expect(vocab.get('[SEP]')).toBe(102);
  });

  for (const vec of GOLDEN) {
    it(`matches input_ids for: ${vec.text}`, () => {
      const { ids } = encode(vec.text, vocab);
      expect(ids).toEqual(vec.ids);
    });

    it(`matches word alignment for: ${vec.text}`, () => {
      const { wordIds } = encode(vec.text, vocab);
      const asInts = wordIds.map(w => (w === null ? -1 : w));
      expect(asInts).toEqual(vec.wordIds);
    });

    it(`matches char offsets for: ${vec.text}`, () => {
      const { offsets } = encode(vec.text, vocab);
      expect(offsets).toEqual(vec.offsets);
    });
  }
});
