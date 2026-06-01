/**
 * aggregate.ts
 *
 * Port of the shared, backend-agnostic NER logic from
 * `inference/v01/inference_utils.py`:
 *   - aggregate_token_predictions_to_words  (filter specials + token->word)
 *   - _aggregate_token_labels_to_word        (BIO label aggregation per word)
 *   - word_labels_to_spans                   (BIO word labels -> char spans)
 *
 * This is the same logic the PyTorch and ONNX Python paths share; porting it
 * once here keeps the on-device app bit-for-bit consistent with the training /
 * eval pipeline. Pure functions only (no model, no native deps) so they are
 * verified in Node against the Python reference.
 *
 * Only the `return_bio=True` behaviour of _aggregate_token_labels_to_word is
 * ported, because word_labels_to_spans is the sole consumer and always needs
 * BIO labels; the collapsed-label branch is unused by this pipeline.
 */

/** A character-level entity span over the original text. */
export type Span = {
  start: number;
  end: number;
  text: string;
  label: string;
};

/** Word-level aggregation result, parallel arrays aligned by word. */
export type WordLevelResult = {
  /** Word surface strings sliced from the original text. */
  words: string[];
  /** One aggregated BIO label per word. */
  wordLabels: string[];
  /** One (start, end) char offset per word, into the original text. */
  wordOffsets: [number, number][];
};

/**
 * Aggregates token-level labels of a single word into one word-level BIO label.
 * Mirrors _aggregate_token_labels_to_word (return_bio=True):
 *  - all 'O' -> 'O'
 *  - single non-O polarity -> the B-* label if any token is B-*, else first I-*
 *  - mixed polarities -> 'CONFLICT-<labels joined by '-'>'
 */
function aggregateTokenLabelsToWord(tokenLabels: string[]): string {
  const polarities: string[] = [];
  const bioLabels: string[] = [];

  for (const label of tokenLabels) {
    if (label === 'O') {
      polarities.push('O');
    } else {
      polarities.push(label.split('_').pop() as string); // POS / NEG
      bioLabels.push(label);
    }
  }

  if (polarities.every(p => p === 'O')) {
    return 'O';
  }

  const nonOPolarities = polarities.filter(p => p !== 'O');

  if (new Set(nonOPolarities).size === 1) {
    for (const lbl of bioLabels) {
      if (lbl.startsWith('B-')) {
        return lbl;
      }
    }
    return bioLabels[0];
  }

  return `CONFLICT-${tokenLabels.join('-')}`;
}

/**
 * Filters special tokens (wordId === null) and aggregates token-level
 * predictions up to word level. Mirrors aggregate_token_predictions_to_words;
 * returns only the word-level arrays needed downstream (the filtered_* tuples in
 * Python are intermediate and unused by the span step).
 */
export function aggregateTokenPredictionsToWords(
  text: string,
  wordIds: (number | null)[],
  offsets: [number, number][],
  predTokenLabels: string[],
): WordLevelResult {
  const words: string[] = [];
  const wordLabels: string[] = [];
  const wordOffsets: [number, number][] = [];

  let currentWordId: number | null = null;
  let currentTokenLabels: string[] = [];
  let currentStart = 0;
  let currentEnd = 0;
  let inWord = false;

  const flush = (): void => {
    if (inWord) {
      wordLabels.push(aggregateTokenLabelsToWord(currentTokenLabels));
      wordOffsets.push([currentStart, currentEnd]);
      words.push(text.slice(currentStart, currentEnd));
    }
  };

  for (let i = 0; i < wordIds.length; i++) {
    const wordId = wordIds[i];
    if (wordId === null) {
      continue; // skip [CLS]/[SEP] and padding, in lockstep
    }
    const [tokStart, tokEnd] = offsets[i];

    if (wordId !== currentWordId) {
      flush();
      currentWordId = wordId;
      currentTokenLabels = [predTokenLabels[i]];
      currentStart = tokStart;
      currentEnd = tokEnd;
      inWord = true;
    } else {
      currentTokenLabels.push(predTokenLabels[i]);
      currentEnd = tokEnd;
    }
  }
  flush();

  return { words, wordLabels, wordOffsets };
}

/**
 * Converts word-level BIO labels into character-level spans over `text`.
 * Mirrors word_labels_to_spans, including: 'O' emitted as its own span, 'B-'
 * always starts a new entity, 'I-' extends a matching entity or otherwise starts
 * a new one, and unexpected label formats emitted as standalone spans.
 */
export function wordLabelsToSpans(
  text: string,
  wordOffsets: [number, number][],
  wordLabels: string[],
): Span[] {
  if (wordOffsets.length !== wordLabels.length) {
    throw new Error('wordOffsets and wordLabels must be same length');
  }

  const spans: Span[] = [];
  let currentLabel: string | null = null;
  let currentStart = 0;
  let currentEnd = 0;

  const flushCurrent = (): void => {
    if (currentLabel !== null) {
      spans.push({
        start: currentStart,
        end: currentEnd,
        text: text.slice(currentStart, currentEnd),
        label: currentLabel,
      });
      currentLabel = null;
    }
  };

  for (let i = 0; i < wordLabels.length; i++) {
    const rawLabel = wordLabels[i];
    const [wStart, wEnd] = wordOffsets[i];

    if (rawLabel === 'O') {
      flushCurrent();
      spans.push({ start: wStart, end: wEnd, text: text.slice(wStart, wEnd), label: 'O' });
      continue;
    }

    const dashIdx = rawLabel.indexOf('-');
    const prefix = dashIdx === -1 ? rawLabel : rawLabel.slice(0, dashIdx);
    const norm = dashIdx === -1 ? '' : rawLabel.slice(dashIdx + 1);

    if (prefix === 'B') {
      flushCurrent();
      currentLabel = norm;
      currentStart = wStart;
      currentEnd = wEnd;
    } else if (prefix === 'I') {
      if (currentLabel === norm) {
        currentEnd = wEnd;
      } else {
        flushCurrent();
        currentLabel = norm;
        currentStart = wStart;
        currentEnd = wEnd;
      }
    } else {
      flushCurrent();
      spans.push({ start: wStart, end: wEnd, text: text.slice(wStart, wEnd), label: rawLabel });
    }
  }

  flushCurrent();
  return spans;
}
