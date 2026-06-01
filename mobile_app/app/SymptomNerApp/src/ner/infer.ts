/**
 * infer.ts
 *
 * On-device counterpart to `mobile_app/model_prep/onnx_inference.py`
 * (predict_word_level_onnx). ONLY the forward pass lives here; the filtering and
 * token->word aggregation that follow are the shared, backend-agnostic logic in
 * aggregate.ts (ported once from inference_utils.py). This mirrors the Python
 * split exactly so the app reproduces the training/eval pipeline's results.
 *
 * This module covers Python lines 45-83: tokenize -> build feeds from the
 * graph's declared inputs -> run -> argmax over labels -> label strings.
 */

import { Tensor } from 'onnxruntime-react-native';
import { encode } from './tokenizer';
import { MODEL_CONFIG, VOCAB } from './modelConfig';
import { getSession } from './session';
import {
  aggregateTokenPredictionsToWords,
  wordLabelsToSpans,
  type Span,
} from './aggregate';

/** Per-token forward-pass result, aligned element-by-element. */
export type TokenLevelResult = {
  /** Token surface strings, including [CLS]/[SEP]. */
  tokens: string[];
  /** Per-token original-word index; null for special tokens. */
  wordIds: (number | null)[];
  /** Per-token (start, end) char offsets into the original text. */
  offsets: [number, number][];
  /** Per-token predicted label string (via id2label). */
  predTokenLabels: string[];
};

/**
 * Runs token-classification inference via ONNX Runtime and returns per-token
 * predicted label strings (plus the alignment needed for aggregation). Mirrors
 * the forward pass of predict_word_level_onnx; aggregation to words/spans is
 * applied separately by the caller using aggregate.ts.
 */
export async function predictTokenLabels(
  text: string,
): Promise<TokenLevelResult> {
  // 1. Tokenize (ids + alignment + offsets).
  const { ids, tokens, wordIds, offsets } = encode(text, VOCAB);
  const seqLen = ids.length;

  // 2. Build int64 tensors. ONNX Runtime RN represents int64 as BigInt64Array.
  //    Single, unpadded sequence: attention_mask all 1s, token_type_ids all 0s.
  const dims = [1, seqLen];
  const inputIds = BigInt64Array.from(ids, id => BigInt(id));
  const attentionMask = new BigInt64Array(seqLen).fill(1n);
  const tokenTypeIds = new BigInt64Array(seqLen); // zero-filled

  const tensorByName: Record<string, Tensor> = {
    input_ids: new Tensor('int64', inputIds, dims),
    attention_mask: new Tensor('int64', attentionMask, dims),
    token_type_ids: new Tensor('int64', tokenTypeIds, dims),
  };

  // Feed exactly the inputs the graph declares (mirrors Python's name filter).
  const { session, inputNames, outputNames } = await getSession();
  const feeds: Record<string, Tensor> = {};
  for (const name of inputNames) {
    if (tensorByName[name] !== undefined) {
      feeds[name] = tensorByName[name];
    }
  }

  // 3. Run. logits shape (1, seqLen, numLabels), flat in row-major order.
  const output = await session.run(feeds);
  const logits = output[outputNames[0]];
  const data = logits.data as Float32Array;
  const numLabels = MODEL_CONFIG.numLabels;

  // 4. argmax over the label axis per token -> label string.
  const predTokenLabels: string[] = [];
  for (let t = 0; t < seqLen; t++) {
    let bestId = 0;
    let bestVal = -Infinity;
    for (let l = 0; l < numLabels; l++) {
      const v = data[t * numLabels + l];
      if (v > bestVal) {
        bestVal = v;
        bestId = l;
      }
    }
    predTokenLabels.push(MODEL_CONFIG.id2label[bestId]);
  }

  return { tokens, wordIds, offsets, predTokenLabels };
}

/**
 * Full on-device NER for a piece of text: forward pass -> filter/aggregate to
 * words -> BIO word labels to character spans. Equivalent to running
 * predict_word_level_onnx followed by word_labels_to_spans in Python.
 */
export async function predictSpans(text: string): Promise<Span[]> {
  const { wordIds, offsets, predTokenLabels } = await predictTokenLabels(text);
  const { wordLabels, wordOffsets } = aggregateTokenPredictionsToWords(
    text,
    wordIds,
    offsets,
    predTokenLabels,
  );
  return wordLabelsToSpans(text, wordOffsets, wordLabels);
}
