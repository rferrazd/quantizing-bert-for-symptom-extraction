/**
 * modelConfig.ts
 *
 * Loads model-version-specific configuration from the bundled `config.json`
 * (written by bundle_app_assets.py). Everything that changes between model
 * versions — the label scheme, label count, max sequence length — is derived
 * here at runtime, so app code contains no hardcoded V05 assumptions. Switching
 * to V06 only requires re-bundling assets, not editing this file.
 */

import rawConfig from '../../assets/model/config.json';
import vocabTokens from '../../assets/model/vocab.json';
import { vocabFromArray } from './tokenizer';

/** Shape of the fields we read out of a HuggingFace BERT config.json. */
type RawConfig = {
  id2label: Record<string, string>;
  max_position_embeddings?: number;
};

/** Resolved, app-facing model configuration. */
export type ModelConfig = {
  /** Label strings indexed by class id (e.g. ['O','B-SYMPTOM_POS', ...]). */
  id2label: string[];
  /** Number of output classes. */
  numLabels: number;
  /** Maximum sequence length the model supports (for truncation). */
  maxSeqLen: number;
};

/**
 * Converts the raw `id2label` object ({"0":"O",...}) into a dense array ordered
 * by numeric class id. Throws if ids are non-contiguous, which would indicate a
 * malformed config.
 */
function buildId2Label(raw: Record<string, string>): string[] {
  const entries = Object.entries(raw).map(
    ([k, v]) => [Number(k), v] as [number, string],
  );
  entries.sort((a, b) => a[0] - b[0]);
  const labels = entries.map(([, label]) => label);
  for (let i = 0; i < entries.length; i++) {
    if (entries[i][0] !== i) {
      throw new Error(
        `config.json id2label is not contiguous from 0: got id ${entries[i][0]} at position ${i}`,
      );
    }
  }
  return labels;
}

/** Parses the bundled config.json into a ModelConfig. */
function buildModelConfig(): ModelConfig {
  const raw = rawConfig as RawConfig;
  const id2label = buildId2Label(raw.id2label);
  return {
    id2label,
    numLabels: id2label.length,
    maxSeqLen: raw.max_position_embeddings ?? 512,
  };
}

/** Resolved configuration for the currently-bundled model version. */
export const MODEL_CONFIG: ModelConfig = buildModelConfig();

/** The currently-bundled tokenizer vocabulary (token → id). */
export const VOCAB: Map<string, number> = vocabFromArray(
  vocabTokens as string[],
);
