"""
wordpiece_alignment.py

Aligns word-level BIO labels to BioBERT WordPiece subword tokens for the V04
dataset. Reads the three word-level tokenized JSONL files produced by
dataset_generator.py (train, template_ood, symptom_ood) and writes parallel
wordpiece-aligned JSONL files ready for training.

Alignment rules (from V03):
  - Special tokens ([CLS], [SEP]) -> label "None" / id -100 (ignored in loss).
  - First subword of a word inherits the word's BIO label.
  - Continuation subwords inherit the label, with B- converted to I-.

The label set is fixed and hardcoded; LABEL2ID is not discovered from data so
that all three splits share the same mapping even if an OOD split happens to
lack a label.
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase


PROJECT_ROOT = Path("/Users/robertagarcia/Desktop/learning/bert_symptom_ner")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

LABEL_LIST: list[str] = [
    "O",
    "B-SYMPTOM_POS",
    "I-SYMPTOM_POS",
    "B-SYMPTOM_NEG",
    "I-SYMPTOM_NEG",
]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_DIR = PROJECT_ROOT / settings.VERSION

MAX_LENGTH = 512

SPLITS: dict[str, tuple[str, str]] = {
    "train": ("train_tokenized.jsonl", "train_wordpiece.jsonl"),
    "template_ood": ("template_ood_tokenized.jsonl", "template_ood_wordpiece.jsonl"),
    "symptom_ood": ("symptom_ood_tokenized.jsonl", "symptom_ood_wordpiece.jsonl"),
}


def align_sample(row: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Tokenizes a single word-level row with WordPiece and aligns BIO labels to
    subword tokens. Returns the input row augmented with wordpiece-level
    fields: tokens, input_ids, attention_mask, token_labels, token_label_ids.
    """
    words: list[str] = row["word_tokens"]
    word_labels: list[str] = row["word_labels"]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids: list[int] = encoding["input_ids"]
    attention_mask: list[int] = encoding["attention_mask"]
    word_ids: list[int | None] = encoding.word_ids()
    tokens: list[str] = tokenizer.convert_ids_to_tokens(input_ids)

    token_labels: list[str] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            token_labels.append("None")
        elif word_id != previous_word_id:
            token_labels.append(word_labels[word_id])
        else:
            # when will we fall in this case?
            original = word_labels[word_id]
            if isinstance(original, str) and original.startswith("B-"):
                token_labels.append("I-" + original[2:])
            else:
                token_labels.append(original)
        previous_word_id = word_id

    token_label_ids: list[int] = [
        -100 if lbl == "None" else LABEL2ID[lbl] for lbl in token_labels
    ]
    # what are the input_ids?
    return {
        **row,
        "tokens": tokens,
        "input_ids": input_ids, 
        "attention_mask": attention_mask,
        "token_labels": token_labels,
        "token_label_ids": token_label_ids,
    }


def align_file(
    input_path: Path,
    output_path: Path,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, int]:
    """
    Reads a word-level JSONL file, aligns every sample to WordPiece tokens,
    and writes the result to output_path. Returns (sample_count, truncated_count)
    where truncated_count is the number of samples that hit MAX_LENGTH.
    """
    sample_count = 0
    truncated_count = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aligned = align_sample(row, tokenizer)
            if len(aligned["input_ids"]) == MAX_LENGTH:
                truncated_count += 1
            fout.write(json.dumps(aligned) + "\n")
            sample_count += 1

    return sample_count, truncated_count


def save_label_mappings(out_dir: Path) -> None:
    """Persists LABEL2ID and ID2LABEL as JSON files for use by the training script."""
    with open(out_dir / "label2id.json", "w") as f:
        json.dump(LABEL2ID, f, indent=2)
    with open(out_dir / "id2label.json", "w") as f:
        json.dump(ID2LABEL, f, indent=2)
