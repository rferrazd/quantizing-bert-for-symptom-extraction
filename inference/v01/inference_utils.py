"""File with functions for performing inference"""

import torch, re
from typing import List, Dict

def predict_token_level(text, model, tokenizer, device="cpu"):
    """
    Predict token-level label IDs for `text`.
    Returns filtered tokens (no special tokens) and corresponding label IDs (ints).
    """

    # 1) Tokenize and get word_ids (word_ids == None for special tokens)
    encoding = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_ids = encoding.word_ids(batch_index=0)  # list aligned to tokens
    
    # 2) Move inputs to device and run model
    inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy().tolist()  # list of ints
    
    # 3) Filter everything by word_id is not None (keeps positional alignment)
    #    word_ids is the single source of truth for aligning tokenizer tokens 
    #    to original words — filtering by it prevents silent misalignment bugs

    filtered_tokens = []
    filtered_preds = []
    for tok, wid, pid in zip(tokens, word_ids, pred_ids):
        if wid is None:
            # skip [CLS], [SEP], and any special/padding tokens (consistent filtering)
            continue
        filtered_tokens.append(tok)
        filtered_preds.append(int(pid)) 
    
    # 4) Return token strings and integer label ids (aligned)
    return filtered_tokens, filtered_preds


def predict_word_level(
    text: str,
    model,
    tokenizer,
    id2label: dict,
    device: str,
    ):
    """
    Runs token-level inference and aggregates predictions to the word level.

    Two-stage logic:
    1) Token-level predictions (model output)
    2) Token → word aggregation using tokenizer.word_ids()

    Returns:
        - tokens (filtered, aligned with word_ids)
        - token-level labels
        - word_ids (one per token)
        - words (word strings aligned to tokenizer word_ids)
        - word-level labels (one per word)
        - word_offsets (one (start,end) pair per word, into original `text`)
    """

    # -------------------------------
    # 1. Tokenization
    # -------------------------------
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # word_ids maps each token → original word index
    # Special tokens ([CLS], [SEP]) get word_id = None
    word_ids = encoding.word_ids(batch_index=0)

    # Offsets (start, end) per token into the ORIGINAL `text`.
    # Note: requires a "fast" tokenizer (HF tokenizers-backed); BioBERT tokenizers are typically fast.
    offsets = encoding["offset_mapping"][0].tolist()

    # -------------------------------
    # 2. Model inference (token-level)
    # -------------------------------
    # Exclude 'offset_mapping' from inputs - it's only for span extraction, not model inference
    inputs = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

    # Convert predicted label IDs → label strings
    pred_token_labels = [id2label[p.item()] for p in predictions]

    # -------------------------------
    # 3. FILTER EVERYTHING TOGETHER
    # -------------------------------
    # This is CRITICAL.
    # We remove special tokens by checking word_id is None.
    # ALL lists must be filtered in lockstep to avoid misalignment.
    filtered_tokens = []
    filtered_word_ids = []
    filtered_token_labels = []
    filtered_offsets = []

    for token, word_id, label, offset in zip(tokens, word_ids, pred_token_labels, offsets):
        if word_id is None:
            continue
        filtered_tokens.append(token)
        filtered_word_ids.append(word_id)
        filtered_token_labels.append(label)
        filtered_offsets.append(tuple(offset))

    # Sanity check
    assert len(filtered_tokens) == len(filtered_word_ids) == len(filtered_token_labels) == len(filtered_offsets)

    # -------------------------------
    # 4. Aggregate TOKENS → WORDS
    # -------------------------------
    word_labels = []
    words = []
    word_offsets = []

    current_word_id = None
    current_word_token_labels = []
    current_word_start = None
    current_word_end = None

    for idx, word_id in enumerate(filtered_word_ids):
        token_label = filtered_token_labels[idx]
        tok_start, tok_end = filtered_offsets[idx]

        # New word encountered
        if word_id != current_word_id:
            # Save label for previous word (if exists)
            if current_word_id is not None:
                word_label = _aggregate_token_labels_to_word(
                    current_word_token_labels
                )
                word_labels.append(word_label)
                word_offsets.append((current_word_start, current_word_end))
                words.append(text[current_word_start:current_word_end])

            # Start collecting labels for new word
            current_word_id = word_id
            current_word_token_labels = [token_label]
            current_word_start = tok_start
            current_word_end = tok_end

        else:
            # Same word → multiple subword tokens
            current_word_token_labels.append(token_label)
            # extend to end of latest subtoken
            current_word_end = tok_end

    # Handle last word
    if current_word_token_labels:
        word_label = _aggregate_token_labels_to_word(current_word_token_labels)
        word_labels.append(word_label)
        word_offsets.append((current_word_start, current_word_end))
        words.append(text[current_word_start:current_word_end])

    return (
        filtered_tokens,
        filtered_token_labels,
        filtered_word_ids,
        words,
        word_labels,
        word_offsets,
    )

def _aggregate_token_labels_to_word(
    token_labels: List[str],
    return_bio: bool = True,
) -> str:
    """
    Aggregate token-level labels belonging to the same word into one word-level label.

    This function supports TWO modes:
    1) return_bio=True  → preserves BIO (used for entity span extraction)
    2) return_bio=False → collapses to SYMPTOM_POS / SYMPTOM_NEG (used earlier)

    Examples:
        ['B-SYMPTOM_POS', 'I-SYMPTOM_POS'] → 'B-SYMPTOM_POS'
        ['I-SYMPTOM_POS', 'I-SYMPTOM_POS'] → 'I-SYMPTOM_POS'
        ['O', 'O'] → 'O'
    """

    # Collect useful information
    polarities = []
    bio_labels = []

    for label in token_labels:
        if label == "O":
            polarities.append("O")
        else:
            polarity = label.split("_")[-1]   # POS / NEG
            polarities.append(polarity)
            bio_labels.append(label)

    # --------------------------------------------------
    # Case 1: all tokens are 'O'
    # --------------------------------------------------
    if all(p == "O" for p in polarities):
        return "O"

    # Remove 'O' for polarity checks
    non_o_polarities = [p for p in polarities if p != "O"]

    # --------------------------------------------------
    # Case 2: same polarity (POS or NEG)
    # --------------------------------------------------
    if len(set(non_o_polarities)) == 1:
        polarity = non_o_polarities[0]

        if return_bio:
            # BIO PRIORITY RULE:
            # If ANY token is B-*, the word should be B-*
            for lbl in bio_labels:
                if lbl.startswith("B-"):
                    return lbl

            # Otherwise, fall back to I-*
            return bio_labels[0]

        # Old behavior (collapsed label)
        return f"SYMPTOM_{polarity}"

    # --------------------------------------------------
    # Case 3: conflicting polarities inside the same word
    # --------------------------------------------------
    return f"CONFLICT-{'-'.join(token_labels)}"




def word_labels_to_spans(text: str, word_offsets: List[tuple], word_labels: List[str]) -> List[Dict]:
    """
    Convert word-level BIO labels into character-level spans over the ORIGINAL `text`.

    - Input labels are like: "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "B-SYMPTOM_NEG", ...
    - Output spans use **character offsets** into the provided `text`

    Behavior:
    - "O" becomes its own single-word span (per your earlier behavior).
      (If you want to drop O spans, we can remove that.)
    - "B-XXX" starts a new entity span of type XXX.
    - "I-XXX" continues the current entity if it matches XXX; otherwise it starts a new entity
      (robust to "I" appearing without a matching previous "B").
    """

    # Sanity check: labels must align 1:1 with word offsets
    if len(word_offsets) != len(word_labels):
        print(f"ERROR: Length mismatch!")
        print(f"  word_offsets ({len(word_offsets)}): {word_offsets}")
        print(f"  word_labels ({len(word_labels)}): {word_labels}")
        raise AssertionError("word_offsets and word_labels must be same length")

    spans: List[Dict] = []

    # -------------------------------------------------------------------------
    # Step 1) Use the provided word-level (start,end) offsets into the original text
    positions = list(word_offsets)

    # -------------------------------------------------------------------------
    # Step 2) Track the "current" entity we are building as we scan tokens.
    # If current_label is None, we are not currently inside an entity.
    # -------------------------------------------------------------------------
    current_label = None    # normalized label like "SYMPTOM_POS" (no "B-" / "I-")
    current_start = None    # char start of the entity in `text`
    current_end = None      # char end (exclusive) of the entity in `text`

    # Helper: when an entity ends, emit it into spans and clear state
    def flush_current():
        # `nonlocal` allows this nested function to modify variables from the outer function's scope.
        # Without it, assigning to these variables would create new local variables instead of
        # modifying the outer function's current_label, current_start, and current_end.
        nonlocal current_label, current_start, current_end
        if current_label is not None:
            spans.append({
                "start": current_start,
                "end": current_end,
                "text": text[current_start:current_end],
                "label": current_label,
            })
            current_label = None
            current_start = None
            current_end = None

    # -------------------------------------------------------------------------
    # Step 3) Scan tokens and apply BIO rules (state machine)
    # -------------------------------------------------------------------------
    for raw_label, (w_start, w_end) in zip(word_labels, positions):
        word = text[w_start:w_end]

        # Case A) Outside: close any open entity, and optionally emit an "O" span
        if raw_label == "O":
            flush_current()
            spans.append({
                "start": w_start,
                "end": w_end,
                "text": word,
                "label": "O",
            })
            continue

        # For BIO labels like "B-SYMPTOM_POS" / "I-SYMPTOM_POS":
        # prefix = "B" or "I"
        # norm   = "SYMPTOM_POS" (everything after the first "-")
        prefix, norm = raw_label.split("-", 1)

        # Case B) Begin: always start a new entity (but flush any previous first)
        if prefix == "B":
            flush_current()
            current_label = norm
            current_start = w_start
            current_end = w_end

        # Case C) Inside: extend if it matches; otherwise start a new one (robust behavior)
        elif prefix == "I":
            if current_label == norm:
                # Same entity continues: just extend the end pointer
                current_end = w_end
            else:
                # Mismatch or "I" without prior "B": treat as a new entity start
                flush_current()
                current_label = norm
                current_start = w_start
                current_end = w_end

        # Case D) Unexpected label format: treat as a standalone span
        else:
            flush_current()
            spans.append({
                "start": w_start,
                "end": w_end,
                "text": word,
                "label": raw_label,
            })

    # If we ended while still inside an entity, append the last entity span
    flush_current()
    return spans
