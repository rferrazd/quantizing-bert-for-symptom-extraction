"""
onnx_inference.py

ONNX Runtime version of the word-level NER inference pipeline.

This is the ONNX counterpart to `inference.v01.inference_utils.predict_word_level`.
The ONLY thing that differs between the two is the forward pass: this module
tokenizes to numpy arrays and runs the model through an onnxruntime session,
whereas the PyTorch version uses `model(**inputs)`. Everything after the forward
pass (filtering special tokens, aggregating tokens -> words) is delegated to the
shared, backend-agnostic helper `aggregate_token_predictions_to_words`, so the
critical alignment logic lives in exactly one place.

This function is the reference the mobile app's on-device code must reproduce.
"""

import numpy as np
import onnxruntime as ort

# Reuse the shared steps 3+4 (filter + token->word aggregation). Do NOT duplicate.
from inference.v01.inference_utils import aggregate_token_predictions_to_words


def predict_word_level_onnx(
    text: str,
    session: ort.InferenceSession,
    tokenizer,
    id2label: dict,
) -> tuple:
    """
    Run token-classification inference via ONNX Runtime and aggregate the
    token-level predictions up to word level.

    Returns the SAME 6-tuple as inference_utils.predict_word_level so that
    word_labels_to_spans can consume it unchanged:
        (filtered_tokens, filtered_token_labels, filtered_word_ids,
         words, word_labels, word_offsets)
    """

    # -------------------------------------------------------------------
    # 1. Tokenize.
    #    return_tensors="np" because ONNX Runtime consumes numpy arrays.
    #    return_offsets_mapping=True gives (start, end) char positions per token.
    # -------------------------------------------------------------------
    encoding = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # word_ids maps each token -> original word index; None for [CLS]/[SEP].
    word_ids = encoding.word_ids(batch_index=0)

    # Offsets (start, end) into the ORIGINAL text, one pair per token.
    offsets = encoding["offset_mapping"][0].tolist()

    # -------------------------------------------------------------------
    # 2. Run the ONNX model.
    #    Build the feed dict from exactly the input names the graph declares
    #    (input_ids, attention_mask, token_type_ids). offset_mapping is for us,
    #    not the model, so it is excluded automatically by this filter.
    # -------------------------------------------------------------------
    model_input_names = {i.name for i in session.get_inputs()}
    onnx_inputs = {
        name: encoding[name]
        for name in model_input_names
        if name in encoding
    }

    # session.run(None, feed) -> None means "return all outputs".
    # outputs[0] is the logits array of shape (1, seq_len, num_labels).
    logits = session.run(None, onnx_inputs)[0]

    # argmax over the label dimension (last axis) -> predicted label id per token.
    pred_ids = np.argmax(logits, axis=-1)[0]  # shape (seq_len,)

    # Convert label ids -> label strings. int() coerces numpy ints AND guards
    # against id2label keys that may be str-ints when loaded from raw config.json.
    pred_token_labels = [id2label[int(p)] for p in pred_ids]

    # -------------------------------------------------------------------
    # 3 + 4. Shared filter + token->word aggregation.
    # -------------------------------------------------------------------
    return aggregate_token_predictions_to_words(
        text=text,
        tokens=tokens,
        word_ids=word_ids,
        offsets=offsets,
        pred_token_labels=pred_token_labels,
    )
