
# Production inference pipeline

## Phase 1 - Inference flow

1. Raw text → tokenizer

Token-level prediction

Token → word aggregation

Word BIO → entity spans ← this is what you’re building

Validate outputs using the cases above

✅ Goal: “Does the model behave well on real sentences?”


# Phase 2 — Stress & usability checks

6. Test longer inputs (multi-sentence clinical notes)

7. Measure:

     - max input length before truncation hurts (where it starts loosing performance)
     - latency on CPU

8. Decide on such techniques:

    - tackle sentence chunking
    - sliding window