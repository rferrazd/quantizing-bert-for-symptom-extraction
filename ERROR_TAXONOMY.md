# Error Taxonomy — Clinical Symptom NER

## Overview

This document defines a **shared vocabulary for categorizing errors** produced by the symptom NER model during inference.

Rather than immediately fixing or optimizing the model, this taxonomy helps us  **understand *what kind* of mistakes the system is making** , so improvements are deliberate, measurable, and aligned with real-world usage.

This taxonomy was introduced during  **v0.1** , when behavior is still exploratory and unstable.

---

## Why This Exists

In early-stage ML systems, not all errors are equal.

Some errors indicate:

* Fundamental modeling issues
* Dataset gaps
* Tokenization artifacts
* Expected limitations at the current version

Without clear categorization, it’s easy to:

* Overfit fixes to individual examples
* Add premature heuristics
* Optimize the wrong layer (model vs data vs inference)

**The error taxonomy ensures we fix the *right problems* at the  *right time* .**

---

## How This Is Used

For every failing or surprising inference case:

1. **Assign one or more error categories** from this taxonomy
2. Log the error alongside:
   * Input text
   * Predicted spans
   * Expected spans
3. Track frequency of each category across test cases
4. Use patterns (not anecdotes) to decide:
   * Whether to adjust inference logic
   * Enrich the dataset
   * Accept behavior for the current version

No fixes should be implemented without first classifying the error.

---

## Error Categories

### 1. Model Overgeneralization

The model predicts a symptom where none exists.

**Examples**

* Predicting `SYMPTOM_POS` for:
  * Person names (`Marie`, `Claire`)
  * General states (`well`, `stable`)
  * Demographics (`65`)

**Implication**

* Model has learned overly broad symptom cues
* Often caused by dataset bias or insufficient negative examples

---

### 2. Boundary Errors

#### 2a. Boundary Overreach

The model captures a symptom  **plus unrelated surrounding words** .

**Examples**

* `blisters on his head`
* `large blister on`

**Implication**

* Core entity detected correctly
* Span precision is low
* Often acceptable in early versions if recall is prioritized

#### 2b. Boundary Undereach

**Examples:**

* `chest` `pain` instead of `chest pain`


---

### 3. BIO Sequencing Errors

Incorrect or inconsistent BIO tag transitions.

**Examples**

* `I-SYMPTOM_POS` without a preceding `B-`
* New symptom starting with `I-` instead of `B-`

**Implication**

* Model uncertainty at entity boundaries
* Should be handled robustly in span construction

---

### 4. Incorrect Polarity Assignment

Incorrect polarity assignment in the presence of negation.

**Examples**

* `muscle cramps` labeled as `SYMPTOM_NEG` when context implies presence

* Negation scope bleeding across conjunctions. Example: Patient does not have any `edemas` only `rashes`

**Implication**

* Model struggles with negation scope and contrastive clauses
* Usually requires dataset enrichment, not heuristics

---

### 5. Tokenization Artifacts

Errors caused by subword splits influencing predictions.

**Examples**

* Mixed labels across subwords of a single word
* Conflicting labels aggregated into `CONFLICT-*`

**Implication**

* Expected behavior with WordPiece/BPE tokenizers
* Must be handled gracefully at aggregation time

---

### 6. Missing Expected Entity

A clinically relevant symptom is not detected at all.

**Examples**

* `exanthema` not extracted
* Rare or domain-specific terms missed

**Implication**

* Vocabulary or representation gap
* Strong signal for dataset enrichment

---

## Initial Versioning Philosophy (what inspired this document)

* **v0.1** : Error discovery and categorization
* **v0.x** : Reduce frequency of *blocking* categories
* **v1.0** : Error patterns are rare, predictable, and documented

Some categories may remain acceptable depending on product goals.

---

## Guiding Principle

> *Do not fix what you do not yet understand.*



Add a template for logging errors (ERROR_LOG.md)

Or mark which categories are blocking vs acceptable for v0.1 → v1.0