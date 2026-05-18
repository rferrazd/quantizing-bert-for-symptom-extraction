---
name: generate_hda_template
description: Generate a new HDA (História da Doença Atual) template for the v04 NER dataset, following project conventions for placeholder usage, vocabulary diversity, demographic variety, and POS/NEG ordering balance.
---

# generate_hda_template

Use this skill when the user asks to add new HDA templates to `v04/dataset_templates.py`. HDA templates are multi-sentence clinical paragraphs that mix patient-affirmed symptoms (`SYMPTOM_POS`), patient-denied symptoms (`SYMPTOM_NEG`), and varied clinical phrasing. They train BioBERT to distinguish symptom polarity in realistic Brazilian ED-style notes.

## Goal

Produce templates that prevent the model from overfitting to surface patterns — specific cue words, fixed orderings, fixed demographics, or fixed POS/NEG ratios.

## Rubric

### 1. Structure (Tier 2 or Tier 3 only)

- **Tier 2:** 3–4 sentences. Total symptoms: 3–5, distributed across POS and NEG slots.
- **Tier 3:** 4–6 sentences. Total symptoms: 5–8, distributed across POS and NEG slots.

### 2. Placeholders

- Use **indexed** placeholders only: `{SYMPTOM_POS_1}`, `{SYMPTOM_POS_2}`, `{SYMPTOM_NEG_1}`, `{SYMPTOM_NEG_2}`, etc.
- The builder draws a **different** symptom for each unique index. Two slots with the same index would fill with the same symptom — don't do this inside HDA paragraphs.
- Don't use `{SYMPTOM_O}` inside HDA templates — distractor context belongs in DISTRACTOR_TEMPLATES.

### 3. POS/NEG ratio diversity (critical)

Across the full template set, ratios must vary. Do NOT make every template POS-heavy or NEG-heavy.

- **POS-heavy** templates (e.g., 4 POS / 1 NEG, 5 POS / 1 NEG): patient reports many active symptoms, few denials.
- **Balanced** templates (e.g., 3 POS / 3 NEG, 2 POS / 2 NEG): even split.
- **NEG-heavy** templates (e.g., 1 POS / 4 NEG, 2 POS / 5 NEG): mostly denials with one or two reported symptoms. Realistic for screening visits or rule-out evaluations.

Target distribution when adding N new templates: roughly equal counts of POS-heavy, balanced, and NEG-heavy.

### 4. POS/NEG ordering diversity

Vary the order in which POS and NEG symptoms appear across templates. No single ordering should dominate.

Patterns to rotate:

- `POS → NEG` (chief complaint first, denials after)
- `NEG → POS` (rule-outs first, complaints revealed on questioning)
- `POS → NEG → POS` (interleaved with secondary symptoms at the end)
- `NEG → POS → NEG` (denials bracket the complaint)
- `POS → NEG → POS → NEG` (fully interleaved)

### 5. Vocabulary diversity (anti-overfitting to cue words)

Rotate affirmation and negation cue phrases across templates. Do NOT let "denies" or any other single cue dominate.

**Affirmation cues** (from `AFFIRMED_TEMPLATES`):
acknowledges, admits to, complains of, clinical history reveals, onset of X began, complaint of X was elicited, patient indicates experiencing, noted alongside, has been experiencing, endorses, presents with, reports, mentions, additionally reports, identified on review, also mentions

**Negation cues** (from `NEGATED_TEMPLATES`):
currently without, was not observed, ROS negative for, absent on current assessment, no X at this time, physical exam negative for, no X was identified, confirmed absence of, no X reported, is not present at evaluation, there is no indication of, assessment shows absence of, denies, review of systems: X denied, findings do not support presence of

Each template should use 2–4 distinct cue phrases. Document them in the inline comment above the template.

### 6. Demographic diversity (anti-memorization-of-openings)

Vary the opening across templates. Don't repeat the same demographic string verbatim more than once.

Variables to rotate:
- **Presence/absence of demographics:** ~30% no demographics, ~70% with demographics
- **Age group:** pediatric (3–12y), adolescent (13–17y), young adult (18–35y), adult (36–55y), older adult (56–70y), elderly (71+)
- **Gender:** male, female, unspecified
- **Comorbidities:** none ("previously healthy"), hypertension, type 2 diabetes, CKD, asthma, hypothyroidism, multiple comorbidities
- **Accompaniment / arrival:** walk-in, accompanied by spouse/parent/child/caregiver, brought by family, brought to ED, referred by GP

### 7. Informant diversity

The informant is the person reporting the symptoms. Vary across templates:
- Patient themselves
- Mother, father, parents
- Daughter, son
- Wife, husband, spouse
- Caregiver

**Important scope rules:**
- Family member reporting **current patient symptoms** → label as `SYMPTOM_POS` / `SYMPTOM_NEG` (the symptom is the patient's).
- Family member's **own** symptoms (family history) → would be `O`, but these belong in `DISTRACTOR_TEMPLATES`, NOT HDA templates.

### 8. Scope rules (out-of-scope content)

Never include in HDA templates:
- **Past history of the patient labeled as SYMPTOM_NEG/POS.** "No prior history of X" is out-of-scope (label O). Past resolved symptoms are O.
- **Family history of relatives** ("father had X"). O.
- **Hypothetical / educational content** ("if X develops, seek care"). O.
- **Literature / research mentions.** O.

If the scenario you want to express requires any of these, use `DISTRACTOR_TEMPLATES` instead.

### 9. Clinical realism

- Follow Amplimed-style HDA structure loosely: onset → progression → associated symptoms → denied symptoms. Variation is encouraged — strict adherence creates positional shortcuts.
- Allow telegraphic clinical phrasing in some sentences, full sentences in others.
- Avoid overly literary language. These should read like ED clinician notes.

### 10. Hold-out separation

`sample_hdas.py` contains hold-out HDAs (hda6–hda10) used to validate generalization. New training templates must:
- NOT copy exact phrasings from those examples (preserves them as a true generalization test).
- Use the same medical vocabulary (expected — same domain).
- Use distinct openings (the hold-outs all open with "Patient reports..." — avoid that exact opener in training templates).

### 11. Inline documentation

Above each template, write a brief comment with:
- Template ID (e.g., T21)
- Tier (2 or 3)
- POS/NEG count and balance category (POS-heavy / balanced / NEG-heavy)
- Ordering pattern
- Demographic / informant choices
- Key affirmation and negation vocab used

Example:
```python
# T21 — Tier 3, balanced (3 POS / 3 NEG), adult female with diabetes, NEG→POS→NEG.
# Negation vocab: "currently without" / "ROS negative for" / "confirmed absence of".
# Affirmation vocab: "acknowledges" / "complains of" / "has been experiencing".
```

## Workflow

1. **Confirm count and balance target** with the user before writing. (How many templates? Any specific balance / demographic gaps to fill?)
2. **Audit existing templates** in `v04/dataset_templates.py` HDA_TEMPLATES section. Count current ratios, orderings, demographics. Identify gaps.
3. **Draft templates** following the rubric. Each template gets an inline comment header (rule 11).
4. **Self-audit** the new batch:
   - POS-heavy / balanced / NEG-heavy roughly balanced?
   - No single ordering > 40% of new templates?
   - No single demographic / informant repeated identically?
   - "Denies" used in ≤ 25% of templates?
5. **Update the inventory header** in `v04/dataset_templates.py`:
   ```
   #   HDA_TEMPLATES:        N   — multi-sentence HDA paragraphs, mixed POS/NEG (growing)
   ```
6. **Append the new templates** to the `HDA_TEMPLATES` list. Do not reorder existing templates.
7. **Report to the user** with a short summary of: count added, balance distribution, ordering distribution, demographic spread.

## Anti-patterns to reject

- All new templates POS-heavy (or all NEG-heavy).
- All new templates open with "Patient" or "Female patient" or any single string.
- Every negation uses "denies" or any single cue word.
- POS symptoms always precede NEG symptoms.
- Templates that mimic `sample_hdas.py` phrasings.
- Templates that include past history, family history, or hypothetical scenarios labeled as POS/NEG (those belong in DISTRACTOR_TEMPLATES with O labels).
- Single-symptom templates (those belong in AFFIRMED/NEGATED_TEMPLATES).
