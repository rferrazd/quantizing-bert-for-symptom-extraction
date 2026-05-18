"""
test_pipeline_walkthrough.py

A guided walkthrough of every function in data_preparation/dataset_generator.py.
Each section runs ONE function with a concrete example, prints inputs and outputs,
and annotates what to look for.

Run with:    python3 test_pipeline_walkthrough.py
Or run sections interactively (copy-paste into REPL).

The pipeline order, top to bottom:

    Helpers (module-level):
        1. normalize_token
        2. extract_placeholders
        3. draw_symptom
        4. build_fill_map          (uses draw_symptom + extract_placeholders)
        5. fill_template           (uses output of build_fill_map)
        6. build_symptoms_metadata (uses output of build_fill_map)
        7. whitespace_tokenize

    Class methods (DatasetGenerator):
        8.  tokenize_with_spans
        9.  find_subsequence
        10. tokenize_sample        (combines 7, 8, 9)
        11. generate_raw_synthetic_samples (combines 3, 4, 5, 6)
        12. tokenize_all_samples   (loops 10)
"""

import sys
import pandas as pd

sys.path.insert(0, "/Users/robertagarcia/Desktop/learning/bert_symptom_ner")

from data_preparation.dataset_generator import (
    DatasetGenerator,
    normalize_token,
    extract_placeholders,
    draw_symptom,
    build_fill_map,
    fill_template,
    build_symptoms_metadata,
    whitespace_tokenize,
)


def section(title: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# A tiny, deterministic-ish symptoms_df for the walkthrough.
SYMPTOMS_DF = pd.DataFrame({
    "id":        ["s01",   "s02",   "s03",       "s04",   "s05",          "s06",            "s07"],
    "prefLabel": ["fever", "cough", "joint pain","rash",  "abdominal pain","nasal congestion","chest pain"],
})


# =====================================================================
# 1) normalize_token
# =====================================================================
section("1) normalize_token(tok) — lowercases a single token")

print("Input:  'Fever'")
print("Output:", repr(normalize_token("Fever")))

print("\nInput:  'COUGH'")
print("Output:", repr(normalize_token("COUGH")))

print("\nInput:  '.'  (punctuation is unchanged)")
print("Output:", repr(normalize_token(".")))

print("""
WHAT TO NOTICE
- Just .lower(). No splitting, no stripping.
- Used inside tokenize_with_spans on each matched token.
""")


# =====================================================================
# 2) extract_placeholders
# =====================================================================
section("2) extract_placeholders(template) — pull placeholder names in order")

t1 = "Patient endorses {SYMPTOM_POS}."
print(f"Input: {t1!r}")
print("Output:", extract_placeholders(t1))

t2 = "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}. Denies {SYMPTOM_NEG_1}."
print(f"\nInput: {t2!r}")
print("Output:", extract_placeholders(t2))

t3 = "Has {SYMPTOM_POS} and continues to report {SYMPTOM_POS}."
print(f"\nInput (same placeholder twice): {t3!r}")
print("Output:", extract_placeholders(t3))

print("""
WHAT TO NOTICE
- Returns unique placeholder NAMES (no braces).
- Order is preserved (dict.fromkeys trick).
- Duplicates collapse to a single entry — they will be filled with the same draw later.
""")


# =====================================================================
# 3) draw_symptom
# =====================================================================
section("3) draw_symptom(symptoms_df) — random (id, prefLabel) draw")

for i in range(4):
    sid, text = draw_symptom(SYMPTOMS_DF)
    print(f"Draw {i+1}: id={sid!r}, prefLabel={text!r}")

print("""
WHAT TO NOTICE
- Random each call. Run again — you'll see different draws.
- Returns a TUPLE (id, prefLabel). The id is what we dedup on; prefLabel is the text we splice into the template.
""")


# =====================================================================
# 4) build_fill_map
# =====================================================================
section("4) build_fill_map(template, group_name, symptoms_df) — fill every placeholder")

template = "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}. Denies {SYMPTOM_NEG_1}."
print(f"Template: {template!r}")
print("Group:    'hda'")
fm = build_fill_map(template, "hda", SYMPTOMS_DF)
print("\nfill_map:")
for ph, (sid, text) in fm.items():
    print(f"  {ph:20s} -> ({sid!r}, {text!r})")

# Verify the cross-polarity dedup is working:
ids_in_map = [sid for sid, _ in fm.values()]
assert len(set(ids_in_map)) == len(ids_in_map), "DUPLICATE ID — Bug 4 regressed!"
print("\nDedup check: all symptom_ids are unique across POS and NEG ✓")

print("\n--- Bare repeated placeholder (same symptom fills both) ---")
bare = "Has {SYMPTOM_POS} and continues to report {SYMPTOM_POS}."
fm2 = build_fill_map(bare, "affirmed_repeat", SYMPTOMS_DF)
print(f"Template: {bare!r}")
print("fill_map:", dict(fm2))
print("Both slots get ONE entry in fill_map; fill_template will substitute it everywhere.")

print("""
WHAT TO NOTICE
- For indexed placeholders (_1, _2): different ids per slot, no collisions across POS/NEG.
- For bare repeated placeholders: only one fill_map entry — both occurrences in the template will be replaced with the same symptom.
- The 'group_name' argument isn't used by the function anymore (after dead-code removal), but it stays in the signature for clarity.
""")


# =====================================================================
# 5) fill_template
# =====================================================================
section("5) fill_template(template, fill_map) — substitute placeholders with text")

print(f"Template: {template!r}")
print("\nUsing the fill_map from step 4:")
filled = fill_template(template, fm)
print(f"Filled:   {filled!r}")

print("""
WHAT TO NOTICE
- Each "{PLACEHOLDER}" is replaced with the symptom prefLabel.
- The fill_map's id is NOT used here (only the text). The id lives on in symptoms metadata.
""")


# =====================================================================
# 6) build_symptoms_metadata
# =====================================================================
section("6) build_symptoms_metadata(fill_map) — POS/NEG list with polarity tags")

meta = build_symptoms_metadata(fm)
print("Metadata for the HDA fill_map:")
for entry in meta:
    print(f"  {entry}")

print("\n--- Distractor example: SYMPTOM_O is EXCLUDED ---")
distract_template = "Literature describes {SYMPTOM_O}."
distract_fm = build_fill_map(distract_template, "distractor", SYMPTOMS_DF)
print(f"distract_fm: {dict(distract_fm)}")
print(f"metadata:    {build_symptoms_metadata(distract_fm)}")

print("""
WHAT TO NOTICE
- Only POS and NEG slots produce metadata entries. SYMPTOM_O is intentionally excluded — those tokens stay 'O' at tokenize time.
- This is what drives label assignment in tokenize_sample.
""")


# =====================================================================
# 7) whitespace_tokenize
# =====================================================================
section("7) whitespace_tokenize(symptom_text) — split symptom on whitespace, lowercase")

for s in ["fever", "joint pain", "Nasal Congestion", "  spaced  out  "]:
    print(f"Input: {s!r:30s}  →  {whitespace_tokenize(s)}")

print("""
WHAT TO NOTICE
- Splits on any whitespace (multiple spaces fine).
- Lowercases each part.
- Returns a list — this is the 'target' we'll search for in the tokenized template text.
- Punctuation that's part of the symptom string would survive here; we'd handle it as part of the word token.
""")


# =====================================================================
# Setup for class-method tests
# =====================================================================
section("Setting up DatasetGenerator for class-method tests")

gen = DatasetGenerator(
    symptoms_df=SYMPTOMS_DF,
    dataset_templates=[],  # unused for these tests
)
print("gen = DatasetGenerator(symptoms_df=..., dataset_templates=[])")


# =====================================================================
# 8) tokenize_with_spans
# =====================================================================
section("8) gen.tokenize_with_spans(text) — split into (token, start, end) triples")

text = "Patient reports joint pain. Denies cough."
print(f"Text: {text!r}\n")
triples = gen.tokenize_with_spans(text)
for tok, start, end in triples:
    print(f"  ({tok!r:14s}, start={start:3d}, end={end:3d})  text[{start}:{end}]={text[start:end]!r}")

print("""
WHAT TO NOTICE
- Each word becomes ONE token. "joint pain" → two tokens ("joint", "pain").
- Punctuation gets its OWN token (e.g. ".").
- start/end are char-level offsets into the original text — useful later for span-based eval.
""")


# =====================================================================
# 9) find_subsequence
# =====================================================================
section("9) gen.find_subsequence(tokens, target, start_from=0) — locate target in token list")

tokens = [tok for (tok, _, _) in triples]
print("tokens =", tokens)

for target in [["joint", "pain"], ["cough"], ["fever"]]:
    idx = gen.find_subsequence(tokens, target)
    print(f"find_subsequence(tokens, {target}) → {idx}")

# Demo start_from: find a token, then search again after it
print("\n--- start_from in action ---")
tokens_with_dup = ["fever", "and", "cough", "with", "fever"]
print(f"tokens: {tokens_with_dup}")
first  = gen.find_subsequence(tokens_with_dup, ["fever"], start_from=0)
second = gen.find_subsequence(tokens_with_dup, ["fever"], start_from=first + 1)
print(f"first  occurrence of ['fever']: index {first}")
print(f"second occurrence of ['fever']: index {second}")

print("""
WHAT TO NOTICE
- Returns the START index where target fully matches, or None.
- Multi-token targets (like ["joint", "pain"]) must match a contiguous slice.
- start_from lets you skip past an earlier match — useful if a token appears more than once.
""")


# =====================================================================
# 10) tokenize_sample
# =====================================================================
section("10) gen.tokenize_sample(raw_sample) — produce BIO labels for one sample")

# Build a raw sample by hand (normally generate_raw_synthetic_samples produces these)
raw_sample = {
    "text":           "Reports joint pain and fever. Denies cough.",
    "template_group": "hda",
    "template":       "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}. Denies {SYMPTOM_NEG_1}.",
    "symptoms": [
        {"symptom_id": "s03", "symptom_text": "joint pain", "polarity": "POS"},
        {"symptom_id": "s01", "symptom_text": "fever",      "polarity": "POS"},
        {"symptom_id": "s02", "symptom_text": "cough",      "polarity": "NEG"},
    ],
}

print("Raw sample:")
print(f"  text:     {raw_sample['text']!r}")
print(f"  symptoms: {raw_sample['symptoms']}")

tokenized, not_found = gen.tokenize_sample(raw_sample)
print("\nTokenized output:")
print(f"  word_tokens: {tokenized['word_tokens']}")
print(f"  word_labels: {tokenized['word_labels']}")
print(f"  not_found:   {not_found}")

print("\nToken-by-token view:")
for tok, lbl in zip(tokenized['word_tokens'], tokenized['word_labels']):
    print(f"  {tok:14s} → {lbl}")

print("""
WHAT TO NOTICE
- Two-word symptom 'joint pain' gets B-SYMPTOM_POS on 'joint' and I-SYMPTOM_POS on 'pain'.
- Single-word symptoms get just B-SYMPTOM_POS or B-SYMPTOM_NEG.
- All other tokens (including '.') stay 'O'.
- not_found is empty here — every symptom in symptoms[] was located in the tokens.
""")


# =====================================================================
# 10b) tokenize_sample — distractor case (all O)
# =====================================================================
section("10b) tokenize_sample on a DISTRACTOR sample — all labels should be O")

distractor_sample = {
    "text":           "Patient was advised to monitor for fever after discharge.",
    "template_group": "distractor",
    "template":       "Patient was advised to monitor for {SYMPTOM_O} after discharge.",
    "symptoms":       [],  # no POS or NEG entries — SYMPTOM_O is excluded from metadata
}

print(f"Text: {distractor_sample['text']!r}")
tok_d, nf_d = gen.tokenize_sample(distractor_sample)
print(f"word_tokens: {tok_d['word_tokens']}")
print(f"word_labels: {tok_d['word_labels']}")
assert all(l == "O" for l in tok_d['word_labels']), "Distractor should be all O!"
print("✓ All labels are 'O' as expected.")

print("""
WHAT TO NOTICE
- Distractor metadata is empty, so the symptom loop in tokenize_sample never executes.
- Every token stays 'O', including the word 'fever' — that's the whole point of distractor training.
""")


# =====================================================================
# 11) generate_raw_synthetic_samples — end-to-end (small)
# =====================================================================
section("11) gen.generate_raw_synthetic_samples(K) — full generation on a mini template set")

mini_templates = [
    ("affirmed",   ["Patient endorses {SYMPTOM_POS}."]),
    ("negated",    ["Denies {SYMPTOM_NEG}."]),
    ("distractor", ["Literature describes {SYMPTOM_O}."]),
    ("hda",        ["Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}. Denies {SYMPTOM_NEG_1}."]),
]
gen_mini = DatasetGenerator(symptoms_df=SYMPTOMS_DF, dataset_templates=mini_templates)
samples = gen_mini.generate_raw_synthetic_samples(K=3)

print(f"Generated {len(samples)} samples total.")
print(f"  affirmed:   {sum(1 for s in samples if s['template_group']=='affirmed')}  (expect 7 = pool size)")
print(f"  negated:    {sum(1 for s in samples if s['template_group']=='negated')}   (expect 7 = pool size)")
print(f"  distractor: {sum(1 for s in samples if s['template_group']=='distractor')}  (expect 7 = pool size)")
print(f"  hda:        {sum(1 for s in samples if s['template_group']=='hda')}   (expect K=3)")

print("\nOne sample from each group:")
for group in ["affirmed", "negated", "distractor", "hda"]:
    s = next(s for s in samples if s["template_group"] == group)
    print(f"\n  [{group}]")
    print(f"    text:     {s['text']!r}")
    print(f"    symptoms: {s['symptoms']}")

print("""
WHAT TO NOTICE
- Single-symptom groups (affirmed, negated, distractor): EXHAUSTIVE — one sample per symptom in the pool. With 7 symptoms in our toy pool, each group has 7 samples.
- Multi-symptom group (hda): K=3 random draws per template.
- 'symptoms' is empty for distractor (no POS/NEG entries to track).
""")


# =====================================================================
# 12) tokenize_all_samples — end-to-end (small)
# =====================================================================
section("12) gen.tokenize_all_samples(samples) — tokenize every sample")

tokenized, not_found = gen_mini.tokenize_all_samples(samples)
print(f"Tokenized {len(tokenized)} samples.")
print(f"not_found list: {len(not_found)} entries (expect 0 for our clean toy data)")

print("\nOne tokenized HDA sample, token-by-token:")
hda_tok = next(t for t in tokenized if t["template_group"] == "hda")
print(f"  text: {hda_tok['text']!r}")
for tok, lbl in zip(hda_tok["word_tokens"], hda_tok["word_labels"]):
    marker = "  ← labeled" if lbl != "O" else ""
    print(f"    {tok:18s} → {lbl}{marker}")

print("""
WHAT TO NOTICE
- tokenize_all_samples just loops tokenize_sample over the raw samples.
- The returned 'not_found' list is your debugging signal: if any symptom_text fails to match a token subsequence, it'll show up here. With clean synthetic data this should be empty.
- The token-by-token view is what the HuggingFace tokenizer will later align WordPiece IDs to via word_ids().
""")


print("\n" + "=" * 70)
print("  WALKTHROUGH COMPLETE")
print("=" * 70)
