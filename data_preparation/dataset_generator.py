"""
Goal of this class is to take the list of symptoms and the dataset_templates and generate it into a ready to use and upload to huggingface training dataset
"""

# Postpones evaluating type hints (they become strings at runtime). That avoids errors
# when a hint mentions a class defined later in the file and can speed up import time.
from __future__ import annotations

from typing import Iterator, List, Tuple, Dict, Optional, Set
import re, random, json
import pandas as pd


# Each entry is (logical group name, list of template strings). Pass v04 TEMPLATE_GROUPS or a subset.
TemplateGroups = List[Tuple[str, List[str]]]

# template for parcing SYMPTOM_
PLACEHOLDER_RE = re.compile(r'\{(SYMPTOM[^}]*)\}')
SINGLE_SYMPTOM_GROUPS = {"affirmed", "negated", "distractor"}
# =============
# HELPERS 
# =============
def normalize_token(tok):
    """Lowercase normalization for matching (leave punctuation tokens as-is)."""
    return tok.lower()

def extract_placeholders(template:str):
    """Return unique placeholder names in template, preserving order.Don't use set() it won't preserve the order"""
    return list(dict.fromkeys(PLACEHOLDER_RE.findall(template)))

def draw_symptom(symptoms_df, 
                id_column_name:str="id",
                prefLabel_column_name:str="prefLabel") -> Tuple[str, str]:
    """Return (symptom_id, symptom_text) sampled at random."""
    row = symptoms_df.sample(1).iloc[0]
    return row[id_column_name], row[prefLabel_column_name]

def build_fill_map(template:str,
group_name:str, symptoms_df:Dict):
    """
    Return {placeholder_name: (symptom_id, symptom_text)} for every slot.

    Drawing rules:
      - Bare repeated placeholder  → one draw shared across all occurrences
      - Indexed (_1, _2, _3 …)     → different draws, no repeat within same base type
      - word_collision group        → SYMPTOM_O gets the same draw as SYMPTOM_POS/NEG
    """

    # get a list of the placeholder names like ['SYMPTOM_POS_1', 'SYMPTOM_NEG']
    placeholders = extract_placeholders(template)
    fill_map: Dict[str, Tuple[str,str]] = {}

    # Single dedup scope across ALL indexed slots, regardless of polarity.
    # Prevents the same symptom_id from filling SYMPTOM_POS_1 and SYMPTOM_NEG_1
    # in the same HDA (which would be clinically incoherent and also break
    # tokenize_sample, which finds the first token match per symptom).
    used_ids: Set[str] = set()
    MAX_DRAW_ATTEMPTS = 100

    for plh in placeholders:
        if plh in fill_map:
            continue

        # Indexed placeholder, e.g. "SYMPTOM_POS_1": draw a fresh symptom that
        # hasn't been used yet anywhere in this fill_map.
        index_match = re.match(r'^(SYMPTOM_(?:POS|NEG))_(\d+)$', plh)

        if index_match:
            for attempt in range(MAX_DRAW_ATTEMPTS):
                sid, symptom_text = draw_symptom(symptoms_df)
                if sid not in used_ids:
                    break
            else:
                raise RuntimeError(
                    f"Could not find an unused symptom for placeholder {plh!r} "
                    f"after {MAX_DRAW_ATTEMPTS} attempts. "
                    f"Pool may be smaller than the number of indexed slots."
                )
            used_ids.add(sid)
            fill_map[plh] = (sid, symptom_text)
        else:
            # "Bare" means the placeholder has no numeric index: {SYMPTOM_POS}, not {SYMPTOM_POS_1}.
            # Some templates use the same placeholder twice to mean the SAME symptom both times,
            # e.g. "denies improvement in {SYMPTOM_POS}" where {SYMPTOM_POS} appears earlier too.
            # The `if plh in fill_map: continue` above already skips the second occurrence,
            # so we only draw once and both slots get the same symptom automatically.
            fill_map[plh] = draw_symptom(symptoms_df)

    return fill_map


def fill_template(template: str, fill_map: Dict[str, Tuple[str, str]]) -> str:
    """Use the created fill map to efficiently replace the template with an actual symptom"""
    text = template
    for ph, (_, symptom_text) in fill_map.items():
        text = text.replace(f"{{{ph}}}", symptom_text)
    return text


def build_symptoms_metadata(fill_map: Dict[str, Tuple[str, str]]) -> List[dict]:
    """POS and NEG slots only — SYMPTOM_O is intentionally excluded."""
    symptoms = []
    for ph, (sid, symptom_text) in fill_map.items():
        if "POS" in ph:
            symptoms.append({"symptom_id": sid, "symptom_text": symptom_text, "polarity": "POS"})
        elif "NEG" in ph:
            symptoms.append({"symptom_id": sid, "symptom_text": symptom_text, "polarity": "NEG"})
    return symptoms

def whitespace_tokenize(symptom_text: str) -> List[str]:
    return [p.lower() for p in re.split(r"\s+", symptom_text.strip()) if p]

# =========

class DatasetGenerator:
    def __init__(
        self,
        symptoms_df: pd.DataFrame,
        dataset_templates: TemplateGroups,
    ):
        self.symptoms_df = symptoms_df
        self.dataset_templates = dataset_templates
        # Two alternating branches separated by |:
        #   [A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*  — a word, optionally joined by hyphens or
        #                                          apostrophes (e.g. "well-being", "it's")
        #   [^\sA-Za-z0-9]                       — any single non-whitespace, non-word char
        #                                          (e.g. ".", ",", ":")
        # Result: "shortness of breath." → ["shortness", "of", "breath", "."]
        self.TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


    # Iterator[...]: callers get items one-by-one (memory-friendly); Tuple[str, str] is
    # each item's shape: two strings (group name, then template text).
    def iter_grouped_templates(self) -> Iterator[Tuple[str, str]]:
        """Yield (template_group, template) for every template in order."""
        # dataset_templates is [(group_name, [tmpl, tmpl, ...]), ...]; unpacking assigns
        for group_name, templates in self.dataset_templates:
            for template in templates:
                # yield: pause here and give this pair to the caller; resume on next iteration.
                yield group_name, template


    def generate_raw_synthetic_samples(self, K: int | None = 40, save_in_jsonl_path: Optional[str] = None) -> List[dict]:
        """
        Generate the raw (un-tokenized) synthetic dataset by filling every template
        with concrete symptoms drawn from `self.symptoms_df`.

        How many samples per template?

            - Single-symptom groups (affirmed, negated, distractor): EXHAUSTIVE.
              Each template has exactly ONE placeholder. We produce one sample
              per symptom in the pool. So with N symptoms, every such template
              contributes N samples.

            - Multi-symptom groups (hda): K random draws per template.
              Each template has SEVERAL indexed placeholders (e.g. SYMPTOM_POS_1,
              SYMPTOM_POS_2, SYMPTOM_NEG_1...). True exhaustive coverage would be
              combinatorial in the pool size (N × (N-1) × (N-2) × ...) — infeasible.
              Instead we sample K independent draws, each one fills every slot
              randomly.

        What does K mean?

            K is the number of random draws we generate PER multi-symptom template.

            - K = 40 (default):  40 random samples per HDA template.
                                  With 20 HDA templates → 800 HDA samples total.
            - K = None:          K is set to len(symptoms_df) automatically.
                                  Produces the same sample count as a single-symptom
                                  template would (one per symptom). Roughly balances
                                  HDA sample count with affirmed/negated/distractor.

        Args:
            K: Random draws per multi-symptom template (None → pool size).
            save_in_jsonl_path: If provided, write all samples to this path as JSONL.

        Returns:
            List of sample dicts. Each dict has keys: text, template_group,
            template, symptoms.
        """
        samples: List[dict] = []

        # K = None → use pool size as K (effectively "one HDA sample per symptom per template").
        K = K if K > 0 else len(self.symptoms_df)

        # Outer loop: walk every (group_name, [template, ...]) pair from dataset_templates.
        for group_name, templates in self.dataset_templates:
            for template in templates:

                if group_name in SINGLE_SYMPTOM_GROUPS:
                    # ---- EXHAUSTIVE PATH ----
                    # Single-symptom template: one placeholder, one sample per symptom.

                    # Pick the right placeholder name for this group.
                    # affirmed → SYMPTOM_POS, negated → SYMPTOM_NEG, distractor → SYMPTOM_O.
                    ph = {"affirmed": "SYMPTOM_POS",
                        "negated":  "SYMPTOM_NEG",
                        "distractor": "SYMPTOM_O"}[group_name]

                    # Iterate every row of the symptoms DataFrame — one sample each.
                    for _, row in self.symptoms_df.iterrows():
                        fill_map = {ph: (row["id"], row["prefLabel"])}
                        samples.append({
                            "text":           fill_template(template, fill_map),
                            "template_group": group_name,
                            "template":       template,
                            "symptoms":       build_symptoms_metadata(fill_map),
                        })

                else:
                    # ---- SAMPLED PATH (multi-symptom, e.g. hda) ----
                    # K random draws; each draw picks fresh symptoms for every indexed slot.
                    for _ in range(K):
                        fill_map = build_fill_map(template, group_name, self.symptoms_df)
                        samples.append({
                            "text":           fill_template(template, fill_map),
                            "template_group": group_name,
                            "template":       template,
                            "symptoms":       build_symptoms_metadata(fill_map),
                        })

        if save_in_jsonl_path:
            with open(save_in_jsonl_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"Saved {len(samples)} in {save_in_jsonl_path}")

        return samples

    #    ===============================
    #    ==== Tokeization functions ====
    #    ===============================

    def find_subsequence(self, tokens: List[str], target: List[str], start_from: int = 0) -> Optional[int]:  # BUG FIX: missing `self`
        n = len(target)
        if n == 0:
            return None
        # BUG FIX: was `range(start_from, start_from - n + 1)` which always produces an empty
        # range (start > stop), so the loop never ran. Upper bound must be len(tokens) - n + 1.
        for i in range(start_from, len(tokens) - n + 1):
            if tokens[i:i+n] == target:
                # return the index where the symptom entity starts in the token list
                return i
        return None


    def tokenize_with_spans(self,text:str) -> List[Tuple]:
        """
        Return list of (token, start_char, end_char) using TOKEN_PATTERN.
        Example: "stridor." -> [("stridor", idx, idx+7), (".", idx+7, idx+8)]
        """
        tokens = []
        # .finditer() scans `text` left-to-right and yields a Match object for each
        # non-overlapping occurrence of TOKEN_PATTERN. Unlike .findall() it gives you
        # the match position (m.start(), m.end()) not just the matched string. 
        # With finditer we iterate over the match object 
        for m in self.TOKEN_PATTERN.finditer(text):
            tok = m.group(0)
            tokens.append((normalize_token(tok), m.start(), m.end()))
        return tokens
    
    def tokenize_sample(self,raw_sample:dict):
        text       = raw_sample["text"]
        group_name = raw_sample["template_group"]
        symptoms   = raw_sample["symptoms"]
        
        tokens_with_spans = self.tokenize_with_spans(text)
        tokens      = [t for (t, _, _) in tokens_with_spans]
        #token_norms = [normalize_token(t) for t in tokens]
        labels = ["O"] * len(tokens)
        not_found = []

        # go through the list of symptom entities for that sample
        for symptom in symptoms:
            target = whitespace_tokenize(symptom["symptom_text"])  # BUG FIX: module-level function, not a method — drop `self.`
            polarity = symptom["polarity"]

            start_idx = self.find_subsequence(tokens, target)
            
            if start_idx is not None:
                for k in range(len(target)):
                    pos = start_idx + k
                    labels[pos] = f"B-SYMPTOM_{polarity}" if k == 0 else f"I-SYMPTOM_{polarity}"
            else:
                # claude we must save what was not found in a log file
                not_found.append({
                    "text": text,
                    "symptom_id": symptom["symptom_id"],
                    "symptom_text": symptom["symptom_text"],
                })
        tokenized = {
        "text":           text,
        "template_group": group_name,
        "word_tokens":    tokens,
        "word_labels":    labels,
        "symptoms":       symptoms,
    }
        return tokenized, not_found

    # Yes — only word tokenization so far. TOKEN_PATTERN splits on whitespace and punctuation
    # into whole words. BERT sub-word tokenization (WordPiece) happens later during training,
    # applied by the HuggingFace DataCollator / tokenizer with word_ids() alignment.
    def tokenize_all_samples(self, raw_samples: List[dict], save_in_jsonl_path: Optional[str] = None) -> Tuple[List[dict], List[dict]]:
        tokenized, all_not_found = [], []
        for raw in raw_samples:
            sample, not_found = self.tokenize_sample(raw)
            assert len(sample["word_tokens"]) == len(sample["word_labels"])
            tokenized.append(sample)
            all_not_found.extend(not_found)

        if save_in_jsonl_path:
            with open(save_in_jsonl_path, "w") as f:
                for s in tokenized:  # BUG FIX: was `samples` which doesn't exist in this scope
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"Saved {len(tokenized)} in {save_in_jsonl_path}")
        

        return tokenized, all_not_found



if __name__ == "__main__":

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path("/Users/robertagarcia/Desktop/learning/bert_symptom_ner")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # local imports 
    from config import settings
    from v04.dataset_templates import TEMPLATE_GROUPS
    from data_preparation.dataset_split import (
        save_split_artifacts,
        split_symptoms_df,
        split_template_groups,
    )

    VERSION = settings.VERSION  # e.g. "v04"
    SPLITS_DIR = f"{VERSION}/data/splits"

    symptoms_df = pd.read_csv("base_symptom_dict.csv")
    print(f"VERSION: {VERSION}  |  symptoms pool: {len(symptoms_df)}")

    # ---------------------------------------------------------------
    # 6b  Load-or-create deterministic splits
    # ---------------------------------------------------------------
    train_template_groups, heldout_template_groups, heldout_template_indices = (
        split_template_groups(TEMPLATE_GROUPS)
    )
    train_symptoms_df, heldout_symptoms_df = split_symptoms_df(symptoms_df)

    save_split_artifacts(
            heldout_template_indices,
            heldout_symptoms_df["id"].tolist(),
            SPLITS_DIR,
        )
    print(f"Split artifacts saved to {SPLITS_DIR}/split_artifacts.json")
  
    print("\nHeld-out template indices:")
    for group, idxs in heldout_template_indices.items():
        print(f"  {group:12s}: {idxs}")
    print(f"Held-out symptom count: {len(heldout_symptoms_df)}")
    print(f"Train symptom count:    {len(train_symptoms_df)}")

    # ---------------------------------------------------------------
    # 6c  Instantiate three generators
    # ---------------------------------------------------------------
    gen_train = DatasetGenerator(train_symptoms_df, train_template_groups)
    gen_template_ood = DatasetGenerator(train_symptoms_df, heldout_template_groups)
    gen_symptom_ood = DatasetGenerator(heldout_symptoms_df, train_template_groups)

    # ---------------------------------------------------------------
    # 6d  Generate + tokenize + save each split
    # ---------------------------------------------------------------
    all_not_found: List[dict] = []

    splits_config = [
        # (label,          generator,        K_hda, raw_path,                  tok_path)
        ("train",         gen_train,         40,    f"{SPLITS_DIR}/train_raw.jsonl",        f"{SPLITS_DIR}/train_tokenized.jsonl"),
        ("template_ood",  gen_template_ood,  20,    f"{SPLITS_DIR}/template_ood_raw.jsonl", f"{SPLITS_DIR}/template_ood_tokenized.jsonl"),
        ("symptom_ood",   gen_symptom_ood,   20,    f"{SPLITS_DIR}/symptom_ood_raw.jsonl",  f"{SPLITS_DIR}/symptom_ood_tokenized.jsonl"),
    ]

    summary_rows = []

    for label, gen, K_hda, raw_path, tok_path in splits_config:
        print(f"\n--- {label} (HDA K={K_hda}) ---")

        raw = gen.generate_raw_synthetic_samples(K=K_hda, save_in_jsonl_path=raw_path)
        print(f"  raw:       {len(raw):>7,} samples  →  {raw_path}")

        tok, not_found = gen.tokenize_all_samples(raw, save_in_jsonl_path=tok_path)
        print(f"  tokenized: {len(tok):>7,} samples  →  {tok_path}")
        print(f"  not_found: {len(not_found)}")

        for nf in not_found:
            nf["split"] = label
        all_not_found.extend(not_found)

        summary_rows.append((label, len(raw), len(tok), len(not_found)))

    # ---------------------------------------------------------------
    # 6d (cont.)  Write combined not_found log
    # ---------------------------------------------------------------
    not_found_path = f"{VERSION}/data/not_found.jsonl"
    if all_not_found:
        with open(not_found_path, "w") as f:
            for nf in all_not_found:
                f.write(json.dumps(nf, ensure_ascii=False) + "\n")
        print(f"\nNot-found log: {len(all_not_found)} entries  →  {not_found_path}")
    else:
        print("Hurray! no issues with the dataset generation all samples were properly whitespce tokenized and BIO word labeled!")

    # ---------------------------------------------------------------
    # 6e  Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 58)
    print(f"  {'split':<16} {'raw':>9} {'tokenized':>10} {'not_found':>10}")
    print("=" * 58)
    for label, n_raw, n_tok, n_nf in summary_rows:
        print(f"  {label:<16} {n_raw:>9,} {n_tok:>10,} {n_nf:>10,}")
    print("=" * 58)
    print(f"  {'TOTAL':<16} {sum(r for _,r,_,_ in summary_rows):>9,} "
          f"{sum(t for _,_,t,_ in summary_rows):>10,} "
          f"{sum(n for _,_,_,n in summary_rows):>10,}")


    # ---------------------------------------------------------------
    #  7       WORDPIECE TOKENIZATION
    # ---------------------------------------------------------------
    from transformers import AutoTokenizer
    from data_preparation.wordpiece_alignment import align_file, save_label_mappings, LABEL_LIST
    wordpiece_tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL)

    Path(SPLITS_DIR).mkdir(parents=True, exist_ok=True)

    save_label_mappings(out_dir=Path(f"{VERSION}/data"))
    print(f"Saved label2id.json and id2label.json to {f"{VERSION}/data"}/")
    print(f"Labels ({len(LABEL_LIST)}): {LABEL_LIST}\n")

    for label, gen, K_hda, raw_path, tok_path in splits_config:
        wordpiece_path = f"{SPLITS_DIR}/{label}_wordpiece.jsonl"
        sample_count, truncated_count = align_file(
            input_path=Path(tok_path),
            output_path=Path(wordpiece_path),
            tokenizer=wordpiece_tokenizer,
        )
        msg = f"[{label}] {sample_count:,} samples -> {wordpiece_path}"
        if truncated_count:
            msg += f"  (WARNING: {truncated_count} samples hit max_length=512)"
        print(msg)

        



