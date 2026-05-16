"""
Goal of this class is to take the list of symptoms and the dataset_templates and generate it into a ready to use and upload to huggingface training dataset
"""

# Postpones evaluating type hints (they become strings at runtime). That avoids errors
# when a hint mentions a class defined later in the file and can speed up import time.
from __future__ import annotations

from typing import Iterator, List, Tuple, Dict, Optional
import re, random
import pandas as pd


# Each entry is (logical group name, list of template strings). Pass v04 TEMPLATE_GROUPS or a subset.
TemplateGroups = List[Tuple[str, List[str]]]

# template for parcing SYMPTOM_
PLACEHOLDER_RE = re.compile(r'\{(SYMPTOM[^}]*)\}')
SINGLE_SYMPTOM_GROUPS = {"affirmed", "negated", "distractor"}
# =============
# HELPERS 
# =============


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
    if group_name == "word_collision":
        # collision = Tuple(id, prefLabel)
        collision = draw_symptom(symptoms_df)
        for plh in placeholders:
            fill_map[plh] = collision

        return fill_map
    
    already_used_ids: Dict[str, List[str]] = {}  # base_type -> [symptom_id, ...]
    for plh in placeholders:
        if plh in fill_map:
            continue

        # matches: "{SYMPTOM_POS_1}" but NOT "{SYMPTOM_POS_1}"
        index_match = re.match(r'^(SYMPTOM_(?:POS|NEG|O))_(\d+)$', plh)
        
        if index_match:
            base_type = index_match.group(1)
            already_used_ids.setdefault(base_type, [])
            # Keep drawing until we get a symptom not already used for this base type
            while True:
                sid, symptom_text = draw_symptom(symptoms_df)
                if sid not in already_used_ids[base_type]:
                    break
            already_used_ids[base_type].append(sid)
            fill_map[plh] = (sid, symptom_text)
        else:
            # not sure I understood this bare placeholder and when we will fall in this case
            # Bare placeholder — one draw, reused for all occurrences in template
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

# =========

class DatasetGenerator:
    def __init__(
        self,
        symptoms_df: pd.DataFrame,
        dataset_templates: TemplateGroups,
    ):
        self.symptoms_df = symptoms_df
        self.dataset_templates = dataset_templates

    # Iterator[...]: callers get items one-by-one (memory-friendly); Tuple[str, str] is
    # each item's shape: two strings (group name, then template text).
    def iter_grouped_templates(self) -> Iterator[Tuple[str, str]]:
        """Yield (template_group, template) for every template in order."""
        # dataset_templates is [(group_name, [tmpl, tmpl, ...]), ...]; unpacking assigns
        for group_name, templates in self.dataset_templates:
            for template in templates:
                # yield: pause here and give this pair to the caller; resume on next iteration.
                yield group_name, template




    def generate_raw_synthetic_samples(self, K: int = 20) -> List[dict]:
        """
        Single-symptom groups: one sample per symptom × per template (exhaustive, like v03).
        Multi-symptom groups:  K random draws per template.
        """
        samples: List[dict] = []

        for group_name, templates in self.dataset_templates:
            for template in templates:

                if group_name in SINGLE_SYMPTOM_GROUPS:

                    # I did not understand this syntax
                    ph = {"affirmed": "SYMPTOM_POS",
                        "negated":  "SYMPTOM_NEG",
                        "distractor": "SYMPTOM_O"}[group_name]

                    for _, row in self.symptoms_df.iterrows():
                        fill_map = {ph: (row["id"], row["prefLabel"])}
                        samples.append({
                            "text":           fill_template(template, fill_map),
                            "template_group": group_name,
                            "template":       template,
                            "symptoms":       build_symptoms_metadata(fill_map),
                        })

                else:
                    for _ in range(K):
                        fill_map = build_fill_map(template, group_name, symptoms_df)
                        samples.append({
                            "text":           fill_template(template, fill_map),
                            "template_group": group_name,
                            "template":       template,
                            "symptoms":       build_symptoms_metadata(fill_map),
                        })

        return samples


if __name__ == "__main__":
    
    import  sys, json
    from pathlib import Path
    import pandas as pd
    
    PROJECT_ROOT = Path("/Users/robertagarcia/Desktop/learning/bert_symptom_ner")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # local imports
    from config import settings
    from v04.dataset_templates import TEMPLATE_GROUPS
    

    def save_jsonl(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


    symptoms_df = pd.read_csv('base_symptom_dict.csv')



    print("VERSION: ", settings.VERSION)

    folder_version = settings.VERSION

    # 1) Generate synthetic dataset
    # ------------------------
    file_path = f"{settings.VERSION}/synthethic_data.jsonl"
    samples = DatasetGenerator(
        symptoms_df = symptoms_df,
        dataset_templates = TEMPLATE_GROUPS
    ).generate_raw_synthetic_samples()

    save_jsonl(file_path, samples)
    print(f"Step 1 completed: Raw Synthetic Samples were created. Total of {len(samples)} samples")


 




