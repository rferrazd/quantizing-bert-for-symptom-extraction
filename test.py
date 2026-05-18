"""
test.py

Smoke checks for the curated common-symptom list against base_symptom_dict.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOP_40_COMMON_SYMPTOMS: list[str] = [
    # Fever family
    "fever",
    "mild fever",
    "lowgrade fever",
    "high fever",
    "very high fever",
    "hyperpyrexia",
    "hyperthermia",
    "sudden onset of fever",
    "prolonged fever",
    "continuous fever",
    "transient fever",
    "remittent fever",
    "relapsing fever",
    "cyclic fever",
    "pelepstein fever",
    "afebrile",
    "febrile convulsion",
    # Other high-yield
    "cough",
    "fatigue",
    "headache",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal pain",
    "chest pain",
    "dyspnea",
    "throat pain",
    "backache",
    "weakness",
    "malaise",
    "nasal congestion",
    "rhinorrhea",
    "rash",
    "joint pain",
    "dizziness",
    "chills",
    "anorexia",
    "constipation",
    "dysuria",
    "wheezing",
]


def verify_symptoms_in_csv(
    symptoms: list[str],
    csv_path: Path,
    pref_label_column: str = "prefLabel",
) -> tuple[list[str], list[str]]:
    """
    Check that each symptom string exists as a prefLabel in the CSV.

    Returns (found, missing) where found preserves input order.
    """
    df = pd.read_csv(csv_path)
    labels_in_csv = set(df[pref_label_column].astype(str))
    print(f"TOTAL NUMBER OF SYMPTOMS: {len(labels_in_csv)}")
    found: list[str] = []
    missing: list[str] = []
    for symptom in symptoms:
        if symptom in labels_in_csv:
            found.append(symptom)
        else:
            missing.append(symptom)
    return found, missing


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "base_symptom_dict.csv"

    found, missing = verify_symptoms_in_csv(TOP_40_COMMON_SYMPTOMS, csv_path)

    print(f"Checked {len(TOP_40_COMMON_SYMPTOMS)} curated symptoms against {csv_path.name}")
    print(f"  Found:   {len(found)}")
    print(f"  Missing: {len(missing)}")

    if missing:
        print("\nMissing prefLabels (not in CSV):")
        for label in missing:
            print(f"  - {label}")
        sys.exit(1)

    print("\nAll curated symptoms are present in the CSV.")
    for label in found:
        print(f"  ok  {label}")
