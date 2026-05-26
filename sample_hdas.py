"""
Sample transcriptions for V03/V04 inference stress-testing.

Levels:
    - easy   : at most one {SYMPTOM} per sentence
    - medium : multi-symptom HDA paragraphs (mixed polarity, telegraphic clinical style)

Placeholders:
    {SYMPTOM_POS} — patient-reported positive symptom (should be tagged SYMPTOM_POS)
    {SYMPTOM_NEG} — explicitly denied symptom     (should be tagged SYMPTOM_NEG)

Usage: replace placeholders with real symptom strings before inference.

TODO (deferred, not P1):
    Add 10–15 more hold-out HDAs to broaden OOD coverage. Specifically need:
      - NEG-first openings (current hold-outs are all POS → NEG)
      - Demographic openings (current hold-outs have none)
      - Varied negation cues beyond "denies" (current hold-outs lean almost entirely on "denies")
      - Interleaved POS/NEG orderings
      - Family-as-informant variants
      - Adversarial non-symptom denials labeled O (e.g. "denies known drug allergies" — already
        present in hda6/8/9/10 but never explicitly excluded; keep this convention)
"""

transcriptions_and_hdas = [

    # ------------------------------------------------------------------
    # EASY — at most one symptom per sentence, one clear polarity cue each
    # ------------------------------------------------------------------
    {
        "level": "easy",
        "case": "easy_allergic",
        "template": (
            "Female patient, 38 years old, previously healthy, presents referred by her GP "
            "with a 5-day history of allergic reaction. "
            "She reports gradual onset with no identifiable triggering event and no recent "
            "changes to medication or diet. "
            "She denies jaundice, and there is no history of similar episodes. "
            "She also denies edema. "
            "On direct questioning she endorses maculopapular rash, which she describes as "
            "intermittent and worse in the evenings. "
            "No change in skin texture,, and no recent travel are reported."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                # Char spans [start, end) — same slice semantics as template[start:end]
                "jaundice": [241, 249],
                "edema": [312, 317],
                "change in skin texture": [440, 462],
            },
            "SYMPTOM_POS": {
                "maculopapular rash": [354, 372],
                "allergic reaction": [102, 119],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — up to two symptoms per sentence, mixed polarity in same clause
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "medium_rash",
        # Instantiation of the placeholder pattern using SYMPTOM_POOL strings
        # ({SYMPTOM_POS}×5, {SYMPTOM_NEG}×4) — left-to-right fill order matches list order below.
        "template": (
            "Male patient, 54 years old, with a background of hypertension and type 2 diabetes, "
            "presents to the emergency department accompanied by his wife. "
            "She reports that over the past ten days he has developed rash and "
            "ankle rash, both progressive in nature. "
            "The patient denies jaundice but acknowledges papular rash that has "
            "worsened since last week. "
            "His wife confirms maculopapular rash observed at home over the last three days; "
            "the patient himself denies facial edema when asked directly. "
            "Review of systems is notable for allergic reaction; cyanosis is denied. "
            "No change in skin texture or inflammation is reported by either the patient or his wife."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "jaundice": [270, 278],
                "facial edema": [451, 463],
                "cyanosis": [537, 545],
                "change in skin texture": [560, 582],
            },
            "SYMPTOM_POS": {
                "rash": [202, 206],
                "ankle rash": [211, 221],
                "papular rash": [296, 308],
                "maculopapular rash": [362, 380],
                "allergic reaction": [518, 535],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — hda6: dengue-like presentation (high fever, mosquito exposure)
    # Telegraphic style, POS → NEG ordering, no demographics, "denies"-heavy
    # Adversarial O cases: "denies previous comorbidities", "denies known drug allergies"
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "hda6",
        "template": (
            "Patient reports sudden onset of high fever 4 days ago, associated with intense headache and chills. "
            "She describes diffuse joint pain and fatigue that worsened progressively over the last two days. "
            "Patient also reports nausea and anorexia, with reduced food intake. "
            "Patient denies cough, throat pain, rhinorrhea, or nasal congestion. "
            "Patient denies vomiting or diarrhea. "
            "Patient mentions recent exposure to mosquitoes while traveling to a rural area. "
            "Patient denies previous comorbidities and continuous medication use. "
            "Patient denies known drug allergies."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "cough": [280, 285],
                "throat pain": [287, 298],
                "rhinorrhea": [300, 310],
                "nasal congestion": [315, 331],
                "vomiting": [348, 356],
                "diarrhea": [360, 368],
            },
            "SYMPTOM_POS": {
                "high fever": [32, 42],
                "headache": [79, 87],
                "chills": [92, 98],
                "joint pain": [122, 132],
                "fatigue": [137, 144],
                "nausea": [218, 224],
                "anorexia": [229, 237],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — hda7: respiratory presentation (cough, wheezing, dyspnea)
    # Adversarial O case: "previous history of childhood asthma" (past history → O)
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "hda7",
        "template": (
            "Patient reports cough with onset approximately 5 days ago, associated with wheezing and dyspnea during moderate physical activity. \n"
            "Reports mild fever, fatigue, and chest pain that worsens during coughing episodes.\n"
            "Patient mentions nasal congestion and rhinorrhea predominantly during the night. Patient denies nausea, vomiting, abdominal pain, or diarrhea. \n"
            "Patient denies rash or joint pain. Patient has a previous history of childhood asthma but is not currently using regular medications."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "nausea": [311, 317],
                "vomiting": [319, 327],
                "abdominal pain": [329, 343],
                "diarrhea": [348, 356],
                "rash": [374, 378],
                "joint pain": [382, 392],
            },
            "SYMPTOM_POS": {
                "cough": [16, 21],
                "wheezing": [75, 83],
                "dyspnea": [88, 95],
                "mild fever": [140, 150],
                "fatigue": [152, 159],
                "chest pain": [165, 175],
                "nasal congestion": [232, 248],
                "rhinorrhea": [253, 263],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — hda8: abdominal presentation
    # Adversarial cases: "mentioned an absense of" (typo + alt-negation cue), "denies known drug allergies"
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "hda8",
        "template": (
            "Patient reports abdominal pain with onset 2 days ago, located diffusely throughout the abdomen and associated with nausea and vomiting. Patient describes associated weakness and dizziness, especially when standing up rapidly. \n"
            "Patient also reports transient fever and anorexia. Patient denies diarrhea or constipation. \n"
            "When asked, the patient mentioned an absense of dysuria or backache. No past surgeries or chronic illnesses. Patient denies known drug allergies."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "diarrhea": [293, 301],
                "constipation": [305, 317],
                "dysuria": [368, 375],
                "backache": [379, 387],
            },
            "SYMPTOM_POS": {
                "abdominal pain": [16, 30],
                "nausea": [115, 121],
                "vomiting": [126, 134],
                "weakness": [165, 173],
                "dizziness": [178, 187],
                "transient fever": [248, 263],
                "anorexia": [268, 276],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — hda9: headache + URI presentation
    # Adversarial cues: "does not present any signs of" / "neither X, Y, nor Z are present"
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "hda9",
        "template": (
            "Patient reports headache of moderate intensity for approximately 3 days, associated with dizziness and malaise. \n"
            "Patient also reports nasal congestion and rhinorrhea, predominantly in the morning. Patient mentions mild fever and throat pain with progressive worsening since symptom onset. \n"
            "Patient does not present any signs of cough, chest pain, or dyspnea. Neither nausea, vomiting, nor abdominal pain are present. \n"
            "Patient reports increased fatigue during daily activities. \n"
            "Patient denies previous comorbidities and continuous medication use."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "cough": [328, 333],
                "chest pain": [335, 345],
                "dyspnea": [350, 357],
                "nausea": [367, 373],
                "vomiting": [375, 383],
                "abdominal pain": [389, 403],
            },
            "SYMPTOM_POS": {
                "headache": [16, 24],
                "dizziness": [89, 98],
                "malaise": [103, 110],
                "nasal congestion": [134, 150],
                "rhinorrhea": [155, 165],
                "mild fever": [214, 224],
                "throat pain": [229, 240],
                "fatigue": [444, 451],
            },
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — hda10: prolonged fever, systemic presentation
    # Adversarial: "without vomiting" (alt-negation cue after a POS clause)
    # ------------------------------------------------------------------
    {
        "level": "medium",
        "case": "hda10",
        "template": (
            "Patient reports prolonged fever for approximately 7 days, associated with fatigue and weakness. \n"
            "He describes chills occurring mainly during the evening and reports diffuse backache and joint pain. \n"
            "Patient also mentions anorexia and nausea without vomiting. Patient denies cough, wheezing, or dyspnea. \n"
            "Patient denies dysuria or abdominal pain. Patient reports no recent travel history. \n"
            "Patient denies previous chronic diseases and denies known drug allergies."
        ),
        "entities": {
            "SYMPTOM_NEG": {
                "vomiting": [249, 257],
                "cough": [274, 279],
                "wheezing": [281, 289],
                "dyspnea": [294, 301],
                "dysuria": [319, 326],
                "abdominal pain": [330, 344],
            },
            "SYMPTOM_POS": {
                "prolonged fever": [16, 31],
                "fatigue": [74, 81],
                "weakness": [86, 94],
                "chills": [110, 116],
                "backache": [173, 181],
                "joint pain": [186, 196],
                "anorexia": [221, 229],
                "nausea": [234, 240],
            },
        },
    },
]

# Symptom pool (sourced from training distribution — use these to swap placeholders)
SYMPTOM_POOL = [
    "jaundice", "cyanosis", "edema", "anasarca", "limb edema",
    "palpebral edema", "facial edema", "change in skin texture",
    "spontaneous ecchymoses", "sweaty", "diaphoresis", "necrotic lesion",
    "rash", "ankle rash", "papular rash", "maculopapular rash", "allergic reaction",
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











def entity_char_span(text: str, phrase: str) -> list[int]:
    """Return [start, end) character indices of the first occurrence of *phrase* in *text*."""
    start = text.find(phrase)
    if start < 0:
        raise ValueError(f"phrase not found: {phrase!r}")
    return [start, start + len(phrase)]


def verify_stored_entity_spans() -> None:
    """Assert each stored [start, end) span matches template[start:end] for that phrase."""
    for entry in transcriptions:
        if "entities" not in entry:
            continue
        text = entry["template"]
        for _label, name_to_span in entry["entities"].items():
            for phrase, span in name_to_span.items():
                start, end = span
                got = text[start:end]
                assert got == phrase, (
                    f"{entry.get('level')!r} — span mismatch for {phrase!r}: "
                    f"text[{start}:{end}]={got!r}"
                )


if __name__ == "__main__":
    verify_stored_entity_spans()
    print("verify_stored_entity_spans: OK")

