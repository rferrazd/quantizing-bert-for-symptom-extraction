"""
Sample transcriptions for V03 inference stress-testing.

Levels:
    - easy   : at most one {SYMPTOM} per sentence
    - medium : at most two {SYMPTOM} per sentence, mixed polarity

Placeholders:
    {SYMPTOM_POS} — patient-reported positive symptom (should be tagged SYMPTOM_POS)
    {SYMPTOM_NEG} — explicitly denied symptom     (should be tagged SYMPTOM_NEG)

Usage: replace placeholders with real symptom strings before inference.
"""

transcriptions = [

    # ------------------------------------------------------------------
    # EASY — at most one symptom per sentence, one clear polarity cue each
    # ------------------------------------------------------------------
    {
        "level": "easy",
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
]


# Symptom pool (sourced from training distribution — use these to swap placeholders)
SYMPTOM_POOL = [
    "jaundice", "cyanosis", "edema", "anasarca", "limb edema",
    "palpebral edema", "facial edema", "change in skin texture",
    "spontaneous ecchymoses", "sweaty", "diaphoresis", "necrotic lesion",
    "rash", "ankle rash", "papular rash", "maculopapular rash", "allergic reaction"
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

