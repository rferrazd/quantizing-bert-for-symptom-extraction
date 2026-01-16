"""
Real-world inference test cases for Symptom NER

Cases were labeled into five different categories. Cases may be multiple labeled.
These cases are designed to cover 5 critical categories:

1. Single symptom
2. Multi-word symptom
3. Multiple symptoms (same polarity)
4. Mixed polarity / negation
5. Messy, real clinical text (punctuation, conjunctions)

Having these cases are crucial to answering the question:

“Does my AI system behave sensibly on messy, clinical, real-world input?”

Note: more cases will be added in the upcoming versions :)
"""


REAL_WORLD_CASES = [
# 1) Single, simple symptom
{
"id": "case_simple_single",
"text": "The patient has a large blister on her toe.",
"expected_entities": [
{"text": "blister", "label": "SYMPTOM_POS"}
]
},

# 2) Single multi-word symptom
{
    "id": "case_multiword_single",
    "text": "The patient has lymphatic system symptom.",
    "expected_entities": [
        {"text": "lymphatic system symptom", "label": "SYMPTOM_POS"}
    ]
},

# 3) Multiple independent symptoms (same polarity)
{
    "id": "case_multiple_symptoms",
    "text": "The patient, male 65 years ols reports hair shedding, blisters on his head, and knee pain.",
    "expected_entities": [
        {"text": "hair shedding", "label": "SYMPTOM_POS"},
        {"text": "knee pain", "label": "SYMPTOM_POS"},
        {"text": "exanthema", "label": "SYMPTOM_POS"}
    ]
},

# 4) Mixed polarity (negation handling)
{
    "id": "case_negation",
    "text": "The patient does not have muscle necrosis, only signs of muscle cramps.",
    "expected_entities": [
        {"text": "muscle necrosis", "label": "SYMPTOM_NEG"},
        {"text": "muscle cramps", "label": "SYMPTOM_POS"}
    ]
},

# 5) Adjacent symptoms without punctuation (hard case)
{
    "id": "case_adjacent_entities",
    "text": "The patient has a facial edema rash and knee pain.",
    "expected_entities": [
        {"text": "facial edema", "label": "SYMPTOM_POS"},
        {"text": "rash", "label": "SYMPTOM_POS"},
        {"text": "knee pain", "label": "SYMPTOM_POS"}
    ]
},

# 6) Mixed symptoms with conjunctions
{
    "id": "case_conjunctions",
    "text": "The patient has a rash and is vomiting but no chest pain.",
    "expected_entities": [
        {"text": "rash", "label": "SYMPTOM_POS"},
        {"text": "vomiting", "label": "SYMPTOM_POS"},
        {"text": "chest pain", "label": "SYMPTOM_NEG"}
    ]
},

# 7) No symptoms (control case)
{
    "id": "case_no_symptoms",
    "text": "Marie Claire is feeling well today.",
    "expected_entities": []
}

]



# jaundice
# cyanosis
# edema
# anasarca
# limb edema
# palpebral edema
# facial edema
# change in skin texture
# spontaneous ecchymoses
# sweaty
# diaphoresis
# necrotic lesion
# rash
# ankle rash
# papular rash
# maculopapular rash
# blotchy red rash
# purpuric rash
