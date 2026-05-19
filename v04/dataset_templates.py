# Template inventory:
#   AFFIRMED_TEMPLATES:  40   — single-symptom, patient-affirmed
#   NEGATED_TEMPLATES:   40  — single-symptom, patient-denied (current denials only; past history is O)
#   DISTRACTOR_TEMPLATES: 38  — single-symptom, non-patient context (all O labels)
#   HDA_TEMPLATES:        20  — multi-sentence HDA paragraphs, mixed POS/NEG (growing)

AFFIRMED_TEMPLATES = [
    # Passive / impersonal constructions
    "It was noted that {SYMPTOM_POS} is present.",
    "{SYMPTOM_POS} was documented during evaluation.",
    "{SYMPTOM_POS} was observed on assessment.",
    "It has been recorded that the patient has {SYMPTOM_POS}.",
    # Temporal / onset framing
    "Onset of {SYMPTOM_POS} began recently.",
    "Has been experiencing {SYMPTOM_POS} for several days.",
    "{SYMPTOM_POS} started earlier this week.",
    "Reports ongoing {SYMPTOM_POS} since yesterday.",
    # Family / third-person report
    "Family reports that the patient has {SYMPTOM_POS}.",
    "Caregiver notes {SYMPTOM_POS}.",
    "According to relatives, the patient has {SYMPTOM_POS}.",
    "Third-party observer reports {SYMPTOM_POS}.",
    # Degree / severity qualifiers
    "Severe {SYMPTOM_POS} is reported.",
    "Mild {SYMPTOM_POS} noted during visit.",
    "Patient endorses significant {SYMPTOM_POS}.",
    "Moderate {SYMPTOM_POS} described by the patient.",
    # Multi-symptom context
    "Chief complaint includes {SYMPTOM_POS} among others.",
    "Presentation notable for {SYMPTOM_POS} and additional findings.",
    "Clinical picture includes {SYMPTOM_POS}.",
    "{SYMPTOM_POS} noted alongside other symptoms.",
    # Question/intake response style
    "Admits to {SYMPTOM_POS}.",
    "Endorses having {SYMPTOM_POS}.",
    "Acknowledges {SYMPTOM_POS} when asked.",
    "Confirms presence of {SYMPTOM_POS}.",
    # Incidental / secondary findings
    "Incidentally noted {SYMPTOM_POS}.",
    "Also reports {SYMPTOM_POS}.",
    "Additionally, {SYMPTOM_POS} was mentioned.",
    "{SYMPTOM_POS} identified as a secondary concern.",
    # Verb-last / inverted constructions
    "Present on exam: {SYMPTOM_POS}.",
    "Observed during evaluation: {SYMPTOM_POS}.",
    "Documented in assessment: {SYMPTOM_POS}.",
    "Identified on review: {SYMPTOM_POS}.",
    # Other varied structures
    "The patient indicates experiencing {SYMPTOM_POS}.",
    "Clinical history reveals {SYMPTOM_POS}.",
    "Evaluation uncovered {SYMPTOM_POS}.",
    "The individual is noted to have {SYMPTOM_POS}.",
    "{SYMPTOM_POS} persists at the time of visit.",
    "There is ongoing {SYMPTOM_POS}.",
    "The complaint of {SYMPTOM_POS} was elicited.",
    "{SYMPTOM_POS} remains a current issue.",
]

NEGATED_TEMPLATES = [
    # Conditional negation
    "No {SYMPTOM_NEG} at this time.",
    "Currently without {SYMPTOM_NEG}.",
    "At present, there is no {SYMPTOM_NEG}.",
    "No active {SYMPTOM_NEG} reported.",
    # Exam-finding negation
    "Exam reveals no {SYMPTOM_NEG}.",
    "Physical exam negative for {SYMPTOM_NEG}.",
    "Assessment shows absence of {SYMPTOM_NEG}.",
    "No {SYMPTOM_NEG} detected on examination.",
    # Passive negation
    "No {SYMPTOM_NEG} was identified.",
    "{SYMPTOM_NEG} was not observed.",
    "{SYMPTOM_NEG} was not documented.",
    "It was determined that {SYMPTOM_NEG} is absent.",
    # Qualifier-based negation
    "No significant {SYMPTOM_NEG} reported.",
    "{SYMPTOM_NEG} not clinically significant.",
    "No {SYMPTOM_NEG} identified.",
    # System-review negation
    "ROS negative for {SYMPTOM_NEG}.",
    "Review of systems: {SYMPTOM_NEG} denied.",
    "System review does not indicate {SYMPTOM_NEG}.",
    "No {SYMPTOM_NEG} reported in review of systems.",
    # Double-check / confirmation negation
    "Confirmed absence of {SYMPTOM_NEG}.",
    "Patient confirms no {SYMPTOM_NEG}.",
    "Absence of {SYMPTOM_NEG} verified.",
    "Negative for {SYMPTOM_NEG} upon confirmation.",
    # Other varied negation structures
    "The patient explicitly denies any {SYMPTOM_NEG}.",
    "There is no indication of {SYMPTOM_NEG}.",
    "{SYMPTOM_NEG} is not present at evaluation.",
    "No complaints related to {SYMPTOM_NEG}.",
    "Findings do not support presence of {SYMPTOM_NEG}.",
    "No observable {SYMPTOM_NEG} noted.",
    "{SYMPTOM_NEG} is absent on current assessment.",
    "No mention of {SYMPTOM_NEG} by the patient.",
    "{SYMPTOM_NEG} has not been experienced.",
    "There are no reports suggesting {SYMPTOM_NEG}.",
    "{SYMPTOM_NEG} excluded based on evaluation.",
    "During the evaluation the patient denied any evidence of {SYMPTOM_NEG}.",
    "Patient is entirely free of {SYMPTOM_NEG} at this time.",
    "Updates indicate a lack of {SYMPTOM_NEG}.",
    "History is negative for recent {SYMPTOM_NEG}.",
    "The patient does not experience {SYMPTOM_NEG}.",
    "Patient denies {SYMPTOM_NEG}.",
]

# ALL TOKENS SHOULD BE LABELED WITH A "O":
DISTRACTOR_TEMPLATES = [

#CATEGORY 1: Generic / educational statements

"{SYMPTOM_O} is a commonly reported manifestation in viral infections.",
"Symptoms such as {SYMPTOM_O} may indicate underlying systemic illness.",
"In clinical practice, {SYMPTOM_O} is often associated with inflammatory processes.",
"{SYMPTOM_O} can occur as a side effect of various medications.",
"Typical presentations include fever, cough, and {SYMPTOM_O}.",

#CATEGORY 2: Epidemiological / population-level statements

"A significant proportion of patients undergoing chemotherapy report {SYMPTOM_O}.",
"""Approximately 20% of individuals with this condition experience {SYMPTOM_O}.""",
"Postoperative patients frequently develop {SYMPTOM_O} within 24 hours.",
"In large cohorts, {SYMPTOM_O} has been observed as a common complaint.",
"Among elderly populations, {SYMPTOM_O} is often underreported.",

#CATEGORY 3: Hypothetical / conditional / instructional

"If {SYMPTOM_O} develops, the patient should seek immediate care.",
"In case of {SYMPTOM_O}, discontinue the medication and reassess.",
"If there is any onset of {SYMPTOM_O}, initiate protocol A.",
"Advise monitoring for {SYMPTOM_O} following discharge.",

#CATEGORY 4: Family history / third-party

"The patient’s father experienced {SYMPTOM_O} prior to his cardiac event.",
"Mother reports a history of chronic {SYMPTOM_O}.",
"A sibling was noted to have recurrent {SYMPTOM_O} during adolescence.",
"Family history significant for {SYMPTOM_O} in first-degree relatives.",
"The patient’s child had episodes of {SYMPTOM_O} last year.",

# CATEGORY 5: Past-resolved / historical patient symptoms

"The patient previously had {SYMPTOM_O}, which has since resolved.",
"History of intermittent {SYMPTOM_O} noted during childhood.",
"{SYMPTOM_O} was reported last year but is no longer present.",
"Patient had experienced {SYMPTOM_O} following surgery, now resolved.",
"Prior episodes of {SYMPTOM_O} have completely subsided.",

# CATEGORY 6: Discharge / post-visit instructions

"Patient was advised to monitor for {SYMPTOM_O} after discharge.",
"Return precautions include onset of {SYMPTOM_O}.",
"Patient educated on warning signs, including {SYMPTOM_O}.",
"Instructions given to seek emergency care if {SYMPTOM_O} develops.",
"Discharge summary notes to watch for {SYMPTOM_O} in the coming days.",

# CATEGORY 7: Clinical reasoning / differential

"Differential includes conditions associated with {SYMPTOM_O}.",
"Clinical reasoning accounts for {SYMPTOM_O} as a potential contributor.",
"Working diagnosis considered in the context of possible {SYMPTOM_O}.",
"Assessment weighs {SYMPTOM_O} as part of the differential.",
"Further workup indicated to rule out {SYMPTOM_O}.",

# CATEGORY 8: Negated or uncertain mentions (tricky non-affirmed contexts)

"Possible {SYMPTOM_O} to be ruled out pending further evaluation.",
"Query regarding {SYMPTOM_O} remains unanswered.",
"Evaluation ongoing to determine presence of {SYMPTOM_O}.",
"Unclear if {SYMPTOM_O} is contributing to the clinical picture."
]


# =============================================================================
# V04 ADDITIONS
# =============================================================================
# Placeholder convention for the multi-slot templates below:
#
#   {SYMPTOM_POS}   -> one symptom slot tagged SYMPTOM_POS
#   {SYMPTOM_NEG}   -> one symptom slot tagged SYMPTOM_NEG
#   {SYMPTOM_O}     -> one symptom slot that must NOT be tagged (label O)
#
#   Indexed variants ({SYMPTOM_POS_1}, {SYMPTOM_POS_2}, {SYMPTOM_NEG_1} …)
#   signal that the builder must draw DIFFERENT symptoms for each index.
#
#   A bare repeated placeholder (e.g. {SYMPTOM_NEG} appearing twice) means
#   the SAME symptom string should fill both slots — the builder replaces all
#   occurrences with one draw (used in narratives where the same symptom is
#   referenced more than once in the same sentence).
#
#   WORD_COLLISION_TEMPLATES are the special case where {SYMPTOM_O} and
#   {SYMPTOM_POS}/{SYMPTOM_NEG} are intentionally filled with the SAME string
#   from the pool (set them equal before substitution).
# =============================================================================

# HDA (História da Doença Atual) paragraphs — multi-sentence, multi-symptom.
# Structure follows Amplimed guidelines: onset → progression → associated → denied.
# Also mixed structure .... 
# Three tiers of complexity; all templates use indexed placeholders so the builder
# draws a DIFFERENT symptom for each slot.
# TIER 2 — Medium: 3–4 sentences, 3–4 POS + 3–4 NEG, temporal/progression language.
# TIER 3 — Complex: 4–5 sentences, 4–5 POS + 4–5 NEG, full HDA structure.
HDA_TEMPLATES = [

    # ── TIER 2 — Medium ─────────────────────────────────────────────────────

    # Order: NEG → POS. Negation opens the note; symptoms revealed on questioning.
    # Negation vocab: "currently without". Affirmation vocab: "acknowledges" / "admits to".
    (
        "Patient currently without {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}. "
        "On direct questioning, acknowledges {SYMPTOM_POS_1} with gradual onset over the past four days. "
        "Also admits to {SYMPTOM_POS_2} and {SYMPTOM_POS_3}, which have worsened since symptom onset."
    ),

    # Order: POS → NEG → POS interleaved. Onset leads, negation mid-note, secondary symptom closes.
    # Negation vocab: "was not observed" / "assessment shows absence of".
    # Affirmation vocab: "complains of" / "clinical history reveals".
    (
        "Patient complains of {SYMPTOM_POS_1} for approximately one week. "
        "{SYMPTOM_NEG_1} was not observed. "
        "Clinical history reveals {SYMPTOM_POS_2} associated with physical exertion. "
        "Assessment shows absence of {SYMPTOM_NEG_2} and {SYMPTOM_NEG_3}."
    ),

    # Order: POS → NEG at end, different vocabulary throughout.
    # Negation vocab: "currently without" / "ROS negative for".
    # Affirmation vocab: "onset of X began" / "associated with".
    (
        "Onset of {SYMPTOM_POS_1} began approximately three days ago, "
        "associated with {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Symptoms remain unchanged since onset. "
        "Currently without {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}. "
        "ROS negative for {SYMPTOM_NEG_3}."
    ),

    # ── TIER 3 — Complex ────────────────────────────────────────────────────

    # Order: NEG → POS → NEG. Demographics open; negation first, then chief complaint elicited,
    # then secondary NEG sweep closes.
    # Negation vocab: "absent on current assessment" / "no X at this time".
    # Affirmation vocab: "complaint of X was elicited" / "patient indicates experiencing" / "noted alongside".
    (
        "Male patient, 67 years old, with type 2 diabetes, brought by family to the emergency department. "
        "{SYMPTOM_NEG_1} and {SYMPTOM_NEG_2} are absent on current assessment. "
        "The complaint of {SYMPTOM_POS_1} was elicited upon questioning, with onset approximately 5 days ago. "
        "Patient indicates experiencing {SYMPTOM_POS_2} and {SYMPTOM_POS_3}, worsening over the last 48 hours. "
        "{SYMPTOM_POS_4} also noted alongside these symptoms. "
        "No {SYMPTOM_NEG_3} or {SYMPTOM_NEG_4} at this time."
    ),

    # Order: POS → NEG → POS → NEG interleaved throughout.
    # Negation vocab: "physical exam negative for" / "no X was identified" / "confirmed absence of".
    # Affirmation vocab: "presents with" / "has been experiencing" / "endorses".
    (
        "Patient presents with {SYMPTOM_POS_1} of sudden onset this morning. "
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Has been experiencing {SYMPTOM_POS_2} and {SYMPTOM_POS_3} since yesterday evening. "
        "No {SYMPTOM_NEG_2} was identified. "
        "Patient also endorses {SYMPTOM_POS_4} that worsens with movement. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),

    # Order: POS → NEG → POS → NEG interleaved, demographics open.
    # Negation vocab: "no X reported" / "is not present at evaluation" / "there is no indication of".
    # Affirmation vocab: "acknowledges" / "onset of X began" / "also mentions".
    (
        "Female patient, 45 years old, previously healthy, walk-in. "
        "Acknowledges {SYMPTOM_POS_1} for approximately 6 days, with gradual onset. "
        "No {SYMPTOM_NEG_1} reported. "
        "Onset of {SYMPTOM_POS_2} began two days after the initial complaint. "
        "{SYMPTOM_NEG_2} is not present at evaluation. "
        "Also mentions {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, predominantly in the evenings. "
        "There is no indication of {SYMPTOM_NEG_3}."
    ),

    # ── Demographic variants ─────────────────────────────────────────────────

    # Pediatric — Tier 2, POS → NEG.
    # Affirmation vocab: "parents report" / "associated with".
    # Negation vocab: "no X observed" / "ROS negative for".
    (
        "Child patient, 7 years old, brought by parents to the emergency department. "
        "Parents report onset of {SYMPTOM_POS_1} approximately two days ago, associated with {SYMPTOM_POS_2}. "
        "No {SYMPTOM_NEG_1} observed by caregivers at home. "
        "ROS negative for {SYMPTOM_NEG_2} and {SYMPTOM_NEG_3}."
    ),

    # Elderly with comorbidities — Tier 3, NEG → POS → NEG.
    # Family member as informant (patient endorses one symptom herself).
    # Affirmation vocab: "daughter reports" / "endorses" / "describes as".
    # Negation vocab: "no X at this time" / "assessment shows absence of".
    (
        "Elderly female patient, 79 years old, with hypertension, brought by daughter. "
        "No {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2} at this time. "
        "Daughter reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2} over the past five days. "
        "Patient herself endorses {SYMPTOM_POS_3}, which she describes as intermittent. "
        "Assessment shows absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),

    # ── Additional templates (T9–T20): balance ratios, demographics, ordering ──

    # T9 — Tier 2, NEG-heavy (1 POS / 4 NEG), no demographics, NEG→POS→NEG.
    # Negation vocab: "physical exam negative for" / "no X identified" / "confirmed absence of".
    # Affirmation vocab: "reports".
    (
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Patient reports {SYMPTOM_POS_1} since yesterday morning. "
        "No {SYMPTOM_NEG_2} identified on assessment. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),

    # T10 — Tier 2, NEG-heavy (1 POS / 3 NEG), adult male walk-in, NEG→POS.
    # Negation vocab: "denies" / "ROS negative for".
    # Affirmation vocab: "endorses".
    (
        "Male patient, 34 years old, walk-in. "
        "Denies {SYMPTOM_NEG_1} and {SYMPTOM_NEG_2}. "
        "ROS negative for {SYMPTOM_NEG_3}. "
        "Endorses {SYMPTOM_POS_1} intermittently over the past week."
    ),

    # T11 — Tier 2, balanced (2 POS / 2 NEG), adult female with comorbidities, POS→NEG interleaved.
    # Negation vocab: "was not observed" / "explicitly denies".
    # Affirmation vocab: "presents with" / "also mentions".
    (
        "Female patient, 52 years old, with hypertension and hypothyroidism. "
        "Presents with {SYMPTOM_POS_1} of 4 days duration. "
        "{SYMPTOM_NEG_1} was not observed on exam. "
        "Also mentions {SYMPTOM_POS_2} worsening at night. "
        "Patient explicitly denies {SYMPTOM_NEG_2}."
    ),

    # T12 — Tier 2, POS-heavy (3 POS / 1 NEG), brought by spouse, POS→NEG.
    # Negation vocab: "no X reported".
    # Affirmation vocab: "complains of" / "confirms" / "observed at home".
    (
        "Patient brought to clinic by husband. "
        "Complains of {SYMPTOM_POS_1} and {SYMPTOM_POS_2} starting four days ago. "
        "Husband confirms {SYMPTOM_POS_3} observed at home over the same period. "
        "No {SYMPTOM_NEG_1} reported."
    ),

    # T13 — Tier 2, balanced (2 POS / 2 NEG), GP referral, POS→NEG.
    # Negation vocab: "is not present at evaluation" / "there is no indication of".
    # Affirmation vocab: "referred by GP for evaluation of" / "has been experiencing".
    (
        "Female patient referred by GP for evaluation of {SYMPTOM_POS_1}. "
        "Has been experiencing {SYMPTOM_POS_2} concurrently for the past several days. "
        "{SYMPTOM_NEG_1} is not present at evaluation. "
        "There is no indication of {SYMPTOM_NEG_2}."
    ),

    # T14 — Tier 3, NEG-heavy (2 POS / 5 NEG), no demographics, NEG→POS→NEG interleaved.
    # Negation vocab: "currently without" / "patient confirms no" / "findings do not support presence of".
    # Affirmation vocab: "admits to" / "also mentioned briefly".
    (
        "Patient presents for evaluation. "
        "Currently without {SYMPTOM_NEG_1}, {SYMPTOM_NEG_2}, or {SYMPTOM_NEG_3}. "
        "On questioning, admits to {SYMPTOM_POS_1} of recent onset. "
        "Patient confirms no {SYMPTOM_NEG_4}. "
        "{SYMPTOM_POS_2} also mentioned briefly during the visit. "
        "Findings do not support presence of {SYMPTOM_NEG_5}."
    ),

    # T15 — Tier 3, POS-heavy (5 POS / 1 NEG), adolescent with father informant, POS→NEG.
    # Negation vocab: "no X reported".
    # Affirmation vocab: "father reports" / "acknowledges" / "onset of X began".
    (
        "Adolescent patient, 14 years old, brought by father. "
        "Father reports {SYMPTOM_POS_1} for one week, with associated {SYMPTOM_POS_2}. "
        "Patient also acknowledges {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, predominantly at night. "
        "Onset of {SYMPTOM_POS_5} began two days ago. "
        "No {SYMPTOM_NEG_1} reported."
    ),

    # T16 — Tier 3, balanced (3 POS / 3 NEG), elderly male with comorbidities, POS→NEG interleaved.
    # Negation vocab: "physical exam negative for" / "assessment shows absence of" / "there is no indication of".
    # Affirmation vocab: "son reports" / "endorses" / "progressing over".
    (
        "Male patient, 73 years old, with type 2 diabetes and chronic kidney disease, brought by son to the emergency department. "
        "Son reports {SYMPTOM_POS_1} progressing over five days. "
        "Patient endorses {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Assessment shows absence of {SYMPTOM_NEG_2}. "
        "There is no indication of {SYMPTOM_NEG_3}."
    ),

    # T17 — Tier 3, NEG-heavy (2 POS / 4 NEG), adult female with asthma history, NEG→POS→NEG.
    # Negation vocab: "is absent on current assessment" / "was not documented" / "confirmed absence of".
    # Affirmation vocab: "patient indicates experiencing" / "also reports".
    (
        "Female patient, 41 years old, with history of asthma, walk-in. "
        "{SYMPTOM_NEG_1} is absent on current assessment. "
        "{SYMPTOM_NEG_2} was not documented during evaluation. "
        "Patient indicates experiencing {SYMPTOM_POS_1} since the morning. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}. "
        "Also reports mild {SYMPTOM_POS_2}."
    ),

    # T18 — Tier 3, slight POS-heavy (4 POS / 3 NEG), no demographics, interleaved.
    # Negation vocab: "no X at this time" / "review of systems: X denied" / "ROS negative for".
    # Affirmation vocab: "acknowledges" / "additionally reports" / "also identified on review".
    (
        "Patient presents to the emergency department this evening. "
        "Acknowledges {SYMPTOM_POS_1} of acute onset. "
        "No {SYMPTOM_NEG_1} at this time. "
        "Additionally reports {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Review of systems: {SYMPTOM_NEG_2} denied. "
        "{SYMPTOM_POS_4} also identified on review. "
        "ROS negative for {SYMPTOM_NEG_3}."
    ),

    # T19 — Tier 3, POS-heavy (5 POS / 1 NEG), pediatric with mother informant, POS→NEG→POS.
    # Negation vocab: "no X observed".
    # Affirmation vocab: "mother describes onset of" / "presents with" / "additionally, X noted".
    (
        "Pediatric patient, 5 years old, brought by mother. "
        "Mother describes onset of {SYMPTOM_POS_1} three days ago, with subsequent development of {SYMPTOM_POS_2}. "
        "Child also presents with {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, worse in the afternoon. "
        "No {SYMPTOM_NEG_1} observed at home. "
        "Additionally, {SYMPTOM_POS_5} noted during evaluation."
    ),

    # T20 — Tier 3, balanced (3 POS / 3 NEG), elderly walk-in, no comorbidities, POS→NEG interleaved.
    # Negation vocab: "currently without" / "no X reported".
    # Affirmation vocab: "complaint of X elicited" / "patient indicates experiencing" / "mentioned in passing".
    (
        "Elderly male patient, 70 years old, previously healthy, walk-in. "
        "Complaint of {SYMPTOM_POS_1} elicited at intake. "
        "Patient indicates experiencing {SYMPTOM_POS_2} over the past three days. "
        "Currently without {SYMPTOM_NEG_1}. "
        "{SYMPTOM_POS_3} mentioned in passing during questioning. "
        "No {SYMPTOM_NEG_2} or {SYMPTOM_NEG_3} reported."
    ),
]




# Hard negatives: same symptom word appears as distractor AND as patient mention
# in the same example. Trains the model off lexical-keyword shortcuts.
# Builder should fill {SYMPTOM_POS}/{SYMPTOM_NEG} and {SYMPTOM_O} with the SAME
# symptom string when generating this template.
# WORD_COLLISION_TEMPLATES = [
#     # --- DISTRACTOR FIRST → POS (original 12) ---
#     "{SYMPTOM_O} is common in viral infections; the patient denies {SYMPTOM_NEG}.",
#     "{SYMPTOM_O} is a known side effect of this medication; the patient currently reports {SYMPTOM_POS}.",
#     "The literature describes {SYMPTOM_O} as a hallmark of this syndrome; on exam the patient has {SYMPTOM_POS}.",
#     "Family history is notable for {SYMPTOM_O}; the patient herself denies {SYMPTOM_NEG}.",
#     "Mother had chronic {SYMPTOM_O} during adolescence; the patient currently endorses {SYMPTOM_POS}.",
#     "Screening form asks about {SYMPTOM_O}; the patient reports {SYMPTOM_POS}.",
#     "Educational material covered warning signs such as {SYMPTOM_O}; the patient now denies {SYMPTOM_NEG}.",
#     "{SYMPTOM_O} was reported last year but has resolved; the patient currently has {SYMPTOM_POS}.",
#     "Discussed possible future symptoms including {SYMPTOM_O}; today the patient denies {SYMPTOM_NEG}.",
#     "Postoperative patients frequently develop {SYMPTOM_O}; this patient denies {SYMPTOM_NEG}.",
#     "Checklist item: {SYMPTOM_O} — marked not applicable. The patient endorses {SYMPTOM_POS}.",
#     "{SYMPTOM_O} can occur as an adverse effect; the patient currently reports {SYMPTOM_POS}.",

# ]


# (group_name, templates) pairs for generators that branch on list membership,
# not on parsing placeholder strings inside each template.
TEMPLATE_GROUPS = [
    ("affirmed",    AFFIRMED_TEMPLATES),
    ("negated",     NEGATED_TEMPLATES),
    ("distractor",  DISTRACTOR_TEMPLATES),
    ("hda",         HDA_TEMPLATES),
]
