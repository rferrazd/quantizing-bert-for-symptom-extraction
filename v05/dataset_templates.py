"""
v05/dataset_templates.py

V05 template inventory. Inherits V04 structure (polarity-specific placeholders,
indexed HDA slots) and adds three targeted fixes based on V04 HDA error analysis:

  P1 — Adversarial-O distractors ("Denied non-symptom history"): teaches the
       model that denial cues do NOT always precede a symptom.
  P2 — Mid-clause negation HDA templates: teaches paraphrases of negation
       ("X without Y", "absence of", "Neither A, B, nor C",
        "does not present any signs of").
  P3 — Modifier-coverage in AFFIRMED + HDA templates: places modifiers
       ("sudden onset of", "intense", "diffuse", "increased", "progressive",
        "worsening", "of moderate intensity") adjacent to {SYMPTOM_POS} so
       the generator labels them O — preventing boundary overreach at inference.

Placeholder convention:
  {SYMPTOM_POS}                 — single slot, label SYMPTOM_POS
  {SYMPTOM_NEG}                 — single slot, label SYMPTOM_NEG
  {SYMPTOM_O}                   — single slot, label O (used in DISTRACTOR_TEMPLATES)
  {NON_SYMPTOM_HISTORY}         — single slot drawn from NON_SYMPTOM_HISTORY_POOL,
                                  label O (NEW for V05 P1)
  Indexed variants ({SYMPTOM_POS_1}, {SYMPTOM_NEG_2}, …) — different draws per slot.

Inventory:
  AFFIRMED_TEMPLATES:            48   (40 V04 + 8 new modifier templates — P3)
  NEGATED_TEMPLATES:             38   (V04, unchanged)
  DISTRACTOR_TEMPLATES:          63   (38 V04 + 25 new denied-non-symptom — P1)
  HDA_TEMPLATES:                 28   (20 V04 + 8 mid-clause negation — P2)
"""

# =============================================================================
# AFFIRMED_TEMPLATES — single-symptom, patient-affirmed. (V04 + V05 P3)
# =============================================================================
AFFIRMED_TEMPLATES = [
    # ---- V04 (unchanged) -----------------------------------------------------
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

    # ---- V05 P3: modifier coverage (modifiers labeled O by generator) -------
    # These teach the model that pre-modifiers adjacent to a symptom are NOT
    # part of the entity. Targets V04 boundary overreach failures.
    "Patient reports sudden onset of {SYMPTOM_POS}.",
    "Describes intense {SYMPTOM_POS} since this morning.",
    "Reports diffuse {SYMPTOM_POS} throughout the day.",
    "Patient endorses increased {SYMPTOM_POS} over the past week.",
    "{SYMPTOM_POS} of moderate intensity reported.",
    "Notes progressive {SYMPTOM_POS} over several days.",
    "Describes worsening {SYMPTOM_POS} since onset.",
    "Acknowledges persistent {SYMPTOM_POS} that has not improved.",
]


# =============================================================================
# NEGATED_TEMPLATES — single-symptom, current denial only. (V04, unchanged)
# =============================================================================
# Past-history negations remain out of scope (labeled O via distractor group).
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


# =============================================================================
# NON_SYMPTOM_HISTORY_POOL — V05 NEW
# =============================================================================
# Phrases that look syntactically like symptoms in a denial context but are
# administrative/historical/contextual. Fills the {NON_SYMPTOM_HISTORY}
# placeholder in DISTRACTOR_TEMPLATES category 9.
# All occurrences in generated text MUST be labeled O.
NON_SYMPTOM_HISTORY_POOL = [
    "drug allergies",
    "known drug allergies",
    "previous comorbidities",
    "chronic illnesses",
    "chronic diseases",
    "past surgeries",
    "prior surgeries",
    "recent travel",
    "recent travel history",
    "regular medications",
    "continuous medication use",
    "similar episodes",
    "prior episodes",
    "triggering event",
    "identifiable triggering event",
    "recent changes to medication",
    "recent changes to diet",
    "family history of cancer",
    "history of hospitalization",
    "history of trauma",
    "smoking history",
    "alcohol use",
]


# =============================================================================
# DISTRACTOR_TEMPLATES — ALL TOKENS LABELED "O".
# (V04 categories 1-8, unchanged) + (V05 P1 category 9: denied non-symptom)
# =============================================================================
DISTRACTOR_TEMPLATES = [

    # ---- V04 unchanged --------------------------------------------------------
    # CATEGORY 1: Generic / educational statements
    "{SYMPTOM_O} is a commonly reported manifestation in viral infections.",
    "Symptoms such as {SYMPTOM_O} may indicate underlying systemic illness.",
    "In clinical practice, {SYMPTOM_O} is often associated with inflammatory processes.",
    "{SYMPTOM_O} can occur as a side effect of various medications.",
    "Typical presentations include fever, cough, and {SYMPTOM_O}.",

    # CATEGORY 2: Epidemiological / population-level statements
    "A significant proportion of patients undergoing chemotherapy report {SYMPTOM_O}.",
    "Approximately 20% of individuals with this condition experience {SYMPTOM_O}.",
    "Postoperative patients frequently develop {SYMPTOM_O} within 24 hours.",
    "In large cohorts, {SYMPTOM_O} has been observed as a common complaint.",
    "Among elderly populations, {SYMPTOM_O} is often underreported.",

    # CATEGORY 3: Hypothetical / conditional / instructional
    "If {SYMPTOM_O} develops, the patient should seek immediate care.",
    "In case of {SYMPTOM_O}, discontinue the medication and reassess.",
    "If there is any onset of {SYMPTOM_O}, initiate protocol A.",
    "Advise monitoring for {SYMPTOM_O} following discharge.",

    # CATEGORY 4: Family history / third-party
    "The patient's father experienced {SYMPTOM_O} prior to his cardiac event.",
    "Mother reports a history of chronic {SYMPTOM_O}.",
    "A sibling was noted to have recurrent {SYMPTOM_O} during adolescence.",
    "Family history significant for {SYMPTOM_O} in first-degree relatives.",
    "The patient's child had episodes of {SYMPTOM_O} last year.",

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
    "Unclear if {SYMPTOM_O} is contributing to the clinical picture.",

    # ---- V05 P1: CATEGORY 9 — Denied non-symptom history (NEW) ---------------
    # Pairs denial cues with non-symptom NPs to break the V04 shortcut
    # "denial cue → next NP is SYMPTOM_NEG". All tokens labeled O.
    # The {NON_SYMPTOM_HISTORY} placeholder is filled from NON_SYMPTOM_HISTORY_POOL.
    "Patient denies {NON_SYMPTOM_HISTORY}.",
    "Patient denies known {NON_SYMPTOM_HISTORY}.",
    "Patient denies any {NON_SYMPTOM_HISTORY}.",
    "Denies {NON_SYMPTOM_HISTORY}.",
    "No {NON_SYMPTOM_HISTORY} reported.",
    "No {NON_SYMPTOM_HISTORY} at this time.",
    "No history of {NON_SYMPTOM_HISTORY}.",
    "No prior {NON_SYMPTOM_HISTORY}.",
    "No known {NON_SYMPTOM_HISTORY}.",
    "Patient reports no {NON_SYMPTOM_HISTORY}.",
    "Currently without {NON_SYMPTOM_HISTORY}.",
    "Patient is not currently using {NON_SYMPTOM_HISTORY}.",
    "Not currently on {NON_SYMPTOM_HISTORY}.",
    "Patient denies previous {NON_SYMPTOM_HISTORY}.",
    "No mention of {NON_SYMPTOM_HISTORY}.",
    "Patient denies any recent {NON_SYMPTOM_HISTORY}.",
    "ROS negative for {NON_SYMPTOM_HISTORY}.",
    "Physical exam negative for {NON_SYMPTOM_HISTORY}.",
    "Patient explicitly denies {NON_SYMPTOM_HISTORY}.",
    "Confirmed absence of {NON_SYMPTOM_HISTORY}.",
    "Patient confirms no {NON_SYMPTOM_HISTORY}.",
    "There is no indication of {NON_SYMPTOM_HISTORY}.",
    "Patient does not report {NON_SYMPTOM_HISTORY}.",
    "Denies any {NON_SYMPTOM_HISTORY} on intake.",
    "Patient has no significant {NON_SYMPTOM_HISTORY}.",
]


# =============================================================================
# HDA_TEMPLATES — multi-sentence clinical paragraphs, mixed POS/NEG.
# (V04 T1-T20, unchanged) + (V05 P2/P3 T21-T28, NEW)
# =============================================================================
# Placeholder convention:
#   {SYMPTOM_POS}, {SYMPTOM_NEG}, {SYMPTOM_O}                  — single draws
#   {NON_SYMPTOM_HISTORY}                                      — V05 NEW, label O
#   Indexed ({SYMPTOM_POS_1}, {SYMPTOM_NEG_2} …)              — different draws
HDA_TEMPLATES = [

    # ====================== V04 templates T1–T20 (unchanged) =================

    # T1 — Tier 2, NEG → POS
    (
        "Patient currently without {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}. "
        "On direct questioning, acknowledges {SYMPTOM_POS_1} with gradual onset over the past four days. "
        "Also admits to {SYMPTOM_POS_2} and {SYMPTOM_POS_3}, which have worsened since symptom onset."
    ),
    # T2 — Tier 2, POS → NEG → POS interleaved
    (
        "Patient complains of {SYMPTOM_POS_1} for approximately one week. "
        "{SYMPTOM_NEG_1} was not observed. "
        "Clinical history reveals {SYMPTOM_POS_2} associated with physical exertion. "
        "Assessment shows absence of {SYMPTOM_NEG_2} and {SYMPTOM_NEG_3}."
    ),
    # T3 — Tier 2, POS → NEG at end
    (
        "Onset of {SYMPTOM_POS_1} began approximately three days ago, "
        "associated with {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Symptoms remain unchanged since onset. "
        "Currently without {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}. "
        "ROS negative for {SYMPTOM_NEG_3}."
    ),
    # T4 — Tier 3, NEG → POS → NEG, demographics
    (
        "Male patient, 67 years old, with type 2 diabetes, brought by family to the emergency department. "
        "{SYMPTOM_NEG_1} and {SYMPTOM_NEG_2} are absent on current assessment. "
        "The complaint of {SYMPTOM_POS_1} was elicited upon questioning, with onset approximately 5 days ago. "
        "Patient indicates experiencing {SYMPTOM_POS_2} and {SYMPTOM_POS_3}, worsening over the last 48 hours. "
        "{SYMPTOM_POS_4} also noted alongside these symptoms. "
        "No {SYMPTOM_NEG_3} or {SYMPTOM_NEG_4} at this time."
    ),
    # T5 — Tier 3, POS → NEG → POS → NEG interleaved
    (
        "Patient presents with {SYMPTOM_POS_1} of sudden onset this morning. "
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Has been experiencing {SYMPTOM_POS_2} and {SYMPTOM_POS_3} since yesterday evening. "
        "No {SYMPTOM_NEG_2} was identified. "
        "Patient also endorses {SYMPTOM_POS_4} that worsens with movement. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),
    # T6 — Tier 3, POS → NEG → POS → NEG interleaved, demographics
    (
        "Female patient, 45 years old, previously healthy, walk-in. "
        "Acknowledges {SYMPTOM_POS_1} for approximately 6 days, with gradual onset. "
        "No {SYMPTOM_NEG_1} reported. "
        "Onset of {SYMPTOM_POS_2} began two days after the initial complaint. "
        "{SYMPTOM_NEG_2} is not present at evaluation. "
        "Also mentions {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, predominantly in the evenings. "
        "There is no indication of {SYMPTOM_NEG_3}."
    ),
    # T7 — Pediatric, POS → NEG
    (
        "Child patient, 7 years old, brought by parents to the emergency department. "
        "Parents report onset of {SYMPTOM_POS_1} approximately two days ago, associated with {SYMPTOM_POS_2}. "
        "No {SYMPTOM_NEG_1} observed by caregivers at home. "
        "ROS negative for {SYMPTOM_NEG_2} and {SYMPTOM_NEG_3}."
    ),
    # T8 — Elderly with comorbidities, NEG → POS → NEG
    (
        "Elderly female patient, 79 years old, with hypertension, brought by daughter. "
        "No {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2} at this time. "
        "Daughter reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2} over the past five days. "
        "Patient herself endorses {SYMPTOM_POS_3}, which she describes as intermittent. "
        "Assessment shows absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),
    # T9 — NEG-heavy
    (
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Patient reports {SYMPTOM_POS_1} since yesterday morning. "
        "No {SYMPTOM_NEG_2} identified on assessment. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}."
    ),
    # T10 — Adult male walk-in, NEG → POS
    (
        "Male patient, 34 years old, walk-in. "
        "Denies {SYMPTOM_NEG_1} and {SYMPTOM_NEG_2}. "
        "ROS negative for {SYMPTOM_NEG_3}. "
        "Endorses {SYMPTOM_POS_1} intermittently over the past week."
    ),
    # T11 — Balanced, POS → NEG interleaved
    (
        "Female patient, 52 years old, with hypertension and hypothyroidism. "
        "Presents with {SYMPTOM_POS_1} of 4 days duration. "
        "{SYMPTOM_NEG_1} was not observed on exam. "
        "Also mentions {SYMPTOM_POS_2} worsening at night. "
        "Patient explicitly denies {SYMPTOM_NEG_2}."
    ),
    # T12 — POS-heavy, brought by spouse
    (
        "Patient brought to clinic by husband. "
        "Complains of {SYMPTOM_POS_1} and {SYMPTOM_POS_2} starting four days ago. "
        "Husband confirms {SYMPTOM_POS_3} observed at home over the same period. "
        "No {SYMPTOM_NEG_1} reported."
    ),
    # T13 — GP referral, POS → NEG
    (
        "Female patient referred by GP for evaluation of {SYMPTOM_POS_1}. "
        "Has been experiencing {SYMPTOM_POS_2} concurrently for the past several days. "
        "{SYMPTOM_NEG_1} is not present at evaluation. "
        "There is no indication of {SYMPTOM_NEG_2}."
    ),
    # T14 — NEG-heavy, NEG → POS → NEG interleaved
    (
        "Patient presents for evaluation. "
        "Currently without {SYMPTOM_NEG_1}, {SYMPTOM_NEG_2}, or {SYMPTOM_NEG_3}. "
        "On questioning, admits to {SYMPTOM_POS_1} of recent onset. "
        "Patient confirms no {SYMPTOM_NEG_4}. "
        "{SYMPTOM_POS_2} also mentioned briefly during the visit. "
        "Findings do not support presence of {SYMPTOM_NEG_5}."
    ),
    # T15 — POS-heavy, adolescent
    (
        "Adolescent patient, 14 years old, brought by father. "
        "Father reports {SYMPTOM_POS_1} for one week, with associated {SYMPTOM_POS_2}. "
        "Patient also acknowledges {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, predominantly at night. "
        "Onset of {SYMPTOM_POS_5} began two days ago. "
        "No {SYMPTOM_NEG_1} reported."
    ),
    # T16 — Balanced, elderly male
    (
        "Male patient, 73 years old, with type 2 diabetes and chronic kidney disease, brought by son to the emergency department. "
        "Son reports {SYMPTOM_POS_1} progressing over five days. "
        "Patient endorses {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Physical exam negative for {SYMPTOM_NEG_1}. "
        "Assessment shows absence of {SYMPTOM_NEG_2}. "
        "There is no indication of {SYMPTOM_NEG_3}."
    ),
    # T17 — NEG-heavy, asthma history
    (
        "Female patient, 41 years old, with history of asthma, walk-in. "
        "{SYMPTOM_NEG_1} is absent on current assessment. "
        "{SYMPTOM_NEG_2} was not documented during evaluation. "
        "Patient indicates experiencing {SYMPTOM_POS_1} since the morning. "
        "Confirmed absence of {SYMPTOM_NEG_3} and {SYMPTOM_NEG_4}. "
        "Also reports mild {SYMPTOM_POS_2}."
    ),
    # T18 — Slight POS-heavy, interleaved
    (
        "Patient presents to the emergency department this evening. "
        "Acknowledges {SYMPTOM_POS_1} of acute onset. "
        "No {SYMPTOM_NEG_1} at this time. "
        "Additionally reports {SYMPTOM_POS_2} and {SYMPTOM_POS_3}. "
        "Review of systems: {SYMPTOM_NEG_2} denied. "
        "{SYMPTOM_POS_4} also identified on review. "
        "ROS negative for {SYMPTOM_NEG_3}."
    ),
    # T19 — POS-heavy, pediatric with mother informant
    (
        "Pediatric patient, 5 years old, brought by mother. "
        "Mother describes onset of {SYMPTOM_POS_1} three days ago, with subsequent development of {SYMPTOM_POS_2}. "
        "Child also presents with {SYMPTOM_POS_3} and {SYMPTOM_POS_4}, worse in the afternoon. "
        "No {SYMPTOM_NEG_1} observed at home. "
        "Additionally, {SYMPTOM_POS_5} noted during evaluation."
    ),
    # T20 — Balanced, elderly walk-in
    (
        "Elderly male patient, 70 years old, previously healthy, walk-in. "
        "Complaint of {SYMPTOM_POS_1} elicited at intake. "
        "Patient indicates experiencing {SYMPTOM_POS_2} over the past three days. "
        "Currently without {SYMPTOM_NEG_1}. "
        "{SYMPTOM_POS_3} mentioned in passing during questioning. "
        "No {SYMPTOM_NEG_2} or {SYMPTOM_NEG_3} reported."
    ),

    # ====================== V05 P2/P3 — NEW templates T21–T28 ================

    # T21 — V05 P2: "X without Y" mid-clause negation (directly counters hda10).
    (
        "Patient reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2} without {SYMPTOM_NEG_1}. "
        "Onset approximately three days ago, with progressive worsening. "
        "Denies {NON_SYMPTOM_HISTORY_1} and {NON_SYMPTOM_HISTORY_2}."
    ),
    # T22 — V05 P2: "mentioned an absence of" mid-clause (counters hda8).
    (
        "Patient describes {SYMPTOM_POS_1} of acute onset. "
        "When asked, the patient mentioned an absence of {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}. "
        "No {NON_SYMPTOM_HISTORY_1} reported."
    ),
    # T23 — V05 P2: "Neither A, B, nor C" + "does not present any signs of" (counters hda9).
    (
        "Patient reports {SYMPTOM_POS_1} for the past five days, associated with {SYMPTOM_POS_2}. "
        "Patient does not present any signs of {SYMPTOM_NEG_1}, {SYMPTOM_NEG_2}, or {SYMPTOM_NEG_3}. "
        "Neither {SYMPTOM_NEG_4}, {SYMPTOM_NEG_5}, nor {SYMPTOM_NEG_6} are present. "
        "Patient denies {NON_SYMPTOM_HISTORY_1}."
    ),
    # T24 — V05 P2: "X but no Y" alternative negation cue.
    (
        "Patient endorses {SYMPTOM_POS_1} but no {SYMPTOM_NEG_1}. "
        "Also reports {SYMPTOM_POS_2} progressing over the past week. "
        "No {NON_SYMPTOM_HISTORY_1} or {NON_SYMPTOM_HISTORY_2}."
    ),
    # T25 — V05 P3: modifier-heavy POS (counters boundary overreach on
    #              "sudden onset of", "intense", "diffuse", "increased").
    # Modifiers are NOT inside placeholders → generator labels them O.
    (
        "Patient describes sudden onset of {SYMPTOM_POS_1} approximately two days ago, "
        "associated with intense {SYMPTOM_POS_2} and diffuse {SYMPTOM_POS_3}. "
        "Patient also reports increased {SYMPTOM_POS_4} during evening hours. "
        "No {SYMPTOM_NEG_1} reported. "
        "Denies {NON_SYMPTOM_HISTORY_1}."
    ),
    # T26 — V05 P3: postmodifier "of moderate intensity" + "progressive".
    (
        "Patient reports {SYMPTOM_POS_1} of moderate intensity for the past four days. "
        "Progressive {SYMPTOM_POS_2} noted over the same period. "
        "{SYMPTOM_NEG_1} was not observed on exam. "
        "Confirmed absence of {SYMPTOM_NEG_2}. "
        "No history of {NON_SYMPTOM_HISTORY_1}."
    ),
    # T27 — V05 P1+P2 combined: heavy adversarial-O denial chain
    #       (directly counters hda6/hda10 "denies known drug allergies" etc.).
    (
        "Patient presents with {SYMPTOM_POS_1} and {SYMPTOM_POS_2} for the past three days. "
        "Denies {SYMPTOM_NEG_1}, {SYMPTOM_NEG_2}, or {SYMPTOM_NEG_3}. "
        "Patient denies previous {NON_SYMPTOM_HISTORY_1} and continuous {NON_SYMPTOM_HISTORY_2}. "
        "Patient denies known {NON_SYMPTOM_HISTORY_3}. "
        "No recent {NON_SYMPTOM_HISTORY_4}."
    ),
    # T28 — V05 P2+P3: mid-clause negation + worsening modifier.
    (
        "Patient describes worsening {SYMPTOM_POS_1} since onset, "
        "with associated {SYMPTOM_POS_2} without {SYMPTOM_NEG_1}. "
        "Mentioned an absence of {SYMPTOM_NEG_2} when asked directly. "
        "Patient denies {NON_SYMPTOM_HISTORY_1} and {NON_SYMPTOM_HISTORY_2}."
    ),
]


# =============================================================================
# (group_name, templates) pairs for generators.
# =============================================================================
TEMPLATE_GROUPS = [
    ("affirmed",    AFFIRMED_TEMPLATES),
    ("negated",     NEGATED_TEMPLATES),
    ("distractor",  DISTRACTOR_TEMPLATES),
    ("hda",         HDA_TEMPLATES),
]
