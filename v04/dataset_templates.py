# Template inventory (import module and sum *_TEMPLATES lists; 211 total):
#   AFFIRMED_TEMPLATES: 40
#   NEGATED_TEMPLATES: 40          (-1: removed "zero instances" unnatural phrasing)
#   DISTRACTOR_TEMPLATES: 38
#   MULTI_SYMPTOM_TEMPLATES: 19
#   NARRATIVE_TEMPLATES: 30
#   MIXED_POLARITY_CLAUSE_TEMPLATES: 10
#   WORD_COLLISION_TEMPLATES: 12
#   TRIAGE_HDA_TEMPLATES: 22       (new — Brazilian ED/HDA telegraphic style)
#
# Placeholder indexing convention (v04):
#   Bare {SYMPTOM_POS} / {SYMPTOM_NEG} repeated in one template → SAME symptom fills all.
#   {SYMPTOM_POS_1}, {SYMPTOM_POS_2} … → builder draws DIFFERENT symptoms per index.

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
    # History-framed negation
    "No prior history of {SYMPTOM_NEG}.",
    "Patient has never experienced {SYMPTOM_NEG}.",
    "The family mentioned that the patient has never had {SYMPTOM_NEG}.",
    "No previous episodes of {SYMPTOM_NEG}.",
    "Denies any past occurrence of {SYMPTOM_NEG}.",
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
"{SYMPTOM} was reported last year but is no longer present.",
"Patient had experienced {SYMPTOM_O} following surgery, now resolved.",
"Prior episodes of {SYMPTOM_O} have completely subsided.",

#CATEGORY 6: Research / literature / citation context

"Recent studies indicate that {SYMPTOM_O} is a frequent adverse effect.",
"The literature describes {SYMPTOM_O} as a key feature of this syndrome.",
"Clinical trials have identified {SYMPTOM_O} in a subset of participants.",
"According to published data, {SYMPTOM_O} correlates with disease severity.",
"Evidence suggests that {SYMPTOM_O} may predict poorer outcomes.",

# CATEGORY 7: Administrative / procedural / non-clinical context

"Checklist item: {SYMPTOM_O} — mark if applicable.",
"Screening form includes assessment of {SYMPTOM_O}.",
"Please indicate presence of {SYMPTOM_O} in the section below.",
"Symptom review section lists {SYMPTOM_O} among standard entries.",
"Documentation field for {SYMPTOM_O} remains incomplete.",

# CATEGORY 8: Negated or uncertain mentions (tricky non-affirmed contexts)

"Possible {SYMPTOM_O} to be ruled out pending further evaluation.",
"Query regarding {SYMPTOM_O} remains unanswered.",
"Evaluation ongoing to determine presence of {SYMPTOM_O}.",
"Unclear if {SYMPSYMPTOM_O} is contributing to the clinical picture."
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

# 2-4 symptom placeholders per sentence, mixed polarity.
# Mirrors deployment style: multiple symptoms surfacing in one clinical sentence.
# TODO: review if cases are balanced meaning if we have sentences with 3 POS and 1 NEG, we need also ones with 3 NEG and 1 POS
MULTI_SYMPTOM_TEMPLATES = [
    # Two-symptom, same polarity (_1/_2 = different draws required)
    "Patient reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}.",
    "Complains of {SYMPTOM_POS_1} along with {SYMPTOM_POS_2} since yesterday.",
    "Denies {SYMPTOM_NEG_1} and {SYMPTOM_NEG_2}.",
    "No {SYMPTOM_NEG_1}, no {SYMPTOM_NEG_2} on review.",
    # Two-symptom, mixed polarity
    "Patient endorses {SYMPTOM_POS} but denies {SYMPTOM_NEG}.",
    "Reports {SYMPTOM_POS}; {SYMPTOM_NEG} is denied.",
    "Acknowledges {SYMPTOM_POS} while denying any {SYMPTOM_NEG}.",
    "Positive for {SYMPTOM_POS}, negative for {SYMPTOM_NEG}.",
    "{SYMPTOM_POS} is present; {SYMPTOM_NEG} is absent.",
    # Three-symptom, mixed
    "Complains of {SYMPTOM_POS_1} and {SYMPTOM_POS_2}, but denies {SYMPTOM_NEG}.",
    "Reports {SYMPTOM_POS}; denies {SYMPTOM_NEG_1} and {SYMPTOM_NEG_2}.",
    "On ROS, positive for {SYMPTOM_POS_1} and {SYMPTOM_POS_2}, negative for {SYMPTOM_NEG}.",
    "Endorses {SYMPTOM_POS_1} along with {SYMPTOM_POS_2}; denies {SYMPTOM_NEG}.",
    # Four-symptom, mixed
    "Patient describes {SYMPTOM_POS_1}, {SYMPTOM_POS_2}, and {SYMPTOM_POS_3}; denies {SYMPTOM_NEG}.",
    "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}; denies {SYMPTOM_NEG_1} and {SYMPTOM_NEG_2}.",
    # With a distractor mixed in
    "Patient reports {SYMPTOM_POS}; family history is notable for {SYMPTOM_O}.",
    "Endorses {SYMPTOM_POS}; educational handout discussed warning signs including {SYMPTOM_O}.",
    "Denies {SYMPTOM_NEG}; screening questionnaire lists {SYMPTOM_O} among standard items.",
    "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2}; {SYMPTOM_O} is described in the educational pamphlet.",
]

# Long, narrative-style clinical sentences with subordinate clauses.
# Modeled on the "Long, realistic clinical sentences" bucket of
# behavioural_template_bank.json and the easy/medium medical_transcriptions.
NARRATIVE_TEMPLATES = [

    # examples starting with a POS symptom followed by a NEG symptom
    "The patient reports {SYMPTOM_POS} that started approximately three days ago after physical exertion, associated with mild fatigue.",
    "Over the past week, the patient has experienced intermittent {SYMPTOM_POS}, which tends to worsen in the evenings and partially improves with rest.",
    "Patient describes persistent {SYMPTOM_POS} without clear triggers, and denies {SYMPTOM_NEG} or recent infections.",
    "Since the last visit, the patient notes worsening {SYMPTOM_POS}, especially when walking long distances, but denies {SYMPTOM_NEG}.",
    "Patient presents today due to ongoing {SYMPTOM_POS}, which has been affecting daily activities despite over-the-counter medications.",
    "She reports gradual onset of {SYMPTOM_POS} with no identifiable triggering event, and denies {SYMPTOM_NEG} or recent travel.",
    "He has been experiencing {SYMPTOM_POS} for the last five days, with no relief from rest; denies {SYMPTOM_NEG_1} or {SYMPTOM_NEG_2}.",
    "On direct questioning he endorses {SYMPTOM_POS}, which he describes as intermittent and worse in the evenings.",
    "The patient, referred by her GP with a 5-day history of {SYMPTOM_POS}, denies {SYMPTOM_NEG} and reports no history of similar episodes.",
    "Over the last 48 hours the patient developed {SYMPTOM_POS}, progressive in nature, while explicitly denying {SYMPTOM_NEG}.",
    "Wife reports that the patient has had {SYMPTOM_POS} for the past week; the patient himself denies {SYMPTOM_NEG} when asked directly.",
    "The patient presents with a chief complaint of {SYMPTOM_POS_1}; on systems review he also endorses {SYMPTOM_POS_2} but denies {SYMPTOM_NEG}.",
    "On admission, the patient was noted to have {SYMPTOM_POS}; subsequent evaluation ruled out {SYMPTOM_NEG}.",
    "Despite recent treatment, the patient continues to report {SYMPTOM_POS} and denies improvement in {SYMPTOM_POS}.",
    "Family brought the patient in due to {SYMPTOM_POS} observed at home over three days, although the patient denies {SYMPTOM_NEG}.",
    # examples starting with SYMPTOM_NEG first, then additional symptom(s)
    "At triage he disputes any link between today's minor injury and {SYMPTOM_NEG}; once in the exam room he shifts attention to steadily worsening {SYMPTOM_POS} that began after midnight and now limits weight-bearing on the affected side.",
    "She opens by correcting chart text that implied active {SYMPTOM_NEG}, insisting the wording never matched her report, then spends most of the visit dissecting throbbing {SYMPTOM_POS} that peaks after prolonged standing and eases modestly once she is off her feet.",
    "Overnight telephone nursing documented absence of {SYMPTOM_NEG}; this afternoon she confirms {SYMPTOM_NEG} remains absent yet adds {SYMPTOM_POS} that follows exertion, worsens with hill climbing, and is accompanied by symptoms she describes as 'air hunger' after heavy meals.",
    "On 0400 vitals reassessment he denied {SYMPTOM_NEG} while febrile; by daylight he reaffirms no {SYMPTOM_NEG} but volunteers band-like {SYMPTOM_POS} across the upper back that emerged after sleeping upright in a recliner.",
    "Preoperative holding paperwork records a negative screen for {SYMPTOM_NEG}; minutes before induction he repeats that {SYMPTOM_NEG} has not been an issue and attributes his discomfort to new-onset {SYMPTOM_POS} that anesthesia attributed to prolonged shoulder abduction on the gurney.",
    "Through an interpreter she rejects the referral summary's emphasis on {SYMPTOM_NEG}, stating the letter overstated a single offhand comment, then requests guidance about relentless {SYMPTOM_POS} that fluctuates with dose changes to her home antihypertensive regimen.",
    "Case management yesterday annotated {SYMPTOM_NEG} as absent on the facility checklist; at bedside today he corroborates lack of {SYMPTOM_NEG}, then discloses mounting {SYMPTOM_POS} that his daughter noticed before overriding his preference to stay home.",
    "Behavioral health intake probes {SYMPTOM_NEG} because prior forms flagged it; the patient minimizes {SYMPTOM_NEG} as checkbox noise and redirects toward intrusive {SYMPTOM_POS} that derails focus during back-to-back video conferences.",
    "In the return-to-play lane he downplays any recurrence of {SYMPTOM_NEG} since resuming drills, then reproduces focal {SYMPTOM_POS} with resisted loading—a pattern that was absent before last weekend's out-of-state tournament travel.",
    "Pharmacy reconciliation records an explicit denial of {SYMPTOM_NEG}; back in the exam room he nuances the story, acknowledging exertional {SYMPTOM_POS} only beyond two flights of stairs while still denying {SYMPTOM_NEG} at rest or during sleep.",
    "EMS paperwork listed concern for {SYMPTOM_NEG} en route; in the ED bay he clarifies he has not experienced {SYMPTOM_NEG} and identifies {SYMPTOM_POS} as the driver of today's visit—cramping discomfort that surfaced after aggressive field fluids and failed to resolve with oral analgesia.",
    "Resident sign-out emphasized overnight resolution of {SYMPTOM_NEG_1} after a single conservative intervention; at follow-up he concurs that {SYMPTOM_NEG_2} has not recurred, then introduces breakthrough {SYMPTOM_POS} whenever he reclines within an hour of a heavy meal, now prompting sleep-position experiments at home.",
    "The consultant's terse prior note alluded to neglected {SYMPTOM_NEG_1}; in person today she dismantles that characterization regarding {SYMPTOM_NEG_2} and instead produces a dated symptom log centered on paroxysmal {SYMPTOM_POS} triggered by cold exposure, brisk walking, and arguments with her adult children.",
    "He begins the video visit by striking {SYMPTOM_NEG} from the problem list he prepared with his spouse, citing wearable trendlines, then narrates exertional {SYMPTOM_POS} that reliably eases when he sits yet returns with painting ceilings or unloading groceries from overhead shelves.",
    "Occupational health transmitted a screening remark about {SYMPTOM_NEG_1}; during the clinical interview he challenges whether {SYMPTOM_NEG_2} belongs on a worker's compensation template at all, pivoting to disruptive {SYMPTOM_POS} that jerks him awake and only settles after pacing his apartment for ten to fifteen minutes.",
]


# Mixed polarity in a single clause — the model must not let one cue dominate.
MIXED_POLARITY_CLAUSE_TEMPLATES = [
    "Patient denies {SYMPTOM_NEG} but reports {SYMPTOM_POS}.",
    "Reports {SYMPTOM_POS} though denies {SYMPTOM_NEG}.",
    "{SYMPTOM_POS} is present; {SYMPTOM_NEG} is denied.",
    "Acknowledges {SYMPTOM_POS} while denying {SYMPTOM_NEG}.",
    "Positive for {SYMPTOM_POS}, negative for {SYMPTOM_NEG}.",
    "Endorses {SYMPTOM_POS} on questioning, no {SYMPTOM_NEG}.",
    "{SYMPTOM_POS} confirmed; {SYMPTOM_NEG} ruled out.",
    "Currently has {SYMPTOM_POS}; has never had {SYMPTOM_NEG}.",
    "On exam: {SYMPTOM_POS} observed; no {SYMPTOM_NEG}.",
    "Patient confirms {SYMPTOM_POS} but denies any {SYMPTOM_NEG}.",
]

# Hard negatives: same symptom word appears as distractor AND as patient mention
# in the same example. Trains the model off lexical-keyword shortcuts.
# Builder should fill {SYMPTOM_POS}/{SYMPTOM_NEG} and {SYMPTOM_O} with the SAME
# symptom string when generating this template.
WORD_COLLISION_TEMPLATES = [
    "{SYMPTOM_O} is common in viral infections; the patient denies {SYMPTOM_NEG}.",
    "{SYMPTOM_O} is a known side effect of this medication; the patient currently reports {SYMPTOM_POS}.",
    "The literature describes {SYMPTOM_O} as a hallmark of this syndrome; on exam the patient has {SYMPTOM_POS}.",
    "Family history is notable for {SYMPTOM_O}; the patient herself denies {SYMPTOM_NEG}.",
    "Mother had chronic {SYMPTOM_O} during adolescence; the patient currently endorses {SYMPTOM_POS}.",
    "Screening form asks about {SYMPTOM_O}; the patient reports {SYMPTOM_POS}.",
    "Educational material covered warning signs such as {SYMPTOM_O}; the patient now denies {SYMPTOM_NEG}.",
    "{SYMPTOM_O} was reported last year but has resolved; the patient currently has {SYMPTOM_POS}.",
    "Discussed possible future symptoms including {SYMPTOM_O}; today the patient denies {SYMPTOM_NEG}.",
    "Postoperative patients frequently develop {SYMPTOM_O}; this patient denies {SYMPTOM_NEG}.",
    "Checklist item: {SYMPTOM_O} — marked not applicable. The patient endorses {SYMPTOM_POS}.",
    "{SYMPTOM_O} can occur as an adverse effect; the patient currently reports {SYMPTOM_POS}.",
]

# Short, telegraphic HDA templates modeled on Brazilian emergency department
# documentation style (HDA — história da doença atual).
#
# Structural signature of Brazilian ED/UPA HDA:
#   1. Age + sex + relevant comorbidities up front (almost always present)
#   2. Referral origin: walk-in, UBS, UPA transfer, brought by family
#   3. Chief complaint (queixa principal) with duration
#   4. Brief negation sweep at the end — often just "Denies X." or "No X."
#
# These short forms are completely absent from earlier versions and represent
# the most common ED note register. They are the hardest for models trained
# only on longer templates because polarity cues are minimal and close together.
TRIAGE_HDA_TEMPLATES = [

    # CATEGORY A: Single-line triage (queixa principal + nega)
    "CC: {SYMPTOM_POS}. Denies {SYMPTOM_NEG}.",
    "{SYMPTOM_POS} x 3 days. No {SYMPTOM_NEG}.",
    "Reports {SYMPTOM_POS}. Denies {SYMPTOM_NEG}.",
    "Presents with {SYMPTOM_POS}. No {SYMPTOM_NEG} reported.",
    "{SYMPTOM_POS}, onset 2 days ago. {SYMPTOM_NEG} denied.",

    # CATEGORY B: Demographics + symptom + negation
    # Brazilian HDA opens with age, sex, and relevant comorbidities
    "Female, 38yo, no comorbidities. Reports {SYMPTOM_POS} for 5 days. Denies {SYMPTOM_NEG}.",
    "Male, 54yo, HTN. Presents with {SYMPTOM_POS}. No {SYMPTOM_NEG}.",
    "Patient, 67yo, DM2. {SYMPTOM_POS} since yesterday. Denies {SYMPTOM_NEG}.",
    "Female, 29yo, previously healthy. {SYMPTOM_POS} x 1 week. No {SYMPTOM_NEG}.",
    "Male, 72yo, HTN, DM2, prior AMI. Referred with {SYMPTOM_POS}. Denies {SYMPTOM_NEG}.",

    # CATEGORY C: Referral origin (encaminhamento — routine in Brazilian SUS)
    "Referred by GP with {SYMPTOM_POS}. No {SYMPTOM_NEG} on intake.",
    "Brought by family. Reports {SYMPTOM_POS}. Denies {SYMPTOM_NEG}.",
    "Transferred from UPA with {SYMPTOM_POS}. {SYMPTOM_NEG} absent.",
    "Walk-in. CC: {SYMPTOM_POS}. Denies {SYMPTOM_NEG}.",
    "Self-referred. {SYMPTOM_POS} for 2 days. No {SYMPTOM_NEG}.",
    "Referred from UBS. {SYMPTOM_POS}, worsening. Denies {SYMPTOM_NEG}.",

    # CATEGORY D: Onset + evolution + negation sweep
    "{SYMPTOM_POS}, gradual onset, 4 days. No {SYMPTOM_NEG}.",
    "{SYMPTOM_POS}, sudden onset this morning. Denies {SYMPTOM_NEG}.",
    "{SYMPTOM_POS}, worsening over 48h. No {SYMPTOM_NEG}.",
    "{SYMPTOM_POS} since last night, no improvement. {SYMPTOM_NEG} denied.",

    # CATEGORY E: Multi-symptom telegraphic (brief ROS sweep at triage)
    "CC: {SYMPTOM_POS_1} and {SYMPTOM_POS_2}. Denies {SYMPTOM_NEG}.",
    "Reports {SYMPTOM_POS_1} and {SYMPTOM_POS_2} x 3 days. No {SYMPTOM_NEG}.",
]

# (group_name, templates) pairs for generators that branch on list membership,
# not on parsing placeholder strings inside each template.
TEMPLATE_GROUPS = [
    ("affirmed", AFFIRMED_TEMPLATES),
    ("negated", NEGATED_TEMPLATES),
    ("distractor", DISTRACTOR_TEMPLATES),
    ("multi_symptom", MULTI_SYMPTOM_TEMPLATES),
    ("narrative", NARRATIVE_TEMPLATES),
    ("mixed_polarity_clause", MIXED_POLARITY_CLAUSE_TEMPLATES),
    ("word_collision", WORD_COLLISION_TEMPLATES),
    ("triage_hda", TRIAGE_HDA_TEMPLATES),
]
