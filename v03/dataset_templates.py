AFFIRMED_TEMPLATES = [
    # Passive / impersonal constructions
    "It was noted that {SYMPTOM} is present.",
    "{SYMPTOM} was documented during evaluation.",
    "{SYMPTOM} was observed on assessment.",
    "It has been recorded that the patient has {SYMPTOM}.",
    # Temporal / onset framing
    "Onset of {SYMPTOM} began recently.",
    "Has been experiencing {SYMPTOM} for several days.",
    "{SYMPTOM} started earlier this week.",
    "Reports ongoing {SYMPTOM} since yesterday.",
    # Family / third-person report
    "Family reports that the patient has {SYMPTOM}.",
    "Caregiver notes {SYMPTOM}.",
    "According to relatives, the patient has {SYMPTOM}.",
    "Third-party observer reports {SYMPTOM}.",
    # Degree / severity qualifiers
    "Severe {SYMPTOM} is reported.",
    "Mild {SYMPTOM} noted during visit.",
    "Patient endorses significant {SYMPTOM}.",
    "Moderate {SYMPTOM} described by the patient.",
    # Multi-symptom context
    "Chief complaint includes {SYMPTOM} among others.",
    "Presentation notable for {SYMPTOM} and additional findings.",
    "Clinical picture includes {SYMPTOM}.",
    "{SYMPTOM} noted alongside other symptoms.",
    # Question/intake response style
    "Admits to {SYMPTOM}.",
    "Endorses having {SYMPTOM}.",
    "Acknowledges {SYMPTOM} when asked.",
    "Confirms presence of {SYMPTOM}.",
    # Incidental / secondary findings
    "Incidentally noted {SYMPTOM}.",
    "Also reports {SYMPTOM}.",
    "Additionally, {SYMPTOM} was mentioned.",
    "{SYMPTOM} identified as a secondary concern.",
    # Verb-last / inverted constructions
    "Present on exam: {SYMPTOM}.",
    "Observed during evaluation: {SYMPTOM}.",
    "Documented in assessment: {SYMPTOM}.",
    "Identified on review: {SYMPTOM}.",
    # Other varied structures
    "The patient indicates experiencing {SYMPTOM}.",
    "Clinical history reveals {SYMPTOM}.",
    "Evaluation uncovered {SYMPTOM}.",
    "The individual is noted to have {SYMPTOM}.",
    "{SYMPTOM} persists at the time of visit.",
    "There is ongoing {SYMPTOM}.",
    "The complaint of {SYMPTOM} was elicited.",
    "{SYMPTOM} remains a current issue.",
]

NEGATED_TEMPLATES = [
    # Conditional negation
    "No {SYMPTOM} at this time.",
    "Currently without {SYMPTOM}.",
    "At present, there is no {SYMPTOM}.",
    "No active {SYMPTOM} reported.",
    # History-framed negation
    "No prior history of {SYMPTOM}.",
    "Patient has never experienced {SYMPTOM}.",
    "The family mentioned that the patient has never had {SYMPTOM}.",
    "No previous episodes of {SYMPTOM}.",
    "Denies any past occurrence of {SYMPTOM}.",
    # Exam-finding negation
    "Exam reveals no {SYMPTOM}.",
    "Physical exam negative for {SYMPTOM}.",
    "Assessment shows absence of {SYMPTOM}.",
    "No {SYMPTOM} detected on examination.",
    # Passive negation
    "No {SYMPTOM} was identified.",
    "{SYMPTOM} was not observed.",
    "{SYMPTOM} was not documented.",
    "It was determined that {SYMPTOM} is absent.",
    # Qualifier-based negation
    "No significant {SYMPTOM} reported.",
    "{SYMPTOM} not clinically significant.",
    "No concerning {SYMPTOM} identified.",
    "{SYMPTOM} deemed negligible.",
    # System-review negation
    "ROS negative for {SYMPTOM}.",
    "Review of systems: {SYMPTOM} denied.",
    "System review does not indicate {SYMPTOM}.",
    "No {SYMPTOM} reported in review of systems.",
    # Double-check / confirmation negation
    "Confirmed absence of {SYMPTOM}.",
    "Patient confirms no {SYMPTOM}.",
    "Absence of {SYMPTOM} verified.",
    "Negative for {SYMPTOM} upon confirmation.",
    # Other varied negation structures
    "The patient explicitly denies any {SYMPTOM}.",
    "There is no indication of {SYMPTOM}.",
    "{SYMPTOM} is not present at evaluation.",
    "No complaints related to {SYMPTOM}.",
    "The patient reports zero instances of {SYMPTOM}.",
    "Findings do not support presence of {SYMPTOM}.",
    "No observable {SYMPTOM} noted.",
    "{SYMPTOM} is absent on current assessment.",
    "No mention of {SYMPTOM} by the patient.",
    "{SYMPTOM} has not been experienced.",
    "There are no reports suggesting {SYMPTOM}.",
    "{SYMPTOM} excluded based on evaluation.",
]

# ALL TOKENS SHOULD BE LABELED AT "O"
DISTRACTOR_TEMPLATES = [

#CATEGORY 1: Generic / educational statements

"{SYMPTOM} is a commonly reported manifestation in viral infections.",
"Symptoms such as {SYMPTOM} may indicate underlying systemic illness.",
"In clinical practice, {SYMPTOM} is often associated with inflammatory processes.",
"{SYMPTOM} can occur as a side effect of various medications.",
"Typical presentations include fever, cough, and {SYMPTOM}.",

#CATEGORY 2: Epidemiological / population-level statements

"A significant proportion of patients undergoing chemotherapy report {SYMPTOM}.",
"Approximately 20% of individuals with this condition experience {SYMPTOM}.",
"Postoperative patients frequently develop {SYMPTOM} within 24 hours.",
"In large cohorts, {SYMPTOM} has been observed as a common complaint.",
"Among elderly populations, {SYMPTOM} is often underreported.",

#CATEGORY 3: Hypothetical / conditional / instructional

"If {SYMPTOM} develops, the patient should seek immediate care.",
"In case of {SYMPTOM}, discontinue the medication and reassess.",
"If there is any onset of {SYMPTOM}, initiate protocol A.",
"Advise monitoring for {SYMPTOM} following discharge.",

#CATEGORY 4: Family history / third-party

"The patient’s father experienced {SYMPTOM} prior to his cardiac event.",
"Mother reports a history of chronic {SYMPTOM}.",
"A sibling was noted to have recurrent {SYMPTOM} during adolescence.",
"Family history significant for {SYMPTOM} in first-degree relatives.",
"The patient’s child had episodes of {SYMPTOM} last year.",

# CATEGORY 5: Past-resolved / historical patient symptoms

"The patient previously had {SYMPTOM}, which has since resolved.",
"History of intermittent {SYMPTOM} noted during childhood.",
"{SYMPTOM} was reported last year but is no longer present.",
"Patient had experienced {SYMPTOM} following surgery, now resolved.",
"Prior episodes of {SYMPTOM} have completely subsided.",

#CATEGORY 6: Research / literature / citation context

"Recent studies indicate that {SYMPTOM} is a frequent adverse effect.",
"The literature describes {SYMPTOM} as a key feature of this syndrome.",
"Clinical trials have identified {SYMPTOM} in a subset of participants.",
"According to published data, {SYMPTOM} correlates with disease severity.",
"Evidence suggests that {SYMPTOM} may predict poorer outcomes.",

# CATEGORY 7: Administrative / procedural / non-clinical context

"Checklist item: {SYMPTOM} — mark if applicable.",
"Screening form includes assessment of {SYMPTOM}.",
"Please indicate presence of {SYMPTOM} in the section below.",
"Symptom review section lists {SYMPTOM} among standard entries.",
"Documentation field for {SYMPTOM} remains incomplete.",

# CATEGORY 8: Negated or uncertain mentions (tricky non-affirmed contexts)

"Possible {SYMPTOM} to be ruled out pending further evaluation.",
"Query regarding {SYMPTOM} remains unanswered.",
"Evaluation ongoing to determine presence of {SYMPTOM}.",
"Unclear if {SYMPTOM} is contributing to the clinical picture."
]