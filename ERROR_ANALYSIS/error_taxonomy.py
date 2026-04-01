"""
Dataclasses for the error analysis pipeline
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


@dataclass
class Spans:
    start: int
    end: int
    text: str
    label: str

@dataclass 
class ErrorTaxonomy: 
    type_0 : str = "Irrelevant Span Mislabeling"
    type_1: str = "Missing Expected Entity"
    type_2: str = "Tokenization Artifacts"
    type_3: str = "Incorrect Polarity Assignment"
    type_4: str = "BIO Sequencing Errors"
    type_5: str = "Boundary Overreach"
    type_6: str = "Boundary Undereach"
    type_7: str = "Model Overgeneralization"

ERROR_TAXONOMY = {
    "0":ErrorTaxonomy.type_0,
    "1":ErrorTaxonomy.type_1,
    "2":ErrorTaxonomy.type_2,
    "3":ErrorTaxonomy.type_3,
    "4":ErrorTaxonomy.type_4,
    "5":ErrorTaxonomy.type_5,
    "6":ErrorTaxonomy.type_6,
    "7":ErrorTaxonomy.type_7
}

@dataclass
class BehaviouralExample:
    example: str
    entities_with_labels: List[Spans]


# Not being used yet
class Reviewer(Enum):
    HUMAN = "HUMAN"
    AI = "AI"
    CODE = "CODE"
    
@dataclass
class LogEntry:
    date: str
    reviewer: Reviewer  # Only Reviewer.HUMAN or Reviewer.AI
    input_text: str
    tokens: List[str]
    token_level_labels: List[str]
    word_level_labels: List[str]  
    predicted_spans: List[Spans]
    expected_entities: List[Spans]
    error_type: Optional[Union[List[ErrorTaxonomy], ErrorTaxonomy]]
    reasoning: str 
    model: str 
    model_version: str 
    inference_pipeline_version: str

