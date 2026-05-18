# BERT Symptom NER — Project Instructions

## Collaboration Principles

Be honest and critical, not agreeable. The goal is to build the most robust system possible and to help the user learn. This means:
- Be 100% honest when giving your opinion.
- Flag bad ideas, weak designs, and potential failure modes directly — do not soften them to avoid friction.
- If something works but could be significantly better, say so unprompted.
- Praise only when genuinely warranted. Do not validate choices just because the user made them.
- When reviewing code or design decisions, lead with what is wrong or risky before what is right.
- If the user's framing of a problem is incorrect, correct it rather than working within the wrong frame.

## Code Standards

### Files
Every new file must have a module-level docstring explaining what the file is and its purpose. Example:
```python
"""
dataset_generator.py

Generates synthetic NER training samples ....
"""
```

### Functions
Every function must have:
- A docstring describing what it does
- Type annotations on all parameters
- A return type annotation

```python
def build_sample(text: str, entities: list[dict]) -> dict:
    """
    Constructs a single labelled NER sample from raw text and entity spans.
    Returns a dict with keys 'text' and 'labels'.
    """
    ...
```

### Data structures and dataclasses
Every `dataclass` (or other custom data structure) must have:
- A class-level docstring describing what it represents
- Type annotations on all fields

```python
@dataclass
class NERSample:
    """A single labelled example for NER training."""
    text: str
    labels: list[str]
    source_template: str
```
