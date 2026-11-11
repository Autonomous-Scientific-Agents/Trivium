from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FactType(str, Enum):
    attribute = "attribute"
    definition = "definition"
    quantity = "quantity"
    event = "event"
    relationship = "relationship"
    comparison = "comparison"
    classification = "classification"
    temporal = "temporal"
    location = "location"
    causal = "causal"
    procedure = "procedure"
    result = "result"
    derived = "derived"  # computed from other facts (design §10.4)


def fmt_number(value: float) -> str:
    """Human-friendly number: integers without exponent, otherwise up to 6 significant digits."""
    if abs(value) < 1e15 and float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


# pseudo-units that read badly after a number ("2021 year"); kept on the fact, hidden in display
_NO_DISPLAY_UNITS = {"year", "years", "yr", "date", "count", "number", "none", "n/a"}


class FactObject(BaseModel):
    text: str
    value: float | None = None
    unit: str | None = None

    @property
    def is_numeric(self) -> bool:
        return self.value is not None

    def display(self) -> str:
        if self.value is not None:
            v = fmt_number(self.value)
            unit = self.unit if self.unit and self.unit.lower() not in _NO_DISPLAY_UNITS else None
            return f"{v} {unit}" if unit else v
        return self.text


class FactQuality(BaseModel):
    entailment: float = 0.0
    atomicity: float = 0.0
    explicitness: float = 0.0
    ambiguity: float = 1.0  # lower is better
    overall: float = 0.0
    notes: str = ""


class Fact(BaseModel):
    id: str
    subject: str
    predicate: str
    object: FactObject
    qualifiers: dict[str, str] = Field(default_factory=dict)
    fact_type: FactType = FactType.attribute
    explicit: bool = True
    document_id: str
    section_id: str = ""
    chunk_id: str
    evidence: str
    confidence: float = 1.0
    quality: FactQuality | None = None
    trivia_score: float | None = None  # how good a trivia nugget this fact is (trivia mode)
    derived_from: list[str] = Field(default_factory=list)
    derivation: str | None = None  # e.g. "difference(fact_1, fact_2)"

    @field_validator("subject", "predicate")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def statement(self) -> str:
        """Render the fact as a natural-language sentence."""
        pred = self.predicate.replace("_", " ")
        s = f"{self.subject} — {pred}: {self.object.display()}"
        if self.qualifiers:
            q = ", ".join(f"{k}={v}" for k, v in self.qualifiers.items())
            s += f" ({q})"
        return s

    @property
    def quality_score(self) -> float:
        return self.quality.overall if self.quality else self.confidence


_PRONOUNS = re.compile(
    r"^(it|they|this|that|these|those|he|she|we|the (?:model|method|authors?|paper|study))$", re.I
)


# configuration values: an environment variable, a command-line flag, or "X is set to 12".
# Comparing or doing arithmetic on them ("NRANKS_PER_NODE divided by NTHREADS") makes filler.
_SETTING_SUBJECT = re.compile(r"^\$?\{?[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\}?$|^--?[A-Za-z]")
_SETTING_PREDICATE = re.compile(
    r"^(?:is_)?(?:set|sets|set_to|set_as|defaults?|defaults?_to|default_value|value|has_value|"
    r"equals?|assigned(?:_to)?|configured(?:_as|_to)?|exported(?:_as|_to)?)(?:_to)?$"
)


def is_setting(fact: Fact) -> bool:
    """A configuration value rather than a quantity worth comparing or computing with."""
    return bool(
        _SETTING_SUBJECT.match(fact.subject.strip())
        or _SETTING_PREDICATE.match(fact.predicate.strip().lower())
    )


def looks_unresolved(subject: str) -> bool:
    return bool(_PRONOUNS.match(subject.strip()))
