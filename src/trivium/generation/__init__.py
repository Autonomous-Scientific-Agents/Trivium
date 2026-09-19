from trivium.generation.atomic import AtomicGenerator
from trivium.generation.base import Candidate, Feedback, GenerationContext, Generator
from trivium.generation.comparison import ComparisonGenerator
from trivium.generation.cross_document import CrossDocumentGenerator
from trivium.generation.derived import DerivedGenerator
from trivium.generation.multihop import MultiHopGenerator
from trivium.generation.multiple_choice import MultipleChoiceConverter
from trivium.generation.relationship import RelationshipGenerator
from trivium.generation.true_false import TrueFalseGenerator

GENERATORS: dict[str, type[Generator]] = {
    g.name: g
    for g in (
        AtomicGenerator,
        RelationshipGenerator,
        ComparisonGenerator,
        DerivedGenerator,
        MultiHopGenerator,
        CrossDocumentGenerator,
        TrueFalseGenerator,
    )
}

__all__ = [
    "Candidate",
    "Feedback",
    "GenerationContext",
    "Generator",
    "GENERATORS",
    "MultipleChoiceConverter",
]
