"""trivium — adaptive, grounded question generation for trivia and RAG evaluation."""

__version__ = "0.1.0"

from trivium.config import Config  # noqa: E402
from trivium.orchestration.controller import generate_dataset  # noqa: E402

__all__ = ["Config", "generate_dataset", "__version__"]
