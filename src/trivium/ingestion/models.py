from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Section(BaseModel):
    id: str
    title: str
    level: int = 1
    start_char: int
    end_char: int


class Document(BaseModel):
    id: str
    title: str
    source: str  # path or URL
    source_type: str = "file"  # file | url | topic
    retrieved_at: datetime = Field(default_factory=datetime.now)
    text: str
    sections: list[Section] = Field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    id: str
    document_id: str
    section_id: str
    section_title: str
    text: str
    start_char: int
    end_char: int
