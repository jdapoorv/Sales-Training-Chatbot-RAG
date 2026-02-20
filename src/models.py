from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptChunk:
    """Represents a single chunk of a call transcript stored in the vector DB."""
    chunk_id: str
    call_id: str
    call_title: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    """A retrieved chunk with its relevance distance."""
    chunk: TranscriptChunk
    distance: float


@dataclass
class QueryResponse:
    """Full LLM response with source citations."""
    answer: str
    sources: List[SearchResult] = field(default_factory=list)
