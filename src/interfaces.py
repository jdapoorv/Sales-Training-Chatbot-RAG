from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.models import TranscriptChunk, SearchResult

class VectorStore(ABC):
    @abstractmethod
    def ingest_transcript(self, call_id: str, call_title: str, chunks: List[Dict[str, Any]]) -> str:
        """Ingest chunks into the vector store."""
        pass

    @abstractmethod
    def get_call_ids(self) -> List[str]:
        """Return all call IDs."""
        pass

    @abstractmethod
    def search_chunks(self, query: str, call_id: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant chunks."""
        pass

    @abstractmethod
    def get_all_chunks_for_call(self, call_id: str) -> List[TranscriptChunk]:
        """Retrieve all chunks for a call."""
        pass

class LLMProvider(ABC):
    @abstractmethod
    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Generate content from the LLM."""
        pass
