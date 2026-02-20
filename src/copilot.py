from typing import List, Optional
from src.models import QueryResponse, SearchResult
from src.interfaces import VectorStore, LLMProvider

class SalesCopilot:
    def __init__(self, vector_store: VectorStore, llm_provider: LLMProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    def _format_context(self, results: List[SearchResult]) -> str:
        """Build a numbered context block from search results to feed the LLM."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Source {i} | Call: {r.chunk.call_title} | Chunk #{r.chunk.chunk_index}]\n"
                f"{r.chunk.text.strip()}"
            )
        return "\n\n---\n\n".join(lines)

    def _format_full_transcript(self, call_id: str) -> str:
        """Reconstruct the full transcript text from all stored chunks."""
        chunks = self.vector_store.get_all_chunks_for_call(call_id)
        return "\n".join(c.text for c in chunks)

    def answer_question(
        self,
        query: str,
        call_id: Optional[str] = None,
        top_k: int = 5,
    ) -> QueryResponse:
        results = self.vector_store.search_chunks(query, call_id=call_id, top_k=top_k)
        if not results:
            return QueryResponse(
                answer="I could not find any relevant information in the stored transcripts.",
                sources=[],
            )

        context = self._format_context(results)
        user_msg = f"### Transcript Excerpts\n{context}\n\n### Question\n{query}\n\nProvide a clear, direct answer and reference every specific claim with [Source N] where N matches the excerpt number above."
        
        system_prompt = "You are a sales-call analyst AI. Your job is to answer using ONLY the transcript excerpts provided. Cite facts using [Source N]."
        
        answer = self.llm_provider.generate_content(system_prompt, user_msg, temperature=0.2)
        return QueryResponse(answer=answer, sources=results)

    def summarise_call(self, call_id: str) -> QueryResponse:
        transcript = self._format_full_transcript(call_id)
        if not transcript.strip():
            return QueryResponse(answer=f"No transcript found for '{call_id}'.", sources=[])

        system_prompt = "You are an expert sales-call analyst. Summarise the transcript with: Overview, Key Discussion Points, Sentiment/Objections, Next Steps, and Pricing."
        user_msg = f"### Call Transcript\n{transcript}\n\nProduce the structured summary as instructed."
        
        answer = self.llm_provider.generate_content(system_prompt, user_msg, temperature=0.3)
        return QueryResponse(answer=answer, sources=[])
