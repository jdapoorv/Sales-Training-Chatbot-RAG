import os
from typing import List, Optional

from openai import OpenAI
from google import genai

from src.models import QueryResponse, SearchResult
from src.storage import get_all_chunks_for_call, search_chunks

# --- Global State (Lazy Indexed) ---
_client = None
_model_name = None

def _get_provider_info():
    """Retrieve provider and model from environment."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        key = os.getenv("OPENAI_API_KEY")
    elif provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        key = os.getenv("GROQ_API_KEY")
    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3")
        key = "ollama"  # dummy
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        key = os.getenv("GEMINI_API_KEY")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    
    return provider, model, key

def _ensure_client():
    """Lazy initialization of the LLM client."""
    global _client, _model_name
    if _client is not None:
        return _client, _model_name, os.getenv("LLM_PROVIDER", "openai").lower()

    provider, model, key = _get_provider_info()
    _model_name = model

    if not key and provider != "ollama":
        raise ValueError(f"Missing API key for provider '{provider}'. Please check your .env file.")

    if provider == "openai":
        _client = OpenAI(api_key=key)
    elif provider == "groq":
        _client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    elif provider == "ollama":
        _client = OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
    elif provider == "gemini":
        _client = genai.Client(api_key=key)
    
    return _client, _model_name, provider


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_context(results: List[SearchResult]) -> str:
    """Build a numbered context block from search results to feed the LLM."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[Source {i} | Call: {r.chunk.call_title} | Chunk #{r.chunk.chunk_index}]\n"
            f"{r.chunk.text.strip()}"
        )
    return "\n\n---\n\n".join(lines)


def _format_full_transcript(call_id: str) -> str:
    """Reconstruct the full transcript text from all stored chunks."""
    chunks = get_all_chunks_for_call(call_id)
    return "\n".join(c.text for c in chunks)


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Unified wrapper to call different LLM providers."""
    client, model, provider = _ensure_client()

    if provider in ["openai", "groq", "ollama"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    elif provider == "gemini":
        # Using the new google-genai SDK
        full_prompt = f"{system_prompt}\n\nUSER INPUT:\n{user_prompt}"
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config={'temperature': temperature}
        )
        return response.text.strip()
    
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    query: str,
    call_id: Optional[str] = None,
    top_k: int = 5,
) -> QueryResponse:
    results = search_chunks(query, call_id=call_id, top_k=top_k)
    if not results:
        return QueryResponse(
            answer="I could not find any relevant information in the stored transcripts.",
            sources=[],
        )

    context = _format_context(results)
    user_msg = f"### Transcript Excerpts\n{context}\n\n### Question\n{query}\n\nProvide a clear, direct answer and reference every specific claim with [Source N] where N matches the excerpt number above."
    
    system_prompt = "You are a sales-call analyst AI. Your job is to answer using ONLY the transcript excerpts provided. Cite facts using [Source N]."
    
    answer = _call_llm(system_prompt, user_msg, temperature=0.2)
    return QueryResponse(answer=answer, sources=results)


def summarise_call(call_id: str) -> QueryResponse:
    transcript = _format_full_transcript(call_id)
    if not transcript.strip():
        return QueryResponse(answer=f"No transcript found for '{call_id}'.", sources=[])

    system_prompt = "You are an expert sales-call analyst. Summarise the transcript with: Overview, Key Discussion Points, Sentiment/Objections, Next Steps, and Pricing."
    user_msg = f"### Call Transcript\n{transcript}\n\nProduce the structured summary as instructed."
    
    answer = _call_llm(system_prompt, user_msg, temperature=0.3)
    return QueryResponse(answer=answer, sources=[])
