import os
from typing import List, Optional

from openai import OpenAI

from src.models import QueryResponse, SearchResult
from src.storage import get_all_chunks_for_call, search_chunks, get_latest_call_id

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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


def _source_snippet(r: SearchResult) -> str:
    """Return a compact, human-readable snippet for one source chunk."""
    preview = r.chunk.text.strip().replace("\n", " ")[:120]
    return (
        f'  • [{r.chunk.call_title} | Chunk #{r.chunk.chunk_index}] '
        f'"{preview}…"'
    )


# ---------------------------------------------------------------------------
# Q&A over all (or a specific) call(s)
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are a sales-call analyst AI. Your job is to answer the user's question \
using ONLY the transcript excerpts provided below. 

Rules:
1. Base your answer strictly on the provided context; do not invent facts.
2. If the context does not contain enough information, say so clearly.
3. Be concise and precise.
4. When you cite a fact, reference the source in-line as [Source N].
"""

QA_USER_TEMPLATE = """\
### Transcript Excerpts
{context}

### Question
{question}

Provide a clear, direct answer and reference every specific claim with [Source N] \
where N matches the excerpt number above.
"""


def answer_question(
    query: str,
    call_id: Optional[str] = None,
    top_k: int = 5,
) -> QueryResponse:
    """
    RAG-based Q&A: retrieve relevant chunks then ask the LLM.

    Parameters
    ----------
    query   : the user's natural-language question
    call_id : if provided, restrict search to this specific call
    top_k   : number of chunks to retrieve
    """
    results = search_chunks(query, call_id=call_id, top_k=top_k)
    if not results:
        return QueryResponse(
            answer="I could not find any relevant information in the stored transcripts.",
            sources=[],
        )

    context = _format_context(results)
    user_msg = QA_USER_TEMPLATE.format(context=context, question=query)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return QueryResponse(answer=answer, sources=results)


# ---------------------------------------------------------------------------
# Summarisation of a specific call
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """\
You are an expert sales-call analyst. Summarise the provided call transcript \
in a structured way. Your summary MUST include:

1. **Call Overview** – participants (if mentioned), date/time (if mentioned), \
   and overall topic.
2. **Key Discussion Points** – bullet list of the main topics discussed.
3. **Sentiment & Objections** – overall tone, any concerns or objections raised \
   by the prospect.
4. **Next Steps / Action Items** – concrete follow-ups mentioned in the call.
5. **Pricing / Commercial Discussion** – any pricing, discounts, or budget topics \
   (mark "None mentioned" if absent).

Be factual. Do not embellish or invent.
"""

SUMMARY_USER_TEMPLATE = """\
### Call Transcript
{transcript}

Produce the structured summary as instructed.
"""


def summarise_call(call_id: str) -> QueryResponse:
    """
    Summarise an entire call transcript.

    For short transcripts we use the full text directly;  
    structured sections give the LLM enough context for a quality summary.
    """
    transcript = _format_full_transcript(call_id)
    if not transcript.strip():
        return QueryResponse(
            answer=f"No transcript data found for call '{call_id}'.",
            sources=[],
        )

    user_msg = SUMMARY_USER_TEMPLATE.format(transcript=transcript)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return QueryResponse(answer=answer, sources=[])
