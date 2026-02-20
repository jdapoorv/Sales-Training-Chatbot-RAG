"""
Pure utility functions with no heavy dependencies.
Safe to import in tests without triggering chromadb/pydantic issues.
"""
from typing import List


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[dict]:
    """
    Split text into overlapping chunks, preserving character positions.

    Returns a list of dicts with keys: text, start, end.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer to break on a sentence or line boundary
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
            )
            if boundary > start + overlap:
                end = boundary + 1
        chunks.append({"text": text[start:end], "start": start, "end": end})
        start = end - overlap if end < len(text) else end
    return chunks
