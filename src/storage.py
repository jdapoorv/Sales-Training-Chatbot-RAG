import os
import re
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.models import TranscriptChunk, SearchResult
from src.utils import chunk_text

# --- Configuration ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "call_transcripts"
CHUNK_SIZE = 400       # characters per chunk
CHUNK_OVERLAP = 80     # overlap to preserve context
TOP_K = 5              # default number of chunks to retrieve


def _get_collection() -> chromadb.Collection:
    """Initialise ChromaDB client and return the transcript collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection



def ingest_transcript(file_path: str) -> str:
    """
    Ingest a call transcript file into the vector store.
    Returns the call_id assigned.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    call_title = path.stem  # filename without extension used as title

    # Derive a stable call_id from the stem so re-ingesting same file is idempotent
    call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_title)

    collection = _get_collection()

    # Remove existing chunks for this call_id to allow re-ingestion
    existing = collection.get(where={"call_id": call_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    raw_chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    documents, metadatas, ids = [], [], []

    for idx, chunk_data in enumerate(raw_chunks):
        chunk_id = f"{call_id}__{idx}"
        documents.append(chunk_data["text"])
        metadatas.append({
            "call_id": call_id,
            "call_title": call_title,
            "chunk_index": idx,
            "start_char": chunk_data["start"],
            "end_char": chunk_data["end"],
        })
        ids.append(chunk_id)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return call_id


def get_call_ids() -> List[str]:
    """Return a deduplicated list of all ingested call IDs."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    seen = set()
    calls = []
    for meta in result["metadatas"]:
        cid = meta["call_id"]
        if cid not in seen:
            seen.add(cid)
            calls.append(cid)
    return sorted(calls)


def get_latest_call_id() -> Optional[str]:
    """Return the most-recently-ingested call ID (last alphabetically for now)."""
    ids = get_call_ids()
    return ids[-1] if ids else None


def search_chunks(
    query: str,
    call_id: Optional[str] = None,
    top_k: int = TOP_K,
) -> List[SearchResult]:
    """
    Embed `query` and return the top-k most relevant TranscriptChunks.
    Optionally filter by a specific call_id.
    """
    collection = _get_collection()
    where = {"call_id": call_id} if call_id else None

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    search_results = []
    if not results["ids"] or not results["ids"][0]:
        return search_results

    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        chunk = TranscriptChunk(
            chunk_id=chunk_id,
            call_id=meta["call_id"],
            call_title=meta["call_title"],
            chunk_index=meta["chunk_index"],
            text=results["documents"][0][i],
            start_char=meta["start_char"],
            end_char=meta["end_char"],
        )
        search_results.append(SearchResult(chunk=chunk, distance=results["distances"][0][i]))

    return search_results


def get_all_chunks_for_call(call_id: str) -> List[TranscriptChunk]:
    """Retrieve all chunks belonging to a specific call, ordered by chunk_index."""
    collection = _get_collection()
    result = collection.get(
        where={"call_id": call_id},
        include=["documents", "metadatas"],
    )
    chunks = []
    for i, cid in enumerate(result["ids"]):
        meta = result["metadatas"][i]
        chunks.append(TranscriptChunk(
            chunk_id=cid,
            call_id=meta["call_id"],
            call_title=meta["call_title"],
            chunk_index=meta["chunk_index"],
            text=result["documents"][i],
            start_char=meta["start_char"],
            end_char=meta["end_char"],
        ))
    return sorted(chunks, key=lambda c: c.chunk_index)
