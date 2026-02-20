import os
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from src.models import TranscriptChunk, SearchResult
from src.interfaces import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(
        self, 
        db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        collection_name: str = "call_transcripts"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_transcript(self, call_id: str, call_title: str, chunks: List[Dict[str, Any]]) -> str:
        """Ingest chunks into ChromaDB, replacing any existing ones for the same call_id."""
        existing = self.collection.get(where={"call_id": call_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        documents, metadatas, ids = [], [], []

        for idx, chunk_data in enumerate(chunks):
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

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return call_id

    def get_call_ids(self) -> List[str]:
        """Return a deduplicated list of all ingested call IDs."""
        result = self.collection.get(include=["metadatas"])
        seen = set()
        calls = []
        for meta in result["metadatas"]:
            cid = meta["call_id"]
            if cid not in seen:
                seen.add(cid)
                calls.append(cid)
        return sorted(calls)

    def search_chunks(
        self,
        query: str,
        call_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Embed query and return the top-k most relevant TranscriptChunks."""
        where = {"call_id": call_id} if call_id else None

        results = self.collection.query(
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

    def get_all_chunks_for_call(self, call_id: str) -> List[TranscriptChunk]:
        """Retrieve all chunks belonging to a specific call, ordered by chunk_index."""
        result = self.collection.get(
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

