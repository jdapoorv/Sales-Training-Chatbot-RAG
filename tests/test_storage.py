import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Pre-emptive mock for chromadb before any src imports
mock_chroma = MagicMock()
sys.modules["chromadb"] = mock_chroma
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()

from src.storage import ChromaVectorStore
from src.models import TranscriptChunk

@pytest.fixture
def mock_collection():
    return MagicMock()

@pytest.fixture
def store(mock_collection):
    # Patch the PersistentClient so it returns our mock
    with patch("chromadb.PersistentClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield ChromaVectorStore(db_path=":memory:")

class TestChromaVectorStore:
    def test_ingest_transcript(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": []}
        
        chunks = [
            {"text": "Hello", "start": 0, "end": 5},
            {"text": "World", "start": 6, "end": 11}
        ]
        
        res = store.ingest_transcript("call_1", "Title 1", chunks)
        
        assert res == "call_1"
        assert mock_collection.add.called
        args = mock_collection.add.call_args.kwargs
        assert len(args["ids"]) == 2
        assert args["metadatas"][0]["call_id"] == "call_1"

    def test_get_call_ids(self, store, mock_collection):
        mock_collection.get.return_value = {
            "metadatas": [
                {"call_id": "b"},
                {"call_id": "a"},
                {"call_id": "b"}
            ]
        }
        
        ids = store.get_call_ids()
        assert ids == ["a", "b"]

    def test_search_chunks(self, store, mock_collection):
        mock_collection.query.return_value = {
            "ids": [["c1_0"]],
            "documents": [["content"]],
            "metadatas": [[{
                "call_id": "c1",
                "call_title": "t1",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 7
            }]],
            "distances": [[0.1]]
        }
        
        results = store.search_chunks("query", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.call_id == "c1"
        assert results[0].distance == 0.1
