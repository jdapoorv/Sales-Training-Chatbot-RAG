"""
Tests for ingestion and retrieval logic.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Chunk Utility Tests (pure Python, no chromadb import)
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for the chunk_text utility — importable without chromadb."""

    def test_short_text_single_chunk(self):
        from src.utils import chunk_text
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_long_text_multiple_chunks(self):
        from src.utils import chunk_text
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 3

    def test_chunk_covers_full_text(self):
        from src.utils import chunk_text
        text = "x" * 800
        chunks = chunk_text(text, chunk_size=300, overlap=60)
        covered = set()
        for c in chunks:
            covered.update(range(c["start"], c["end"]))
        assert set(range(len(text))).issubset(covered)

    def test_empty_text(self):
        from src.utils import chunk_text
        chunks = chunk_text("", chunk_size=300, overlap=50)
        assert chunks == []


# ---------------------------------------------------------------------------
# Storage / Ingestion Tests (chromadb mocked at module-import level)
# ---------------------------------------------------------------------------

# Patch chromadb before any src.storage import so Python 3.14 is unaffected
import sys
from unittest.mock import MagicMock as _MM

_chroma_mock = _MM()
sys.modules.setdefault("chromadb", _chroma_mock)
sys.modules.setdefault("chromadb.utils", _MM())
sys.modules.setdefault("chromadb.utils.embedding_functions", _MM())


class TestIngestTranscript:
    """Tests for ingest_transcript with ChromaDB fully mocked."""

    def test_file_not_found(self):
        from src.storage import ingest_transcript
        with pytest.raises(FileNotFoundError):
            ingest_transcript("/nonexistent/path/call.txt")

    def test_ingest_creates_chunks(self, tmp_path):
        sample = tmp_path / "test_call.txt"
        content = "Speaker A: Hello.\nSpeaker B: Hi there, how are you?\n" * 20
        sample.write_text(content)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import ingest_transcript
            call_id = ingest_transcript(str(sample))

        assert call_id == "test_call"
        mock_collection.add.assert_called_once()
        args = mock_collection.add.call_args.kwargs
        assert len(args["documents"]) >= 1
        assert all(m["call_id"] == "test_call" for m in args["metadatas"])

    def test_reingest_deletes_existing(self, tmp_path):
        sample = tmp_path / "my_call.txt"
        sample.write_text("Some content " * 50)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["my_call__0", "my_call__1"]}

        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import ingest_transcript
            ingest_transcript(str(sample))

        mock_collection.delete.assert_called_once_with(ids=["my_call__0", "my_call__1"])


class TestGetCallIds:
    def test_returns_sorted_unique_ids(self):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"call_id": "call_b"},
                {"call_id": "call_a"},
                {"call_id": "call_b"},
            ]
        }
        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import get_call_ids
            ids = get_call_ids()
        assert ids == ["call_a", "call_b"]

    def test_empty_store(self):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"metadatas": []}
        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import get_call_ids
            ids = get_call_ids()
        assert ids == []


class TestSearchChunks:
    def test_returns_search_results(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["call_a__0"]],
            "documents": [["Hello world"]],
            "metadatas": [[{
                "call_id": "call_a",
                "call_title": "call_a",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 11,
            }]],
            "distances": [[0.1]],
        }
        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import search_chunks
            results = search_chunks("hello", top_k=1)

        assert len(results) == 1
        assert results[0].chunk.call_id == "call_a"
        assert results[0].chunk.text == "Hello world"

    def test_empty_results(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        with patch("src.storage._get_collection", return_value=mock_collection):
            from src.storage import search_chunks
            results = search_chunks("anything")
        assert results == []
