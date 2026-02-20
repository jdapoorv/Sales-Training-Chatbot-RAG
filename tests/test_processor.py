import pytest
from unittest.mock import MagicMock, patch
import sys

# Pre-emptive mock for chromadb to avoid issues in Python 3.14 test environments
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

from src.processor import TranscriptProcessor

from src.utils import chunk_text

class TestTranscriptProcessor:
    def test_chunk_text_single_chunk(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_text_multiple_chunks(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 3

    def test_process_file_format(self, tmp_path):
        sample = tmp_path / "1_test_call.txt"
        sample.write_text("Speaker A: Content goes here.")
        
        processor = TranscriptProcessor()
        result = processor.process_file(str(sample))
        
        assert result["call_id"] == "1_test_call"
        assert result["call_title"] == "1_test_call"
        assert len(result["chunks"]) > 0
        assert "text" in result["chunks"][0]
        assert "start" in result["chunks"][0]
