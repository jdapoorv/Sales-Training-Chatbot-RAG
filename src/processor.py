import re
from pathlib import Path
from typing import List, Dict, Any
from src.utils import chunk_text

class TranscriptProcessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Reads a file, generates a call_id, and chunks the text."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        raw_text = path.read_text(encoding="utf-8").strip()
        call_title = path.stem
        call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_title)

        chunks = chunk_text(raw_text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        
        return {
            "call_id": call_id,
            "call_title": call_title,
            "chunks": chunks
        }
