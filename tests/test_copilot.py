import pytest
from unittest.mock import MagicMock
from src.copilot import SalesCopilot
from src.models import SearchResult, TranscriptChunk, QueryResponse

class TestSalesCopilot:
    @pytest.fixture
    def copilot(self):
        self.mock_vs = MagicMock()
        self.mock_llm = MagicMock()
        return SalesCopilot(self.mock_vs, self.mock_llm)

    def test_answer_question_orchestration(self, copilot):
        # Mock search results
        chunk = TranscriptChunk(
            chunk_id="1", call_id="c", call_title="t", chunk_index=0, 
            text="Context", start_char=0, end_char=7
        )
        self.mock_vs.search_chunks.return_value = [SearchResult(chunk=chunk, distance=0.1)]
        self.mock_llm.generate_content.return_value = "The answer [Source 1]"
        
        response = copilot.answer_question("Question?")
        
        assert response.answer == "The answer [Source 1]"
        assert len(response.sources) == 1
        assert self.mock_llm.generate_content.called

    def test_summarise_call_orchestration(self, copilot):
        # Mock retrieval of all chunks
        chunk = TranscriptChunk(
            chunk_id="1", call_id="c", call_title="t", chunk_index=0, 
            text="Full Transcript", start_char=0, end_char=15
        )
        self.mock_vs.get_all_chunks_for_call.return_value = [chunk]
        self.mock_llm.generate_content.return_value = "Summary text"
        
        response = copilot.summarise_call("c")
        
        assert response.answer == "Summary text"
        assert self.mock_vs.get_all_chunks_for_call.assert_called_with("c") is None
