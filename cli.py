#!/usr/bin/env python3
"""
cli.py – MVC Controller for the Sales Call GenAI Chatbot.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before imports
load_dotenv()

from src.storage import ChromaVectorStore
from src.generation import ProviderFactory
from src.copilot import SalesCopilot
from src.processor import TranscriptProcessor
from src.view import ConsoleView
from src.models import QueryResponse

# Default folder that is auto-ingested on every startup
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

class SalesChatbotController:
    def __init__(self):
        self.view = ConsoleView()
        self.processor = TranscriptProcessor()
        self.vector_store = ChromaVectorStore()
        self.llm_provider = ProviderFactory.get_provider()
        self.copilot = SalesCopilot(self.vector_store, self.llm_provider)

    def _handle_ingest_folder(self, folder_path: str, silent: bool = False) -> None:
        """Ingest all .txt files in a folder, delegating to View for progress."""
        folder = Path(folder_path)
        if not folder.is_dir():
            self.view.display_error(f"Folder not found: {folder_path}")
            return
        
        txt_files = sorted(folder.glob("*.txt"))
        if not txt_files:
            if not silent:
                self.view.display_warning(f"No .txt files found in {folder_path}")
            return
        
        if not silent:
            self.view.display_bulk_ingest_start(len(txt_files), folder_path)
        
        ok, failed = 0, 0
        for txt in txt_files:
            try:
                with self.view.show_status(f"  Ingesting [cyan]{txt.name}[/cyan]…"):
                    processed = self.processor.process_file(str(txt))
                    call_id = self.vector_store.ingest_transcript(
                        processed["call_id"], 
                        processed["call_title"], 
                        processed["chunks"]
                    )
                if not silent:
                    self.view.display_success(f"{txt.name} → [bold]{call_id}[/bold]")
                ok += 1
            except Exception as e:
                self.view.display_message(f"  [red]✗[/red] {txt.name}: {e}")
                failed += 1
        
        if not silent:
            self.view.display_bulk_ingest_done(ok, failed)

    def _handle_list(self) -> None:
        ids = self.vector_store.get_call_ids()
        self.view.display_call_ids(ids)

    def _handle_summarise(self, call_id: str) -> None:
        self.view.display_message(f"\n[bold]Summarising call:[/bold] [cyan]{call_id}[/cyan]\n")
        with self.view.show_status("Generating summary…"):
            response = self.copilot.summarise_call(call_id)
        self.view.display_response(response, show_sources=False)

    def _handle_ingest(self, path_str: str) -> None:
        path = path_str.strip().strip("'\"")
        self.view.display_message(f"\n[bold]Ingesting:[/bold] [cyan]{path}[/cyan]")
        try:
            with self.view.show_status("Processing transcript…"):
                processed = self.processor.process_file(path)
                call_id = self.vector_store.ingest_transcript(
                    processed["call_id"], 
                    processed["call_title"], 
                    processed["chunks"]
                )
            self.view.display_success(f"Ingested successfully as call ID: [bold]{call_id}[/bold]")
        except FileNotFoundError as e:
            self.view.display_error(str(e))
        except Exception as e:
            self.view.display_error(f"Unexpected error during ingestion: {e}")

    def _handle_qa(self, query: str) -> None:
        self.view.display_message("")
        with self.view.show_status("Searching transcripts and thinking…"):
            response = self.copilot.answer_question(query)
        self.view.display_response(response, show_sources=True)

    def _parse_and_dispatch(self, user_input: str) -> bool:
        cmd = user_input.strip()
        lower = cmd.lower()

        if lower in ("exit", "quit"):
            self.view.display_message("\n[bold]Goodbye! 👋[/bold]\n")
            return False
        elif lower in ("help",):
            self.view.display_help()
        elif lower == "list my call ids":
            self._handle_list()
        elif lower == "summarise the last call":
            ids = self.vector_store.get_call_ids()
            call_id = ids[-1] if ids else None
            if call_id:
                self._handle_summarise(call_id)
            else:
                self.view.display_warning("No calls ingested yet.")
        elif lower.startswith("summarise call "):
            call_id = cmd[len("summarise call "):].strip()
            self._handle_summarise(call_id)
        elif lower.startswith("ingest a new call transcript from "):
            path_str = cmd[len("ingest a new call transcript from "):].strip()
            self._handle_ingest(path_str)
        elif lower.startswith("ingest all from "):
            folder_str = cmd[len("ingest all from "):].strip()
            self._handle_ingest_folder(folder_str)
        elif lower.startswith("ingest "):
            path_str = cmd[len("ingest "):].strip()
            self._handle_ingest(path_str)
        else:
            self._handle_qa(cmd)
        return True

    def run(self) -> None:
        self.view.display_banner()

        data_folder = Path(DATA_FOLDER)
        if data_folder.is_dir() and any(data_folder.glob("*.txt")):
            self.view.display_message(f"[dim]Auto-ingesting transcripts from [cyan]{DATA_FOLDER}[/cyan]…[/dim]")
            self._handle_ingest_folder(DATA_FOLDER, silent=False)
            self.view.display_message("")

        while True:
            try:
                self.view.display_message("")
                user_input = self.view.get_input()
                if not user_input:
                    continue
                should_continue = self._parse_and_dispatch(user_input)
                if not should_continue:
                    break
            except KeyboardInterrupt:
                self.view.display_message("\n\n[bold]Interrupted. Goodbye! 👋[/bold]\n")
                break
            except Exception as e:
                self.view.display_error(f"An error occurred: {e}")

if __name__ == "__main__":
    controller = SalesChatbotController()
    controller.run()
