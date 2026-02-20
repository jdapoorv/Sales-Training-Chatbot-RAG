from typing import List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from src.models import QueryResponse

class ConsoleView:
    BANNER = """
# 🎙 Sales Call GenAI Chatbot

Type a question, or use one of the built-in commands:

| Command | What it does |
|---|---|
| `list my call ids` | Show all ingested calls |
| `summarise the last call` | Summarise the most recent call |
| `summarise call <call_id> ` | Summarise a specific call |
| `ingest a new call transcript from <path>` | Add a new transcript |
| `ingest all from <folder>` | Ingest all .txt files from a folder |
| `help` | Show this message |
| `exit` / `quit` | Quit |
"""

    def __init__(self):
        self.console = Console()

    def display_banner(self):
        self.console.print(Panel(Markdown(self.BANNER), border_style="blue"))

    def display_help(self):
        self.console.print(Markdown(self.BANNER))

    def display_message(self, message: str, style: Optional[str] = None):
        self.console.print(message, style=style)

    def display_error(self, message: str):
        self.console.print(f"[red]Error:[/red] {message}")

    def display_success(self, message: str):
        self.console.print(f"[green]✓[/green] {message}")

    def display_warning(self, message: str):
        self.console.print(f"[yellow]{message}[/yellow]")

    def get_input(self, prompt: str = "[bold blue]You >[/bold blue] ") -> str:
        return self.console.input(prompt).strip()

    def show_status(self, message: str, spinner: str = "dots"):
        return self.console.status(message, spinner=spinner)

    def display_call_ids(self, ids: List[str]):
        if not ids:
            self.display_warning("No transcripts ingested yet.")
        else:
            self.console.print(f"\n[bold green]Ingested calls ({len(ids)}):[/bold green]")
            for cid in ids:
                self.console.print(f"  • {cid}")

    def display_response(self, response: QueryResponse, show_sources: bool = True):
        self.console.print(Markdown(response.answer))

        if show_sources and response.sources:
            self.console.print()
            self.console.print(Rule("📎 Sources", style="dim"))
            for i, r in enumerate(response.sources, 1):
                preview = r.chunk.text.strip().replace("\n", " ")[:150]
                score = 1 - r.distance  # cosine similarity
                self.console.print(
                    f"  [cyan][Source {i}][/cyan] "
                    f"[yellow]{r.chunk.call_title}[/yellow] | "
                    f"Chunk #{r.chunk.chunk_index} | "
                    f"Similarity: {score:.2f}\n"
                    f'    "[dim]{preview}…[/dim]"'
                )

    def display_bulk_ingest_start(self, count: int, folder_path: str):
        self.console.print(f"\n[bold]Bulk ingesting {count} file(s) from[/bold] [cyan]{folder_path}[/cyan]")

    def display_bulk_ingest_done(self, ok: int, failed: int):
        self.console.print(f"\n[bold green]Done:[/bold green] {ok} ingested, {failed} failed.")
