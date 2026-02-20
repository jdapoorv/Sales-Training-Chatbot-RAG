#!/usr/bin/env python3
"""
cli.py – Interactive command-line chatbot for the Sales Call GenAI Chatbot.

Usage:
    python cli.py

Available commands (type at the prompt):
    list my call ids
    summarise the last call
    summarise call <call_id>
    ingest a new call transcript from <path>
    help
    exit / quit
    <any other text>  →  Q&A over all ingested transcripts
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

# Load .env before importing src modules (OPENAI_API_KEY etc.)
load_dotenv()

# Default folder that is auto-ingested on every startup
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

from src.generation import answer_question, summarise_call
from src.storage import get_call_ids, get_latest_call_id, ingest_transcript
from src.models import QueryResponse

console = Console()

def _handle_ingest_folder(folder_path: str, silent: bool = False) -> None:
    """Ingest all .txt files in a folder, printing progress for each."""
    folder = Path(folder_path)
    if not folder.is_dir():
        console.print(f"[red]Folder not found:[/red] {folder_path}")
        return
    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        if not silent:
            console.print(f"[yellow]No .txt files found in {folder_path}[/yellow]")
        return
    if not silent:
        console.print(f"\n[bold]Bulk ingesting {len(txt_files)} file(s) from[/bold] [cyan]{folder_path}[/cyan]")
    ok, failed = 0, 0
    for txt in txt_files:
        try:
            with console.status(f"  Ingesting [cyan]{txt.name}[/cyan]…", spinner="dots"):
                call_id = ingest_transcript(str(txt))
            if not silent:
                console.print(f"  [green]✓[/green] {txt.name} → [bold]{call_id}[/bold]")
            ok += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {txt.name}: {e}")
            failed += 1
    if not silent:
        console.print(f"\n[bold green]Done:[/bold green] {ok} ingested, {failed} failed.")


BANNER = """
# 🎙 Sales Call GenAI Chatbot

Type a question, or use one of the built-in commands:

| Command | What it does |
|---|---|
| `list my call ids` | Show all ingested calls |
| `summarise the last call` | Summarise the most recent call |
| `summarise call <call_id>` | Summarise a specific call |
| `ingest a new call transcript from <path>` | Add a new transcript |
| `ingest all from <folder>` | Ingest all .txt files from a folder |
| `help` | Show this message |
| `exit` / `quit` | Quit |
"""


def _print_response(response: QueryResponse, show_sources: bool = True) -> None:
    """Render the LLM answer and, optionally, source snippets."""
    console.print(Markdown(response.answer))

    if show_sources and response.sources:
        console.print()
        console.print(Rule("📎 Sources", style="dim"))
        for i, r in enumerate(response.sources, 1):
            preview = r.chunk.text.strip().replace("\n", " ")[:150]
            score = 1 - r.distance  # cosine similarity
            console.print(
                f"  [cyan][Source {i}][/cyan] "
                f"[yellow]{r.chunk.call_title}[/yellow] | "
                f"Chunk #{r.chunk.chunk_index} | "
                f"Similarity: {score:.2f}\n"
                f'    "[dim]{preview}…[/dim]"'
            )


def _handle_list() -> None:
    ids = get_call_ids()
    if not ids:
        console.print("[yellow]No transcripts ingested yet.[/yellow]")
    else:
        console.print(f"\n[bold green]Ingested calls ({len(ids)}):[/bold green]")
        for cid in ids:
            console.print(f"  • {cid}")


def _handle_summarise(call_id: str) -> None:
    console.print(f"\n[bold]Summarising call:[/bold] [cyan]{call_id}[/cyan]\n")
    with console.status("Generating summary…", spinner="dots"):
        response = summarise_call(call_id)
    _print_response(response, show_sources=False)


def _handle_ingest(path_str: str) -> None:
    path = path_str.strip().strip("'\"")
    console.print(f"\n[bold]Ingesting:[/bold] [cyan]{path}[/cyan]")
    try:
        with console.status("Processing transcript…", spinner="dots"):
            call_id = ingest_transcript(path)
        console.print(f"[green]✓ Ingested successfully as call ID:[/green] [bold]{call_id}[/bold]")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]Unexpected error during ingestion:[/red] {e}")


def _handle_qa(query: str) -> None:
    console.print()
    with console.status("Searching transcripts and thinking…", spinner="dots"):
        response = answer_question(query)
    _print_response(response, show_sources=True)


def _parse_and_dispatch(user_input: str) -> bool:
    """
    Route user input to the right handler.
    Returns False if the user wants to exit.
    """
    cmd = user_input.strip()
    lower = cmd.lower()

    if lower in ("exit", "quit"):
        console.print("\n[bold]Goodbye! 👋[/bold]\n")
        return False

    elif lower in ("help",):
        console.print(Markdown(BANNER))

    elif lower == "list my call ids":
        _handle_list()

    elif lower == "summarise the last call":
        call_id = get_latest_call_id()
        if call_id:
            _handle_summarise(call_id)
        else:
            console.print("[yellow]No calls ingested yet.[/yellow]")

    elif lower.startswith("summarise call "):
        call_id = cmd[len("summarise call "):].strip()
        _handle_summarise(call_id)

    elif lower.startswith("ingest a new call transcript from "):
        path_str = cmd[len("ingest a new call transcript from "):].strip()
        _handle_ingest(path_str)

    elif lower.startswith("ingest all from "):
        folder_str = cmd[len("ingest all from "):].strip()
        _handle_ingest_folder(folder_str)

    elif lower.startswith("ingest "):
        # short alias: "ingest <path>"
        path_str = cmd[len("ingest "):].strip()
        _handle_ingest(path_str)

    else:
        _handle_qa(cmd)

    return True


def main() -> None:
    console.print(Panel(Markdown(BANNER), border_style="blue"))

    # Auto-ingest the Data folder on every startup
    data_folder = Path(DATA_FOLDER)
    if data_folder.is_dir() and any(data_folder.glob("*.txt")):
        console.print(f"[dim]Auto-ingesting transcripts from [cyan]{DATA_FOLDER}[/cyan]…[/dim]")
        _handle_ingest_folder(DATA_FOLDER, silent=False)
        console.print()

    while True:
        try:
            console.print()
            user_input = console.input("[bold blue]You >[/bold blue] ").strip()
            if not user_input:
                continue
            should_continue = _parse_and_dispatch(user_input)
            if not should_continue:
                break
        except KeyboardInterrupt:
            console.print("\n\n[bold]Interrupted. Goodbye! 👋[/bold]\n")
            break
        except Exception as e:
            console.print(f"\n[red]An error occurred:[/red] {e}\n")


if __name__ == "__main__":
    main()
