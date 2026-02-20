# Sales Call Copilot 🎙

A CLI chatbot that ingests sales call transcripts, stores them as vector embeddings, and lets you ask natural-language questions — citing the exact conversation segments that informed each answer.

---

## Architecture

```
Clari_Chatbot/
├── cli.py              ← Entry point (interactive CLI)
├── src/
│   ├── models.py       ← Data classes (TranscriptChunk, SearchResult, QueryResponse)
│   ├── storage.py      ← ChromaDB ingestion, chunking, semantic search
│   └── generation.py   ← OpenAI RAG Q&A + structured call summarisation
├── samples/            ← Sample call transcripts (3 included)
├── tests/
│   └── test_storage.py ← Unit tests (pytest, mocked ChromaDB)
├── requirements.txt
├── .env.example
└── setup_commands.md
```

### Storage Design

| Layer | Technology | Why |
|---|---|---|
| **Vector store** | ChromaDB (local persistent) | Zero-infra, embedded, fast cosine similarity, metadata filtering |
| **Embeddings** | ChromaDB default (all-MiniLM-L6-v2) | Fast, local, no extra API cost |
| **Schema** | Each transcript → N overlapping chunks; each chunk stores `call_id`, `call_title`, `chunk_index`, `start_char`, `end_char` as metadata | Enables per-call filtering and precise source highlighting |

**Chunking strategy**: 400-character chunks with 80-character overlap, breaking on sentence/line boundaries where possible. This balances context richness and embedding precision.

---

## Setup

### 1. Clone & create environment
```bash
git clone <repo-url>
cd Clari_Chatbot
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the CLI
```bash
python cli.py
```

---

## Usage

```
You > list my call ids
You > ingest a new call transcript from samples/enterprise-prospect-01.txt
You > summarise the last call
You > summarise call smb-renewal-02
You > Give me all negative comments when pricing was mentioned in the calls
You > What integrations were discussed across all calls?
You > exit
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Assumptions

1. **LLM**: OpenAI API (`gpt-4o-mini` by default, configurable via `OPENAI_MODEL` env var).
2. **Embeddings**: Local sentence-transformer via ChromaDB's default embedding function — no extra API cost.
3. **"Last call"**: Defined as the last call ID alphabetically among all ingested calls (a timestamp-based ordering would require transcript metadata).
4. **Transcripts**: Plain `.txt` files with freeform content. No specific format is enforced.
5. **Re-ingestion**: Ingesting the same file path again replaces the previous version (idempotent by call ID).
6. **Source display**: Up to 5 most semantically relevant chunks shown after each Q&A response as citations.
