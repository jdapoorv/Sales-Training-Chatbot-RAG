# Sales Call GenAI Chatbot 

A CLI chatbot that ingests sales call transcripts, stores them as vector embeddings, and lets you ask natural-language questions — citing the exact conversation segments that informed each answer.

---

## Architecture

The project is built using **SOLID principles** and an **MVC (Model-View-Controller) architecture**, ensuring a clean separation of concerns and a modular codebase.

```
Clari_Chatbot/
├── cli.py              ← Controller (orchestrates user input and commands)
├── src/
│   ├── view.py         ← View (handles all terminal rendering via Rich)
│   ├── copilot.py      ← Orchestrator (manages RAG logic, depends on interfaces)
│   ├── interfaces.py   ← Abstractions (VectorStore and LLMProvider interfaces)
│   ├── processor.py    ← Service (handles file processing and chunking - SRP)
│   ├── storage.py      ← Model (ChromaDB implementation of VectorStore)
│   ├── generation.py   ← Model (LLM Provider implementations & Factory)
│   └── models.py       ← Blueprint (shared data classes)
├── data/               ← Sample call transcripts
├── tests/
└── .env.example
```

### Technical Stack

| Layer | Component | Responsibility |
|---|---|---|
| **View** | `ConsoleView` | Pure presentation layer using `rich`. No business logic. |
| **Model** | `ChromaVectorStore` | Persistent vector storage using ChromaDB. |
| **Model** | `LLMProvider` | Abstracts OpenAI, Groq, Gemini, and Ollama. |
| **Controller** | `SalesChatbotController` | Glues input to the Model and View. |
| **Service** | `TranscriptProcessor` | Handles char-based chunking with smart boundaries. |

---

### Storage Design

- **Vector store**: ChromaDB (local persistent) — zero-infra, fast cosine similarity.
- **Embeddings**: ChromaDB default (`all-MiniLM-L6-v2`) — performant and local.
- **Chunking**: 400-char chunks with 80-char overlap, preserving sentence boundaries.

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
You > ingest a new call transcript from data/1_demo_call.txt
You > summarise the last call
You > summarise call 1_demo_call
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
