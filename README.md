# 📚 Book Q&A — RAG API

Ask natural language questions about any book and get grounded answers with source citations.

Built with **FastAPI**, **LangChain**, **Pinecone**, and **OpenAI**.

---

## How it works

```
Your book (.txt or .pdf)
        │
        ▼
   Split into chunks          ~1000 tokens each, 200 token overlap
        │
        ▼
   OpenAI Embeddings          each chunk → 1536 numbers (its "meaning")
        │
        ▼
   Pinecone                   stores all vectors, indexed for fast search
        │
        ▼ ── when you ask a question ──
        │
   Embed question             question → 1536 numbers
        │
        ▼
   Pinecone search            find 5 chunks closest in meaning to your question
        │
        ▼
   GPT-4o-mini                reads those 5 chunks, writes a grounded answer
        │
        ▼
   Response                   answer + which chunks it came from
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/book-qa-rag.git
cd book-qa-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get your API keys

**OpenAI**
1. Go to [platform.openai.com](https://platform.openai.com) → sign up
2. Click your avatar → **API Keys** → **Create new secret key**
3. Copy it. You won't see it again.

Cost estimate: embedding 3 full novels costs ~$0.10 total. Each question ~$0.001.

**Pinecone**
1. Go to [app.pinecone.io](https://app.pinecone.io) → sign up (free tier is enough)
2. Left sidebar → **API Keys** → copy the default key

### 3. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=book-qa-index
```

### 4. Add your books

Drop any `.txt` or `.pdf` into the `data/` folder:

```
data/
├── count-of-monte-cristo.txt    ← downloaded from gutenberg.org
├── war-and-peace.txt
└── my-own-book.pdf              ← or any PDF you own
```

**Where to get the classics for free:**
- [gutenberg.org/ebooks/1184](https://gutenberg.org/ebooks/1184) — The Count of Monte Cristo
- [gutenberg.org/ebooks/2600](https://gutenberg.org/ebooks/2600) — War and Peace
- [gutenberg.org/ebooks/1342](https://gutenberg.org/ebooks/1342) — Pride and Prejudice

On each page click **"Plain Text UTF-8"** to download.

### 5. Run the server

```bash
uvicorn api:app --reload
```

Open **http://localhost:8000/docs** for the interactive API explorer.

---

## API Endpoints

### `GET /`
Health check. Shows all ingested books.

---

### `GET /books`
List all ingested books.

```json
{
  "books": [
    {
      "book_id": "count-of-monte-cristo",
      "title": "The Count of Monte Cristo",
      "author": "Alexandre Dumas",
      "chunks": 1823,
      "ingested_at": "2024-06-10T14:32:00"
    }
  ]
}
```

---

### `POST /ingest`
Ingest a book from the `data/` folder. Run once per book.

```json
{
  "file_name": "count-of-monte-cristo.txt",
  "book_id":   "count-of-monte-cristo",
  "title":     "The Count of Monte Cristo",
  "author":    "Alexandre Dumas"
}
```

Takes 60–90 seconds for a full novel. Progress logs in the terminal.

---

### `POST /ask`
Ask a question. The book must be ingested first.

```json
{
  "book_id":  "count-of-monte-cristo",
  "question": "How did Dantès escape from prison?"
}
```

Response:

```json
{
  "question": "How did Dantès escape from prison?",
  "answer":   "Dantès escaped by sewing himself into the burial shroud of the Abbé Faria...",
  "book_id":  "count-of-monte-cristo",
  "title":    "The Count of Monte Cristo",
  "author":   "Alexandre Dumas",
  "sources": [
    { "chunk_index": 312, "snippet": "He had substituted himself for the dead man..." },
    { "chunk_index": 318, "snippet": "The gravediggers carried the sack to the edge..." }
  ]
}
```

---

## CLI (alternative to the API)

```bash
# Ingest a book
python main.py ingest count-of-monte-cristo.txt count-of-monte-cristo "The Count of Monte Cristo" "Alexandre Dumas"

# Ask a question
python main.py ask count-of-monte-cristo "How did Dantès escape?"

# List ingested books
python main.py books
```

---

## Project Structure

```
book-qa-rag/
├── api.py              # FastAPI server — all endpoints
├── main.py             # CLI interface
├── src/
│   ├── config.py       # API keys, model settings, paths
│   ├── books.py        # Local registry of ingested books (data/registry.json)
│   ├── ingest.py       # Load → split → embed → upsert to Pinecone
│   ├── query.py        # Embed question → retrieve → LLM answer
│   └── utils.py        # Startup validation
├── data/               # Drop your books here (gitignored except registry.json)
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Orchestration | LangChain 0.2 |
| Vector store | Pinecone serverless |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4o-mini |
| PDF parsing | pypdf |

---

## License

MIT
