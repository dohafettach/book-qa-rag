"""
Book Q&A — FastAPI server.

Endpoints:
  GET  /              → health check + list of ingested books
  GET  /books         → list all ingested books
  POST /ingest        → ingest any .txt or .pdf from the data/ folder
  POST /ask           → ask a question about an ingested book

Run with:
  uvicorn api:app --reload
Then open: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils import validate_env
from src.config import DATA_DIR


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts. Crashes loudly if keys are missing."""
    validate_env()
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[api] Data directory: {DATA_DIR}")
    print("[api] API keys found. Server ready.")
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Book Q&A RAG",
    description=(
        "Ask questions about any book in natural language. "
        "Drop a .txt or .pdf into data/, ingest it once, then query it forever."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    file_name: str  = Field(...,  example="count-of-monte-cristo.txt",
                           description="File name inside the data/ folder. Must be .txt or .pdf.")
    book_id:   str  = Field(...,  example="count-of-monte-cristo",
                           description="Your chosen slug. Used to query this book later.")
    title:     str  = Field(...,  example="The Count of Monte Cristo")
    author:    str  = Field(...,  example="Alexandre Dumas")


class IngestResponse(BaseModel):
    book_id:     str
    title:       str
    author:      str
    chunks:      int
    ingested_at: str


class AskRequest(BaseModel):
    book_id:  str = Field(..., example="count-of-monte-cristo")
    question: str = Field(..., example="Why was Dantès put in prison?")


class AskResponse(BaseModel):
    question: str
    answer:   str
    book_id:  str
    title:    str
    author:   str
    sources:  list


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    """
    Confirms the server is running and shows all ingested books.
    Good first call to make after starting the server.
    """
    from src.books import list_books
    books = list_books()
    return {
        "status":         "ok",
        "ingested_books": len(books),
        "books":          books,
        "docs":           "/docs",
    }


@app.get("/books", summary="List ingested books")
def list_books():
    """
    Returns all books that have been ingested and are ready to query.
    Each entry shows the book_id you'll use in /ask.
    """
    from src.books import list_books as _list
    return {"books": _list()}


@app.post("/ingest", response_model=IngestResponse, summary="Ingest a book")
def ingest_book(body: IngestRequest):
    """
    Ingest a book from the data/ folder into Pinecone.

    **Run once per book.** After ingestion the book is ready to query via /ask.

    Steps this endpoint performs:
    1. Loads the file from data/{file_name}
    2. Splits it into ~1000 token chunks with 200 token overlap
    3. Sends each chunk to OpenAI to get a 1536-dim embedding vector
    4. Upserts all vectors + metadata into Pinecone
    5. Saves the book to the local registry

    For a 500-page novel this takes about 60-90 seconds and costs ~$0.05.

    **How to get books:**
    - Project Gutenberg (gutenberg.org) — free plain text for any pre-1927 book
    - Any PDF you own
    """
    from src.ingest import ingest

    file_path = os.path.join(DATA_DIR, body.file_name)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"File '{body.file_name}' not found in data/ folder. "
                f"Drop the file there first, then call this endpoint."
            ),
        )

    try:
        result = ingest(
            file_path=file_path,
            book_id=body.book_id,
            title=body.title,
            author=body.author,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, summary="Ask a question about a book")
def ask_question(body: AskRequest):
    """
    Ask a natural language question about an ingested book.

    The book must be ingested first via POST /ingest.

    What happens:
    1. Your question is embedded into a 1536-dim vector
    2. Pinecone finds the 5 most semantically similar chunks from your book
    3. Those chunks are sent to GPT-4o-mini with a strict "use only this context" prompt
    4. GPT writes an answer grounded in the actual text
    5. The source chunks are returned alongside the answer so you can verify

    Example:
    ```json
    {
      "book_id": "count-of-monte-cristo",
      "question": "How did Dantès escape from prison?"
    }
    ```
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from src.query import ask
        result = ask(question=body.question, book_id=body.book_id)
        return result

    except ValueError as e:
        # Book not ingested yet — give a helpful message
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
