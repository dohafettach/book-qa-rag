"""
Ingestion pipeline.

Takes any .txt or .pdf file, splits it into overlapping chunks,
embeds each chunk with OpenAI, and upserts the vectors into Pinecone.
Registers the book in the local registry so the API can list it.

Flow:
  file → load → split into chunks → embed → upsert to Pinecone → register
"""

import os
from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _load_txt(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF page by page and join into one string.
    pypdf is already in requirements — no extra install needed.
    """
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_file(file_path: str) -> str:
    """
    Dispatch to the right loader based on file extension.
    Raises a clear error if the format isn't supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return _load_txt(file_path)
    elif ext == ".pdf":
        return _load_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Drop a .txt or .pdf into the data/ folder."
        )


def _split(text: str, book_id: str, title: str, author: str) -> list:
    """
    Split raw text into overlapping chunks.
    Each chunk is wrapped in a LangChain Document with metadata
    so Pinecone stores: the text, the book it came from, and its position.

    Why metadata matters: when querying we filter by book_id so a question
    about Monte Cristo doesn't accidentally retrieve chunks from War and Peace.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    raw_chunks = splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "book_id":     book_id,
                "title":       title,
                "author":      author,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    print(f"[ingest] '{title}' → {len(documents)} chunks")
    return documents


def _ensure_pinecone_index(pc) -> None:
    """
    Create the Pinecone index if it doesn't already exist.
    dimension=1536 must match text-embedding-3-small's output size exactly.
    This is safe to call every time — it's a no-op if the index exists.
    """
    from pinecone import ServerlessSpec

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[ingest] Created Pinecone index '{PINECONE_INDEX_NAME}'")
    else:
        print(f"[ingest] Using existing Pinecone index '{PINECONE_INDEX_NAME}'")


def ingest(file_path: str, book_id: str, title: str, author: str) -> dict:
    """
    Full ingestion pipeline for any .txt or .pdf file.

    Args:
        file_path: path to the file inside data/
        book_id:   your chosen slug, e.g. "count-of-monte-cristo"
        title:     human-readable title, e.g. "The Count of Monte Cristo"
        author:    author name, e.g. "Alexandre Dumas"

    Returns:
        dict with book metadata and number of chunks upserted
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from src.books import register_book

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: '{file_path}'. "
            f"Make sure you dropped it into the data/ folder."
        )

    # 1. Load
    print(f"[ingest] Loading '{file_path}' ...")
    text = _load_file(file_path)
    print(f"[ingest] Loaded {len(text):,} characters")

    # 2. Split
    documents = _split(text, book_id, title, author)

    # 3. Set up embedding model
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    # 4. Set up Pinecone — create index if needed
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc)

    # 5. Embed all chunks and upsert into Pinecone
    # from_documents sends chunks to OpenAI in batches,
    # gets back vectors, and writes them all to Pinecone
    print(f"[ingest] Embedding and uploading {len(documents)} chunks — this takes a minute ...")
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    # 6. Save to local registry so the API can list this book
    entry = register_book(
        book_id=book_id,
        title=title,
        author=author,
        file_path=file_path,
        chunks=len(documents),
    )

    print(f"[ingest] Done. '{title}' is ready to query.")
    return entry
