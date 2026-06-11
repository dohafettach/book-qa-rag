"""
Book registry: tracks which books have been ingested.
The registry is a simple JSON file stored in data/registry.json.

Each entry looks like:
{
  "book-id": {
    "book_id":  "count-of-monte-cristo",
    "title":    "The Count of Monte Cristo",
    "author":   "Alexandre Dumas",
    "file":     "data/count-of-monte-cristo.txt",
    "chunks":   1823,
    "ingested_at": "2024-06-10T14:32:00"
  }
}

This is how the API knows what books exist and what their metadata is.
Without a registry you'd have to ask Pinecone — which only stores vectors,
not human-readable book info.
"""

import os
import json
from datetime import datetime
from src.config import DATA_DIR

REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")


def _load_registry() -> dict:
    """Read the registry file. Return empty dict if it doesn't exist yet."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    """Write the registry back to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def register_book(book_id: str, title: str, author: str, file_path: str, chunks: int) -> dict:
    """Add or update a book entry in the registry after successful ingestion."""
    registry = _load_registry()
    entry = {
        "book_id":     book_id,
        "title":       title,
        "author":      author,
        "file":        file_path,
        "chunks":      chunks,
        "ingested_at": datetime.utcnow().isoformat(),
    }
    registry[book_id] = entry
    _save_registry(registry)
    return entry


def get_book(book_id: str) -> dict | None:
    """Return a single book's metadata, or None if not found."""
    return _load_registry().get(book_id)


def list_books() -> list[dict]:
    """Return all ingested books as a list."""
    return list(_load_registry().values())


def book_exists(book_id: str) -> bool:
    return book_id in _load_registry()
