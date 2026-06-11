"""
Book Q&A RAG — CLI entry point.

Usage:
  python main.py ingest <file_in_data_folder> <book_id> "<Title>" "<Author>"
  python main.py ask <book_id> "<question>"
  python main.py books

Examples:
  python main.py ingest count-of-monte-cristo.txt count-of-monte-cristo "The Count of Monte Cristo" "Alexandre Dumas"
  python main.py ask count-of-monte-cristo "Why was Dantès put in prison?"
  python main.py books
"""

import sys
import os
from src.utils import validate_env
from src.config import DATA_DIR


def main():
    validate_env()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "books":
        from src.books import list_books
        books = list_books()
        if not books:
            print("No books ingested yet.")
        for b in books:
            print(f"  {b['book_id']:30s} {b['title']} — {b['author']} ({b['chunks']} chunks)")

    elif command == "ingest":
        if len(sys.argv) < 6:
            print("Usage: python main.py ingest <file_name> <book_id> <title> <author>")
            sys.exit(1)
        file_name, book_id, title, author = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        file_path = os.path.join(DATA_DIR, file_name)
        from src.ingest import ingest
        result = ingest(file_path=file_path, book_id=book_id, title=title, author=author)
        print(f"\nIngested: {result['title']} ({result['chunks']} chunks)")

    elif command == "ask":
        if len(sys.argv) < 4:
            print("Usage: python main.py ask <book_id> <question>")
            sys.exit(1)
        book_id  = sys.argv[2]
        question = " ".join(sys.argv[3:])
        from src.query import ask
        result = ask(question=question, book_id=book_id)
        print(f"\nQ: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({len(result['sources'])} chunks):")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] chunk {s['chunk_index']}: {s['snippet'][:120]}...")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
