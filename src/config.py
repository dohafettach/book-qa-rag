import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (required) ────────────────────────────────────────────────────────

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "book-qa-index")

# ── Model settings ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"  # produces 1536-dim vectors
LLM_MODEL       = "gpt-4o-mini"             # cheap, fast, accurate enough for Q&A
CHUNK_SIZE      = 1000                       # tokens per chunk (~¾ of a page)
CHUNK_OVERLAP   = 200                        # overlap between consecutive chunks
TOP_K           = 5                          # number of chunks retrieved per query

# ── Data directory ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
