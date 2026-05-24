"""
Configuration module.

Loads all settings from environment variables (via .env) and exposes
them as typed constants used across the entire codebase.
Nothing is hardcoded here — all secrets come from the environment.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root (works whether run from root or a sub-dir)
load_dotenv()


# ─── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")


# ─── Model Settings ────────────────────────────────────────────────────────────
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

# Cohere embed-english-v3.0 always outputs 1024-dimensional vectors.
COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
COHERE_EMBED_DIM: int = 1024
COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")


# ─── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
MIN_CHUNK_LENGTH: int = 50  # Discard chunks shorter than this (e.g., page numbers)


# ─── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))


# ─── File Paths ────────────────────────────────────────────────────────────────
# BASE_DIR is always the project root, regardless of which module imports this.
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
INDEX_DIR: str = os.path.join(BASE_DIR, "data", "index")
METADATA_DIR: str = os.path.join(BASE_DIR, "data", "metadata")


# ─── App Settings ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
APP_TITLE: str = "Research Paper RAG Chatbot"


# ─── Grounding / Trust ─────────────────────────────────────────────────────────
# Evidence score thresholds (0.0–1.0) for answer confidence labels.
# Score is computed from rerank scores and number of supporting chunks.
CONFIDENCE_HIGH_THRESHOLD: float = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.55"))
CONFIDENCE_MEDIUM_THRESHOLD: float = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.20"))

# Below this score the bot emits a soft refusal / uncertainty statement.
# Kept low (0.05) so the LLM gets a chance on borderline queries — the
# LLM system prompt already guards against hallucination.
GROUNDING_REFUSAL_THRESHOLD: float = float(os.getenv("GROUNDING_REFUSAL_THRESHOLD", "0.05"))

# Minimum rerank score to count a chunk as "supporting" evidence.
MIN_SUPPORTING_SCORE: float = float(os.getenv("MIN_SUPPORTING_SCORE", "0.20"))

# Minimum word count for a query before the guardrail will even attempt retrieval.
# Single-word queries are almost always too vague to produce high rerank scores.
MIN_QUERY_WORDS: int = int(os.getenv("MIN_QUERY_WORDS", "2"))
