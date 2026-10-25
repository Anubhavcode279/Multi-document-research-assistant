import os
from dotenv import load_dotenv
load_dotenv()

PDF_DIR = "data/pdfs"
CHROMA_DIR = "data/chroma"
CHUNKS_FILE = "data/chunks.jsonl"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Small and fast on CPU. For better quality (but slower), use "BAAI/bge-reranker-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # ollama | gemini | groq
SMART_MODEL = os.getenv("SMART_MODEL", "qwen2.5:7b")
FAST_MODEL = os.getenv("FAST_MODEL", "qwen2.5:3b")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))          # use 4096 if low on RAM

CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 120
FINAL_K = 4               # chunks kept per retrieval
MAX_SUBQUESTIONS = 3
MAX_HOPS = 1              # query-rewrite retries per sub-question
MAX_REVISIONS = 1         # synthesis retries after a failed self-check