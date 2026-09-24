from pathlib import Path

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# Parâmetros do Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Modelos
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:1.5b"