
# Advanced RAG - Local Document QA System (Nível 2)

A local, privacy-focused Advanced RAG (Retrieval-Augmented Generation) pipeline built to perform precise question answering over long documents without sending data to external APIs.

## 🏗️ Architecture

1. **Hybrid Retrieval (Lexical + Semantic):**
   - **BM25 (Lexical):** Exact keyword and named-entity matching using `rank_bm25`.
   - **ChromaDB (Semantic):** Dense vector search via embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
   - **Reciprocal Rank Fusion (RRF):** Merges both search results into an initial candidate pool (k=35).
2. **Multilingual Re-ranking:**
   - **Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`):** Re-scores candidates to filter out non-relevant context and selects the top-5 highest scoring chunks.
3. **Generation:**
   - **Ollama (`qwen2.5:3b`):** Local LLM inference with strict prompt constraints to mitigate hallucinations.

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.

### Installation

1. Clone the repository:
