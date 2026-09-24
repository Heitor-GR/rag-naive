
# 🧠 Naïve RAG Local — Nível 1 (MVP)

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![Vector DB](https://img.shields.io/badge/ChromaDB-Persistente-red.svg)
![LLM](https://img.shields.io/badge/Ollama-qwen2.5%3A1.5b-orange.svg)

Pipeline de **Retrieval-Augmented Generation (RAG)** 100% local e privado em Python. Permite fazer perguntas e extrair informações de documentos PDF sem envio de dados para APIs externas, otimizado para execução em **CPU e placas de vídeo integradas**.

---

## 📌 Arquitetura do Nível 1

O sistema segue o fluxo clássico de **Naïve RAG**:

```text
[ Documento PDF ]
       │
       ▼ (loader.py - pypdf)
[ Texto Bruto ]
       │
       ▼ (RecursiveCharacterTextSplitter: 1000 chars, overlap 200)
[ Chunks de Texto ]
       │
       ▼ (sentence-transformers/all-MiniLM-L6-v2)
[ Embeddings Vetoriais ]
       │
       ▼ (vectorstore.py)
[ ChromaDB Local ]
       │
       ├────────────────────────┐
       ▼ (pergunta do usuário)   ▼ (k=5 chunks mais similares)
[ Busca Vetorial ] ──► [ Contexto Recuperado ]
                                │
                                ▼ (rag.py + Prompt Estrito)
                         [ Ollama (Qwen2.5:1.5b) ]
                                │
                                ▼
                       [ Resposta Final ]
```
