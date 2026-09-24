import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

def tokenizar_portugues(texto: str) -> List[str]:
    """Tokenização simples: minúsculas e remoção de pontuação."""
    texto = texto.lower()
    return re.findall(r'\b\w+\b', texto)

class HybridRetriever:
    """
    Retriever Híbrido com RERANKER (Nível 2 Completo):
    1. Busca Híbrida (ChromaDB + BM25 unificados via RRF) -> Traz candidatos iniciais (ex: 15).
    2. Reranker (Cross-Encoder) -> Filtra e reordena para entregar apenas os mais relevantes (ex: 3).
    """
    def __init__(self, vector_db, documents: List[Document], k_rrf: int = 60, model_reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_db = vector_db
        self.documents = documents
        self.k_rrf = k_rrf
        
        # 1. BM25 (Busca Léxica)
        print("[*] Indexando chunks para busca léxica (BM25)...")
        corpus_tokenizado = [tokenizar_portugues(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus_tokenizado)
        
        # 2. Reranker (Cross-Encoder)
        print("[*] Carregando modelo de Reranking (Cross-Encoder)...")
        self.reranker = CrossEncoder(model_reranker)

    def _buscar_bm25(self, query: str, top_k: int) -> List[Document]:
        query_tokenizada = tokenizar_portugues(query)
        scores = self.bm25.get_scores(query_tokenizada)
        indices_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[indices_top[i]] for i in range(len(indices_top)) if scores[indices_top[i]] > 0]

    def buscar(self, query: str, top_k: int = 3, candidatos_iniciais: int = 15) -> List[Document]:
        """
        Recupera candidatos via RRF e aplica o Cross-Encoder para entregar apenas o top_k refinado.
        """
        # 1. ETAPA GROSSA: Traz 15 candidatos da Busca Híbrida (BM25 + ChromaDB)
        cand_vetoriais = self.vector_db.similarity_search(query, k=candidatos_iniciais)
        cand_lexicos = self._buscar_bm25(query, top_k=candidatos_iniciais)
        
        # Fusão RRF
        rrf_scores = {}
        def acumular_rrf(docs: List[Document]):
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.page_content
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
                rrf_scores[doc_id]["score"] += 1.0 / (self.k_rrf + rank)

        acumular_rrf(cand_vetoriais)
        acumular_rrf(cand_lexicos)
        
        docs_rrf = [item["doc"] for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)]
        candidatos = docs_rrf[:candidatos_iniciais]
        
        if not candidatos:
            return []

        # 2. ETAPA FINA (RERANKING): O Cross-Encoder avalia a relação Pergunta x Chunk
        pares = [[query, doc.page_content] for doc in candidatos]
        rerank_scores = self.reranker.predict(pares)
        
        # Junta o documento com a nota do Reranker e ordena do maior para o menor
        doc_scores = list(zip(candidatos, rerank_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Devolve apenas os 'top_k' (padrão: 3) melhores e mais limpos
        return [doc for doc, score in doc_scores[:top_k]]