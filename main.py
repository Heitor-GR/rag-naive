from pathlib import Path
from src.config import DATA_RAW_DIR, VECTOR_DB_DIR
from src.loader import carregar_e_dividir_pdf
from src.vectorstore import criar_ou_carregar_vector_db
from src.retriever import HybridRetriever
from src.rag import executar_rag

def main():
    print("=== Nível 2: Advanced RAG (Busca Híbrida BM25 + ChromaDB) ===")
    
    pdfs = list(DATA_RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\n[-] Nenhum PDF encontrado em '{DATA_RAW_DIR}'.")
        return

    pdf_path = pdfs[0]
    print(f"\n[+] PDF detectado: {pdf_path.name}")

    # 1. O BM25 necessita dos chunks em memória para indexação léxica
    print("[*] Carregando e fatiando o documento...")
    chunks = carregar_e_dividir_pdf(pdf_path)
    print(f"[+] Total de chunks disponíveis: {len(chunks)}")

    # 2. Inicializa o banco vetorial
    if not VECTOR_DB_DIR.exists() or not any(VECTOR_DB_DIR.iterdir()):
        print("[*] Gerando embeddings e persistindo no ChromaDB...")
        vector_db = criar_ou_carregar_vector_db(chunks)
    else:
        print("[*] Carregando base vetorial do disco...")
        vector_db = criar_ou_carregar_vector_db()

    # 3. Inicializa o Retriever Híbrido
    hybrid_retriever = HybridRetriever(vector_db=vector_db, documents=chunks)

    # 4. Loop Interativo
    print("\n--- Sistema Pronto! Digite sua pergunta (ou 'sair' para encerrar) ---")
    while True:
        pergunta = input("\nVocê: ").strip()
        if pergunta.lower() in ["sair", "exit", "q"]:
            break
        if not pergunta:
            continue

        print("[*] Consultando LLM com busca híbrida...")
        try:
            resposta, _ = executar_rag(hybrid_retriever, pergunta)
            print(f"\nIA: {resposta}")
        except Exception as e:
            print(f"\n[-] Erro ao executar RAG: {e}")

if __name__ == "__main__":
    main()