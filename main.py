from pathlib import Path
from src.config import DATA_RAW_DIR, VECTOR_DB_DIR
from src.loader import carregar_e_dividir_pdf
from src.vectorstore import criar_ou_carregar_vector_db
from src.rag import executar_rag

def main():
    print("=== Nível 1: Naïve RAG (Ollama + ChromaDB) ===")
    
    # 1. Procura PDFs na pasta data/raw/
    pdfs = list(DATA_RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\n[-] Nenhum PDF encontrado em '{DATA_RAW_DIR}'.")
        print("   Adicione um arquivo .pdf nessa pasta e execute novamente.")
        return

    pdf_path = pdfs[0]
    print(f"\n[+] PDF detectado: {pdf_path.name}")

    # 2. Cria a base vetorial se ainda não existir localmente
    if not VECTOR_DB_DIR.exists() or not any(VECTOR_DB_DIR.iterdir()):
        print("[*] Processando PDF e gerando embeddings no ChromaDB...")
        chunks = carregar_e_dividir_pdf(pdf_path)
        print(f"[+] Total de chunks criados: {len(chunks)}")
        vector_db = criar_ou_carregar_vector_db(chunks)
        print("[+] Base vetorial criada com sucesso!")
    else:
        print("[*] Carregando base vetorial existente do disco...")
        vector_db = criar_ou_carregar_vector_db()

    # 3. Loop interativo no terminal
    print("\n--- Sistema Pronto! Digite sua pergunta (ou 'sair' para encerrar) ---")
    while True:
        pergunta = input("\nVocê: ").strip()
        if pergunta.lower() in ["sair", "exit", "q"]:
            print("Encerrando...")
            break
        if not pergunta:
            continue

        print("[*] Consultando LLM...")
        try:
            resposta, _ = executar_rag(vector_db, pergunta)
            print(f"\nIA: {resposta}")
        except Exception as e:
            print(f"\n[-] Erro ao executar RAG: {e}")
            print("    Certifique-se de que o Ollama está rodando no terminal (`ollama serve`).")

if __name__ == "__main__":
    main()