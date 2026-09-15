from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from src.config import LLM_MODEL, OLLAMA_BASE_URL

PROMPT_TEMPLATE = """Com base EXCLUSIVAMENTE no contexto fornecido abaixo, responda à pergunta de forma objetiva.
Se a informação não estiver presente no contexto, responda apenas: "Não encontrei essa informação no documento."

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""

def executar_rag(vector_db, pergunta: str, top_k: int = 5):
    """
    Recupera os chunks mais semelhantes, exibe no terminal para debug e gera a resposta.
    """
    # 1. Recuperação (Retrieval)
    docs_relevantes = vector_db.similarity_search(pergunta, k=top_k)
    
    # Exibe no terminal exatamente o que o ChromaDB encontrou
    print("\n" + "="*50)
    print("🔍 [DEBUG] TRECHOS ENCONTRADOS NO CHROMADB:")
    for i, doc in enumerate(docs_relevantes, 1):
        preview = doc.page_content.replace("\n", " ")[:150]
        print(f"   [Chunk {i}]: {preview}...")
    print("="*50 + "\n")
    
    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
    
    # 2. Formatação e Geração
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm
    
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
    return resposta, docs_relevantes