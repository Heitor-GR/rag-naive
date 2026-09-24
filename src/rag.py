from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.retriever import HybridRetriever

PROMPT_TEMPLATE = """Você é um assistente estrito de leitura.
ATENÇÃO: Responda à pergunta usando EXCLUSIVAMENTE as informações do Contexto abaixo.
Se o contexto não disser explicitamente a resposta para a pergunta, responda APENAS: "Não encontrei essa informação no documento."

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""

def executar_rag(retriever: HybridRetriever, pergunta: str, top_k: int = 3):
    # O retriever agora faz o Reranking e entrega apenas os 3 melhores chunks
    docs_relevantes = retriever.buscar(pergunta, top_k=top_k)
    
    print("\n" + "="*50)
    print("🔍 [DEBUG RAG NÍVEL 2 + RERANKER] CHUNKS SELECIONADOS:")
    for i, doc in enumerate(docs_relevantes, 1):
        preview = doc.page_content.replace("\n", " ")[:150]
        print(f"   [Chunk {i}]: {preview}...")
    print("="*50 + "\n")
    
    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm
    
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
    return resposta, docs_relevantes