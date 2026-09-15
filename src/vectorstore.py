from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR

def inicializar_embeddings():
    """Instancia o modelo leve de embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def criar_ou_carregar_vector_db(chunks=None):
    """
    Se receber chunks, persiste no ChromaDB local.
    Se não receber nada, apenas carrega a base vetorial já existente em disco.
    """
    embeddings = inicializar_embeddings()
    
    if chunks:
        # Cria a base e persiste em disco
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_DB_DIR)
        )
    else:
        # Carrega a base existente
        vector_db = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings
        )
        
    return vector_db