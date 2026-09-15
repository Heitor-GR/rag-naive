from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def carregar_e_dividir_pdf(caminho_pdf: Path):
    """
    Carrega um arquivo PDF e o divide em chunks de texto.
    """
    if not caminho_pdf.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_pdf}")
    
    # 1. Carrega o texto bruto do PDF
    loader = PyPDFLoader(str(caminho_pdf))
    documentos = loader.load()
    
    # 2. Divide em pedaços menores (chunks) com sobreposição
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documentos)
    return chunks