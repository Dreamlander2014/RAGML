from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str = "./Data"):
    """Loads all PDFs in data_dir and splits them into chunks."""
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No PDF documents found in '{data_dir}' directory.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[Ingest] Loaded {len(documents)} pages and created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Sample chunk metadata: {chunks[0].metadata}")