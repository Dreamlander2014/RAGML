from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str = "./data"):
    """Loads all PDFs in data_dir and splits them into chunks."""
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No PDF documents found in '{data_dir}' directory.")

    # Splitting the document sequentially (paragraph),(Lines),(Sentence),(Words),(Characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

# --- RBAC Metadata Stamping ---
    for chunk in chunks:
        source_name = chunk.metadata.get("source", "").lower()
        content_lower = chunk.page_content.lower()

        # Check for clinical/restricted keywords in the chunk text
        restricted_keywords = ["dosage", "mg/kg", "contraindication", "administer", "prescription", "icd-10"]
        
        # Rule 1: Explicitly check filename first
        if "patient" in source_name or "public-guide" in source_name:
            chunk.metadata["access_level"] = "public"
        elif any(keyword in source_name for keyword in ["primary-care", "long-term-care", "chs", "outpatient-dialysis"]):
            chunk.metadata["access_level"] = "staff"
        
        # Rule 2: If filename doesn't match, inspect the text content using your keywords
        elif any(keyword in content_lower for keyword in restricted_keywords):
            chunk.metadata["access_level"] = "staff"
        
        # Rule 3: Fail-closed default for everything else
        else:
            chunk.metadata["access_level"] = "staff"

    print(f"[Ingest] Loaded {len(documents)} pages and created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Sample chunk metadata: {chunks[0].metadata}")
