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

    # --- ADD THIS: RBAC Metadata Stamping ---
    for chunk in chunks:
        source_name = chunk.metadata.get("source", "").lower()

        # Check for clinical/restricted keywords
        restricted_keywords = ["dosage", "mg/kg", "contraindication", "administer", "prescription", "icd-10"]
        
        # Rule: If the file name indicates a patient guide or public document, tag as public.
        # Otherwise, default to restricted staff access.
        # Explicitly map your known files
        if "primary-care" in source_name or "long-term-care" in source_name or "chs" in source_name:
            chunk.metadata["access_level"] = "staff"
        elif "patient" in source_name or "public-guide" in source_name:
            chunk.metadata["access_level"] = "public"
        else:
            # Fail-closed default for any unknown file added later
            chunk.metadata["access_level"] = "staff"

    print(f"[Ingest] Loaded {len(documents)} pages and created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Sample chunk metadata: {chunks[0].metadata}")
