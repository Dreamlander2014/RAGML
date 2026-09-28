from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def run_evaluation():
    print("[Evaluation] Running retrieval experiments...")
    chunks = load_and_chunk_documents("./Data")
    
    # 1. Initialize Hybrid Retriever
    hybrid_retriever = get_hybrid_retriever(chunks)
    
    # 2. Initialize Dense-Only Retriever (Baseline)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Test queries with exact drug names/dosages vs. semantic questions
    test_queries = [
        "What is the exact dosage for IV Nitroglycerin?",
        "What are the steps to manage acute respiratory failure?"
    ]

    for q in test_queries:
        print(f"\n================ Question: {q} ================")
        
        dense_results = dense_retriever.invoke(q)
        print(f"\n--- Dense Only (Top Result) ---")
        print(dense_results[0].page_content[:200] if dense_results else "No results")

        hybrid_results = hybrid_retriever.invoke(q)
        print(f"\n--- Hybrid BM25 + Dense (Top Result) ---")
        print(hybrid_results[0].page_content[:200] if hybrid_results else "No results")

if __name__ == "__main__":
    run_evaluation()