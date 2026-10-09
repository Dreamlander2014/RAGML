from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(chunks, persist_dir: str = "./chroma_db"):
    """Builds an Ensemble Retriever combining BM25 keyword and Chroma vector search."""
    # Local embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 1. Vector Search (Chroma)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2. Keyword Search (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # 3. Hybrid Search (RRF)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever
