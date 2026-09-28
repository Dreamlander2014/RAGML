from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "SOP")
        page = doc.metadata.get("page", 0)
        formatted.append(f"[Document: {source} | Page: {page + 1}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def main():
    print("Initializing Clinical Assistant...")
    chunks = load_and_chunk_documents("./Data")
    retriever = get_hybrid_retriever(chunks)

    # Local Ollama instance
    llm = ChatOllama(model="gemma4:e4b", temperature=0.0)

    prompt = ChatPromptTemplate.from_template("""
    You are a clinical SOP operational assistant. Answer the medical query strictly using ONLY the provided context.
    If the answer cannot be determined directly from the context, state:
    "Information not available in current clinical guidelines."

    Context:
    {context}

    Query: {question}

    Answer with source citation:""")

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nClinical Assistant Ready! (Type 'exit' to quit)\n" + "-"*50)
    while True:
        user_input = input("\nEnter query: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = rag_chain.invoke(user_input)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()