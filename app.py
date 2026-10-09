import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever

# 1. Initialize RAG once
chunks = load_and_chunk_documents("./data")
# Grab both the ensemble retriever and vectorstore reference
retriever, vectorstore = get_hybrid_retriever(chunks)
# The temperature determines the level of creativity
llm = ChatOllama(model="gemma4:e4b", temperature=0.0)

prompt = ChatPromptTemplate.from_template("""
Answer strictly using context:
{context}

Question: {question}
Answer with source citation and include the relevance score percentage shown in the document header:""")

def format_docs_with_scores(docs):
    # Get top 4 documents AND their raw vector similarity scores (0.0 to 1.0)
    docs_and_scores = vectorstore.similarity_search_with_score(docs, k=4)
    """Formats context for LLM prompt and includes rank markers for UI citations."""
    formatted = []
    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "SOP")
        page = doc.metadata.get("page", 0)

        # Convert L2 distance to a rough similarity percentage (clamped between 0% and 100%)
        # Lower distance means higher score
        similarity_percentage = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        
        header = f" [Doc: {source} | Page: {page + 1} | Score: {similarity_percentage:.2%}]"
        formatted.append(f"{header}\n{doc.page_content}")
        
    return "\n\n".join(formatted)

# 2. Main processing function called by Gradio UI
def process_question(question, history):

    # Related to the prompt in the ChatPromptTemplate
    # Sends the user prompt to retriever | format_docs to build {context}
    # while RunnablePassthrough() forwards the raw string to {question}
    
    # Format docs into string with score headers
    context_str = format_docs_with_scores(question)
    
    # Invoke model chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})
    
    return answer

# 3. Launch UI locally
demo = gr.ChatInterface(fn=process_question, 
                        title="Local Clinical Assistant")

if __name__ == "__main__":
    demo.launch(share=False, debug=True)