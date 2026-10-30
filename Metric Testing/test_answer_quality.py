
import csv
import time
import shutil
from pathlib import Path

from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DATA_DIR = "./Data"
EVAL_DB = Path("./chroma_answer_eval")
MODEL = "gemma4:e2b"
K = 4
TEMPERATURE = 0

QUESTIONS = [
    ("Q1", "How can hospital staff stop germs from spreading between patients?"),
    ("Q2", "What are standard precautions in infection prevention and control?"),
    ("Q3", "When should healthcare workers clean their hands during patient care?"),
    ("Q4", "What are the guidelines for personal protective equipment (PPE)?"),
    ("Q5", "What PPE should nurses use to prevent infection during patient care?")
]

# Same prompt for both retrieval methods
prompt = ChatPromptTemplate.from_template("""
You are a clinical information assistant.

Answer the question using ONLY the provided hospital
infection prevention and control guidelines.

Give a clear and concise answer.
Cite the source document and page number.

If the retrieved context does not contain sufficient
information, clearly state that you cannot answer
from the available documents.

Context:
{context}

Question:
{question}

Answer:
""")


def format_context(docs):
    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        formatted.append(
            f"[Source: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def main():

    print("1. Loading hospital guidelines...")
    chunks = load_and_chunk_documents(DATA_DIR)

    print("2. Initializing retrievers...")

    # Fresh evaluation database
    if EVAL_DB.exists():
        shutil.rmtree(EVAL_DB)

    hybrid, vectorstore = get_hybrid_retriever(
        chunks,
        persist_dir=str(EVAL_DB)
    )

    dense = vectorstore.as_retriever(
        search_kwargs={"k": K}
    )

    methods = {
        "Dense": dense,
        "Hybrid": hybrid
    }

    print("3. Initializing local LLM...")
    llm = ChatOllama(
        model=MODEL,
        temperature=TEMPERATURE
    )

    chain = prompt | llm

    results = []

    print("4. Running answer-quality evaluation...")

    for qid, question in QUESTIONS:

        print(f"\n{qid}: {question}")

        for method, retriever in methods.items():

            # Retrieve evidence
            start = time.perf_counter()
            docs = retriever.invoke(question)[:K]
            retrieval_time = time.perf_counter() - start

            context = format_context(docs)

            # Generate answer
            start = time.perf_counter()

            response = chain.invoke({
                "context": context,
                "question": question
            })

            generation_time = time.perf_counter() - start

            answer = response.content

            results.append({
                "question_id": qid,
                "question": question,
                "method": method,
                "answer": answer,
                "retrieved_context": context,
                "retrieval_seconds": round(retrieval_time, 3),
                "generation_seconds": round(generation_time, 3),
                "correctness_score": "",
                "faithfulness_score": "",
                "completeness_score": ""
            })

            print(f"\n{method} answer:")
            print(answer[:400])

    print("\n5. Saving results...")

    with open(
        "answer_quality_results.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nTEST COMPLETE!")
    print("Saved: answer_quality_results.csv")


if __name__ == "__main__":
    main()
