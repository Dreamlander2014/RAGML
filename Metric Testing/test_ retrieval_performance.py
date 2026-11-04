
import csv
import time
import shutil
import statistics
from pathlib import Path

from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever


DATA_DIR = "./Data" 
EVAL_DB = Path("./chroma_test2")
K = 4
REPEATS = 5

QUESTIONS = [
    ("Q1", "semantic",
     "How can hospital staff stop germs from spreading between patients?"),

    ("Q2", "exact",
     "What are standard precautions in infection prevention and control?"),

    ("Q3", "semantic",
     "When should healthcare workers clean their hands during patient care?"),

    ("Q4", "exact",
     "What are the guidelines for personal protective equipment (PPE)?"),

    ("Q5", "mixed",
     "What PPE should nurses use to prevent infection during patient care?")
]


def retrieve_and_time(retriever, question):
    start = time.perf_counter()
    docs = retriever.invoke(question)[:K]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return docs, elapsed_ms


def main():

    print("1. Loading hospital documents...")
    chunks = load_and_chunk_documents(DATA_DIR)

    print("2. Building fresh evaluation database...")

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

    # Warm up both retrievers
    print("3. Warming up...")
    for retriever in methods.values():
        retriever.invoke(QUESTIONS[0][2])

    summary = []
    passages = []

    print("4. Running retrieval tests...")

    for qid, query_type, question in QUESTIONS:

        print(f"\n{qid}: {question}")

        for method, retriever in methods.items():

            times = []
            docs = []

            for repeat in range(REPEATS):
                docs, elapsed = retrieve_and_time(
                    retriever, question
                )
                times.append(elapsed)

            unique_sources = len({
                d.metadata.get("source", "")
                for d in docs
            })

            summary.append({
                "question_id": qid,
                "query_type": query_type,
                "question": question,
                "method": method,
                "mean_latency_ms": round(
                    statistics.mean(times), 2
                ),
                "median_latency_ms": round(
                    statistics.median(times), 2
                ),
                "unique_sources": unique_sources,
                "context_characters": sum(
                    len(d.page_content) for d in docs
                )
            })

            for rank, doc in enumerate(docs, start=1):

                passages.append({
                    "question_id": qid,
                    "query_type": query_type,
                    "question": question,
                    "method": method,
                    "rank": rank,
                    "source": doc.metadata.get(
                        "source", ""
                    ),
                    "page": doc.metadata.get(
                        "page", 0
                    ) + 1,
                    "full_passage": doc.page_content,
                    "relevant": ""
                })

            print(
                f"  {method}: "
                f"{statistics.median(times):.2f} ms median | "
                f"{unique_sources} unique PDFs"
            )

    print("\n5. Exporting results...")

    with open(
        "test2_results.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=summary[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary)

    with open(
        "test2_passages.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=passages[0].keys()
        )
        writer.writeheader()
        writer.writerows(passages)

    print("\nTEST COMPLETE!")
    print("Saved: test2_results.csv")
    print("Saved: test2_passages.csv")
    print("Ready for relevance evaluation.")


if __name__ == "__main__":
    main()
