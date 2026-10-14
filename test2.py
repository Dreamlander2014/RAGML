# test hybrid vs dense retrievel

import csv
import time
import statistics
import shutil
from pathlib import Path

from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever

# Experiment settings
K = 4
REPEATS = 5
EVAL_DB = Path("./chroma_test2")

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


def count_unique_sources(docs):
    return len({
        doc.metadata.get("source", "unknown")
        for doc in docs
    })


def run_test():
    print("STEP 1: Loading hospital PDFs")
    chunks = load_and_chunk_documents("./data")

    print("\nSTEP 2: Building a fresh evaluation index")

    # Only delete the dedicated Test 2 index.
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

    print("\nSTEP 3: Warming up retrievers")
    for retriever in methods.values():
        retriever.invoke(QUESTIONS[0][2])

    results = []
    details = []

    print("\nSTEP 4: Running experiment")

    for qid, query_type, question in QUESTIONS:
        print(f"\n{qid}: {question}")

        for method, retriever in methods.items():

            times = []
            docs = []

            for repeat in range(REPEATS):
                start = time.perf_counter()
                current_docs = retriever.invoke(question)[:K]
                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                times.append(elapsed_ms)
                docs = current_docs

            diversity = count_unique_sources(docs)

            results.append({
                "question_id": qid,
                "query_type": query_type,
                "method": method,
                "mean_latency_ms": round(statistics.mean(times), 2),
                "median_latency_ms": round(statistics.median(times), 2),
                "unique_sources": diversity,
                "context_characters": sum(
                    len(d.page_content) for d in docs
                )
            })

            print(
                f"  {method}: "
                f"{statistics.mean(times):.2f} ms | "
                f"{diversity} unique PDFs"
            )

            for rank, doc in enumerate(docs, start=1):
                details.append({
                    "question_id": qid,
                    "method": method,
                    "rank": rank,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", 0) + 1,
                    "passage": doc.page_content[:500]
                })

    print("\nSTEP 5: Results")

    with open("test2_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    with open("test2_passages.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=details[0].keys()
        )
        writer.writeheader()
        writer.writerows(details)

    print("Created test2_results.csv")
    print("Created test2_passages.csv")


if __name__ == "__main__":
    run_test()
