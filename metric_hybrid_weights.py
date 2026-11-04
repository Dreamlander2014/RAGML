from ingest import load_and_chunk_documents
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import csv
import time
import statistics


import shutil
from pathlib import Path

EVAL_DB = Path("./chroma_weight_eval")

if EVAL_DB.exists():
    shutil.rmtree(EVAL_DB)

chunks = load_and_chunk_documents("./Data")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(EVAL_DB)
)

dense = vectorstore.as_retriever(search_kwargs={"k": 4})

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 4


WEIGHTS = {
    "Hybrid_90_10": [0.1, 0.9],
    "Hybrid_75_25": [0.25, 0.75],
    "Hybrid_50_50": [0.5, 0.5],
    "Hybrid_25_75": [0.75, 0.25],
}

retrievers = {"Dense": dense}

for name, weights in WEIGHTS.items():
    retrievers[name] = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=weights
    )





# METRIC TEST: HYBRID WEIGHT OPTIMIZATION

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

passage_results = []
timing_results = []

print("\nWarming up retrievers...")

for retriever in retrievers.values():
    retriever.invoke(QUESTIONS[0][2])

print("\nRunning Hybrid Weight Optimization...")

for qid, query_type, question in QUESTIONS:

    print(f"\n{qid}: {question}")

    for method, retriever in retrievers.items():

        times = []
        docs = []

        for repeat in range(REPEATS):

            start = time.perf_counter()
            docs = retriever.invoke(question)[:K]
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            times.append(elapsed_ms)

        timing_results.append({
            "question_id": qid,
            "query_type": query_type,
            "method": method,
            "mean_latency_ms": round(
                statistics.mean(times), 2
            ),
            "median_latency_ms": round(
                statistics.median(times), 2
            )
        })

        for rank, doc in enumerate(docs, start=1):

            passage_results.append({
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
            f"{statistics.median(times):.2f} ms"
        )


with open(
    "weight_passages.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=passage_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(passage_results)


with open(
    "weight_timings.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=timing_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(timing_results)


print("\nTEST 5 COMPLETE!")
print("Saved: weight_passages.csv")
print("Saved: weight_timings.csv")
