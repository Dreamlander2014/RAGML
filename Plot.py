import time
import numpy as np
import matplotlib.pyplot as plt
from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever
import pandas as pd

def benchmark_pipeline():
    all_chunks = load_and_chunk_documents("./data")
    
    # More granular and progressive steps
    steps = [50, 200, 500, 1000, 2000, len(all_chunks)]
    steps = [s for s in steps if s <= len(all_chunks)]
    
    corpus_sizes = []
    mean_latencies = []
    std_latencies = []
    
    query = "What is the exact dosage for IV Nitroglycerin?"
    
    for size in steps:
        subset_chunks = all_chunks[:size]
        
        # 1. Initialize retriever outside query loop (Index building is offline)
        retriever, vectorstore = get_hybrid_retriever(subset_chunks)
        
        # 2. Warm-up run (eliminates cold-start bias)
        _ = retriever.invoke(query)
        
        # 3. Multi-iteration runs to average out system noise
        iterations = 5
        run_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            _ = retriever.invoke(query)
            elapsed = (time.time() - start_time) * 1000  # ms
            run_times.append(elapsed)
            
        corpus_sizes.append(size)
        mean_latencies.append(np.mean(run_times))
        std_latencies.append(np.std(run_times))
        print(f"Chunks: {size:4d} | Mean Latency: {np.mean(run_times):.2f} ms")

    # Plotting with error bars for scientific accuracy
    plt.figure(figsize=(8, 5))
    plt.errorbar(corpus_sizes, mean_latencies, yerr=std_latencies, fmt='-o', color='b', capsize=5)
    plt.title("Smoothed Hybrid Retrieval Latency vs. Corpus Size")
    plt.xlabel("Number of Chunks (C)")
    plt.ylabel("Query Latency (ms)")
    plt.grid(True)
    plt.savefig("retrieval_latency_corrected.png")
    print("\nCorrected graph saved as 'retrieval_latency_corrected.png'")

    # After collecting your data into corpus_sizes and mean_latencies:
    df = pd.DataFrame({
        "Chunks (C)": corpus_sizes,
        "Mean Latency (ms)": mean_latencies,
        "Std Dev (ms)": std_latencies
    })

    # Save to CSV (which Excel opens natively)
    df.to_csv("retrieval_benchmark.csv", index=False)
    print("\nData saved as 'retrieval_benchmark.csv'. Open this file directly in Excel!")

if __name__ == "__main__":
    benchmark_pipeline()