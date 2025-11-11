# FAISS Index Profiler & Benchmark Tool

## Summary/Description

The FAISS Index Profiler is a Python script designed to provide a comprehensive analysis of FAISS (Facebook AI Similarity Search) index files. It goes beyond basic properties by offering detailed insights into the index structure, potential health issues, tuning suggestions for search parameters, and optional performance benchmarks for both CPU (multi-threaded) and GPU (if available and `faiss-gpu` is installed).

This tool helps users understand, diagnose, and optimize their FAISS indexes, whether they created them or received them from others.

## Why use this tool?

* **Understand Index Internals:** Decode the structure, parameters, and metadata of any FAISS index file.
* **Diagnose Issues:** Identify potential misconfigurations, inefficiencies, or areas for improvement with the "Index Health Check."
* **Get Tuning Advice:** Receive suggestions for search-time parameters (like `nprobe` for IVF or `efSearch` for HNSW) to balance speed and recall based on common presets.
* **Evaluate Performance:** Measure CPU search performance (QPS, latency) across different thread counts and estimate the process memory footprint.
* **Assess GPU Viability:** Check if an index is compatible with GPU acceleration and benchmark its performance (requires `faiss-gpu`).
* **Ensure Compatibility:** Check build versions (if metadata is provided) against your current FAISS environment.

## When to use this tool?

* When you receive a FAISS index from a colleague or an external source and need to understand its characteristics.
* Before deploying a newly created FAISS index to production, to verify its configuration and baseline its performance.
* When troubleshooting an existing index that isn't performing as expected (e.g., slow search, low recall, high memory usage).
* When trying to optimize search parameters for a specific use case (e.g., prioritizing low latency or high recall).
* To compare and contrast different index configurations or the impact of different FAISS versions.
* As a learning tool to better understand how different FAISS index parameters affect its behavior and performance.

## Features

* **Comprehensive Static Profiling:**
    * Basic information: Index type, dimensionality, total vectors, training status.
    * Build information: FAISS version used for index creation (from metadata), factory string (from metadata).
    * Serialization details and CPU/GPU load status.
    * Detection of auxiliary/metadata files (e.g., `.pkl`, `.npy`) in the index directory.
    * Inferred index structure if factory string is not available.
    * Training data notes and recommendations based on index type (IVF, PQ).
    * PQ compression ratio calculation.
    * HNSW graph parameter reporting.
* **Index Health Check:**
    * Flags potential issues like insufficient training data relative to `nlist` (for IVF), untrained indexes, very low vector counts, and mismatches with expected dimensions (if metadata is provided).
    * Reports on-disk size per vector.
* **Tuning Parameter Suggestions:**
    * Provides presets and advice for tuning search-time parameters like `nprobe` (for IVF indexes) and `efSearch` (for HNSW indexes) for different goals (low-latency, balanced, high-recall).
* **CPU Performance Benchmarks (Optional):**
    * Measures search latency and Queries Per Second (QPS).
    * Supports multi-threaded benchmarks to evaluate scalability.
    * Reports peak process memory (RSS) during the benchmark.
    * Allows using auto-generated random queries or a user-provided query file.
* **GPU Compatibility & Benchmarks (Optional):**
    * Checks for GPU availability and `faiss-gpu` extensions.
    * Attempts to move the index to specified GPU(s).
    * If successful, benchmarks search performance on the GPU.
    * Gracefully handles errors if GPU transfer or search fails.

## Requirements

* Python 3.x
* FAISS:
    * `faiss-cpu` for CPU-only profiling and benchmarks.
    * `faiss-gpu` for GPU-related features (availability check, GPU benchmarks).
* NumPy: `numpy` (usually a FAISS dependency)
* psutil: `psutil` (for memory usage reporting in benchmarks)

You can install the necessary packages using pip:
```bash
# For CPU-only:
pip install faiss-cpu numpy psutil

# Or, for GPU support (requires NVIDIA drivers, CUDA toolkit compatible with the faiss-gpu wheel):
pip install faiss-gpu numpy psutil

How To Use:
python3 faiss_profiler.py <path_to_your_faiss_index_file>
Example: python3 faiss_profiler.py my_indexes/my_ivfpq_index.faiss

Enabling Benchmarks
To run performance and memory benchmarks, use the --enable-benchmarks flag. Several options allow you to configure the benchmarks:
python3 faiss_profiler.py <index_file> --enable-benchmarks [benchmark_options]
Benchmark Options:

--bm-k <K>: Number of nearest neighbors to search for during benchmarks. (Default: 10)
--bm-nq <NQ>: Number of query vectors to use for benchmarks. If --bm-query-file is not provided, this many random queries will be generated. If a query file is provided, this argument can be overridden by the number of queries in the file. (Default: 1000)
--bm-threads "<T1,T2,...>": Comma-separated string of CPU thread counts to use for the CPU benchmark. Example: "1,4,8". (Default: "1")
--bm-query-file <path.npy>: Path to a .npy file containing query vectors (should be a 2D float32 NumPy array of shape (nq, d) where d is the index dimension). If provided, these queries are used instead of random ones.
--bm-gpu-ids "<ID1,ID2,...>": Comma-separated string of GPU IDs to use for GPU benchmarks. Example: "0" or "0,1". (Default: "0")

Example:
Run CPU benchmark with default settings (1000 queries, k=10, 1 thread):
python3 faiss_profiler.py my_index.faiss --enable-benchmarks

Run CPU benchmark with 5000 random queries, k=5, and on 1, 2, and 4 CPU threads:
python3 faiss_profiler.py my_index.faiss --enable-benchmarks --bm-nq 5000 --bm-k 5 --bm-threads "1,2,4"

Run benchmarks using queries from a file and attempt GPU benchmark on GPU 0:
python3 faiss_profiler.py my_index.faiss --enable-benchmarks --bm-query-file queries.npy --bm-gpu-ids "0"

HELP!:
python3 faiss_profiler.py -h