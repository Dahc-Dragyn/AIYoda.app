# faiss_profiler.py

import argparse
import faiss
import os
import json
import numpy as np
import glob
import time
import psutil # For memory usage

# [2025-03-17] RULE: When we code AI apps allways use the cheapest gemini model.
# currently that is gemini-2.0-flash-lite-001.

def print_section_header(title):
    """Prints a formatted section header."""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

def print_info(label, value, indent=0):
    """Prints a formatted info line."""
    prefix = " " * indent
    print(f"{prefix}{label:<30}: {value}")

# --- BENCHMARKING HELPER FUNCTIONS ---

def generate_random_queries(d, nq, seed=None):
    """Generates nq random query vectors of dimension d."""
    if seed is not None:
        np.random.seed(seed)
    queries = np.random.random((nq, d)).astype('float32')
    return queries

def benchmark_cpu_search(index, queries, k, num_threads_list):
    """
    Benchmarks CPU search performance for a list of thread counts.
    Reports QPS and peak memory usage.
    """
    print_info("CPU Search Benchmark", f"(k={k}, {queries.shape[0]} queries)")
    
    original_num_threads = faiss.omp_get_max_threads() 
    process = psutil.Process(os.getpid())

    for num_threads in num_threads_list:
        print_info(f"Threads: {num_threads}", "", indent=2)
        faiss.omp_set_num_threads(num_threads)
        
        if queries.shape[0] > 0 :
             _, _ = index.search(queries[:min(10, queries.shape[0])], k) 

        start_time = time.perf_counter()
        if queries.shape[0] > 0 :
            distances, labels = index.search(queries, k)
        else: 
            distances, labels = np.array([]), np.array([])
            print_info("Search Time", "N/A (0 queries)", indent=4)
            print_info("QPS", "N/A (0 queries)", indent=4)
            print_info("Peak Process Memory (RSS)", f"{process.memory_info().rss / (1024**2):.2f} MB", indent=4)
            print_info("FAISS OMP Threads Active", faiss.omp_get_max_threads(), indent=4)
            continue 

        end_time = time.perf_counter()
        
        search_duration_sec = end_time - start_time
        qps = queries.shape[0] / search_duration_sec if search_duration_sec > 0 else 0
        
        mem_after = process.memory_info().rss

        print_info("Search Time", f"{search_duration_sec * 1000:.2f} ms (for {queries.shape[0]} queries)", indent=4)
        if queries.shape[0] > 0:
            print_info("Avg. Query Time", f"{(search_duration_sec / queries.shape[0]) * 1000:.4f} ms", indent=4)
        else:
            print_info("Avg. Query Time", "N/A (0 queries)", indent=4)
        print_info("QPS (Queries Per Second)", f"{qps:.2f}", indent=4)
        print_info("Peak Process Memory (RSS)", f"{mem_after / (1024**2):.2f} MB", indent=4)
        print_info("FAISS OMP Threads Active", faiss.omp_get_max_threads(), indent=4)

    faiss.omp_set_num_threads(original_num_threads)

def benchmark_gpu_search(cpu_index, queries, k, gpu_ids_list):
    """
    Attempts to benchmark search on specified GPU(s).
    """
    print_info("GPU Search Benchmark", f"(k={k}, {queries.shape[0]} queries)")

    if queries.shape[0] == 0:
        print_info("Status", "Skipped (0 queries to benchmark).", indent=2)
        return

    for gpu_id in gpu_ids_list:
        print_info(f"GPU ID: {gpu_id}", "", indent=2)
        gpu_index = None
        res = None
        try:
            res = faiss.StandardGpuResources()
            print_info("Status", "Attempting to move index to GPU...", indent=4)
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            print_info("Status", "Index successfully moved to GPU.", indent=4)
            
            try:
                if hasattr(res, 'getMemoryUsage'): 
                     print_info("FAISS GPU Resources Memory", f"{res.getMemoryUsage() / (1024**2):.2f} MB", indent=4)
            except AttributeError: pass 

            _, _ = gpu_index.search(queries[:min(10, queries.shape[0])], k)

            start_time = time.perf_counter()
            distances, labels = gpu_index.search(queries, k)
            end_time = time.perf_counter()

            search_duration_sec = end_time - start_time
            qps = queries.shape[0] / search_duration_sec if search_duration_sec > 0 else 0

            print_info("Search Time", f"{search_duration_sec * 1000:.2f} ms", indent=4)
            print_info("Avg. Query Time", f"{(search_duration_sec / queries.shape[0]) * 1000:.4f} ms", indent=4)
            print_info("QPS (Queries Per Second)", f"{qps:.2f}", indent=4)

        except AttributeError as ae: 
            print_info("Error", f"GPU operations failed. Is faiss-gpu installed? (Details: {ae})", indent=4)
        except Exception as e: 
            print_info("Error", f"Failed to benchmark on GPU {gpu_id}: {e}", indent=4)
        finally:
            if gpu_index: del gpu_index
            if res: del res

# --- MAIN BENCHMARK ORCHESTRATOR ---

def perform_benchmarks(index, args):
    """
    Main function to orchestrate different benchmarks based on args.
    """
    print_section_header("Performance & Memory Benchmarks")

    if index.d <= 0:
        print_info("Warning", "Index dimensionality is 0 or invalid. Skipping benchmarks.")
        return
    if args.bm_nq <= 0 and not (args.bm_query_file and os.path.exists(args.bm_query_file)):
        print_info("Info", "Number of benchmark queries (--bm-nq) is <= 0 and no query file given. Search benchmarks will be limited or skipped.")
    
    queries = None
    query_source_printed = False
    if args.bm_query_file and os.path.exists(args.bm_query_file):
        try:
            print_info("Query Source", f"Loading from file: {args.bm_query_file}", indent=2)
            query_source_printed = True
            queries_loaded = np.load(args.bm_query_file)
            if not isinstance(queries_loaded, np.ndarray) or queries_loaded.ndim != 2 or queries_loaded.shape[1] != index.d:
                raise ValueError(f"Query file must contain a 2D NumPy array of shape (nq, {index.d}). Found shape {queries_loaded.shape}.")
            
            if args.bm_nq > 0 and queries_loaded.shape[0] != args.bm_nq:
                print_info("Note", f"Query file has {queries_loaded.shape[0]} queries. Overriding --bm-nq ({args.bm_nq}) with this count.", indent=4)
            
            args.bm_nq = queries_loaded.shape[0] 
            queries = queries_loaded

            if queries.dtype != np.float32:
                print_info("Note", "Converting query data to float32.", indent=4)
                queries = queries.astype(np.float32)
        except Exception as e:
            print_info("Error loading query file", str(e), indent=2)
            if args.bm_nq > 0:
                if not query_source_printed: print_info("Query Source", "", indent=2); query_source_printed = True # Ensure header is printed once
                print_info("Generation", f"Falling back to {args.bm_nq} auto-generated random queries.", indent=4)
                queries = generate_random_queries(index.d, args.bm_nq)
                print_info("Status", f"Generated {args.bm_nq} random queries (dim={index.d}).", indent=6)
            else:
                 if not query_source_printed: print_info("Query Source", "", indent=2); query_source_printed = True
                 print_info("Status", "No queries loaded or generated (bm_nq <= 0 or file error).", indent=4)
                 queries = np.empty((0, index.d), dtype=np.float32)
    elif args.bm_nq > 0 :
        print_info("Query Source", f"Auto-generating {args.bm_nq} random queries.", indent=2)
        query_source_printed = True
        queries = generate_random_queries(index.d, args.bm_nq)
        print_info("Status", f"Generated {args.bm_nq} random queries (dim={index.d}).", indent=4)
    else: 
        print_info("Query Source", "0 queries specified (--bm-nq) and no query file. Search benchmarks will be limited.", indent=2)
        query_source_printed = True
        queries = np.empty((0, index.d), dtype=np.float32)

    if args.bm_threads:
        try:
            cpu_threads_list = [int(t.strip()) for t in args.bm_threads.split(',') if t.strip().isdigit() and int(t.strip()) > 0]
            if not cpu_threads_list:
                cpu_threads_list = [1] 
                print_info("Warning", "No valid CPU threads in --bm-threads, defaulting to 1 thread.", indent=2)
        except ValueError:
            cpu_threads_list = [1]
            print_info("Warning", "Invalid format in --bm-threads, defaulting to 1 thread.", indent=2)
        
        benchmark_cpu_search(index, queries, args.bm_k, cpu_threads_list)

    num_gpus_detected = 0
    try:
        if hasattr(faiss, 'get_num_gpus'): num_gpus_detected = faiss.get_num_gpus()
    except AttributeError: pass
    except Exception as e: print_info("GPU Check Error", str(e), indent=0)

    if num_gpus_detected > 0 and hasattr(faiss, 'StandardGpuResources'):
        try:
            gpu_ids_to_test = [int(g.strip()) for g in args.bm_gpu_ids.split(',') if g.strip().isdigit()]
            if not gpu_ids_to_test: gpu_ids_to_test = [0]
        except ValueError:
            gpu_ids_to_test = [0]
            print_info("Warning", "Invalid format in --bm-gpu-ids, defaulting to GPU ID 0.", indent=2)
        
        valid_gpu_ids_to_test = [gid for gid in gpu_ids_to_test if gid < num_gpus_detected]
        if not valid_gpu_ids_to_test and gpu_ids_to_test :
             print_info("Warning", f"Specified GPU IDs ({gpu_ids_to_test}) are out of range for detected GPUs ({num_gpus_detected}). Skipping GPU benchmark.", indent=0)
        elif not valid_gpu_ids_to_test and num_gpus_detected > 0 and not gpu_ids_to_test : # Auto-default if no user input for IDs
             valid_gpu_ids_to_test = [0]
             print_info("Note", "Defaulting to GPU ID 0 for benchmark.", indent=2)


        if valid_gpu_ids_to_test:
             benchmark_gpu_search(index, queries, args.bm_k, valid_gpu_ids_to_test)
        elif num_gpus_detected > 0 : 
            print_info("GPU Search Benchmark", "(Skipped - No valid GPU IDs selected for benchmark)", indent=0)
    else:
        gpu_skip_reason = "(Skipped - "
        if not hasattr(faiss, 'StandardGpuResources'): gpu_skip_reason += "FAISS GPU extensions not found"
        elif num_gpus_detected == 0: gpu_skip_reason += "No GPUs detected by FAISS"
        else: gpu_skip_reason += "Unknown reason" # Should be covered
        gpu_skip_reason += ")"
        print_info("GPU Search Benchmark", gpu_skip_reason, indent=0)

# --- HEALTH CHECK AND TUNING SUGGESTIONS ---

def perform_index_health_check(index, metadata, faiss_index_path):
    findings = []
    if not index.is_trained:
        findings.append({"severity": "Critical", "message": "Index is NOT trained. Most index types require training."})
    if 0 < index.ntotal < 100:
        findings.append({"severity": "Info", "message": f"Index contains very few vectors (ntotal={index.ntotal}). Simpler indexes might be more efficient."})
    elif index.ntotal == 0 and index.is_trained:
         findings.append({"severity": "Info", "message": "Index is trained but contains 0 vectors. Ready for data addition."})
    elif index.ntotal == 0 and not index.is_trained:
        findings.append({"severity": "Info", "message": "Index is not trained and contains 0 vectors."})

    if hasattr(index, 'nlist') and index.nlist > 0:
        min_vec_per_list = 30 
        if index.ntotal > 0 and (index.ntotal / index.nlist < min_vec_per_list):
            avg_vec_per_list = index.ntotal / index.nlist
            findings.append({"severity": "Warning", "message": f"IVF Index: Avg vectors per list low ({avg_vec_per_list:.2f} = ntotal {index.ntotal} / nlist {index.nlist}). Consider if `nlist` is too high."})
        if index.ntotal > 0 and index.nlist > index.ntotal:
            findings.append({"severity": "Warning", "message": f"IVF Index: `nlist` ({index.nlist}) > `ntotal` ({index.ntotal}). Many lists will be empty. Reduce `nlist`."})

    if os.path.exists(faiss_index_path) and index.ntotal > 0:
        try:
            file_size_bytes = os.path.getsize(faiss_index_path)
            bytes_per_vector_on_disk = file_size_bytes / index.ntotal
            msg = f"On-disk size: ~{bytes_per_vector_on_disk:.1f} bytes/vector. (Dim={index.d}; raw float32: {index.d * 4} bytes)."
            if hasattr(index, 'pq') and hasattr(index.pq, 'code_size'): msg += f" PQ code_size: {index.pq.code_size} bytes."
            findings.append({"severity": "Info", "message": msg})
            if hasattr(index, 'pq') and bytes_per_vector_on_disk > (index.pq.code_size * 3) and index.pq.code_size > 0 :
                 findings.append({"severity": "Info", "message": f"On-disk size/vector ({bytes_per_vector_on_disk:.1f}) is >3x PQ code size ({index.pq.code_size}). May indicate overheads."})
        except Exception as e: findings.append({"severity": "Info", "message": f"Could not calc on-disk size/vector: {e}"})

    expected_dim = metadata.get("expected_dimension") or metadata.get("source_dimension")
    if expected_dim is not None:
        try:
            if int(expected_dim) != index.d:
                findings.append({"severity": "Warning", "message": f"Index dim `d`={index.d} != metadata `expected_dimension`={int(expected_dim)}."})
        except ValueError: findings.append({"severity": "Info", "message": f"Could not parse `expected_dimension` ('{expected_dim}') from metadata."})
            
    if hasattr(index, 'hnsw'):
        if index.hnsw.efConstruction < index.hnsw.M:
            findings.append({"severity": "Info", "message": f"HNSW: `efConstruction` ({index.hnsw.efConstruction}) < `M` ({index.hnsw.M}). Often `efConstruction` >= `M`."})
    return findings

def suggest_tuning_presets(index):
    suggestions = []
    is_ivf = hasattr(index, 'nlist') and index.nlist > 0
    is_hnsw = hasattr(index, 'hnsw')
    is_flat = isinstance(index, (faiss.IndexFlatL2, faiss.IndexFlatIP))
    uses_pq = hasattr(index, 'pq'); uses_sq = "ScalarQuantizer" in type(index).__name__

    suggestions.append({"level": "Header", "message": "Search-Time Parameter Tuning Suggestions:"})

    if is_ivf:
        current_nprobe = index.nprobe if hasattr(index, 'nprobe') else 'N/A'
        suggestions.append({"level": "Info", "message": f"IVF (nlist={index.nlist}): Tune `nprobe` (lists to visit). Current `index.nprobe`={current_nprobe}."})
        low_nprobe = max(1, int(index.nlist * 0.005)) if index.nlist > 200 else 1
        bal_nprobe1 = max(1, int(np.sqrt(index.nlist))); bal_nprobe2 = max(1, int(index.nlist * 0.02))
        high_nprobe = max(1, int(index.nlist * 0.1))
        suggestions.append({"level": "Preset (IVF - Low Latency)", "message": f"Try small `nprobe` (e.g., {low_nprobe}-{max(low_nprobe + 4, bal_nprobe1 // 2 )}). Start with 1."})
        suggestions.append({"level": "Preset (IVF - Balanced)", "message": f"Try `nprobe` ~sqrt(nlist) (e.g., ~{bal_nprobe1}) or 1-5% of nlist (e.g., ~{bal_nprobe2}-{max(bal_nprobe2+1, high_nprobe//2)})."})
        suggestions.append({"level": "Preset (IVF - High Recall)", "message": f"Try larger `nprobe` (e.g., {high_nprobe}-{max(high_nprobe+1, int(index.nlist*0.25))}, up to `nlist`)."})
        suggestions.append({"level": "Note", "message": "Optimal `nprobe` is data-dependent. Set `index.nprobe = value`. Test."})

    if is_hnsw:
        current_efS = index.hnsw.efSearch if hasattr(index.hnsw, 'efSearch') else 'N/A'
        efC = index.hnsw.efConstruction if hasattr(index.hnsw, 'efConstruction') else 'N/A'
        suggestions.append({"level": "Info", "message": f"HNSW: Tune `efSearch`. Build `efC`={efC}. Current `efS`={current_efS}."})
        bal_efS = efC if isinstance(efC, int) else 64; high_efS = bal_efS * 2 if isinstance(bal_efS, int) else 128
        suggestions.append({"level": "Preset (HNSW - Low Latency)", "message": f"Try smaller `efSearch` (e.g., K to {bal_efS // 2 if isinstance(bal_efS, int) else 32}). Must be >= K."})
        suggestions.append({"level": "Preset (HNSW - Balanced)", "message": f"Try `efSearch` ~`efC` or higher (e.g., {bal_efS}-{high_efS})."})
        suggestions.append({"level": "Preset (HNSW - High Recall)", "message": f"Try larger `efSearch` (e.g., {high_efS}, {high_efS*2}+)."})
        suggestions.append({"level": "Note", "message": "Set `index.hnsw.efSearch = value`. Higher is more accurate, slower."})

    if is_flat:
        suggestions.append({"level": "Info (Flat Index)", "message": "Exact search. No approximation parameters like `nprobe`/`efSearch`."})
    
    if not is_ivf and not is_hnsw and not is_flat:
        suggestions.append({"level": "Info", "message": f"Tuning for '{type(index).__name__}' depends on specifics. See FAISS docs."})

    if uses_pq or uses_sq:
        quant_type = "PQ" if uses_pq else "SQ"
        suggestions.append({"level": "Note (Quantization)", "message": f"Index uses {quant_type}. Impacts accuracy/memory (fixed at construction)."})
    
    if len(suggestions) == 1: 
        suggestions.append({"level": "Info", "message": "No specific search-time tuning presets identified."})
    return suggestions

# --- MAIN PROFILING FUNCTION ---
def profile_faiss_index(faiss_index_path):
    loaded_index_obj = None 
    initial_load_error_message = ""

    if not os.path.exists(faiss_index_path):
        print_section_header("FAISS Index Profiler")
        print_info("Profiler FAISS Version", faiss.__version__)
        print_info("Index File Path", faiss_index_path)
        print_info("Error", "Index file not found at the specified path.")
        return None

    try:
        loaded_index_obj = faiss.read_index(faiss_index_path)
        index_load_successful_for_summary = True
    except Exception as e:
        index_load_successful_for_summary = False
        initial_load_error_message = str(e)

    print_section_header("FAISS Index Profiler")
    print_info("Profiler FAISS Version", faiss.__version__)
    print_info("Index File Path", faiss_index_path)

    if index_load_successful_for_summary and loaded_index_obj:
        summary_text = (f"Type: {type(loaded_index_obj).__name__}, Dims: {loaded_index_obj.d}, "
                        f"Vectors: {loaded_index_obj.ntotal}, Trained: {loaded_index_obj.is_trained}")
        print_info("Index Summary", summary_text)
    else:
        print_info("Index Summary", "Could not load index to provide summary (see error below).")

    print_section_header("Phase 1: Basic Index Information")
    if not index_load_successful_for_summary or not loaded_index_obj:
        print_info("Status", "Index loading failed.")
        print_info("Error details", initial_load_error_message if initial_load_error_message else "Unknown loading error.")
        return None 
    
    index = loaded_index_obj 
    print_info("Status", "Index loaded successfully.")
    print_info("Index Type", type(index).__name__)
    print_info("Dimensionality (d)", index.d)
    print_info("Total Vectors (ntotal)", index.ntotal)
    print_info("Is Trained?", index.is_trained)

    print_section_header("Phase 2: Enhanced Profiling")
    metadata_file_path = faiss_index_path + ".meta.json"; metadata = {} 
    if os.path.exists(metadata_file_path):
        try:
            with open(metadata_file_path, 'r') as f: metadata = json.load(f)
            print_info("Build Metadata File", f"Loaded '{metadata_file_path}'")
            built_with_version = metadata.get("faiss_version")
            if built_with_version:
                print_info("Index Built with FAISS (meta)", built_with_version)
                # Version compatibility logic would go here if desired based on current faiss.__version__
            else: print_info("Index Built with FAISS (meta)", "Build version not specified in metadata.")
            factory_string_meta = metadata.get("factory_string") 
            if factory_string_meta: print_info("Factory String (from meta)", factory_string_meta)
            else: print_info("Factory String (from meta)", "Not specified in metadata.")
        except Exception as e:
            print_info("Build Metadata File Error", f"Could not load or parse '{metadata_file_path}': {e}")
            print_info("Index Built with FAISS (meta)", "Unknown (metadata error).")
            print_info("Factory String (from meta)", "Not readable from metadata due to file error.") 
    else:
        print_info("Build Metadata File", "Not found (expected at '.meta.json' suffix).")
        print_info("Index Built with FAISS (meta)", "Unknown (no metadata file).")
        print_info("Factory String (from meta)", "N/A (no metadata file).")

    print_info("Serialization (Load)", "Index loaded onto CPU by default.")
    try:
        if not hasattr(faiss, 'StandardGpuResources'): 
            print_info("GPU Availability", "FAISS GPU extensions not found (likely faiss-cpu build).")
        else:
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0: print_info("GPU Availability", f"{num_gpus} GPU(s) detected by FAISS.")
            else: print_info("GPU Availability", "No GPUs detected by FAISS (but GPU extensions seem present).")
    except Exception as e: print_info("GPU Availability Check Error", str(e))

    index_dir_abs = os.path.abspath(os.path.dirname(faiss_index_path))
    abs_metadata_file_path = os.path.abspath(metadata_file_path) 
    excluded_files = [os.path.abspath(faiss_index_path), abs_metadata_file_path]
    abs_detected_paths_set = set()
    for pattern in ["*.pkl", "*.npy", "*.json", "*.txt", "*.csv", "*.tsv", "*.dat"]:
        for found_file_path_abs in glob.glob(os.path.join(index_dir_abs, pattern)):
            is_excluded = False
            for excluded_path in excluded_files:
                try: 
                    if os.path.samefile(found_file_path_abs, excluded_path): is_excluded = True; break
                except FileNotFoundError: continue 
            if is_excluded: continue
            abs_detected_paths_set.add(found_file_path_abs)
    detected_aux_files_display = []
    for abs_path in sorted(list(abs_detected_paths_set)):
        try:
            rel_path = os.path.relpath(abs_path, start=index_dir_abs)
            if ".." in rel_path : rel_path = os.path.relpath(abs_path, start=os.getcwd())
        except ValueError: rel_path = abs_path 
        detected_aux_files_display.append(rel_path)
    if detected_aux_files_display:
        print_info("Detected Auxiliary Files", f"(in {index_dir_abs})")
        for f_path in detected_aux_files_display: print(f"{'':<30}  - {f_path}")
    else: print_info("Detected Auxiliary Files", "None found matching common patterns in the index directory.")

    if not metadata.get("factory_string"): 
        inferred_structure_parts = []
        index_type_name = type(index).__name__
        if "Flat" in index_type_name: inferred_structure_parts.append("Flat (Exact Search)")
        if "L2" in index_type_name: inferred_structure_parts.append("L2 Distance")
        elif "IP" in index_type_name: inferred_structure_parts.append("Inner Product Distance")
        if hasattr(index, 'nlist'):
            inferred_structure_parts.append(f"IVF (nlist={index.nlist})")
            if hasattr(index, 'quantizer'): inferred_structure_parts.append(f"IVF Quantizer: {type(index.quantizer).__name__}")
        if hasattr(index, 'pq'):
            inferred_structure_parts.append(f"PQ (M={index.pq.M}, nbits={index.pq.nbits}, dsub={index.pq.dsub}, code_size={index.pq.code_size} bytes)")
        if hasattr(index, 'hnsw'):
             inferred_structure_parts.append(f"HNSW (M={index.hnsw.M}, efConstruction={index.hnsw.efConstruction}, efSearch={index.hnsw.efSearch})")
        if "ScalarQuantizer" in index_type_name and hasattr(index, 'sq_type'): 
            sq_map = {faiss.ScalarQuantizer.SQfp16: "SQfp16", faiss.ScalarQuantizer.SQ8: "SQ8", faiss.ScalarQuantizer.SQ4: "SQ4", faiss.ScalarQuantizer.SQ6: "SQ6" }
            sq_type_str = sq_map.get(index.sq_type, "Unknown SQ type")
            inferred_structure_parts.append(f"ScalarQuantizer ({sq_type_str})")
        if not inferred_structure_parts and index_type_name: inferred_structure_parts.append(f"Base Type: {index_type_name}")
        elif not inferred_structure_parts: inferred_structure_parts.append("Could not infer detailed structure.")
        print_info("Inferred Structure", ", ".join(inferred_structure_parts) if inferred_structure_parts else "Details not inferred.")

    print_info("Training Data Notes", "")
    if not index.is_trained: print_info("Status", "Index reports it is NOT trained.", indent=2)
    if isinstance(index, (faiss.IndexFlatL2, faiss.IndexFlatIP)):
        print_info("Type", "Flat indexes do not require a separate training phase beyond adding vectors.", indent=2)
    ivf_training_note_done = False
    if hasattr(index, 'nlist'):
        nlist = index.nlist; min_train_rec, max_train_rec = nlist * 39, nlist * 256 
        print_info("IVF Specifics", f"Index has 'nlist' property (nlist={nlist}).", indent=2)
        print_info("", f"Recommended training vectors for IVF: {min_train_rec} to {max_train_rec}.", indent=4)
        if index.is_trained and index.ntotal == 0 and nlist > 0: print_info("", "Note: Index is trained but empty. IVF training needs a separate dataset.", indent=4)
        ivf_training_note_done = True
    if hasattr(index, 'pq'):
        print_info("PQ Specifics", "Index uses Product Quantization.", indent=2)
        print_info("", "PQ training typically requires thousands to tens of thousands of vectors.", indent=4)
        if not ivf_training_note_done and index.is_trained and index.ntotal == 0: print_info("", "Note: Index is trained but empty. PQ training needs a representative dataset.", indent=4)
    if not (hasattr(index, 'nlist') or hasattr(index, 'pq') or isinstance(index, (faiss.IndexFlatL2, faiss.IndexFlatIP))):
        if index.is_trained: print_info("Generic Training", "Index is trained. Specific recommendations depend on the exact index type.", indent=2)
        else: print_info("Generic Training", f"Index not trained. Consult docs for: {type(index).__name__}", indent=2)

    if hasattr(index, 'pq'):
        print_info("PQ Compression", "") 
        try:
            d = index.d; original_vec_bytes = d * 4
            print_info("Original Vector Size (est.)", f"{original_vec_bytes} bytes (assuming float32 input)", indent=2)
            compressed_vec_bytes = index.pq.code_size 
            print_info("PQ Compressed Code Size", f"{compressed_vec_bytes} bytes per vector", indent=2)
            if compressed_vec_bytes > 0:
                ratio = original_vec_bytes / compressed_vec_bytes
                print_info("Compression Ratio (vec data)", f"~{ratio:.1f}x", indent=2)
            else: print_info("Compression Ratio (vec data)", "N/A (compressed size is zero or invalid)", indent=2)
        except Exception as e: print_info("PQ Compression Error", f"Could not calculate: {e}", indent=2)
    else: print_info("PQ Compression", "Not applicable (index does not use Product Quantization).")

    if hasattr(index, 'hnsw'):
        print_info("HNSW Parameters", "")
        try:
            print_info("M (max connections/layer)", index.hnsw.M, indent=2)
            print_info("efConstruction (build quality)", index.hnsw.efConstruction, indent=2)
            print_info("efSearch (search quality)", index.hnsw.efSearch, indent=2) 
            print_info("Note", "Actual average edges/node not directly available via std API.", indent=2)
        except Exception as e: print_info("HNSW Parameters Error", f"Could not retrieve: {e}", indent=2)
    else: print_info("HNSW Parameters", "Not applicable (index does not use HNSW structure).")

    print_section_header("Index Health Check")
    health_findings = perform_index_health_check(index, metadata, faiss_index_path)
    if health_findings:
        for finding in health_findings: print_info(f"{finding['severity']}", finding['message'])
    else: print_info("Status", "No specific health concerns flagged by current checks.")
    
    print_section_header("Tuning Parameter Suggestions")
    tuning_suggestions = suggest_tuning_presets(index) 
    if tuning_suggestions:
        for suggestion in tuning_suggestions:
            if suggestion['level'] == "Header": print_info(suggestion['message'], "")
            else: print_info(f"{suggestion['level']}", suggestion['message'])
    else: print_info("Status", "No tuning suggestions available for this index type.")

    return loaded_index_obj

# --- SCRIPT ENTRY POINT ---
def main():
    parser = argparse.ArgumentParser(
        description="FAISS Index Profiler: Analyzes a FAISS index and provides detailed information.\nCan also perform optional performance benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("faiss_index", help="Path to the FAISS index file (e.g., my_index.index or my_index.faiss)")
    
    benchmark_group = parser.add_argument_group('Benchmark Options')
    benchmark_group.add_argument("--enable-benchmarks", action="store_true", help="Enable performance and memory benchmarks.")
    benchmark_group.add_argument("--bm-k", type=int, default=10, help="K for nearest neighbor search benchmarks (default: 10).")
    benchmark_group.add_argument("--bm-nq", type=int, default=1000, help="Number of queries for benchmarks (default: 1000). Set to 0 to skip search timing if no query file.")
    benchmark_group.add_argument("--bm-threads", type=str, default="1", help="Comma-separated list of CPU threads for benchmark (e.g., \"1,4,8\", default: 1). Only positive integers.")
    benchmark_group.add_argument("--bm-query-file", type=str, default=None, help="Path to NumPy .npy query file (nq x d, float32). Default: auto-generate random queries.")
    benchmark_group.add_argument("--bm-gpu-ids", type=str, default="0", help="Comma-separated list of GPU IDs for benchmark (e.g., \"0,1\", default: 0 if GPU is used).")

    args = parser.parse_args()
    
    loaded_index = profile_faiss_index(args.faiss_index)

    if args.enable_benchmarks:
        if loaded_index:
            perform_benchmarks(loaded_index, args)
        else:
            print_section_header("Performance & Memory Benchmarks") 
            print_info("Benchmarks Skipped", "Index could not be loaded for profiling.")
            
if __name__ == "__main__":
    main()