#!/usr/bin/env python3
"""
Llamacpp-Latency-Profiler: Minimal benchmarking tool for llama.cpp models.

Optimized for low-overhead latency measurement on resource-constrained devices
like Raspberry Pi. All non-essential monitoring removed to maximize inference speed.
"""

import argparse
import time
import json
import platform
import os
import sys
import glob
from datetime import datetime, timezone
from pathlib import Path

from llama_cpp import Llama


def discover_models(args):
    """Discover model(s) to benchmark based on arguments."""
    if args.model_dir:
        # Find all GGUF files in the directory
        pattern = os.path.join(args.model_dir, "*.gguf")
        models = sorted(glob.glob(pattern))
        if not models:
            print(f"Error: No .gguf files found in: {args.model_dir}")
            sys.exit(1)
        return models
    else:
        # Single model mode
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        return [args.model]


def get_hardware_info():
    """Collect basic device metadata (called once, before benchmark)."""
    info = {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get board model (for Raspberry Pi, etc.)
    model_path = "/proc/device-tree/model"
    if os.path.isfile(model_path):
        try:
            with open(model_path) as f:
                info["hardware_model"] = f.read().strip().rstrip('\x00')
        except Exception:
            pass

    return info


def get_model_info(model_path):
    """Extract basic model file metadata."""
    info = {
        "filename": os.path.basename(model_path),
    }
    try:
        info["size_gb"] = round(os.stat(model_path).st_size / (1024 * 1024 * 1024), 3)
    except Exception:
        pass
    return info


def calculate_statistics(values):
    """Calculate statistics using pure Python (no numpy overhead)."""
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "median": round(sorted_vals[n // 2], 4),
    }


def benchmark_single_model(model_path, args, output_dir, device_info, model_index, total_models):
    """Benchmark a single model and return results."""
    # Get timestamps
    start_datetime = datetime.now(timezone.utc)
    timestamp_str = start_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    model_info = get_model_info(model_path)

    # Load model
    print(f"\n[{model_index}/{total_models}] Loading model: {model_info['filename']}")

    load_start = time.time()
    llm = Llama(
        model_path=model_path,
        n_threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        verbose=False
    )
    load_time = time.time() - load_start

    print(f"[{model_index}/{total_models}] Model loaded in {load_time:.2f}s")

    # Warmup runs (silent, minimal overhead)
    if args.warmup > 0:
        print(f"[{model_index}/{total_models}] Warmup ({args.warmup} runs)...")
        for _ in range(args.warmup):
            llm(args.prompt, max_tokens=min(args.max_tokens, 50), temperature=0)

    # Benchmark runs - MINIMAL OVERHEAD LOOP
    print(f"[{model_index}/{total_models}] Benchmarking ({args.runs} runs)...")

    elapsed_times = []
    tokens_per_sec_list = []
    completion_tokens_list = []

    benchmark_start = time.time()

    for _ in range(args.runs):
        # Tight timing loop - only measure inference
        start_time = time.time()
        output = llm(args.prompt, max_tokens=args.max_tokens, temperature=0)
        elapsed = time.time() - start_time

        # Extract metrics (minimal post-processing)
        completion_tokens = output["usage"]["completion_tokens"]
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        elapsed_times.append(elapsed)
        tokens_per_sec_list.append(tokens_per_sec)
        completion_tokens_list.append(completion_tokens)

    benchmark_duration = time.time() - benchmark_start

    # Calculate statistics (after all runs complete)
    stats = {
        "elapsed_time_sec": calculate_statistics(elapsed_times),
        "tokens_per_second": calculate_statistics(tokens_per_sec_list),
        "completion_tokens": calculate_statistics(completion_tokens_list),
    }

    # Build results structure
    results = {
        "meta": {
            "timestamp": timestamp_str,
            "profiler_version": "2.1.0-minimal",
        },
        "config": {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "threads": args.threads,
            "n_gpu_layers": args.n_gpu_layers,
            "ctx_size": args.ctx_size,
            "warmup_runs": args.warmup,
            "benchmark_runs": args.runs,
        },
        "model_info": model_info,
        "device_info": device_info,
        "timing": {
            "model_load_sec": round(load_time, 4),
            "total_benchmark_sec": round(benchmark_duration, 4),
        },
        "statistics": stats,
        "runs": [{"elapsed_sec": round(t, 4), "tokens_per_sec": round(tps, 4)}
                 for t, tps in zip(elapsed_times, tokens_per_sec_list)],
    }

    # Save results
    model_name = Path(model_path).stem
    out_filename = f"benchmark_{model_name}_{timestamp_str}.json"
    out_path = output_dir / out_filename

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"[{model_index}/{total_models}] Results: {stats['tokens_per_second']['mean']:.2f} tok/s avg "
          f"(min: {stats['tokens_per_second']['min']:.2f}, max: {stats['tokens_per_second']['max']:.2f})")
    print(f"[{model_index}/{total_models}] Saved: {out_path}")

    # Clean up to free memory before loading next model
    del llm

    return results


def run_benchmark(args):
    """Run the benchmark with minimal overhead for accurate latency measurement."""
    script_dir = Path(__file__).parent.resolve()

    # Discover models to benchmark
    model_paths = discover_models(args)
    total_models = len(model_paths)

    print(f"Discovered {total_models} model(s) to benchmark")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect system info once before benchmark
    device_info = get_hardware_info()

    # Benchmark each model
    all_results = []
    for i, model_path in enumerate(model_paths, 1):
        result = benchmark_single_model(
            model_path, args, output_dir, device_info, i, total_models
        )
        all_results.append(result)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Completed benchmarking {total_models} model(s)")
    print(f"{'='*60}")
    for result in all_results:
        model_name = result["model_info"]["filename"]
        tps = result["statistics"]["tokens_per_second"]["mean"]
        print(f"  {model_name}: {tps:.2f} tok/s")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Llamacpp-Latency-Profiler: Minimal overhead benchmark for llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model model.gguf
  %(prog)s --model model.gguf --runs 10 --warmup 2
  %(prog)s --model-dir /path/to/models --threads 4
        """
    )

    # Model selection (mutually exclusive: single model or directory)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", type=str,
                             help="Path to a single GGUF model file")
    model_group.add_argument("--model-dir", "-d", type=str,
                             help="Path to directory containing GGUF models (benchmarks all)")
    parser.add_argument("--prompt", "-p", type=str,
                        default="Explain the concept of machine learning in simple terms.",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", "-t", type=int, default=200,
                        help="Maximum tokens to generate per run (default: 200)")
    parser.add_argument("--ctx-size", "-c", type=int, default=2048,
                        help="Context size (default: 2048)")

    # Hardware options
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of CPU threads (default: 4)")
    parser.add_argument("--n-gpu-layers", type=int, default=0,
                        help="Number of layers to offload to GPU (default: 0)")

    # Benchmark options
    parser.add_argument("--runs", "-r", type=int, default=10,
                        help="Number of benchmark runs (default: 10)")
    parser.add_argument("--warmup", "-w", type=int, default=1,
                        help="Number of warmup runs before benchmark (default: 1)")

    # Output options
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Output directory for results (default: results)")

    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
