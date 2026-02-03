# Llamacpp-Latency-Profiler

A comprehensive benchmarking tool for [llama.cpp](https://github.com/ggerganov/llama.cpp) based GGUF models. Measures generation latency, tokens per second, and captures detailed device and model metadata across diverse hardware platforms.

## Features

- **Cross-Platform Support**: Works on CPU, GPU (CUDA), Raspberry Pi, Jetson, and other ARM devices
- **Detailed Statistics**: Mean, std, min, max, median, and percentiles (p5, p25, p75, p95)
- **Device Profiling**: Captures CPU frequency, temperature, memory usage, and hardware model
- **Model Metadata**: Extracts file size, quantization type, and model information
- **Warmup Runs**: Configurable warmup iterations for consistent measurements
- **Memory Tracking**: Monitors RAM usage before, during, and after benchmarking
- **Timestamped Results**: ISO 8601 timestamps with detailed per-run timing
- **JSON Output**: Structured results for easy analysis and comparison

## Installation

### Requirements

```bash
pip install llama-cpp-python numpy
```

Optional (for CUDA device info):
```bash
pip install torch
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/Llamacpp-Latency-Profiler.git
cd Llamacpp-Latency-Profiler
```

## Usage

### Basic Usage

```bash
python main.py --model /path/to/model.gguf
```

### Full Options

```bash
python main.py --model /path/to/model.gguf \
    --prompt "Your prompt here" \
    --max-tokens 200 \
    --runs 10 \
    --warmup 2 \
    --threads 4 \
    --verbose
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | *required* | Path to GGUF model file |
| `--prompt` | `-p` | "Explain the concept..." | Prompt for text generation |
| `--max-tokens` | `-t` | 200 | Maximum tokens to generate per run |
| `--ctx-size` | `-c` | 2048 | Context size |
| `--threads` | | 4 | Number of CPU threads |
| `--n-gpu-layers` | | 0 | Number of layers to offload to GPU |
| `--runs` | `-r` | 10 | Number of benchmark runs |
| `--warmup` | `-w` | 1 | Number of warmup runs |
| `--output-dir` | `-o` | results | Output directory for results |
| `--save-outputs` | | false | Save generated text in results |
| `--verbose` | `-v` | false | Enable verbose output |
| `--quiet` | `-q` | false | Minimal output (only summary) |

### Examples

**Quick benchmark with default settings:**
```bash
python main.py -m model.gguf
```

**Thorough benchmark with warmup:**
```bash
python main.py -m model.gguf -r 30 -w 3 -v
```

**GPU-accelerated benchmark:**
```bash
python main.py -m model.gguf --n-gpu-layers 35 --threads 8
```

**Custom prompt with saved outputs:**
```bash
python main.py -m model.gguf -p "Write a poem about AI" --save-outputs
```

## Output Format

Results are saved as JSON files in the `results/` directory with the naming pattern:
```
benchmark_{model_name}_{timestamp}.json
```

### JSON Structure

```json
{
  "meta": {
    "timestamp": "2024-01-15_14-30-00",
    "timestamp_iso": "2024-01-15T14:30:00+00:00",
    "end_timestamp_iso": "2024-01-15T14:35:00+00:00",
    "profiler_version": "2.0.0"
  },
  "config": {
    "model_path": "/path/to/model.gguf",
    "prompt": "...",
    "max_tokens": 200,
    "temperature": 0,
    "threads": 4,
    "n_gpu_layers": 0,
    "ctx_size": 2048,
    "warmup_runs": 1,
    "benchmark_runs": 10
  },
  "model_info": {
    "path": "/path/to/model.gguf",
    "filename": "model.gguf",
    "size_bytes": 2104932768,
    "size_mb": 2007.42,
    "size_gb": 1.96,
    "quantization": "Q4_K_M"
  },
  "device_info": {
    "hostname": "raspberrypi",
    "system": "Linux",
    "release": "6.12.47+rpt-rpi-2712",
    "machine": "aarch64",
    "processor": "",
    "platform": "Linux-6.12.47+rpt-rpi-2712-aarch64-with-glibc2.36",
    "python_version": "3.11.2",
    "cpu_count": 4,
    "cpu_freq": {"current_mhz": 2400, "max_mhz": 2400},
    "cpu_temp_celsius": 52.5,
    "hardware_model": "Raspberry Pi 5 Model B Rev 1.0"
  },
  "memory": {
    "initial": {"total_mb": 8192, "available_mb": 6000, "usage_percent": 26.76},
    "post_model_load": {"total_mb": 8192, "available_mb": 4000, "usage_percent": 51.17},
    "final": {"total_mb": 8192, "available_mb": 4100, "usage_percent": 49.95}
  },
  "timing": {
    "model_load_sec": 2.5432,
    "total_benchmark_sec": 180.5,
    "total_benchmark_human": "3m 0.5s"
  },
  "statistics": {
    "elapsed_time_sec": {
      "count": 10,
      "mean": 18.05,
      "std": 0.23,
      "min": 17.8,
      "max": 18.4,
      "median": 18.02,
      "p5": 17.82,
      "p25": 17.9,
      "p75": 18.15,
      "p95": 18.35
    },
    "tokens_per_second": {
      "count": 10,
      "mean": 11.08,
      "std": 0.14,
      "min": 10.87,
      "max": 11.24,
      "median": 11.09,
      "p5": 10.89,
      "p25": 10.98,
      "p75": 11.18,
      "p95": 11.22
    },
    "completion_tokens": {
      "count": 10,
      "mean": 200,
      "std": 0,
      "min": 200,
      "max": 200,
      "median": 200,
      "p5": 200,
      "p25": 200,
      "p75": 200,
      "p95": 200
    }
  },
  "runs": [
    {
      "run_number": 1,
      "timestamp": "2024-01-15T14:30:05+00:00",
      "elapsed_sec": 18.05,
      "prompt_tokens": 12,
      "completion_tokens": 200,
      "total_tokens": 212,
      "tokens_per_sec": 11.08,
      "cpu_temp_start": 52.0,
      "cpu_temp_end": 55.0,
      "memory_usage_percent": 51.2
    }
  ]
}
```

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| x86_64 Linux | ✅ | Full support |
| x86_64 macOS | ✅ | Full support |
| Raspberry Pi 4/5 | ✅ | Temperature and CPU freq monitoring |
| NVIDIA Jetson | ✅ | Tegra info capture |
| CUDA GPUs | ✅ | VRAM and nvidia-smi support |
| Windows | ⚠️ | Basic support (no temp monitoring) |

## Tips for Accurate Benchmarking

1. **Use warmup runs**: At least 1-2 warmup runs help stabilize measurements
2. **Close other applications**: Minimize background processes for consistent results
3. **Temperature stability**: On Raspberry Pi, consider cooling; high temps cause throttling
4. **Sufficient runs**: Use 10+ runs for statistically meaningful results
5. **Consistent prompts**: Use the same prompt when comparing models

## Analyzing Results

Results can be loaded and analyzed in Python:

```python
import json
import glob

# Load latest result
files = sorted(glob.glob("results/benchmark_*.json"))
with open(files[-1]) as f:
    data = json.load(f)

# Print summary
stats = data["statistics"]["tokens_per_second"]
print(f"Model: {data['model_info']['filename']}")
print(f"Speed: {stats['mean']:.2f} ± {stats['std']:.2f} tokens/sec")
print(f"Device: {data['device_info'].get('hardware_model', 'Unknown')}")
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
