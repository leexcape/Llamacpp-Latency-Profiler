# Llamacpp-Latency-Profiler

A comprehensive benchmarking and analysis tool for [llama.cpp](https://github.com/ggerganov/llama.cpp) based GGUF models. Measures generation latency, tokens per second, power consumption, and provides publication-ready analysis of speculative decoding performance across diverse edge hardware platforms.

## Features

- **Cross-Platform Profiling**: Benchmarks on CPU, GPU (CUDA), Raspberry Pi 4B/5, Jetson AGX Orin, and other ARM devices
- **Detailed Statistics**: Mean, std, min, max, median, and percentiles (p5, p25, p75, p95) for latency and throughput
- **Device Profiling**: Captures CPU frequency, temperature, memory usage, and hardware model
- **Power Monitoring**: Real-time power measurement on Raspberry Pi via INA219 sensor (`Rpi_power_monitor.py`)
- **Model Metadata**: Extracts file size, quantization type, and model information
- **Speculative Decoding Analysis**: Goodput, cost efficiency, energy efficiency, and speculative-length optimisation

## Analysis Scripts

All analysis scripts produce publication-quality figures (PDF + PNG) with a consistent visual style.

### `plot_goodput.py` — Goodput Analysis

Computes the **goodput** (accepted tokens/s) for each draft-model + device combination using the speculative decoding throughput model:

```
Goodput(K) = (K * alpha(K) + 1) / (K / draft_tps + T_verify)
```

- Loads acceptance-rate sweep data from profiling CSVs
- For each configuration, finds the **optimal speculative length K\*** that maximises goodput
- `T_VERIFY` (verification latency) is configurable at the top of the script
- **Outputs**: scatter tradeoff plots (acceptance rate vs draft speed with iso-goodput curves) and grouped bar charts, split by target model family (Llama-3.1-70B, Qwen3-32B)

### `plot_spec_length_analysis.py` — Speculative Length Analysis

Analyses the trade-off between speculative length K and throughput:

1. **Acceptance Rate Decay** (`acceptance_rate_vs_speclen`) — shows how alpha drops as K increases
2. **Tokens per Round** (`tokens_per_round_vs_speclen`) — shows sub-linear growth of K*alpha(K), with ideal reference line and wasted-token shading
3. **Goodput vs K** (`goodput_vs_speclen`) — 2x3 grid (target x device) showing goodput curves with optimal K\* marked, at a configurable T_verify
4. **Optimal K\* vs T_verify** (`optimal_speclen_vs_Tverify`) — 2x3 grid showing how the optimal speculative length shifts with verification latency
5. **Summary CSV** (`optimal_speclen_summary.csv`) — optimal K\* and goodput for representative T_verify values across all 120 configurations

### `plot_cost_efficiency.py` — Cost Efficiency Analysis

Computes **accepted tokens per dollar** using cloud API pricing (device/quant independent since draft speed cancels out):

```
tokens_per_$ = alpha * 1e6 / price_per_M_output
```

- Pricing: Fireworks (Llama-70B) @ $0.90/M output tokens, Groq (Qwen3-32B) @ $0.59/M
- **Outputs**: per-target bar charts of cost efficiency

### `plot_energy_efficiency.py` — Energy Efficiency Analysis

Computes **joules per verified token** from measured power draw:

```
J/verified_tok = power_avg_w / (draft_tps * alpha)
```

- Uses profiling CSVs with power measurements (RPi 5, Jetson AGX Orin)
- **Outputs**: grouped bar charts (J/tok) and scatter plots (goodput vs J/tok with iso-power curves)

### `Rpi_power_monitor.py` — Power Monitor

Real-time power measurement for Raspberry Pi using the INA219 current sensor over I2C.

## Data

### Profiling Results (`results/`)

| File | Description |
|------|-------------|
| `pi5_llamacpp_profile_19models.csv` | RPi 5 latency + power profiling of 19 GGUF models |
| `jetson_agx_orin_llamacpp_profile_19models.csv` | Jetson AGX Orin profiling of 19 GGUF models |
| `profile_results_2026-02-20_11-42-25.csv` | Acceptance-rate sweep (K=2..10) for Qwen3-32B target |
| `profile_results_2026-02-20_13-30-21.csv` | Acceptance-rate sweep (K=2..10) for Llama-3.1-70B target |

### Model Families

| Target Model | Draft Models | Devices |
|-------------|-------------|---------|
| Meta-Llama-3.1-70B-Instruct | Llama-3.2-1B, 1B-Instruct, 3B-Instruct, 3.1-8B-Instruct | RPi 4B, RPi 5, Jetson AGX Orin |
| Qwen3-32B | Qwen3-0.6B, 1.7B, 4B, 8B | RPi 4B, RPi 5, Jetson AGX Orin |

## Installation

### Requirements

```bash
pip install llama-cpp-python numpy pandas matplotlib
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

### Profiling

```bash
python main.py --model /path/to/model.gguf
```

Full options:

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

### Running Analysis Scripts

```bash
# Goodput analysis (uses optimal K* and new goodput model)
python plot_goodput.py

# Speculative-length trade-off analysis
python plot_spec_length_analysis.py

# Cost efficiency analysis
python plot_cost_efficiency.py

# Energy efficiency analysis
python plot_energy_efficiency.py
```

Each script writes timestamped output to `results/<analysis_name>/<timestamp>/`.

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| x86_64 Linux | Supported | Full support |
| x86_64 macOS | Supported | Full support |
| Raspberry Pi 4/5 | Supported | Temperature, CPU freq, and power monitoring |
| NVIDIA Jetson AGX Orin | Supported | Tegra info and power monitoring |
| CUDA GPUs | Supported | VRAM and nvidia-smi support |

## Tips for Accurate Benchmarking

1. **Use warmup runs**: At least 1-2 warmup runs help stabilise measurements
2. **Close other applications**: Minimise background processes for consistent results
3. **Temperature stability**: On Raspberry Pi, consider cooling; high temps cause throttling
4. **Sufficient runs**: Use 10+ runs for statistically meaningful results
5. **Consistent prompts**: Use the same prompt when comparing models

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
