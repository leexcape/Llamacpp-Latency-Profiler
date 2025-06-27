import argparse
import torch
import time
import json
import platform
import subprocess
import os
from datetime import datetime
from llama_cpp import Llama
import numpy as np

n_runs = 30

def get_hardware_info(device):
    info = {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform()
    }

    # Try to get board model (for Raspberry Pi, Jetson, etc.)
    model_path = "/proc/device-tree/model"
    if os.path.isfile(model_path):
        try:
            with open(model_path) as f:
                info["hardware_model"] = f.read().strip()
        except Exception:
            info["hardware_model"] = "unreadable"

    # Try to get Jetson version info
    jetson_release = "/etc/nv_tegra_release"
    if os.path.isfile(jetson_release):
        try:
            with open(jetson_release) as f:
                info["jetson_info"] = f.read().strip()
        except Exception:
            info["jetson_info"] = "unreadable"

    # If CUDA is available, add device info
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            info.update({
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_MB": props.total_memory // (1024 * 1024),
            })
        except Exception:
            info["cuda_error"] = "Unable to get torch.cuda properties"

        # Try to run nvidia-smi
        try:
            smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"], encoding="utf-8")
            info["nvidia_smi"] = smi_output.strip()
        except Exception:
            info["nvidia_smi"] = "nvidia-smi not available or failed"

    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/lxc/MyWorkspace/models/llama-3.2-1b-q4_k_m.gguf", help="gguf model path")
    parser.add_argument("--prompt", type=str, default="Give me some suggestions to advance my python skill.", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type (e.g., float32, float16, bfloat16)")
    args = parser.parse_args()

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    llm = Llama(model_path=args.model, n_threads=4, n_gpu_layer=0)
    print("Model loaded, ctx=", llm.n_ctx())

    generated_text_rec = []
    elapsed_rec = []
    tokens_per_sec_rec = []
    for i in range(n_runs):
        # Decode
        start_time = time.time()
        generated_text = llm(args.prompt, max_tokens=args.max_new_tokens, temperature=0)
        generated_text_rec.append(generated_text)
        end_time = time.time()

        # Stats
        elapsed_rec.append(end_time - start_time)
        tokens_generated = generated_text["usage"]["completion_tokens"]
        tokens_per_sec_rec.append(tokens_generated / (end_time - start_time))
        print(f"{i + 1}-th Inference done, average tokens per second is:{tokens_generated / (end_time - start_time)}")

    # Collect metadata
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_name": args.model,
        "prompt": args.prompt,
        "generated_text": generated_text_rec,
        "response_length": args.max_new_tokens,
        "elapsed_time_sec": elapsed_rec,
        "tokens_per_second": tokens_per_sec_rec,
        "dtype": args.dtype,
        "device_info": get_hardware_info(device)
    }

    # Save JSON
    out_name = f"results/llm_benchmark_{results['timestamp']}.json"
    with open(out_name, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmark saved to {out_name}")
    print(f"The average generation speed: {np.array(list(tokens_per_sec_rec)).mean()} tokens/second")


if __name__ == "__main__":
    main()
