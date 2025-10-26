from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, pynvml
from utils import save_output, timestamp
import numpy as np

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
model.eval()

# Get GPU stats
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
def get_gpu_stats():
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu, util.memory

user_prompts = []
results = []

for i in range(4):
    user_in = input(f"Enter prompt number {i+1}: ")
    user_prompts.append(user_in)

batch_sizes = [1, 2, 4]
for batch_size in batch_sizes:
    batch_prompts = user_prompts[:batch_size]
    latencies = []
    gpu_mem_used = []
    gpu_util_pct = []

    for prompt in batch_prompts:
        torch.cuda.synchronize()
        start = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate text
        _ = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        torch.cuda.synchronize()
        end = time.time()
        lat = end - start
        latencies.append(lat)

        # GPU stats after generation
        gpu_after, mem_after = get_gpu_stats()
        gpu_util_pct.append(gpu_after)
        gpu_mem_used.append(mem_after)

    # Compute metrics
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    total_tokens = batch_size * 64
    total_time = sum(latencies)
    throughput_tokens = total_tokens / total_time
    throughput_prompts = batch_size / total_time
    avg_gpu_util = np.mean(gpu_util_pct)
    avg_mem_used = np.mean(gpu_mem_used)

    line = (
        f"Batch size={batch_size:<2} | p50={p50:.3f}s | p95={p95:.3f}s | "
        f"Total time taken: {total_time}\n "
        f"Throughput={throughput_tokens:.1f} tok/s | {throughput_prompts:.2f} prompts/s | "
        f"GPU util={avg_gpu_util:.1f}% | Mem used={avg_mem_used:.1f} %"
    )
    print(line)
    results.append(line)

save_output(f"batching_{timestamp()}.txt", "\n".join(results))
