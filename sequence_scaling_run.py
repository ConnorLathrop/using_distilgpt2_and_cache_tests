from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, pynvml
from utils import save_output, timestamp
import matplotlib.pyplot as plt

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
model.eval()

# Prompt lengths to test
seq_lengths = [32, 128, 256]
results = []

for seq_len in seq_lengths:
    # Generate a dummy prompt of repeated 'a'
    prompt = " ".join(["a"] * seq_len)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    mem_used = torch.cuda.max_memory_allocated() / (1024**2) 
    latency_per_token = total_time / seq_len

    line = f"Seq_len={seq_len:<4} | Latency per token={latency_per_token:.6f}s | GPU mem used={mem_used:.2f} MB"
    print(line)
    results.append((seq_len, latency_per_token, mem_used))

# Save results to file
text_results = "\n".join([f"{r[0]} {r[1]:.4f} {r[2]:.1f}" for r in results])
save_output(f"cache_scaling_{timestamp()}.txt", text_results)

# Plot results
seqs, latencies, mems = zip(*results)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(seqs, latencies, marker='o', color='blue')
plt.title("Latency per token vs Sequence length")
plt.xlabel("Sequence length (tokens)")
plt.ylabel("Latency per token (s)")

plt.subplot(1,2,2)
plt.plot(seqs, mems, marker='o', color='orange')
plt.title("GPU memory used vs Sequence length")
plt.xlabel("Sequence length (tokens)")
plt.ylabel("Memory used (MB)")

plt.tight_layout()
plt.savefig(f"outputs/cache_scaling_plot_{timestamp()}.png")
plt.show()
