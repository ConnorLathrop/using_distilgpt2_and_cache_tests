from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time
from utils import save_output, timestamp

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
model.eval()

prompts = []

prompt = "What I learned in boating school is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

def measure(use_cache, tokens):
    torch.cuda.empty_cache()
    start_time = time.time()
    start_mem = torch.cuda.memory_allocated() / (1024**2) 
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=tokens, use_cache=use_cache)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    end_mem = torch.cuda.memory_allocated() / (1024**2)
    return total_time, output, end_mem-start_mem

results = []
num_tokens = int(input("Enter how many tokens to use: "))
for use_cache in [True, False]:
    t,o,m = measure(use_cache, num_tokens)
    r = f"use_cache={use_cache} | total_time={t:.2f}s | tokens/sec={num_tokens/t:.1f} | mem_used={m} "
    print(o)
    print(r)
    results.append(f"tokens used: {num_tokens}\n {r}")

save_output(f"kv_cache_results_{timestamp()}.txt", "\n".join(results))
