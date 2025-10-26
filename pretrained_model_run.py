from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


#from distilgpt2 readme:
MODEL_NAME = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
# simple generation with user input
prompt = input("Please provide input: ")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
