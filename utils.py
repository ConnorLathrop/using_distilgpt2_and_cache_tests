# utils.py
import os
from datetime import datetime

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_output(filename, text):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Saved output to {path}]")
    return path

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
