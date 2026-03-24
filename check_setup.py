# check_setup.py
import torch
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from peft import LoraConfig

print("=== GPU ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | {props.total_memory / 1e9:.0f} GB VRAM")

print("\n=== HuggingFace ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = emb.embed_query("test")
print(f"Embedding model works: vector dim = {len(vec)}")

print("\n=== peft / LoRA ===")
cfg = LoraConfig(r=8, lora_alpha=16)
print(f"LoRA config OK: r={cfg.r}")

print("\nAll good — ready to build!")