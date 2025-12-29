import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimize CPU threading for faster inference (only if not already set)
if not torch.cuda.is_available():
    try:
        num_threads = min(4, os.cpu_count() or 1)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Threading already set or cannot be changed after torch operations
        pass

tokenizer = AutoTokenizer.from_pretrained("MikaSie/LegalBERT_BART_fixed_V1")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "MikaSie/LegalBERT_BART_fixed_V1",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=not torch.cuda.is_available()
)

model.to(device)
model.eval()  # Set to evaluation mode for faster inference

def simplify_summary(text, chunk_size=1024, min_length=100, max_length=200):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]

    chunks = [input_ids[0][i:i + chunk_size] for i in range(0, input_ids.size(1), chunk_size)]
    simplified_chunks = []

    for chunk in chunks:
        chunk_inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
        with torch.no_grad():  # Disable gradient computation for faster inference
            outputs = model.generate(
                chunk_inputs["input_ids"],
                max_length=max_length,
                num_beams=1 if not torch.cuda.is_available() else 2,  # Greedy (1) for CPU, small beam for GPU
                min_length=min_length,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                do_sample=False,  # Greedy decoding is faster
                use_cache=True  # Enable KV cache
            )
        simplified_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified_chunks.append(simplified_chunk)
    
    simplified_summary = " ".join(simplified_chunks)
    return simplified_summary
