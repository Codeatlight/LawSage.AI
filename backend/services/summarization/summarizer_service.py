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
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "nsi319/legal-pegasus",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=not torch.cuda.is_available()
)
model = model.to(device)
model.eval()  # Set to evaluation mode for faster inference

def chunk_text(text, tokenizer, max_length=1024):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for start in range(0, len(tokens), max_length - 200):
        chunk_tokens = tokens[start:start + max_length]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
    
    return chunks

def summarize_long_text(text, tokenizer, model, max_chunk_length=1024):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    chunks = chunk_text(text, tokenizer, max_chunk_length)
    
    chunk_summaries = []
    # Use greedy decoding (num_beams=1) for fastest inference, or beam search if quality is priority
    # For CPU, use greedy; for GPU, can use small beam search
    num_beams = 1 if not torch.cuda.is_available() else 3
    
    for chunk in chunks:
        input_tokenized = tokenizer.encode(chunk, return_tensors='pt', 
                                           max_length=max_chunk_length,
                                           truncation=True).to(device)
        
        with torch.no_grad():  # Disable gradient computation for faster inference
            summary_ids = model.generate(input_tokenized,
                                         num_beams=num_beams,
                                         no_repeat_ngram_size=3,
                                         length_penalty=2.0,
                                         min_length=50,
                                         max_length=300,
                                         early_stopping=True,  # Enable early stopping for speed
                                         do_sample=False,  # Greedy decoding is faster
                                         use_cache=True)  # Enable KV cache
        
        summary = tokenizer.decode(summary_ids[0], 
                                   skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False)
        chunk_summaries.append(summary)
    
    combined_summary = " ".join(chunk_summaries)
    final_input = tokenizer.encode(combined_summary, 
                                   return_tensors='pt', 
                                   max_length=max_chunk_length,
                                   truncation=True).to(device)
    
    with torch.no_grad():  # Disable gradient computation for faster inference
        final_summary_ids = model.generate(final_input,
                                           num_beams=num_beams,
                                           no_repeat_ngram_size=3,
                                           length_penalty=2.0,
                                           min_length=100,
                                           max_length=500,
                                           early_stopping=True,  # Enable early stopping for speed
                                           do_sample=False,  # Greedy decoding is faster
                                           use_cache=True)  # Enable KV cache
    
    final_summary = tokenizer.decode(final_summary_ids[0], 
                                     skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False)
    
    return final_summary

def summarize_text(text, min_length=150, max_length=250):
    if not text or len(text.strip()) == 0:
        print("Input text is empty or only whitespace.")
        return "Input text is empty or only whitespace."
    
    return summarize_long_text(text, tokenizer, model)

