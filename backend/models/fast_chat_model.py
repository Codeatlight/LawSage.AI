"""
Fast Chat Model - Alternative implementation using smaller, faster models
This module provides faster response times compared to Phi-3.5-mini-instruct
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
import logging
import os
import time
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)

# Model options - ordered by speed (fastest first)
MODEL_OPTIONS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Very fast, 1.1B parameters - Best for speed",
        "max_tokens": 512,
        "recommended": True
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "description": "Ultra fast, 82M parameters - Fastest option",
        "max_tokens": 300,
        "recommended": False  # Less capable but fastest
    },
    "gpt2": {
        "name": "gpt2",
        "description": "Fast, 124M parameters - Good balance",
        "max_tokens": 400,
        "recommended": True
    },
    "phi3": {
        "name": "microsoft/Phi-3.5-mini-instruct",
        "description": "Slower but more capable, 3.8B parameters - Original model",
        "max_tokens": 1000,
        "recommended": False
    }
}

# Get model from environment variable or use default
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "tinyllama")  # Default to fastest

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Optimize CPU threading
if not torch.cuda.is_available():
    num_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    logging.info(f"CPU optimization: Using {num_threads} threads")

# Global model variables
model = None
tokenizer = None
chat_pipeline = None
current_model_name = None
model_config = None


def load_model(model_key: str = DEFAULT_MODEL) -> bool:
    """
    Load a chat model based on the model key.
    
    Args:
        model_key: Key from MODEL_OPTIONS dict
        
    Returns:
        True if model loaded successfully, False otherwise
    """
    global model, tokenizer, chat_pipeline, current_model_name, model_config
    
    if model_key not in MODEL_OPTIONS:
        logging.error(f"Unknown model key: {model_key}. Available: {list(MODEL_OPTIONS.keys())}")
        return False
    
    model_config = MODEL_OPTIONS[model_key]
    model_name = model_config["name"]
    
    # If same model already loaded, skip
    if current_model_name == model_name and model is not None:
        logging.info(f"Model {model_name} already loaded")
        return True
    
    # Clear previous model from memory
    if model is not None:
        del model
        del tokenizer
        del chat_pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    try:
        logging.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        low_cpu_mem_usage = not torch.cuda.is_available()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True
        )
        
        # Move to device if not using device_map
        if not torch.cuda.is_available() and model.device.type != device:
            model = model.to(device)
        
        model.eval()
        
        # Create pipeline
        chat_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        current_model_name = model_name
        load_time = time.time() - start_time
        logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return True
        
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def ask_legal_question_fast(prompt: str, model_key: Optional[str] = None) -> str:
    """
    Generate a legal question response using the fast model.
    
    Args:
        prompt: User's question
        model_key: Optional model key to use (defaults to DEFAULT_MODEL)
        
    Returns:
        Generated response text
    """
    global model, tokenizer, chat_pipeline, model_config, current_model_name
    
    # Load model if not loaded or different model requested
    if model_key and model_key != DEFAULT_MODEL:
        if not load_model(model_key):
            return "Error: Could not load the requested model."
    elif model is None or tokenizer is None:
        if not load_model():
            return "Error: Model not initialized. Please check logs."
    
    if model_config is None:
        model_config = MODEL_OPTIONS[DEFAULT_MODEL]
    
    # Ensure current_model_name is set
    if current_model_name is None:
        current_model_name = model_config["name"]
    
    try:
        start_time = time.time()
        
        # Legal assistant system prompt
        system_prompt = "You are LawSage AI, a helpful legal assistant specialized in Indian law. You provide clear, accurate, and helpful answers to legal questions. Always respond professionally and focus on legal matters."
        
        # Format prompt based on model type
        model_name_lower = current_model_name.lower() if current_model_name else ""
        if "tinyllama" in model_name_lower:
            # TinyLlama chat format with system message
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"<|system|>\n{system_prompt}<|user|>\n{prompt}<|assistant|>\n"
        elif "phi-3" in model_name_lower or "phi3" in model_name_lower:
            # Phi-3 format with system message
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            # GPT-2 style format with legal context
            formatted_prompt = f"You are LawSage AI, a legal assistant for Indian law.\n\nLegal Question: {prompt}\n\nAnswer as a legal expert:"
        
        # Determine max tokens
        max_new_tokens = min(model_config["max_tokens"], 300 if not torch.cuda.is_available() else model_config["max_tokens"])
        
        # Generate response
        with torch.no_grad():
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for speed
                temperature=0.7 if torch.cuda.is_available() else 0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
        
        # Decode response
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        response_time = time.time() - start_time
        logging.info(f"Response generated in {response_time:.2f} seconds")
        
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return "An error occurred while generating the response."


def get_model_info() -> Dict:
    """Get information about the currently loaded model."""
    global current_model_name, model_config
    
    if model_config is None:
        return {"status": "No model loaded"}
    
    return {
        "model_name": current_model_name,
        "description": model_config["description"],
        "max_tokens": model_config["max_tokens"],
        "device": device,
        "status": "loaded" if model is not None else "not loaded"
    }


def test_response_time(prompt: str = "What are the fundamental rights in the Indian Constitution?", 
                       model_key: Optional[str] = None) -> Dict:
    """
    Test the response time of the model.
    
    Args:
        prompt: Test prompt
        model_key: Optional model key to test
        
    Returns:
        Dictionary with timing information
    """
    if model_key:
        if not load_model(model_key):
            return {"error": f"Could not load model: {model_key}"}
    
    start_time = time.time()
    response = ask_legal_question_fast(prompt, model_key)
    total_time = time.time() - start_time
    
    return {
        "response": response,
        "response_time": round(total_time, 2),
        "model": current_model_name if current_model_name else "unknown",
        "prompt_length": len(prompt),
        "response_length": len(response)
    }


# Initialize with default model on import
if __name__ != "__main__":
    logging.info(f"Initializing fast chat model with: {DEFAULT_MODEL}")
    load_model()

