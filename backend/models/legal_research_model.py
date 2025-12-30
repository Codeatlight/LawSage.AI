import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.cache_utils import DynamicCache
import gc
import logging
import re
import os

logging.basicConfig(level=logging.INFO)

# Monkey patch DynamicCache to add missing methods for compatibility
# This fixes the compatibility issue with Phi-3 model's custom code
if not hasattr(DynamicCache, 'seen_tokens'):
    def _get_seen_tokens(self):
        if hasattr(self, '_seen_tokens'):
            return self._seen_tokens
        # Calculate from cache if available
        try:
            if hasattr(self, 'key_cache') and len(self.key_cache) > 0 and len(self.key_cache[0]) > 0:
                return self.key_cache[0][0].shape[-2]
        except (AttributeError, IndexError):
            pass
        return 0
    
    DynamicCache.seen_tokens = property(_get_seen_tokens)

# Add get_max_length method if missing
if not hasattr(DynamicCache, 'get_max_length'):
    def _get_max_length(self):
        """Get the maximum length of the cache."""
        try:
            if hasattr(self, 'key_cache') and len(self.key_cache) > 0:
                # Return a large default value if cache exists
                # The actual max length depends on model config, but this should work
                return 32768  # Common max position for many models
        except (AttributeError, IndexError):
            pass
        return 0
    
    DynamicCache.get_max_length = _get_max_length

# Add get_usable_length method if missing
if not hasattr(DynamicCache, 'get_usable_length'):
    def _get_usable_length(self, seq_length, layer_idx=None):
        """Get the usable length of the cache given the current sequence length and optional layer index."""
        try:
            # If layer_idx is provided, try to get length from that specific layer
            if layer_idx is not None and hasattr(self, 'key_cache'):
                try:
                    if len(self.key_cache) > layer_idx and len(self.key_cache[layer_idx]) > 0:
                        return self.key_cache[layer_idx][0].shape[-2]
                except (AttributeError, IndexError):
                    pass
            
            # Return the number of cached tokens (seen_tokens)
            if hasattr(self, '_seen_tokens') and self._seen_tokens > 0:
                return self._seen_tokens
            # Or calculate from cache
            if hasattr(self, 'key_cache') and len(self.key_cache) > 0 and len(self.key_cache[0]) > 0:
                return self.key_cache[0][0].shape[-2]
        except (AttributeError, IndexError):
            pass
        return 0
    
    DynamicCache.get_usable_length = _get_usable_length

# Patch the update method to track seen_tokens
if hasattr(DynamicCache, 'update') and not hasattr(DynamicCache, '_patched_update'):
    original_update = DynamicCache.update
    def patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        if not hasattr(self, '_seen_tokens'):
            self._seen_tokens = 0
        if key_states is not None:
            self._seen_tokens = key_states.shape[-2]
        return result
    DynamicCache.update = patched_update
    DynamicCache._patched_update = True

# Patch __init__ to initialize _seen_tokens
if not hasattr(DynamicCache, '_patched_init'):
    original_init = DynamicCache.__init__
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, '_seen_tokens'):
            self._seen_tokens = 0
    DynamicCache.__init__ = patched_init
    DynamicCache._patched_init = True

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Optimize CPU threading for faster inference
if not torch.cuda.is_available():
    # Set number of threads for CPU inference (adjust based on your CPU cores)
    num_threads = min(4, os.cpu_count() or 1)  # Use up to 4 threads
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)  # Single inter-op thread for better performance
    except RuntimeError:
        # Interop threads can only be set before any torch operations
        pass
    logging.info(f"CPU optimization: Using {num_threads} threads")

# Optimize for CPU if CUDA not available
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
low_cpu_mem_usage = not torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    truncation=True
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        dtype=torch_dtype,  # Use dtype instead of deprecated torch_dtype
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
        attn_implementation="eager"  # Use eager attention to avoid flash-attention warning
    )
    
    # Move to device explicitly if not using device_map
    if not torch.cuda.is_available():
        model = model.to(device)
    
    # Enable optimizations
    model.eval()
    if torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead") if hasattr(torch, 'compile') else model
    
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

legal_assistant = None

try:
    if model:
        legal_assistant = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        logging.info("Pipeline created successfully.")
except Exception as e:
    logging.error(f"Error creating pipeline: {e}")

def determine_max_tokens(prompt):
    num_words = len(prompt.split())
    max_tokens = min(1000, max(150, 3 * num_words))
    return max_tokens

def is_sentence_complete(text):
    return bool(re.search(r'[.!?]["\']?\s*$', text))

def ask_legal_question(prompt):
    logging.info(f"Received prompt: {prompt}")
    
    if not model or not tokenizer:
        logging.error("Model or tokenizer is not initialized. Cannot generate response.")
        return "An error occurred while generating the response."

    try:
        max_new_tokens = determine_max_tokens(prompt)
        
        # For CPU, use fewer tokens to speed up
        if not torch.cuda.is_available():
            max_new_tokens = min(max_new_tokens, 300)
        
        # Legal assistant system prompt
        system_prompt = "You are LawSage AI, a helpful legal assistant specialized in Indian law. You provide clear, accurate, and helpful answers to legal questions. Always respond professionally and focus on legal matters."
        
        # Use chat template if available, otherwise format manually
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback format for Phi-3 instruct model with system prompt
            formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        # Optimized generation parameters for faster inference
        # Note: use_cache=False to avoid compatibility issues with DynamicCache
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            "do_sample": False,  # Greedy decoding is faster than sampling
            "repetition_penalty": 1.1,
            "use_cache": False,  # Disable cache to avoid compatibility issues with Phi-3 model's custom code
        }
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs
            )
        
        # Decode only the new tokens (remove input tokens)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        logging.info(f"Generated response: {generated_text[:100]}...")
        return generated_text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return "An error occurred while generating the response."

torch.cuda.empty_cache()
gc.collect()
