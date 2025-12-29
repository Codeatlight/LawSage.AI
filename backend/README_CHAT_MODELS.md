# Chat Model Configuration Guide

## Overview

This application supports multiple chat models with different speed/quality trade-offs. The default model has been optimized for faster response times.

## Available Models

### 1. TinyLlama (Default - Recommended for Speed)
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: 1.1B parameters
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: Good for general legal questions
- **Best for**: Production use when speed is critical

### 2. GPT-2
- **Model**: `gpt2`
- **Size**: 124M parameters
- **Speed**: ⚡⚡⚡⚡ Ultra Fast
- **Quality**: Basic responses
- **Best for**: Testing or when response quality is less critical

### 3. DistilGPT-2
- **Model**: `distilgpt2`
- **Size**: 82M parameters
- **Speed**: ⚡⚡⚡⚡⚡ Fastest
- **Quality**: Basic responses
- **Best for**: Maximum speed requirements

### 4. Phi-3.5 (Original)
- **Model**: `microsoft/Phi-3.5-mini-instruct`
- **Size**: 3.8B parameters
- **Speed**: ⚡ Slower
- **Quality**: ⭐⭐⭐ Best quality
- **Best for**: When quality is more important than speed

## Configuration

### Option 1: Environment Variables (Recommended)

Set these environment variables before starting the Flask app:

```bash
# Use fast model (default: true)
export USE_FAST_CHAT_MODEL=true

# Choose specific model (default: tinyllama)
export CHAT_MODEL=tinyllama
```

Available model keys: `tinyllama`, `gpt2`, `distilgpt2`, `phi3`

### Option 2: Modify Code

Edit `backend/app.py` and change:
```python
use_fast_model = os.getenv('USE_FAST_CHAT_MODEL', 'true').lower() == 'true'
```

## Testing Model Speed

Run the speed comparison test:

```bash
cd backend
python test_model_speed.py
```

This will:
1. Test all available models
2. Measure response times
3. Recommend the fastest model for your system

## Switching Models

### To use the fast model (default):
```bash
export USE_FAST_CHAT_MODEL=true
export CHAT_MODEL=tinyllama
python app.py
```

### To use the original Phi-3 model:
```bash
export USE_FAST_CHAT_MODEL=false
python app.py
```

### To use a different fast model:
```bash
export USE_FAST_CHAT_MODEL=true
export CHAT_MODEL=gpt2  # or distilgpt2
python app.py
```

## Performance Comparison

Based on typical CPU performance:

| Model | Response Time (CPU) | Response Time (GPU) | Quality |
|-------|---------------------|----------------------|---------|
| DistilGPT-2 | ~1-2s | ~0.5s | Basic |
| GPT-2 | ~2-3s | ~0.8s | Good |
| TinyLlama | ~3-5s | ~1-2s | Very Good |
| Phi-3.5 | ~10-20s | ~3-5s | Excellent |

*Note: Actual times vary based on hardware and prompt complexity*

## Troubleshooting

### Model not loading
- Check internet connection (first download requires download)
- Ensure sufficient disk space (~2-5GB per model)
- Check logs in console for specific errors

### Still slow responses
- Try a smaller model (gpt2 or distilgpt2)
- Ensure you're using the fast model (`USE_FAST_CHAT_MODEL=true`)
- Check if GPU is available and being used

### Want better quality
- Switch to Phi-3.5 model (`USE_FAST_CHAT_MODEL=false`)
- Or use TinyLlama which offers good balance

## API Endpoint

Check which model is currently active:
```bash
GET /api/chat/model-info
```

Returns:
```json
{
  "model_type": "fast",
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "description": "Very fast, 1.1B parameters - Best for speed",
  "status": "loaded",
  "available_models": {...}
}
```



