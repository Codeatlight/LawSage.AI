# Quick Start: Fast Chat Model

## 🚀 Quick Setup (Fastest Response Times)

The application now uses a **faster model by default** (TinyLlama) which should significantly reduce response times.

### Default Configuration
- **Fast model is ENABLED by default**
- **Model**: TinyLlama (1.1B parameters - much faster than Phi-3.5)
- **Expected response time**: 3-5 seconds (vs 10-20 seconds with Phi-3.5)

## 📊 Test Model Speed

To compare all models and find the fastest for your system:

```bash
cd backend
python test_model_speed.py
```

This will:
1. Test all available models
2. Show response times for each
3. Recommend the fastest model

## ⚙️ Configuration Options

### Option 1: Use Fast Model (Default - Already Active)
No configuration needed! The fast model is already enabled.

### Option 2: Switch to Original Phi-3 Model
If you need better quality (but slower):

**Windows (PowerShell):**
```powershell
$env:USE_FAST_CHAT_MODEL="false"
python app.py
```

**Linux/Mac:**
```bash
export USE_FAST_CHAT_MODEL=false
python app.py
```

### Option 3: Choose Specific Fast Model

**Windows (PowerShell):**
```powershell
$env:USE_FAST_CHAT_MODEL="true"
$env:CHAT_MODEL="gpt2"  # Options: tinyllama, gpt2, distilgpt2
python app.py
```

**Linux/Mac:**
```bash
export USE_FAST_CHAT_MODEL=true
export CHAT_MODEL=gpt2
python app.py
```

## 📈 Expected Performance

| Model | Response Time | Quality |
|-------|---------------|---------|
| **TinyLlama** (Default) | 3-5s | ⭐⭐⭐ Good |
| GPT-2 | 2-3s | ⭐⭐ Basic |
| DistilGPT-2 | 1-2s | ⭐ Basic |
| Phi-3.5 (Original) | 10-20s | ⭐⭐⭐⭐ Excellent |

*Times are approximate and vary based on hardware*

## 🔍 Check Current Model

Visit: `http://localhost:5000/api/chat/model-info` (requires login)

Or check the console logs when starting the app.

## 💡 Recommendations

1. **For Production**: Use TinyLlama (default) - best balance of speed and quality
2. **For Maximum Speed**: Use GPT-2 or DistilGPT-2
3. **For Best Quality**: Use Phi-3.5 (original model)

## 🐛 Troubleshooting

### Still getting slow responses?
1. Make sure fast model is enabled: `USE_FAST_CHAT_MODEL=true`
2. Run the speed test: `python test_model_speed.py`
3. Try a smaller model: `CHAT_MODEL=gpt2`

### Model download issues?
- First run will download the model (~2-5GB)
- Ensure stable internet connection
- Check available disk space

### Want to switch back to original?
```bash
export USE_FAST_CHAT_MODEL=false
```

## 📝 Notes

- The fast model loads automatically on first use
- Models are cached after first download
- You can switch models without restarting (though first load takes time)
- GPU users will see even better performance



