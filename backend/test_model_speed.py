"""
Test script to compare response times of different chat models.
Run this to find the fastest model for your system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fast_chat_model import (
    MODEL_OPTIONS, 
    load_model, 
    test_response_time,
    get_model_info
)
import time

def test_all_models():
    """Test all available models and compare their response times."""
    test_prompt = "What are the fundamental rights in the Indian Constitution?"
    
    print("=" * 80)
    print("CHAT MODEL SPEED COMPARISON TEST")
    print("=" * 80)
    print(f"\nTest Prompt: {test_prompt}\n")
    
    results = []
    
    for model_key, config in MODEL_OPTIONS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")
        
        try:
            # Load model
            print("Loading model...")
            load_start = time.time()
            success = load_model(model_key)
            load_time = time.time() - load_start
            
            if not success:
                print(f"❌ Failed to load model: {model_key}")
                results.append({
                    "model": model_key,
                    "status": "failed",
                    "load_time": None,
                    "response_time": None
                })
                continue
            
            print(f"✓ Model loaded in {load_time:.2f} seconds")
            
            # Test response time
            print("Generating response...")
            result = test_response_time(test_prompt, model_key)
            
            print(f"✓ Response generated in {result['response_time']} seconds")
            print(f"  Response length: {result['response_length']} characters")
            print(f"  Response preview: {result['response'][:100]}...")
            
            results.append({
                "model": model_key,
                "model_name": config['name'],
                "status": "success",
                "load_time": round(load_time, 2),
                "response_time": result['response_time'],
                "response_length": result['response_length']
            })
            
        except Exception as e:
            print(f"❌ Error testing {model_key}: {e}")
            results.append({
                "model": model_key,
                "status": "error",
                "error": str(e)
            })
        
        # Clear memory between tests
        import gc
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        time.sleep(2)  # Brief pause between tests
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - RESPONSE TIME COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Load Time (s)':<15} {'Response Time (s)':<18} {'Status':<10}")
    print("-" * 80)
    
    successful_results = [r for r in results if r.get('status') == 'success']
    successful_results.sort(key=lambda x: x.get('response_time', float('inf')))
    
    for result in results:
        if result.get('status') == 'success':
            print(f"{result['model']:<20} {result['load_time']:<15} {result['response_time']:<18} ✓")
        else:
            print(f"{result['model']:<20} {'N/A':<15} {'N/A':<18} {result.get('status', 'unknown')}")
    
    if successful_results:
        fastest = successful_results[0]
        print(f"\n🏆 FASTEST MODEL: {fastest['model']} ({fastest['model_name']})")
        print(f"   Response Time: {fastest['response_time']} seconds")
        print(f"\n💡 Recommendation: Use '{fastest['model']}' for fastest responses")
        print(f"   Set environment variable: CHAT_MODEL={fastest['model']}")
    
    return results


if __name__ == "__main__":
    print("\nStarting model speed comparison...")
    print("This may take several minutes as we test each model.\n")
    
    results = test_all_models()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)



