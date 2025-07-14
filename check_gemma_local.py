#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier l'installation Gemma 3n locale
"""
import os
import sys
from pathlib import Path
import json

def check_gemma_installation():
    """Vérifie l'installation Gemma 3n"""
    print("🔍 DIAGNOSTIC GEMMA 3N LOCAL")
    print("="*50)
    
    # Vérifier le dossier models
    models_dir = Path("models/gemma-3n")
    
    print(f"📁 Checking directory: {models_dir.absolute()}")
    
    if not models_dir.exists():
        print("❌ Directory does not exist!")
        print("💡 Please ensure Gemma 3n is installed in models/gemma-3n/")
        return False
    
    print("✅ Directory exists")
    
    # Lister tous les fichiers
    print(f"\n📋 Files in {models_dir}:")
    files = list(models_dir.rglob("*"))
    
    if not files:
        print("❌ Directory is empty!")
        return False
    
    total_size = 0
    important_files = {
        'config.json': False,
        'tokenizer.json': False,
        'tokenizer_config.json': False,
        'model files': False
    }
    
    for file_path in files:
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            
            print(f"  📄 {file_path.name} ({size_mb:.1f} MB)")
            
            # Vérifier les fichiers importants
            if file_path.name == "config.json":
                important_files['config.json'] = True
            elif file_path.name == "tokenizer.json":
                important_files['tokenizer.json'] = True
            elif file_path.name == "tokenizer_config.json":
                important_files['tokenizer_config.json'] = True
            elif any(ext in file_path.name.lower() for ext in ['.safetensors', '.bin', '.pt']):
                important_files['model files'] = True
    
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"\n📊 Total size: {total_size_gb:.2f} GB")
    
    # Vérifier les fichiers essentiels
    print(f"\n🔍 Essential files check:")
    for file_type, found in important_files.items():
        status = "✅" if found else "❌"
        print(f"  {status} {file_type}")
    
    # Analyser config.json si disponible
    config_path = models_dir / "config.json"
    if config_path.exists():
        print(f"\n⚙️ Model configuration:")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"  Model type: {config.get('model_type', 'Unknown')}")
            print(f"  Architecture: {config.get('architectures', ['Unknown'])[0]}")
            print(f"  Vocab size: {config.get('vocab_size', 'Unknown')}")
            print(f"  Hidden size: {config.get('hidden_size', 'Unknown')}")
            
        except Exception as e:
            print(f"  ❌ Error reading config: {e}")
    
    return all(important_files.values())

def test_loading():
    """Test de chargement du modèle"""
    print(f"\n🧪 TESTING MODEL LOADING")
    print("="*30)
    
    try:
        print("📦 Checking dependencies...")
        
        try:
            import torch
            print(f"  ✅ PyTorch {torch.__version__}")
            print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("  ❌ PyTorch not available")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            print("  ✅ Transformers library")
        except ImportError:
            print("  ❌ Transformers not available")
            return False
        
        print(f"\n🔄 Attempting to load model...")
        model_path = "models/gemma-3n"
        
        # Test config loading
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            print(f"  ✅ Config loaded: {config.model_type}")
        except Exception as e:
            print(f"  ❌ Config loading failed: {e}")
            return False
        
        # Test tokenizer loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"  ✅ Tokenizer loaded: {len(tokenizer)} tokens")
        except Exception as e:
            print(f"  ❌ Tokenizer loading failed: {e}")
            return False
        
        # Test model loading (with caution for memory)
        print(f"  🔄 Loading model (may take time)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"  ✅ Model loaded successfully!")
            
            # Test simple generation
            print(f"  🧪 Testing generation...")
            test_input = "Medical analysis:"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"  ✅ Generation test successful: '{response}'")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Model loading failed: {e}")
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  💡 Try running with less memory or on CPU")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def get_installation_info():
    """Affiche les informations d'installation"""
    print(f"\n📋 INSTALLATION GUIDE")
    print("="*30)
    print("If Gemma 3n is not properly installed, here are the options:")
    print()
    print("1. 🔗 HuggingFace Hub (recommended):")
    print("   from transformers import AutoModelForCausalLM")
    print("   model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')")
    print("   model.save_pretrained('models/gemma-3n')")
    print()
    print("2. 📁 Manual download from Kaggle:")
    print("   - Download from Kaggle competition files")
    print("   - Extract to models/gemma-3n/")
    print()
    print("3. 🔧 Ollama (alternative):")
    print("   ollama pull gemma:3n")
    print("   # Then configure for use with transformers")
    print()

def main():
    """Fonction principale de diagnostic"""
    print("🏥 RETINOBLASTOGAMMA - GEMMA 3N DIAGNOSTIC")
    print("="*60)
    
    # Étape 1: Vérifier l'installation
    installation_ok = check_gemma_installation()
    
    if not installation_ok:
        print(f"\n❌ INSTALLATION INCOMPLETE")
        get_installation_info()
        return
    
    # Étape 2: Test de chargement
    loading_ok = test_loading()
    
    if loading_ok:
        print(f"\n🎉 SUCCESS!")
        print("✅ Gemma 3n is properly installed and working")
        print("✅ Ready for RetinoblastoGemma application")
        print()
        print("🚀 Next steps:")
        print("1. Run: python main_local_gemma.py")
        print("2. Load a medical image")
        print("3. Click 'Analyze for Retinoblastoma'")
    else:
        print(f"\n⚠️ INSTALLATION ISSUES DETECTED")
        print("❌ Model files found but loading failed")
        print()
        print("🔧 Possible solutions:")
        print("1. Check if you have enough RAM (8GB+ recommended)")
        print("2. Install missing dependencies: pip install torch transformers")
        print("3. Try CPU-only mode if GPU memory insufficient")
        get_installation_info()

if __name__ == "__main__":
    main()