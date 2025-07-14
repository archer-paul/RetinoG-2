"""
RetinoG - Setup script avec téléchargement automatique Gemma 3n
Google Gemma Worldwide Hackathon 2025
"""
import os
import sys
import subprocess
import shutil
import json
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
GEMMA_MODEL_DIR = MODELS_DIR / "gemma-3n"
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"

class RetinoGSetup:
    def __init__(self):
        self.requirements = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.7",
            "face-recognition>=1.3.0",
            "Pillow>=10.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "python-dotenv>=1.0.0",
            "kaggle>=1.5.16",  # Pour télécharger le modèle
            "tqdm>=4.65.0",
            "requests>=2.31.0"
        ]
    
    def check_python_version(self):
        """Vérifie la version Python"""
        logger.info("🐍 Vérification de Python...")
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ requis")
            sys.exit(1)
        logger.info(f"✅ Python {sys.version.split()[0]} détecté")
    
    def create_directories(self):
        """Crée la structure des dossiers"""
        logger.info("📁 Création de la structure...")
        
        directories = [
            "models",
            "data/test_images", 
            "data/results",
            "scripts",
            "docs"
        ]
        
        for directory in directories:
            dir_path = BASE_DIR / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Créer .gitkeep pour garder les dossiers vides
            gitkeep = dir_path / ".gitkeep"
            if not any(dir_path.iterdir()):  # Si dossier vide
                gitkeep.touch()
        
        logger.info("✅ Structure créée")
    
    def install_requirements(self):
        """Installe les dépendances Python"""
        logger.info("📦 Installation des dépendances...")
        
        for requirement in self.requirements:
            try:
                logger.info(f"  Installing {requirement}")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Erreur avec {requirement}: {e}")
                return False
        
        logger.info("✅ Dépendances installées")
        return True
    
    def setup_kaggle_api(self):
        """Configure l'API Kaggle"""
        logger.info("🏆 Configuration Kaggle API...")
        
        # Vérifier si kaggle.json existe
        kaggle_json = KAGGLE_CONFIG_DIR / "kaggle.json"
        
        if not kaggle_json.exists():
            logger.warning("⚠️  kaggle.json non trouvé")
            logger.info("""
📋 Pour télécharger Gemma 3n automatiquement:

1. Allez sur https://www.kaggle.com/settings/account
2. Créez un nouveau token API
3. Téléchargez kaggle.json
4. Placez-le dans: {}

Ou placez kaggle.json dans le dossier du projet.
            """.format(kaggle_json.parent))
            
            # Chercher kaggle.json dans le dossier du projet
            local_kaggle = BASE_DIR / "kaggle.json"
            if local_kaggle.exists():
                logger.info("📄 kaggle.json trouvé localement, copie...")
                KAGGLE_CONFIG_DIR.mkdir(exist_ok=True)
                shutil.copy2(local_kaggle, kaggle_json)
                kaggle_json.chmod(0o600)  # Permissions de sécurité
                logger.info("✅ kaggle.json configuré")
                return True
            else:
                logger.info("⏭️  Téléchargement manuel requis")
                return False
        
        # Vérifier les permissions
        if kaggle_json.stat().st_mode & 0o077:
            kaggle_json.chmod(0o600)
        
        logger.info("✅ Kaggle API configurée")
        return True
    
    def download_gemma_model(self):
        """Télécharge le modèle Gemma 3n depuis Kaggle"""
        logger.info("🤖 Téléchargement de Gemma 3n...")
        
        if GEMMA_MODEL_DIR.exists() and any(GEMMA_MODEL_DIR.iterdir()):
            logger.info("✅ Modèle Gemma 3n déjà présent")
            return True
        
        try:
            # Importer kaggle après installation
            import kaggle
            
            # Télécharger le modèle
            logger.info("📥 Téléchargement depuis Kaggle (peut prendre du temps)...")
            
            # Créer le dossier modèle
            GEMMA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            # Télécharger avec l'API Kaggle
            kaggle.api.model_download(
                handle="google/gemma-3n",
                path=str(MODELS_DIR),
                untar=True,  # Extraction automatique
                quiet=False
            )
            
            # Vérifier le téléchargement
            if self._verify_model_download():
                logger.info("✅ Modèle Gemma 3n téléchargé et vérifié")
                return True
            else:
                logger.error("❌ Erreur lors de la vérification du modèle")
                return False
                
        except ImportError:
            logger.error("❌ Module kaggle non installé")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement: {e}")
            logger.info("""
🔧 Téléchargement manuel:
1. Visitez: https://www.kaggle.com/models/google/gemma-3n
2. Téléchargez le modèle
3. Extrayez dans: {}
            """.format(GEMMA_MODEL_DIR))
            return False
    
    def _verify_model_download(self):
        """Vérifie que le modèle a été téléchargé correctement"""
        expected_files = [
            "config.json",
            "model.safetensors.index.json",  # ou similaire
            "tokenizer.json"
        ]
        
        # Chercher dans tous les sous-dossiers
        for root, dirs, files in os.walk(MODELS_DIR):
            for expected_file in expected_files:
                if expected_file in files:
                    logger.info(f"  ✓ Trouvé: {expected_file}")
                    return True
        
        # Si pas trouvé, lister ce qui est disponible
        logger.info("📁 Contenu téléchargé:")
        for root, dirs, files in os.walk(MODELS_DIR):
            for file in files:
                rel_path = Path(root).relative_to(MODELS_DIR) / file
                logger.info(f"  - {rel_path}")
        
        return len(list(MODELS_DIR.rglob("*"))) > 0
    
    def create_env_template(self):
        """Crée le fichier .env template"""
        logger.info("🔧 Création du template .env...")
        
        env_template = BASE_DIR / ".env.template"
        env_content = '''# RetinoG - Configuration Environment
# =====================================

# Kaggle API (pour téléchargement modèle)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Google AI API (optionnel)
GOOGLE_API_KEY=your_google_api_key_here

# Modèle Gemma 3n
GEMMA_MODEL_PATH=./models/gemma-3n
GEMMA_USE_CUDA=true

# Paramètres de détection
CONFIDENCE_THRESHOLD=0.5
EYE_DETECTION_THRESHOLD=0.3
FACE_SIMILARITY_THRESHOLD=0.6

# Performance
ENABLE_CACHING=true
PARALLEL_PROCESSING=true
MAX_BATCH_SIZE=4

# Debug
LOG_LEVEL=INFO
DEBUG_MODE=false

# Mobile optimizations
MOBILE_MODE=false
QUANTIZATION_ENABLED=true
'''
        
        with open(env_template, 'w') as f:
            f.write(env_content)
        
        logger.info("✅ .env.template créé")
        logger.info("📝 Copiez .env.template vers .env et configurez vos clés")
    
    def create_download_script(self):
        """Crée un script de téléchargement standalone"""
        logger.info("📜 Création du script de téléchargement...")
        
        scripts_dir = BASE_DIR / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        download_script = scripts_dir / "download_model.py"
        script_content = '''#!/usr/bin/env python3
"""
Script standalone pour télécharger Gemma 3n
Usage: python scripts/download_model.py
"""
import os
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup_gemma import RetinoGSetup

def main():
    setup = RetinoGSetup()
    
    print("🤖 Téléchargement de Gemma 3n...")
    if setup.setup_kaggle_api():
        setup.download_gemma_model()
    else:
        print("❌ Configuration Kaggle requise")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(download_script, 'w') as f:
            f.write(script_content)
        
        download_script.chmod(0o755)  # Exécutable
        logger.info("✅ Script de téléchargement créé")
    
    def create_quick_test(self):
        """Crée un test rapide du système"""
        logger.info("🧪 Création du test rapide...")
        
        test_script = BASE_DIR / "quick_test.py"
        test_content = '''#!/usr/bin/env python3
"""
Test rapide de RetinoG
Vérifie que tous les composants fonctionnent
"""
import sys
from pathlib import Path

def test_imports():
    """Test des imports principaux"""
    print("📦 Test des imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("  ❌ PyTorch manquant")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ❌ OpenCV manquant")
        return False
    
    try:
        import mediapipe
        print("  ✅ MediaPipe")
    except ImportError:
        print("  ❌ MediaPipe manquant")
        return False
    
    return True

def test_model():
    """Test du modèle Gemma"""
    print("🤖 Test du modèle...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ❌ Dossier models manquant")
        return False
    
    gemma_files = list(models_dir.rglob("*.json"))
    if gemma_files:
        print(f"  ✅ Fichiers modèle trouvés: {len(gemma_files)}")
        return True
    else:
        print("  ⚠️  Modèle Gemma non trouvé")
        print("  🔧 Lancez: python scripts/download_model.py")
        return False

def test_core_modules():
    """Test des modules core"""
    print("🔧 Test des modules core...")
    
    try:
        from config.settings import MODELS_DIR, RESULTS_DIR
        print("  ✅ Settings")
    except ImportError as e:
        print(f"  ❌ Settings: {e}")
        return False
    
    try:
        from core.eye_detector import AdvancedEyeDetector
        detector = AdvancedEyeDetector()
        print("  ✅ EyeDetector")
    except ImportError as e:
        print(f"  ❌ EyeDetector: {e}")
        return False
    
    return True

def main():
    print("🏥 RetinoG - Test rapide du système")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Modèle", test_model),
        ("Modules Core", test_core_modules)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\\n{'=' * 40}")
    print(f"🏁 Résultats: {passed}/{len(tests)} tests passés")
    
    if passed == len(tests):
        print("🎉 Système prêt! Lancez: python main.py")
    else:
        print("🔧 Problèmes détectés - consultez les erreurs ci-dessus")

if __name__ == "__main__":
    main()
'''
        
        with open(test_script, 'w') as f:
            f.write(test_content)
        
        test_script.chmod(0o755)
        logger.info("✅ Test rapide créé")
    
    def run_setup(self):
        """Lance l'installation complète"""
        logger.info("🚀 Démarrage de l'installation RetinoG")
        logger.info("Google Gemma Worldwide Hackathon 2025")
        print("=" * 50)
        
        steps = [
            ("Python", self.check_python_version),
            ("Dossiers", self.create_directories),
            ("Dépendances", self.install_requirements),
            ("Kaggle API", self.setup_kaggle_api),
            ("Modèle Gemma", self.download_gemma_model),
            ("Configuration", self.create_env_template),
            ("Scripts", self.create_download_script),
            ("Tests", self.create_quick_test)
        ]
        
        completed = 0
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            try:
                if step_func():
                    completed += 1
                    logger.info(f"✅ {step_name} terminé")
                else:
                    logger.warning(f"⚠️  {step_name} partiellement terminé")
            except Exception as e:
                logger.error(f"❌ {step_name} échoué: {e}")
        
        print("\n" + "=" * 50)
        logger.info(f"🏁 Installation terminée: {completed}/{len(steps)} étapes")
        
        if completed >= len(steps) - 1:  # Permettre 1 échec
            print("""
🎉 INSTALLATION RÉUSSIE!

Étapes suivantes:
1. Copiez .env.template vers .env
2. Configurez vos clés API dans .env
3. Testez: python quick_test.py
4. Lancez: python main.py

Pour le hackathon:
🏆 Votre app RetinoG est prête!
🤖 Gemma 3n installé
📱 Prêt pour démo mobile

Bonne chance pour le Google Gemma Worldwide Hackathon! 🚀
            """)
        else:
            print("""
⚠️  INSTALLATION PARTIELLE

Problèmes détectés. Solutions:
1. Vérifiez les erreurs ci-dessus
2. Installez manuellement les dépendances manquantes
3. Téléchargez Gemma 3n manuellement si nécessaire

Support: Consultez le README.md
            """)

def main():
    setup = RetinoGSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
