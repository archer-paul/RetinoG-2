"""
Script de configuration du modèle Gemma 3n téléchargé depuis Kaggle
Version optimisée pour Google AI Edge Prize
"""
import tarfile
import shutil
import json
import logging
from pathlib import Path
import os
import sys

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import MODELS_DIR, GEMMA_LOCAL_PATH
except ImportError:
    # Fallback si config pas encore accessible
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    GEMMA_LOCAL_PATH = MODELS_DIR / "gemma-3n"
    MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaSetup:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.gemma_dir = GEMMA_LOCAL_PATH
        
    def extract_gemma_archive(self, archive_path: str):
        """
        Extrait l'archive Gemma 3n téléchargée depuis Kaggle
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive non trouvée: {archive_path}")
        
        logger.info(f"Extraction de l'archive: {archive_path}")
        
        # Créer le dossier de destination
        self.gemma_dir.mkdir(parents=True, exist_ok=True)
        
        # Extraire l'archive
        try:
            with tarfile.open(archive_path, 'r:*') as tar:
                members = tar.getnames()
                logger.info(f"Contenu de l'archive: {len(members)} fichiers")
                tar.extractall(path=self.gemma_dir)
                logger.info(f"✓ Extraction terminée dans: {self.gemma_dir}")
                self._show_extracted_structure()
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            raise
    
    def _show_extracted_structure(self):
        """Affiche la structure des fichiers extraits"""
        logger.info("Structure extraite:")
        for root, dirs, files in os.walk(self.gemma_dir):
            level = root.replace(str(self.gemma_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files:
                file_size = os.path.getsize(os.path.join(root, file)) / (1024*1024)
                logger.info(f"{sub_indent}{file} ({file_size:.1f} MB)")
    
    def setup_model_config(self):
        """Configure les paramètres du modèle selon la structure extraite"""
        import time
        
        config = {
            "model_path": str(self.gemma_dir),
            "model_type": "gemma-3n",
            "extracted_at": time.time(),
            "ready_for_use": True
        }
        
        # Détecter automatiquement les fichiers du modèle
        model_files = self._detect_model_files()
        config.update(model_files)
        
        # Sauvegarder la configuration
        config_path = self.gemma_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Configuration sauvegardée: {config_path}")
        return config
    
    def _detect_model_files(self):
        """Détecte automatiquement les fichiers du modèle"""
        files_info = {}
        
        # Chercher les fichiers typiques d'un modèle transformers
        common_files = {
            'config.json': 'config_file',
            'generation_config.json': 'generation_config',
            'model.safetensors.index.json': 'model_index',
            'tokenizer.json': 'tokenizer_file',
            'tokenizer_config.json': 'tokenizer_config',
            'special_tokens_map.json': 'special_tokens',
            'tokenizer.model': 'tokenizer_model',
            'preprocessor_config.json': 'preprocessor_config'
        }
        
        # Chercher aussi les fichiers safetensors
        safetensors_files = []
        
        for root, dirs, files in os.walk(self.gemma_dir):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.gemma_dir)
                
                if file in common_files:
                    files_info[common_files[file]] = str(relative_path)
                    logger.info(f"✓ Trouvé: {file}")
                
                if file.endswith('.safetensors') and 'model-' in file:
                    safetensors_files.append(str(relative_path))
        
        if safetensors_files:
            files_info['model_files'] = safetensors_files
            logger.info(f"✓ Trouvé {len(safetensors_files)} fichiers de modèle safetensors")
        
        # Vérifier les fichiers essentiels
        essential_files = ['config_file', 'tokenizer_file']
        missing_files = [f for f in essential_files if f not in files_info]
        
        if missing_files:
            logger.warning(f"Fichiers manquants: {missing_files}")
        else:
            logger.info("✓ Tous les fichiers essentiels trouvés")
        
        return files_info
    
    def verify_installation(self):
        """Vérifie que l'installation est correcte avec gestion d'erreurs améliorée"""
        try:
            logger.info("Vérification de l'installation...")
            
            # Test d'import transformers
            try:
                from transformers import AutoTokenizer, AutoConfig
                logger.info("✓ Transformers disponible")
            except ImportError as e:
                logger.error(f"✗ Transformers non disponible: {e}")
                logger.error("Installez avec: pip install transformers>=4.35.0")
                return False
            
            # Test de chargement de la configuration
            try:
                config_path = self.gemma_dir
                config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
                logger.info(f"✓ Configuration chargée: {config.model_type}")
            except Exception as e:
                logger.error(f"✗ Erreur de chargement de config: {e}")
                return False
            
            # Test du tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
                logger.info(f"✓ Tokenizer chargé: {len(tokenizer)} tokens")
                
                # Test d'encoding simple
                test_text = "Analyze this eye image for leukocoria signs"
                tokens = tokenizer.encode(test_text)
                logger.info(f"✓ Test tokenization réussi: {len(tokens)} tokens")
                
            except Exception as e:
                logger.error(f"✗ Erreur tokenizer: {e}")
                return False
            
            # Test optionnel du modèle complet (peut être long)
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("! CUDA non disponible - utilisation CPU")
                
                # Ne pas charger le modèle complet dans la vérification (trop lourd)
                logger.info("✓ Configuration modèle validée (chargement complet reporté)")
                
            except ImportError:
                logger.warning("! PyTorch non disponible - installez avec: pip install torch")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Erreur de vérification: {e}")
            return False

def main():
    """Fonction principale d'installation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Gemma 3n pour RetinoblastoGemma")
    parser.add_argument("archive_path", nargs='?', help="Chemin vers l'archive Gemma 3n (.tar.gz)")
    parser.add_argument("--verify", action="store_true", help="Vérifier l'installation après extraction")
    parser.add_argument("--verify-only", action="store_true", help="Vérifier seulement (sans extraction)")
    
    args = parser.parse_args()
    
    try:
        setup = GemmaSetup()
        
        if args.verify_only:
            print("🔍 Vérification de l'installation existante...")
            if setup.verify_installation():
                print("\n✅ Installation Gemma 3n validée!")
                print(f"📁 Modèle disponible dans: {GEMMA_LOCAL_PATH}")
                print("🚀 Vous pouvez maintenant lancer: python main.py")
            else:
                print("\n❌ Problèmes détectés lors de la vérification")
                print("💡 Essayez de réinstaller les dépendances:")
                print("   pip install -r requirements.txt")
            return
        
        if not args.archive_path:
            print("❌ Erreur: Chemin vers l'archive requis")
            print("Usage: python scripts/setup_gemma.py votre-archive.tar.gz --verify")
            return
        
        # Extraire l'archive
        setup.extract_gemma_archive(args.archive_path)
        
        # Configurer le modèle
        config = setup.setup_model_config()
        
        # Vérification si demandée
        if args.verify:
            if setup.verify_installation():
                print("\n✅ Installation Gemma 3n réussie!")
                print(f"📁 Modèle installé dans: {GEMMA_LOCAL_PATH}")
                print("🚀 Vous pouvez maintenant lancer: python main.py")
                print("\n📊 Pour tester l'installation complète:")
                print("   python test_system.py")
            else:
                print("\n❌ Problème détecté lors de la vérification")
                print("💡 Vérifiez que toutes les dépendances sont installées:")
                print("   pip install -r requirements.txt")
        else:
            print("\n✅ Extraction terminée!")
            print("💡 Lancez avec --verify pour tester le chargement:")
            print(f"   python scripts/setup_gemma.py --verify-only")
            
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"\n❌ Erreur: {e}")
        print("\n🔧 Solutions possibles:")
        print("1. Vérifiez que le fichier archive existe")
        print("2. Installez les dépendances: pip install -r requirements.txt")
        print("3. Vérifiez l'espace disque (modèle ~10GB)")
        sys.exit(1)

if __name__ == "__main__":
    main()