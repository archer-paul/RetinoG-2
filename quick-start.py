#!/usr/bin/env python3
"""
Script de démarrage rapide pour RetinoblastoGemma
Résout les problèmes de chargement et offre des options de démarrage
"""
import sys
import time
import logging
import subprocess
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Vérifie la version Python"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ requis")
        return False
    logger.info(f"✅ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Vérifie les dépendances essentielles"""
    essential_deps = [
        'tkinter',
        'PIL',
        'cv2',
        'numpy'
    ]
    
    missing = []
    for dep in essential_deps:
        try:
            __import__(dep)
            logger.info(f"✅ {dep}")
        except ImportError:
            missing.append(dep)
            logger.error(f"❌ {dep} manquant")
    
    if missing:
        logger.error(f"Dépendances manquantes: {missing}")
        logger.info("Installez avec: pip install -r requirements.txt")
        return False
    
    return True

def check_optional_dependencies():
    """Vérifie les dépendances optionnelles"""
    optional_deps = {
        'torch': 'PyTorch (améliore les performances)',
        'transformers': 'Transformers (pour modèle Gemma local)',
        'mediapipe': 'MediaPipe (détection avancée)'
    }
    
    available = []
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            available.append(dep)
            logger.info(f"✅ {dep} - {desc}")
        except ImportError:
            logger.warning(f"⚠️ {dep} non disponible - {desc}")
    
    return available

def create_directories():
    """Crée la structure de dossiers nécessaire"""
    directories = [
        'models',
        'data/test_images',
        'data/results',
        'config'
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 {directory}")
    
    return True

def check_model_availability():
    """Vérifie la disponibilité du modèle Gemma"""
    model_path = Path("models/gemma-3n")
    
    if model_path.exists():
        config_file = model_path / "config.json"
        if config_file.exists():
            logger.info("✅ Modèle Gemma 3n disponible")
            return True
        else:
            logger.warning("⚠️ Modèle Gemma incomplet")
            return False
    else:
        logger.warning("⚠️ Modèle Gemma non trouvé")
        logger.info("💡 L'application fonctionnera en mode simulation")
        return False

def start_application(force_simulation=False):
    """Démarre l'application avec options"""
    try:
        logger.info("🚀 Démarrage de RetinoblastoGemma...")
        
        # Import et démarrage de l'application
        if force_simulation:
            # Mode simulation forcé
            logger.info("🔧 Mode simulation forcé")
            import os
            os.environ['RETINO_FORCE_SIMULATION'] = '1'
        
        # Import de l'application principale
        from main import main
        main()
        
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        logger.info("💡 Vérifiez que tous les modules sont installés")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur de démarrage: {e}")
        return False
    
    return True

def install_dependencies():
    """Installe les dépendances automatiquement"""
    logger.info("📦 Installation des dépendances...")
    
    try:
        # Dépendances essentielles
        essential_packages = [
            "pillow>=10.0.0",
            "opencv-python>=4.8.0", 
            "numpy>=1.24.0",
            "matplotlib>=3.7.0"
        ]
        
        for package in essential_packages:
            logger.info(f"  Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
        
        logger.info("✅ Dépendances essentielles installées")
        
        # Dépendances optionnelles
        optional_packages = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "mediapipe>=0.10.7"
        ]
        
        for package in optional_packages:
            try:
                logger.info(f"  Installing {package} (optionnel)...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning(f"⚠️ Échec installation {package} (optionnel)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur installation: {e}")
        return False

def show_menu():
    """Affiche le menu de démarrage"""
    print("\n" + "="*60)
    print("🏥 RETINOBLASTOGAMMA - EARLY DETECTION SYSTEM")
    print("="*60)
    print("1. 🚀 Démarrage rapide (recommandé)")
    print("2. 🔧 Mode simulation (sans modèle IA)")
    print("3. 📦 Installer les dépendances")
    print("4. 🧪 Tests système")
    print("5. ❌ Quitter")
    print("="*60)

def run_system_tests():
    """Lance les tests système"""
    logger.info("🧪 Lancement des tests système...")
    
    try:
        # Tests basiques
        tests = [
            ("Version Python", check_python_version),
            ("Dépendances", check_dependencies),
            ("Structure dossiers", create_directories),
            ("Modèle Gemma", check_model_availability)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            logger.info(f"\n🔧 Test: {test_name}")
            if test_func():
                passed += 1
        
        logger.info(f"\n📊 Résultats: {passed}/{len(tests)} tests réussis")
        
        if passed == len(tests):
            logger.info("🎉 Système prêt!")
            return True
        else:
            logger.warning("⚠️ Problèmes détectés")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur tests: {e}")
        return False

def main():
    """Fonction principale du démarrage rapide"""
    print("\n🏥 RetinoblastoGemma - Quick Start")
    print("================================")
    
    # Vérifications initiales rapides
    if not check_python_version():
        input("Appuyez sur Entrée pour quitter...")
        return
    
    # Menu principal
    while True:
        show_menu()
        
        try:
            choice = input("\nChoisissez une option (1-5): ").strip()
            
            if choice == '1':
                # Démarrage rapide
                logger.info("🔄 Vérifications préliminaires...")
                
                if not check_dependencies():
                    logger.error("❌ Dépendances manquantes")
                    install_choice = input("Installer automatiquement? (y/n): ")
                    if install_choice.lower() == 'y':
                        if install_dependencies():
                            logger.info("✅ Installation réussie")
                        else:
                            logger.error("❌ Installation échouée")
                            continue
                    else:
                        continue
                
                create_directories()
                check_optional_dependencies()
                model_available = check_model_availability()
                
                # Démarrage avec ou sans modèle
                if start_application(force_simulation=not model_available):
                    break
                else:
                    logger.error("❌ Échec démarrage")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '2':
                # Mode simulation
                logger.info("🔧 Démarrage en mode simulation...")
                create_directories()
                if start_application(force_simulation=True):
                    break
                else:
                    logger.error("❌ Échec démarrage")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '3':
                # Installation dépendances
                if install_dependencies():
                    logger.info("✅ Installation terminée")
                else:
                    logger.error("❌ Installation échouée")
                input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '4':
                # Tests système
                run_system_tests()
                input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '5':
                # Quitter
                logger.info("👋 Au revoir!")
                break
            
            else:
                print("❌ Option invalide. Choisissez 1-5.")
        
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            input("Appuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()