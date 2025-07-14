#!/usr/bin/env python3
"""
Script de démarrage rapide pour RetinoblastoGemma - Version corrigée
Diagnostique et résout les problèmes de chargement
"""
import sys
import time
import logging
import subprocess
from pathlib import Path
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Vérifie la version Python"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ requis")
        return False
    logger.info(f"✅ Python {sys.version.split()[0]}")
    return True

def check_critical_dependencies():
    """Vérifie les dépendances critiques pour l'interface"""
    critical_deps = {
        'tkinter': 'Interface graphique (généralement inclus avec Python)',
        'PIL': 'Pillow pour le traitement d\'images',
        'cv2': 'OpenCV pour la computer vision',
        'numpy': 'NumPy pour les calculs numériques'
    }
    
    missing = []
    for dep, desc in critical_deps.items():
        try:
            __import__(dep)
            logger.info(f"✅ {dep} - {desc}")
        except ImportError:
            missing.append((dep, desc))
            logger.error(f"❌ {dep} manquant - {desc}")
    
    return missing

def check_optional_dependencies():
    """Vérifie les dépendances optionnelles"""
    optional_deps = {
        'torch': 'PyTorch (pour modèles IA)',
        'transformers': 'Transformers (pour Gemma local)',
        'mediapipe': 'MediaPipe (détection avancée)',
        'face_recognition': 'Reconnaissance faciale',
        'matplotlib': 'Graphiques et visualisations'
    }
    
    available = []
    missing = []
    
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            available.append((dep, desc))
            logger.info(f"✅ {dep} - {desc}")
        except ImportError:
            missing.append((dep, desc))
            logger.warning(f"⚠️ {dep} manquant - {desc}")
    
    return available, missing

def install_critical_dependencies():
    """Installe les dépendances critiques"""
    logger.info("📦 Installation des dépendances critiques...")
    
    critical_packages = [
        "pillow>=10.0.0",
        "opencv-python>=4.8.0", 
        "numpy>=1.24.0"
    ]
    
    for package in critical_packages:
        try:
            logger.info(f"  Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            logger.info(f"  ✅ {package} installé")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Échec {package}: {e}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            return False
    
    return True

def create_directories():
    """Crée la structure de dossiers nécessaire"""
    logger.info("📁 Création de la structure de dossiers...")
    
    directories = [
        'models',
        'data/test_images',
        'data/results',
        'config'
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  📁 {directory}")
    
    return True

def test_basic_imports():
    """Test les imports de base requis pour l'application"""
    logger.info("🔄 Test des imports de base...")
    
    try:
        import tkinter as tk
        logger.info("  ✅ tkinter (interface graphique)")
        
        # Test création fenêtre
        root = tk.Tk()
        root.withdraw()  # Cache la fenêtre
        root.destroy()
        logger.info("  ✅ tkinter fonctionne correctement")
        
    except Exception as e:
        logger.error(f"  ❌ tkinter échec: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("  ✅ PIL/Pillow")
    except ImportError as e:
        logger.error(f"  ❌ PIL/Pillow manquant: {e}")
        return False
    
    try:
        import cv2
        logger.info("  ✅ OpenCV")
    except ImportError as e:
        logger.error(f"  ❌ OpenCV manquant: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("  ✅ NumPy")
    except ImportError as e:
        logger.error(f"  ❌ NumPy manquant: {e}")
        return False
    
    return True

def test_config_import():
    """Test l'import de la configuration"""
    logger.info("🔄 Test de la configuration...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from config.settings import MODELS_DIR, DATA_DIR, RESULTS_DIR
        logger.info("  ✅ Configuration chargée")
        logger.info(f"    Models: {MODELS_DIR}")
        logger.info(f"    Data: {DATA_DIR}")
        logger.info(f"    Results: {RESULTS_DIR}")
        return True
    except ImportError as e:
        logger.warning(f"  ⚠️ Configuration manquante: {e}")
        logger.info("  💡 L'application utilisera une configuration par défaut")
        return False

def test_core_modules():
    """Test les modules core (sans les charger complètement)"""
    logger.info("🔄 Test des modules core...")
    
    core_modules = [
        'core.eye_detector',
        'core.face_tracker', 
        'core.visualization',
        'core.gemma_handler'
    ]
    
    importable = []
    failed = []
    
    for module in core_modules:
        try:
            # Test import sans initialisation
            __import__(module)
            importable.append(module)
            logger.info(f"  ✅ {module}")
        except ImportError as e:
            failed.append((module, str(e)))
            logger.warning(f"  ⚠️ {module}: {e}")
    
    if importable:
        logger.info(f"  📊 {len(importable)}/{len(core_modules)} modules importables")
    
    return len(importable) >= 2  # Au moins 2 modules doivent marcher

def run_interface_test():
    """Test que l'interface peut se lancer en mode minimal"""
    logger.info("🔄 Test de l'interface en mode minimal...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Créer une fenêtre de test
        root = tk.Tk()
        root.title("RetinoblastoGemma - Test Interface")
        root.geometry("400x300")
        
        # Ajouter quelques widgets
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(frame, text="Interface Test - OK!").grid(row=0, column=0)
        ttk.Button(frame, text="Close Test", command=root.destroy).grid(row=1, column=0, pady=5)
        
        # Afficher brièvement
        root.update()
        logger.info("  ✅ Interface graphique fonctionnelle")
        
        # Fermer automatiquement après 2 secondes
        root.after(2000, root.destroy)
        root.mainloop()
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Test interface échec: {e}")
        return False

def start_application_safe():
    """Démarre l'application en mode sécurisé"""
    logger.info("🚀 Démarrage de l'application en mode sécurisé...")
    
    try:
        # Forcer le mode simulation
        os.environ['RETINO_FORCE_SIMULATION'] = '1'
        
        # Import et démarrage
        from main_fixed import main
        main()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur de démarrage: {e}")
        return False

def show_diagnostic_menu():
    """Affiche le menu de diagnostic"""
    print("\n" + "="*60)
    print("🏥 RETINOBLASTOGAMMA - DIAGNOSTIC & REPAIR")
    print("="*60)
    print("1. 🔍 Diagnostic complet")
    print("2. 📦 Installer dépendances critiques")
    print("3. 🧪 Test interface graphique")
    print("4. 🚀 Démarrer en mode sécurisé (simulation)")
    print("5. 🚀 Démarrer normalement")
    print("6. ❌ Quitter")
    print("="*60)

def run_full_diagnostic():
    """Lance un diagnostic complet"""
    logger.info("🔍 DIAGNOSTIC COMPLET")
    logger.info("="*50)
    
    tests = [
        ("Version Python", check_python_version),
        ("Dossiers", create_directories),
        ("Imports de base", test_basic_imports),
        ("Configuration", test_config_import),
        ("Modules core", test_core_modules)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n🔧 {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} - OK")
            else:
                logger.warning(f"⚠️ {test_name} - PROBLÈME")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
    
    logger.info(f"\n📊 RÉSULTATS: {passed}/{len(tests)} tests réussis")
    
    # Vérification des dépendances
    logger.info(f"\n📦 VÉRIFICATION DES DÉPENDANCES:")
    missing_critical = check_critical_dependencies()
    available_opt, missing_opt = check_optional_dependencies()
    
    if missing_critical:
        logger.warning(f"⚠️ Dépendances critiques manquantes:")
        for dep, desc in missing_critical:
            logger.warning(f"  - {dep}: {desc}")
        logger.info(f"💡 Lancez l'option 2 pour les installer")
    
    if missing_opt:
        logger.info(f"📋 Dépendances optionnelles manquantes (fonctionnalités limitées):")
        for dep, desc in missing_opt:
            logger.info(f"  - {dep}: {desc}")
    
    # Recommandations
    logger.info(f"\n💡 RECOMMANDATIONS:")
    if passed >= 4:
        logger.info("  🎉 Système prêt! Vous pouvez lancer l'application.")
        if missing_critical:
            logger.info("  💡 Installez les dépendances critiques pour toutes les fonctionnalités")
        else:
            logger.info("  🚀 Essayez l'option 5 (démarrage normal)")
    elif passed >= 2:
        logger.info("  ⚠️ Système partiellement fonctionnel")
        logger.info("  🔧 Essayez l'option 4 (mode sécurisé)")
        if missing_critical:
            logger.info("  📦 Installez d'abord les dépendances critiques (option 2)")
    else:
        logger.warning("  ❌ Problèmes majeurs détectés")
        logger.warning("  🔧 Installez les dépendances (option 2) puis relancez le diagnostic")
    
    input("\nAppuyez sur Entrée pour continuer...")

def main():
    """Fonction principale du diagnostic"""
    print("\n🏥 RetinoblastoGemma - Diagnostic Tool")
    print("=====================================")
    
    # Vérification Python immédiate
    if not check_python_version():
        input("❌ Version Python incompatible. Appuyez sur Entrée pour quitter...")
        return
    
    while True:
        show_diagnostic_menu()
        
        try:
            choice = input("\nChoisissez une option (1-6): ").strip()
            
            if choice == '1':
                run_full_diagnostic()
            
            elif choice == '2':
                logger.info("📦 Installation des dépendances critiques...")
                if install_critical_dependencies():
                    logger.info("✅ Installation réussie!")
                    input("Appuyez sur Entrée pour continuer...")
                else:
                    logger.error("❌ Installation échouée")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '3':
                if run_interface_test():
                    logger.info("✅ Test interface réussi!")
                else:
                    logger.error("❌ Test interface échoué")
                input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '4':
                logger.info("🚀 Démarrage en mode sécurisé...")
                create_directories()
                if start_application_safe():
                    break
                else:
                    logger.error("❌ Démarrage échoué")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '5':
                logger.info("🚀 Démarrage normal...")
                create_directories()
                try:
                    import main_fixed
                    main_fixed.main()
                    break
                except Exception as e:
                    logger.error(f"❌ Démarrage normal échoué: {e}")
                    logger.info("💡 Essayez le mode sécurisé (option 4)")
                    input("Appuyez sur Entrée pour continuer...")
            
            elif choice == '6':
                logger.info("👋 Au revoir!")
                break
            
            else:
                print("❌ Option invalide. Choisissez 1-6.")
        
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            input("Appuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()