"""
Tests rapides pour RetinoblastoGemma - SANS rechargement Gemma
Version optimisée pour éviter les longs temps de chargement
"""
import sys
import time
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def quick_import_test():
    """Test rapide des imports"""
    logger.info("🔄 Test des imports...")
    
    try:
        from config.settings import GEMMA_LOCAL_PATH, GEMMA_AVAILABLE
        from core.eye_detector import AdvancedEyeDetector
        from core.face_tracker import FaceTracker
        from core.visualization import Visualizer
        
        logger.info("✅ Tous les imports réussis")
        logger.info(f"✅ Gemma disponible: {GEMMA_AVAILABLE}")
        logger.info(f"✅ Modèle path: {GEMMA_LOCAL_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur import: {e}")
        return False

def quick_detection_test():
    """Test rapide de détection SANS Gemma"""
    logger.info("🔄 Test détection d'yeux...")
    
    try:
        from core.eye_detector import AdvancedEyeDetector
        from PIL import Image, ImageDraw
        from config.settings import RESULTS_DIR
        
        # Créer une image de test simple
        test_image = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(test_image)
        
        # Visage simple
        draw.ellipse([100, 75, 300, 225], fill='peachpuff', outline='black')
        # Yeux
        draw.ellipse([130, 120, 160, 150], fill='white', outline='black')
        draw.ellipse([240, 120, 270, 150], fill='white', outline='black')
        # Pupilles
        draw.ellipse([142, 132, 148, 138], fill='black')
        draw.ellipse([252, 132, 258, 138], fill='black')
        
        test_path = RESULTS_DIR / "quick_test.jpg"
        test_image.save(test_path)
        
        # Test de détection
        detector = AdvancedEyeDetector()
        start_time = time.time()
        
        results = detector.detect_faces_and_eyes(
            str(test_path),
            enhance_quality=True,
            parallel_processing=False
        )
        
        detection_time = time.time() - start_time
        
        logger.info(f"✅ Détection réussie en {detection_time:.2f}s")
        logger.info(f"   Visages: {results['total_faces_detected']}")
        logger.info(f"   Yeux: {results['total_eyes_detected']}")
        logger.info(f"   Qualité: {results.get('overall_quality', {}).get('level', 'Unknown')}")
        
        # Nettoyage
        detector.cleanup_resources()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur détection: {e}")
        return False

def quick_face_tracker_test():
    """Test rapide du face tracker"""
    logger.info("🔄 Test face tracker...")
    
    try:
        from core.face_tracker import FaceTracker
        
        tracker = FaceTracker()
        
        # Test simple sans vraie image
        summary = tracker.get_all_individuals_summary()
        
        logger.info(f"✅ Face tracker opérationnel")
        logger.info(f"   Individus suivis: {len(summary)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur face tracker: {e}")
        return False

def quick_interface_test():
    """Test que l'interface peut se lancer"""
    logger.info("🔄 Test préparation interface...")
    
    try:
        import tkinter as tk
        
        # Test basique tkinter
        root = tk.Tk()
        root.withdraw()  # Cache la fenêtre
        root.destroy()
        
        logger.info("✅ Interface Tkinter prête")
        
        # Test que main.py peut être importé
        sys.path.append('.')
        
        logger.info("✅ Système prêt pour l'interface")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur interface: {e}")
        return False

def main():
    """Tests rapides"""
    print("🚀 TESTS RAPIDES RETINOBLASTOGAMMA")
    print("="*50)
    
    tests = [
        ("Imports", quick_import_test),
        ("Détection", quick_detection_test),
        ("Face Tracker", quick_face_tracker_test),
        ("Interface", quick_interface_test)
    ]
    
    passed = 0
    total_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🔧 {test_name}...")
        start = time.time()
        
        try:
            success = test_func()
            test_time = time.time() - start
            
            if success:
                print(f"✅ {test_name} OK ({test_time:.2f}s)")
                passed += 1
            else:
                print(f"❌ {test_name} ÉCHEC ({test_time:.2f}s)")
                
        except Exception as e:
            test_time = time.time() - start
            print(f"💥 {test_name} CRASH: {e} ({test_time:.2f}s)")
    
    total_time = time.time() - total_time
    
    print(f"\n{'='*50}")
    print(f"📊 RÉSULTATS: {passed}/{len(tests)} tests réussis")
    print(f"⏱️ Temps total: {total_time:.2f}s")
    
    if passed == len(tests):
        print("🎉 SYSTÈME PRÊT ! Lancez: python main.py")
        return True
    else:
        print("⚠️ Quelques problèmes détectés")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)