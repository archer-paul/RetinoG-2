"""
Script de test complet pour RetinoblastoGemma
Valide toutes les fonctionnalités avant déploiement
"""
import sys
import time
import logging
import json  # Import ajouté pour corriger l'erreur
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Configuration du logging pour les tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Crée une image de test avec des éléments visuels simulant des visages"""
    # Image de base
    img = Image.new('RGB', (800, 600), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Visage 1 - Normal
    draw.ellipse([150, 150, 350, 350], fill='peachpuff', outline='black', width=2)
    # Yeux normaux
    draw.ellipse([180, 200, 220, 240], fill='white', outline='black')
    draw.ellipse([280, 200, 320, 240], fill='white', outline='black')
    # Pupilles normales (sombres)
    draw.ellipse([195, 215, 205, 225], fill='black')
    draw.ellipse([295, 215, 305, 225], fill='black')
    
    # Visage 2 - Avec leucocorie simulée
    draw.ellipse([450, 150, 650, 350], fill='peachpuff', outline='black', width=2)
    # Yeux
    draw.ellipse([480, 200, 520, 240], fill='white', outline='black')
    draw.ellipse([580, 200, 620, 240], fill='white', outline='black')
    # Pupille normale et pupille claire (simulant leucocorie)
    draw.ellipse([495, 215, 505, 225], fill='black')
    draw.ellipse([595, 215, 605, 225], fill='lightgray')  # Pupille claire
    
    return img

def test_imports():
    """Test des imports de modules"""
    logger.info("🔄 Testing module imports...")
    
    try:
        from config.settings import MODELS_DIR, RESULTS_DIR
        logger.info("✅ Settings import successful")
        
        from core.gemma_handler import GemmaHandler, ModelBackend
        logger.info("✅ GemmaHandler import successful")
        
        from core.eye_detector import AdvancedEyeDetector
        logger.info("✅ AdvancedEyeDetector import successful")
        
        from core.face_tracker import FaceTracker
        logger.info("✅ FaceTracker import successful")
        
        from core.visualization import Visualizer
        logger.info("✅ Visualizer import successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_dependencies():
    """Test des dépendances externes"""
    logger.info("🔄 Testing external dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'face_recognition': 'Face Recognition',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing_deps = []
    
    for module_name, friendly_name in dependencies.items():
        try:
            __import__(module_name)
            logger.info(f"✅ {friendly_name} available")
        except ImportError:
            logger.warning(f"⚠️ {friendly_name} not available")
            missing_deps.append(friendly_name)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    return True

def test_directory_structure():
    """Test de la structure des dossiers"""
    logger.info("🔄 Testing directory structure...")
    
    from config.settings import MODELS_DIR, DATA_DIR, RESULTS_DIR, TEST_IMAGES_DIR
    
    directories = [
        (MODELS_DIR, "Models directory"),
        (DATA_DIR, "Data directory"),
        (RESULTS_DIR, "Results directory"),
        (TEST_IMAGES_DIR, "Test images directory")
    ]
    
    for directory, name in directories:
        if directory.exists():
            logger.info(f"✅ {name} exists: {directory}")
        else:
            logger.warning(f"⚠️ {name} missing, creating: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    return True

def test_gemma_handler():
    """Test du gestionnaire Gemma"""
    logger.info("🔄 Testing GemmaHandler...")
    
    try:
        from core.gemma_handler import GemmaHandler
        
        # Initialisation
        start_time = time.time()
        gemma = GemmaHandler()
        init_time = time.time() - start_time
        
        logger.info(f"✅ GemmaHandler initialized in {init_time:.2f}s")
        
        # Test d'analyse simple
        test_image = Image.new('RGB', (128, 128), color='white')
        
        start_time = time.time()
        result = gemma.analyze_eye_region(test_image, 'test')
        analysis_time = time.time() - start_time
        
        logger.info(f"✅ Analysis completed in {analysis_time:.3f}s")
        logger.info(f"   Backend: {result.model_backend}")
        logger.info(f"   Confidence: {result.confidence:.1f}%")
        
        # Test du rapport de performance
        performance_report = gemma.get_performance_report()
        backend = performance_report.get('system_info', {}).get('backend', 'unknown')
        
        logger.info(f"✅ Performance report generated")
        logger.info(f"   Active backend: {backend}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GemmaHandler test failed: {e}")
        return False

def test_eye_detector():
    """Test du détecteur d'yeux"""
    logger.info("🔄 Testing AdvancedEyeDetector...")
    
    try:
        from core.eye_detector import AdvancedEyeDetector
        from config.settings import RESULTS_DIR
        
        # Initialisation
        detector = AdvancedEyeDetector()
        logger.info("✅ AdvancedEyeDetector initialized")
        
        # Créer une image de test
        test_image = create_test_image()
        test_image_path = RESULTS_DIR / "test_detection_image.jpg"
        test_image.save(test_image_path)
        
        # Test de détection
        start_time = time.time()
        results = detector.detect_faces_and_eyes(
            str(test_image_path),
            enhance_quality=True,
            parallel_processing=False
        )
        detection_time = time.time() - start_time
        
        logger.info(f"✅ Detection completed in {detection_time:.2f}s")
        logger.info(f"   Faces detected: {results['total_faces_detected']}")
        logger.info(f"   Eyes detected: {results['total_eyes_detected']}")
        logger.info(f"   Overall quality: {results.get('overall_quality', {}).get('level', 'Unknown')}")
        
        # Test du rapport de performance
        performance_report = detector.get_performance_report()
        success_rate = performance_report.get('detection_statistics', {}).get('success_rate', 0)
        
        logger.info(f"✅ Performance report generated")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AdvancedEyeDetector test failed: {e}")
        return False

def test_face_tracker():
    """Test du traqueur de visages"""
    logger.info("🔄 Testing FaceTracker...")
    
    try:
        from core.face_tracker import FaceTracker
        from config.settings import RESULTS_DIR
        
        # Initialisation
        tracker = FaceTracker()
        logger.info("✅ FaceTracker initialized")
        
        # Test d'enregistrement de visage
        test_image_path = RESULTS_DIR / "test_detection_image.jpg"
        if test_image_path.exists():
            face_id, is_new = tracker.identify_or_register_face(
                str(test_image_path),
                face_box=(100, 100, 200, 200),
                metadata={'test': True}
            )
            
            if face_id:
                logger.info(f"✅ Face {'registered' if is_new else 'recognized'}: {face_id}")
                
                # Test d'ajout d'analyse à l'historique
                test_analysis = {
                    'eyes': [{
                        'position': 'left',
                        'leukocoria_detected': False,
                        'confidence': 75.0,
                        'risk_level': 'low'
                    }]
                }
                
                tracker.add_analysis_to_history(face_id, test_analysis, str(test_image_path))
                logger.info("✅ Analysis added to history")
                
                # Test de progression
                progression = tracker.get_individual_progression(face_id)
                if 'error' not in progression:
                    logger.info(f"✅ Progression analysis completed")
                    logger.info(f"   Total analyses: {progression['total_analyses']}")
                    logger.info(f"   Consistency score: {progression['consistency_score']:.1f}%")
                
                # Sauvegarder la base
                tracker.save_database()
                logger.info("✅ Database saved")
            else:
                logger.warning("⚠️ Face registration failed")
        else:
            logger.warning("⚠️ Test image not found, skipping face registration test")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FaceTracker test failed: {e}")
        return False

def test_visualizer():
    """Test du visualiseur"""
    logger.info("🔄 Testing Visualizer...")
    
    try:
        from core.visualization import Visualizer
        from config.settings import RESULTS_DIR
        
        # Initialisation
        visualizer = Visualizer()
        logger.info("✅ Visualizer initialized")
        
        # Test de création de résumé d'analyse
        test_analysis_results = {
            'faces': [{
                'eyes': [{
                    'position': 'left',
                    'leukocoria_detected': False,
                    'confidence': 85.0,
                    'risk_level': 'low'
                }, {
                    'position': 'right',
                    'leukocoria_detected': True,
                    'confidence': 92.0,
                    'risk_level': 'high'
                }]
            }]
        }
        
        # Test de création de graphique
        try:
            fig = visualizer.create_analysis_summary(test_analysis_results)
            logger.info("✅ Analysis summary chart created")
            
            # Sauvegarder le graphique de test
            test_chart_path = RESULTS_DIR / "test_chart.png"
            fig.savefig(test_chart_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Chart saved: {test_chart_path}")
            
        except Exception as chart_error:
            logger.warning(f"⚠️ Chart creation failed: {chart_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualizer test failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test du workflow complet de bout en bout"""
    logger.info("🔄 Testing end-to-end workflow...")
    
    try:
        from core.gemma_handler import GemmaHandler
        from core.eye_detector import AdvancedEyeDetector
        from core.face_tracker import FaceTracker
        from core.visualization import Visualizer
        from config.settings import RESULTS_DIR
        
        # Initialiser tous les composants
        logger.info("   Initializing all components...")
        gemma = GemmaHandler()
        detector = AdvancedEyeDetector()
        tracker = FaceTracker()
        visualizer = Visualizer()
        
        # Créer et sauvegarder une image de test
        test_image = create_test_image()
        test_image_path = RESULTS_DIR / "end_to_end_test.jpg"
        test_image.save(test_image_path, quality=90)
        
        # Étape 1: Détection
        logger.info("   Step 1: Face and eye detection...")
        start_time = time.time()
        detection_results = detector.detect_faces_and_eyes(
            str(test_image_path),
            enhance_quality=True,
            parallel_processing=True
        )
        detection_time = time.time() - start_time
        
        if not detection_results['faces']:
            logger.warning("⚠️ No faces detected in test image")
            return False
        
        logger.info(f"   ✅ Detection: {len(detection_results['faces'])} faces, {detection_results['total_eyes_detected']} eyes ({detection_time:.2f}s)")
        
        # Étape 2: Analyse avec Gemma
        logger.info("   Step 2: AI analysis with Gemma...")
        analysis_results = {'faces': []}
        tracking_info = {'face_ids': {}}
        
        total_analysis_time = 0
        
        for face_idx, face_detection in enumerate(detection_results['faces']):
            face_analysis = {'eyes': []}
            
            # Reconnaissance faciale
            face_bbox = face_detection.face_bbox
            face_id, is_new = tracker.identify_or_register_face(
                str(test_image_path), 
                face_bbox,
                metadata={'test_workflow': True}
            )
            tracking_info['face_ids'][str(face_idx)] = face_id
            
            # Analyser chaque œil
            for eye_region in face_detection.eyes:
                eye_image = eye_region.enhanced_image if eye_region.enhanced_image else eye_region.image
                
                start_time = time.time()
                eye_result = gemma.analyze_eye_region(eye_image, eye_region.position)
                total_analysis_time += time.time() - start_time
                
                eye_analysis = {
                    'position': eye_region.position,
                    'bbox': eye_region.bbox,
                    'leukocoria_detected': eye_result.leukocoria_detected,
                    'confidence': eye_result.confidence,
                    'risk_level': eye_result.risk_level,
                    'pupil_color': eye_result.pupil_color,
                    'description': eye_result.description,
                    'recommendations': eye_result.recommendations,
                    'processing_time': eye_result.processing_time,
                    'model_backend': eye_result.model_backend
                }
                
                face_analysis['eyes'].append(eye_analysis)
            
            # Ajouter à l'historique
            tracker.add_analysis_to_history(face_id, face_analysis, str(test_image_path))
            analysis_results['faces'].append(face_analysis)
        
        logger.info(f"   ✅ Analysis: {len(analysis_results['faces'])} faces analyzed ({total_analysis_time:.2f}s)")
        
        # Étape 3: Visualisation
        logger.info("   Step 3: Results visualization...")
        
        # Conversion pour le visualizer
        visualization_detection = {
            'faces': [],
            'total_eyes_detected': detection_results['total_eyes_detected'],
            'image_shape': (600, 800)  # Taille de l'image de test
        }
        
        for face_detection in detection_results['faces']:
            face_dict = {
                'face_box': face_detection.face_bbox,
                'eyes': []
            }
            
            for eye_region in face_detection.eyes:
                eye_dict = {
                    'position': eye_region.position,
                    'bbox': eye_region.bbox,
                    'confidence': eye_region.confidence
                }
                face_dict['eyes'].append(eye_dict)
            
            visualization_detection['faces'].append(face_dict)
        
        annotated_image = visualizer.draw_detection_results(
            str(test_image_path),
            visualization_detection,
            analysis_results,
            tracking_info
        )
        
        # Sauvegarder l'image annotée
        output_path = RESULTS_DIR / "end_to_end_result.jpg"
        visualizer.save_annotated_image(annotated_image, str(output_path))
        logger.info(f"   ✅ Visualization: Result saved to {output_path}")
        
        # Étape 4: Rapport final
        logger.info("   Step 4: Generating reports...")
        
        # Sauvegarder tous les résultats
        full_results = {
            'detection': detection_results,
            'analysis': analysis_results,
            'tracking': tracking_info,
            'workflow_metrics': {
                'detection_time': detection_time,
                'analysis_time': total_analysis_time,
                'total_time': detection_time + total_analysis_time
            }
        }
        
        results_path = RESULTS_DIR / "end_to_end_results.json"
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        logger.info(f"   ✅ Results saved: {results_path}")
        
        # Sauvegarder la base de données
        tracker.save_database()
        
        # Résumé final
        positive_detections = sum(
            1 for face in analysis_results['faces']
            for eye in face['eyes']
            if eye.get('leukocoria_detected', False)
        )
        
        total_workflow_time = detection_time + total_analysis_time
        
        logger.info("✅ End-to-end workflow completed successfully!")
        logger.info(f"   Total time: {total_workflow_time:.2f}s")
        logger.info(f"   Faces processed: {len(detection_results['faces'])}")
        logger.info(f"   Eyes analyzed: {detection_results['total_eyes_detected']}")
        logger.info(f"   Positive detections: {positive_detections}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end workflow test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test des benchmarks de performance"""
    logger.info("🔄 Running performance benchmarks...")
    
    try:
        from core.gemma_handler import GemmaHandler
        from core.eye_detector import AdvancedEyeDetector
        import time
        
        # Test de performance Gemma
        logger.info("   Benchmarking Gemma performance...")
        gemma = GemmaHandler()
        
        test_image = Image.new('RGB', (128, 128), color='lightgray')
        times = []
        
        for i in range(5):
            start = time.time()
            result = gemma.analyze_eye_region(test_image, 'benchmark')
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"   ✅ Gemma benchmark: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
        
        # Test de performance détection
        logger.info("   Benchmarking detection performance...")
        detector = AdvancedEyeDetector()
        
        # Créer une image plus complexe pour le benchmark
        complex_image = create_test_image()
        from config.settings import RESULTS_DIR
        benchmark_path = RESULTS_DIR / "benchmark_image.jpg"
        complex_image.save(benchmark_path)
        
        detection_times = []
        
        for i in range(3):  # Moins d'itérations pour la détection
            start = time.time()
            results = detector.detect_faces_and_eyes(str(benchmark_path))
            detection_times.append(time.time() - start)
        
        avg_detection_time = sum(detection_times) / len(detection_times)
        
        logger.info(f"   ✅ Detection benchmark: avg={avg_detection_time:.3f}s")
        
        # Recommandations de performance
        recommendations = []
        
        if avg_time > 3:
            recommendations.append("Gemma analysis is slow - consider using Google AI Edge")
        
        if avg_detection_time > 2:
            recommendations.append("Detection is slow - consider enabling parallel processing")
        
        if not recommendations:
            recommendations.append("Performance is optimal")
        
        logger.info("   Performance recommendations:")
        for rec in recommendations:
            logger.info(f"     • {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def run_all_tests():
    """Lance tous les tests"""
    logger.info("🚀 STARTING RETINOBLASTOGAMMA SYSTEM TESTS")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Tests individuels
    tests = [
        ("Import Tests", test_imports),
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("GemmaHandler", test_gemma_handler),
        ("AdvancedEyeDetector", test_eye_detector),
        ("FaceTracker", test_face_tracker),
        ("Visualizer", test_visualizer),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        logger.info(f"\n{'─' * 20} {test_name} {'─' * 20}")
        
        try:
            start_time = time.time()
            result = test_function()
            test_time = time.time() - start_time
            
            test_results[test_name] = {
                'passed': result,
                'time': test_time
            }
            
            if result:
                logger.info(f"✅ {test_name} PASSED ({test_time:.2f}s)")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            test_results[test_name] = {
                'passed': False,
                'time': 0,
                'error': str(e)
            }
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    total_time = sum(result.get('time', 0) for result in test_results.values())
    logger.info(f"Total test time: {total_time:.2f}s")
    
    # Détail des résultats
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        time_str = f"({result['time']:.2f}s)"
        logger.info(f"  {status} {test_name} {time_str}")
        
        if 'error' in result:
            logger.info(f"    Error: {result['error']}")
    
    # Recommandations finales
    logger.info("\n📋 RECOMMENDATIONS:")
    
    if passed_tests == total_tests:
        logger.info("  🎉 All tests passed! System is ready for use.")
        logger.info("  🚀 You can now run: python main.py")
    else:
        failed_tests = [name for name, result in test_results.items() if not result['passed']]
        logger.info(f"  ⚠️ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        logger.info("  🔧 Please fix the issues before using the system")
    
    # Sauvegarder le rapport de test
    from config.settings import RESULTS_DIR
    report_path = RESULTS_DIR / f"test_report_{int(time.time())}.json"
    
    test_report = {
        'timestamp': time.time(),
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'total_time': total_time
        },
        'results': test_results,
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    try:
        import json  # Import ajouté
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        logger.info(f"\n📄 Test report saved: {report_path}")
    except Exception as e:
        logger.warning(f"Failed to save test report: {e}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)