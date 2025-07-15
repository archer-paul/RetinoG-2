"""
Configuration centrale pour RetinoblastoGemma v6
Hackathon Google Gemma - Paramètres et chemins
"""
from pathlib import Path
import os

# === CHEMINS PRINCIPAUX ===
# Dossier racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Dossiers de données
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" 
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dossiers spécialisés
FACE_TRACKING_DIR = DATA_DIR / "face_tracking"
TEMP_DIR = DATA_DIR / "temp"
EXPORTS_DIR = RESULTS_DIR / "exports"

# === CONFIGURATION GEMMA 3N ===
GEMMA_MODEL_PATH = MODELS_DIR / "gemma-3n"

# Paramètres GPU/CPU
GEMMA_CONFIG = {
    'device': 'auto',  # 'cuda', 'cpu', ou 'auto'
    'use_8bit_quantization': True,
    'max_gpu_memory': "3GB",  # Pour GTX 1650
    'torch_dtype': 'float16',  # 'float16' ou 'float32'
    'low_cpu_mem_usage': True,
    'trust_remote_code': True
}

# === CONFIGURATION EYE DETECTION ===
EYE_DETECTION_CONFIG = {
    'min_detection_confidence': 0.2,
    'max_num_faces': 5,
    'enable_image_enhancement': True,
    'cropped_threshold_ratio': 2.0,
    'fallback_mode': 'computer_vision'
}

# === CONFIGURATION FACE TRACKING ===
FACE_TRACKING_CONFIG = {
    'similarity_threshold': 0.6,
    'max_history_per_face': 10,
    'confidence_boost_factor': 1.2,
    'enable_persistence': True,
    'max_history_days': 365
}

# === CONFIGURATION VISUALISATION ===
VISUALIZATION_CONFIG = {
    'output_quality': 95,
    'max_image_width': 1200,
    'annotation_font_sizes': {
        'title': 24,
        'subtitle': 18,
        'normal': 14,
        'small': 11,
        'tiny': 9
    },
    'color_scheme': 'medical_modern'
}

# === CONFIGURATION ANALYSE ===
ANALYSIS_CONFIG = {
    'default_confidence_threshold': 0.5,
    'high_risk_threshold': 0.7,
    'critical_threshold': 0.85,
    'enable_face_tracking': True,
    'enhanced_detection': True,
    'force_local_mode': True  # Crucial pour hackathon
}

# === CONFIGURATION LOGGING ===
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_to_console': True,
    'max_log_size': '10MB',
    'backup_count': 5
}

# === CONFIGURATION HACKATHON ===
HACKATHON_CONFIG = {
    'competition_name': 'Google Gemma Worldwide Hackathon',
    'project_name': 'RetinoblastoGemma',
    'version': 'v6',
    'team_info': {
        'focus': 'Early retinoblastoma detection',
        'tech_stack': 'Gemma 3n Local + Computer Vision',
        'privacy_approach': '100% Local Processing'
    },
    'target_prizes': [
        'Medical AI Innovation',
        'Privacy-Focused Solutions', 
        'Local AI Implementation',
        'Child Health Impact'
    ]
}

# === CONFIGURATION MÉDICALE ===
MEDICAL_CONFIG = {
    'target_condition': 'Retinoblastoma',
    'primary_sign': 'Leukocoria (white pupil reflex)',
    'target_age_group': 'Children under 6 years',
    'urgency_levels': {
        'routine': 'Continue regular monitoring',
        'soon': 'Schedule appointment within 1 month',
        'urgent': 'Schedule appointment within 1-2 weeks', 
        'immediate': 'Contact pediatric ophthalmologist TODAY'
    },
    'survival_rates': {
        'early_detection': 95,
        'late_detection': 30
    }
}

# === FONCTIONS UTILITAIRES ===
def ensure_directories():
    """Crée tous les dossiers nécessaires"""
    directories = [
        MODELS_DIR, DATA_DIR, RESULTS_DIR, LOGS_DIR,
        FACE_TRACKING_DIR, TEMP_DIR, EXPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_info():
    """Retourne les informations sur le modèle Gemma 3n"""
    model_files = {
        'config': GEMMA_MODEL_PATH / "config.json",
        'tokenizer': GEMMA_MODEL_PATH / "tokenizer.json", 
        'model_files': list(GEMMA_MODEL_PATH.glob("*.safetensors"))
    }
    
    availability = {
        'model_directory_exists': GEMMA_MODEL_PATH.exists(),
        'config_available': model_files['config'].exists(),
        'tokenizer_available': model_files['tokenizer'].exists(),
        'model_files_count': len(model_files['model_files']),
        'total_model_size_gb': sum(f.stat().st_size for f in model_files['model_files']) / (1024**3) if model_files['model_files'] else 0
    }
    
    return {
        'files': model_files,
        'availability': availability,
        'config': GEMMA_CONFIG
    }

def get_system_requirements():
    """Retourne les exigences système recommandées"""
    return {
        'minimum': {
            'ram': '8GB',
            'gpu_memory': '4GB',
            'storage': '15GB',
            'python': '3.8+'
        },
        'recommended': {
            'ram': '16GB',
            'gpu_memory': '6GB+',
            'storage': '20GB',
            'python': '3.10+',
            'gpu': 'NVIDIA GTX 1650+ or equivalent'
        },
        'dependencies': [
            'torch>=2.0.0',
            'transformers>=4.35.0',
            'pillow>=9.0.0',
            'opencv-python>=4.8.0',
            'mediapipe>=0.10.0',
            'numpy>=1.21.0',
            'face-recognition>=1.3.0 (optional)'
        ]
    }

def validate_environment():
    """Valide l'environnement de développement"""
    import sys
    
    validation_results = {
        'python_version_ok': sys.version_info >= (3, 8),
        'directories_exist': True,
        'model_available': False,
        'gpu_available': False,
        'dependencies_ok': True
    }
    
    # Vérifier les dossiers
    try:
        ensure_directories()
    except Exception as e:
        validation_results['directories_exist'] = False
        validation_results['directory_error'] = str(e)
    
    # Vérifier le modèle
    model_info = get_model_info()
    validation_results['model_available'] = model_info['availability']['config_available']
    
    # Vérifier GPU
    try:
        import torch
        validation_results['gpu_available'] = torch.cuda.is_available()
        if validation_results['gpu_available']:
            validation_results['gpu_name'] = torch.cuda.get_device_name(0)
            validation_results['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        validation_results['dependencies_ok'] = False
        validation_results['missing_dependency'] = 'torch'
    
    # Vérifier dépendances critiques
    critical_deps = ['PIL', 'cv2', 'numpy']
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        validation_results['dependencies_ok'] = False
        validation_results['missing_dependencies'] = missing_deps
    
    return validation_results

# === CONFIGURATION PAR ENVIRONNEMENT ===
def get_config_for_environment(env='development'):
    """Retourne la configuration selon l'environnement"""
    
    base_config = {
        'gemma': GEMMA_CONFIG,
        'eye_detection': EYE_DETECTION_CONFIG,
        'face_tracking': FACE_TRACKING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'medical': MEDICAL_CONFIG,
        'hackathon': HACKATHON_CONFIG
    }
    
    if env == 'development':
        # Configuration pour développement
        base_config['logging'] = {**LOGGING_CONFIG, 'level': 'DEBUG'}
        base_config['analysis']['default_confidence_threshold'] = 0.3  # Plus sensible
        
    elif env == 'production':
        # Configuration pour production/démonstration
        base_config['logging'] = {**LOGGING_CONFIG, 'level': 'INFO'}
        base_config['analysis']['default_confidence_threshold'] = 0.5  # Standard
        
    elif env == 'hackathon_demo':
        # Configuration optimisée pour démo hackathon
        base_config['logging'] = {**LOGGING_CONFIG, 'level': 'INFO'}
        base_config['analysis']['default_confidence_threshold'] = 0.4  # Légèrement sensible
        base_config['visualization']['output_quality'] = 100  # Qualité maximale
        base_config['gemma']['use_8bit_quantization'] = True  # Performance optimisée
        
    elif env == 'testing':
        # Configuration pour tests
        base_config['logging'] = {**LOGGING_CONFIG, 'level': 'WARNING'}
        base_config['face_tracking']['enable_persistence'] = False  # Pas de sauvegarde
        base_config['analysis']['enhanced_detection'] = False  # Plus rapide
    
    return base_config

# === TEMPLATES DE MESSAGES ===
MESSAGE_TEMPLATES = {
    'welcome': {
        'title': '🏥 RetinoblastoGemma v6',
        'subtitle': 'AI-Powered Early Detection of Retinoblastoma',
        'description': 'Using local Gemma 3n for 100% private medical analysis',
        'hackathon_badge': '🏆 Google Gemma Worldwide Hackathon Entry'
    },
    
    'analysis_start': {
        'title': '🔍 Starting Retinoblastoma Analysis',
        'description': 'Analyzing image for signs of leukocoria using Gemma 3n...',
        'privacy_note': '🔒 All processing is done locally - no data transmitted'
    },
    
    'positive_detection': {
        'title': '🚨 MEDICAL ALERT: Possible Retinoblastoma Detected',
        'urgent_action': 'IMMEDIATE ACTION REQUIRED',
        'instructions': [
            '1. Contact pediatric ophthalmologist TODAY',
            '2. Bring this analysis and original images to appointment', 
            '3. Do NOT delay seeking professional medical evaluation',
            '4. Emergency: Call your healthcare provider immediately'
        ],
        'disclaimer': 'This is an AI screening result - professional evaluation required'
    },
    
    'negative_detection': {
        'title': '✅ Analysis Complete: No Concerning Findings',
        'description': 'No signs of leukocoria were detected in this analysis',
        'recommendations': [
            'Continue regular pediatric eye monitoring',
            'Take monthly photos under good lighting conditions',
            'Contact healthcare provider if any concerns arise',
            'Repeat screening if visual changes are noticed'
        ]
    },
    
    'system_ready': {
        'title': '✅ System Ready',
        'modules_status': 'All core modules initialized successfully',
        'model_status': 'Gemma 3n loaded and ready for analysis',
        'privacy_reminder': '🔒 100% Local Processing - Privacy Guaranteed'
    },
    
    'error_messages': {
        'model_not_available': {
            'title': '❌ Gemma 3n Model Not Available',
            'description': 'Local model files not found or corrupted',
            'solutions': [
                'Check model files in models/gemma-3n/ directory',
                'Re-download model if necessary',
                'Verify sufficient disk space (10GB+ required)',
                'Contact support if issue persists'
            ]
        },
        
        'gpu_memory_error': {
            'title': '⚠️ GPU Memory Insufficient',
            'description': 'Not enough GPU memory to load Gemma 3n',
            'solutions': [
                'Close other GPU-intensive applications',
                'Enable 8-bit quantization in settings',
                'Use CPU-only mode (slower but functional)',
                'Restart application to clear GPU memory'
            ]
        },
        
        'image_load_error': {
            'title': '❌ Image Loading Failed',
            'description': 'Unable to load or process the selected image',
            'solutions': [
                'Verify image file is not corrupted',
                'Ensure image format is supported (JPG, PNG, etc.)',
                'Check image file permissions',
                'Try with a different image file'
            ]
        }
    }
}

# === CONSTANTES MÉDICALES ===
MEDICAL_CONSTANTS = {
    'retinoblastoma_facts': {
        'incidence': '1 in 15,000-20,000 births',
        'age_group': 'Most common in children under 6 years',
        'survival_rate_early': '95% with early detection and treatment',
        'survival_rate_late': '30-60% with late detection',
        'main_sign': 'Leukocoria (white pupil reflex in photos)',
        'urgency': 'Requires immediate medical attention when suspected'
    },
    
    'leukocoria_description': {
        'definition': 'White, gray, or yellow pupil reflex instead of normal red reflex',
        'visibility': 'Often visible in flash photography',
        'significance': 'May indicate retinoblastoma or other serious eye conditions',
        'normal_appearance': 'Pupils should appear dark or show red reflex in photos'
    },
    
    'risk_factors': [
        'Family history of retinoblastoma',
        'Genetic mutations (RB1 gene)',
        'Previous retinoblastoma in one eye',
        'Certain genetic syndromes'
    ],
    
    'when_to_seek_help': [
        'White, gray, or yellow pupil in photos',
        'Crossed eyes (strabismus)',
        'Eye pain or redness',
        'Vision problems in child',
        'Different colored eyes',
        'Pupil that appears white in normal light'
    ]
}

# === EXPORT DE CONFIGURATION ===
def export_config_summary():
    """Exporte un résumé de configuration pour documentation"""
    summary = {
        'project_info': HACKATHON_CONFIG,
        'system_requirements': get_system_requirements(),
        'model_info': get_model_info(),
        'key_features': [
            '100% Local Processing with Gemma 3n',
            'Multimodal AI Analysis (Vision + Text)',
            'Face Tracking for Longitudinal Monitoring',
            'Real-time Medical Report Generation',
            'Privacy-First Architecture',
            'Child Safety Focused Design'
        ],
        'medical_focus': MEDICAL_CONSTANTS['retinoblastoma_facts'],
        'paths': {
            'models': str(MODELS_DIR),
            'data': str(DATA_DIR),
            'results': str(RESULTS_DIR)
        }
    }
    
    return summary

# Initialisation automatique des dossiers
if __name__ == "__main__":
    ensure_directories()
    print("✅ Configuration initialized and directories created")
    
    # Validation de l'environnement
    validation = validate_environment()
    print("\n🔍 Environment Validation:")
    for key, value in validation.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    # Résumé de configuration
    config_summary = export_config_summary()
    print(f"\n📋 Project: {config_summary['project_info']['project_name']} {config_summary['project_info']['version']}")
    print(f"🏆 Competition: {config_summary['project_info']['competition_name']}")
    print(f"🎯 Focus: {config_summary['project_info']['team_info']['focus']}")
    print(f"🔒 Privacy: {config_summary['project_info']['team_info']['privacy_approach']}")
