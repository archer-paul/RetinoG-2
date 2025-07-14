"""
Configuration settings for RetinoblastoGemma - Version Production
Corrigé pour Windows et intégration Gemma 3n
"""
import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, TEST_IMAGES_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Google AI Configuration pour Gemma 3n
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model settings pour production
GEMMA_MODEL_NAME = "gemini-1.5-pro"  # En attendant gemma-3n sur Google AI
GEMMA_LOCAL_PATH = MODELS_DIR / "gemma-3n"

# Vérifier si le modèle Gemma 3n est disponible
GEMMA_AVAILABLE = GEMMA_LOCAL_PATH.exists() and (GEMMA_LOCAL_PATH / "config.json").exists()

# Configuration Gemma 3n optimisée pour production
GEMMA_CONFIG = {
    "temperature": 0.1,  # Très faible pour cohérence médicale
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "application/json",
}

# Detection settings optimisés pour rétinoblastome
CONFIDENCE_THRESHOLD = 0.5
EYE_DETECTION_THRESHOLD = 0.2  # Plus sensible pour images croppées
FACE_SIMILARITY_THRESHOLD = 0.6

# Visualization settings pour diagnostic médical
COLORS = {
    'normal': (0, 255, 0),      # Vert pour yeux normaux
    'suspicious': (255, 165, 0), # Orange pour suspicion
    'abnormal': (255, 0, 0),    # Rouge pour anomalie détectée
    'face_box': (0, 0, 255),    # Bleu pour le visage
    'high_risk': (139, 0, 0),   # Rouge foncé pour haut risque
}

BOX_THICKNESS = 3
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Logging configuration (sans emojis pour Windows)
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance settings pour Google AI Edge Prize
EDGE_OPTIMIZATION = {
    "enable_quantization": True,
    "use_cache": True,
    "parallel_processing": True,
    "max_cache_size": 100,
    "mobile_mode": False,
    "crop_analysis": True,  # Important pour images croppées
}

# Backend priority pour production
BACKEND_PRIORITY = [
    "google_ai_edge",      # Priorité à Google AI Edge
    "local_transformers",  # Fallback vers modèle local
    "simulation"           # Dernier recours
]

# Configuration spécifique rétinoblastome
RETINOBLASTOMA_CONFIG = {
    "age_range": (0, 6),  # Âge typique des patients
    "typical_signs": [
        "white pupil reflex",
        "leukocoria",
        "strabismus",
        "red eye",
        "vision problems"
    ],
    "urgency_levels": {
        "high": "Immediate medical consultation required",
        "medium": "Schedule ophthalmologist appointment within 1 week",
        "low": "Continue regular monitoring"
    }
}

# Configuration Windows spécifique
if sys.platform == "win32":
    # Encodage pour résoudre les problèmes Unicode
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass  # Ignore si non disponible
    
    # Configuration console Windows
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except:
        pass  # Ignore si échec
