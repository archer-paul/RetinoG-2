"""
Configuration settings for RetinoblastoGemma - Version Google AI Edge
Optimisé pour le hackathon Google Gemma 3n
"""
import os
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

# Model settings - MODIFIÉ pour votre modèle Gemma 3n
GEMMA_MODEL_NAME = "local_gemma_3n"
GEMMA_LOCAL_PATH = MODELS_DIR / "gemma-3n"

# Vérifier si le modèle Gemma 3n est disponible
GEMMA_AVAILABLE = GEMMA_LOCAL_PATH.exists() and (GEMMA_LOCAL_PATH / "config.json").exists()

# Configuration Gemma 3n optimisée
GEMMA_CONFIG = {
    "torch_dtype": "float16",  # Changez en "float32" si pas de GPU NVIDIA
    "device_map": "auto",
    "low_cpu_mem_usage": True,
    "trust_remote_code": True,
    "max_memory": {0: "6GB"} if os.environ.get("CUDA_VISIBLE_DEVICES") else None
}

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
EYE_DETECTION_THRESHOLD = 0.3
FACE_SIMILARITY_THRESHOLD = 0.6

# Visualization settings
COLORS = {
    'normal': (0, 255, 0),      # Vert
    'abnormal': (0, 0, 255),    # Rouge
    'uncertain': (0, 255, 255), # Jaune
    'face_box': (255, 0, 0)     # Bleu pour le visage
}

BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Google AI API (si disponible)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance settings pour Google AI Edge Prize
EDGE_OPTIMIZATION = {
    "enable_quantization": True,
    "use_cache": True,
    "parallel_processing": True,
    "max_cache_size": 100,
    "mobile_mode": False  # Activez pour déploiement mobile
}

# Backend priority (pour sélection automatique)
BACKEND_PRIORITY = [
    "local_transformers",  # Priorité au modèle local Gemma 3n
    "google_ai_edge",      # Fallback vers Google AI si disponible
    "simulation"           # Dernier recours
]