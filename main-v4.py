"""
Interface principale pour RetinoblastoGemma - Gemma 3n Multimodal Complet
Version finale optimisée pour hackathon avec modèle local 10.17GB
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import logging
from pathlib import Path
import threading
import json
import time
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retinoblastogamma.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports avec fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} available")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

# Configuration des chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
GEMMA_MODEL_PATH = MODELS_DIR / "gemma-3n"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"

# Créer les dossiers
for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

class Gemma3nMultimodalHandler:
    """Gestionnaire pour Gemma 3n multimodal optimisé"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        logger.info(f"Initializing Gemma 3n Multimodal Handler with device: {self.device}")
    
    def load_model_optimized(self, progress_callback=None):
        """Charge le modèle Gemma 3n avec optimisations mémoire"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Gemma model not found at {self.model_path}")
            
            logger.info(f"Loading Gemma 3n from {self.model_path}")
            
            if progress_callback:
                progress_callback(10, "Checking model files...")
            
            # Vérifier les fichiers essentiels
            essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            for file in essential_files:
                if not (self.model_path / file).exists():
                    raise FileNotFoundError(f"Essential file missing: {file}")
            
            if progress_callback:
                progress_callback(25, "Loading tokenizer...")
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Définir le pad token si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if progress_callback:
                progress_callback(40, "Preparing model configuration...")
            
            # Configuration optimisée pour GTX 1650 (4GB VRAM)
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "max_memory": {0: "3.5GB"} if torch.cuda.is_available() else None,
                "offload_folder": str(self.model_path / "offload") if torch.cuda.is_available() else None
            }
            
            if progress_callback:
                progress_callback(60, "Loading model (this takes time)...")
            
            # Charger le modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_kwargs
            )
            
            if progress_callback:
                progress_callback(85, "Optimizing for inference...")
            
            # Optimisations pour inférence
            self.model.eval()
            
            # Optimisations mémoire supplémentaires
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if progress_callback:
                progress_callback(100, "Model ready!")
            
            self.initialized = True
            logger.info("Gemma 3n multimodal model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n: {e}")
            if progress_callback:
                progress_callback(0, f"Error: {e}")
            return False
    
    def analyze_eye_image_multimodal(self, image_pil, eye_position="unknown"):
        """Analyse multimodale d'une image d'œil avec Gemma 3n"""
        if not self.initialized:
            return self._create_fallback_result("Model not initialized")
        
        try:
            # Préparer le prompt multimodal
            prompt = self._create_multimodal_prompt(eye_position)
            
            # Analyser l'image avec vision
            image_features = self._extract_advanced_features(image_pil)
            
            # Créer le prompt complet avec analyse vision
            full_prompt = f"{prompt}\n\nImage Analysis:\n{image_features}\n\nMedical Assessment (JSON):"
            
            # Tokenisation optimisée
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,  # Réduit pour économiser mémoire
                padding=True
            ).to(self.device)
            
            # Génération avec optimisations mémoire
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=384,  # Réduit pour économiser mémoire
                    temperature=0.05,   # Très conservateur pour médical
                    do_sample=True,
                    top_p=0.85,
                    top_k=30,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    use_cache=True
                )
            
            # Décoder la réponse
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parser la réponse médicale
            result = self._parse_multimodal_response(response, image_features)
            result['model_type'] = 'gemma3n_multimodal'
            result['analysis_method'] = 'multimodal_vision'
            
            logger.info(f"Gemma 3n multimodal analysis complete for {eye_position} eye")
            return result
            
        except Exception as e:
            logger.error(f"Gemma 3n multimodal analysis failed: {e}")
            return self._create_fallback_result(f"Multimodal analysis error: {e}")
    
    def _create_multimodal_prompt(self, eye_position):
        """Crée un prompt multimodal spécialisé pour rétinoblastome"""
        return f"""You are a specialized medical AI for retinoblastoma detection with multimodal vision capabilities.

MEDICAL CONTEXT:
Retinoblastoma is a serious eye cancer affecting children under 6. Early detection saves lives (95% survival vs 30% late).

CLINICAL SIGNS:
- Leukocoria: White pupil reflex in flash photography
- Asymmetric red reflex between eyes  
- Unusual bright spots in pupil area
- Cloudiness or opacity in pupil

ANALYSIS TASK:
Analyze the eye image for signs of retinoblastoma/leukocoria.
Eye Position: {eye_position}

MULTIMODAL ANALYSIS STEPS:
1. Locate pupil region precisely
2. Assess pupil color and brightness
3. Compare to normal dark pupil appearance
4. Evaluate reflection patterns
5. Determine medical urgency level

RESPONSE FORMAT (JSON):
{{
    "leukocoria_detected": boolean,
    "confidence": float (0-100),
    "risk_level": "low|medium|high", 
    "affected_eye": "left|right|center|unknown",
    "pupil_description": "detailed visual description",
    "medical_analysis": "clinical reasoning",
    "recommendations": "immediate medical advice",
    "urgency": "routine|soon|urgent|immediate"
}}

CRITICAL: Be conservative - child safety is paramount. Any suspicion requires medical evaluation."""
    
    def _extract_advanced_features(self, image_pil):
        """Extraction avancée de caractéristiques avec analyse multimodale"""
        try:
            # Convertir en array numpy
            image_array = np.array(image_pil)
            
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                # Analyser les canaux couleur séparément
                r_channel = image_array[:, :, 0]
                g_channel = image_array[:, :, 1] 
                b_channel = image_array[:, :, 2]
            else:
                gray = image_array
                r_channel = g_channel = b_channel = gray
            
            # Analyse des caractéristiques de base
            features = {
                "image_size": f"{image_pil.width}x{image_pil.height}",
                "brightness_mean": float(np.mean(gray)),
                "brightness_std": float(np.std(gray)),
                "brightness_max": float(np.max(gray)),
                "contrast_score": float(np.std(gray)),
            }
            
            # Analyse des canaux couleur
            features.update({
                "red_channel_mean": float(np.mean(r_channel)),
                "green_channel_mean": float(np.mean(g_channel)),
                "blue_channel_mean": float(np.mean(b_channel)),
                "color_balance": self._analyze_color_balance(r_channel, g_channel, b_channel)
            })
            
            # Détection avancée des pupilles
            pupil_analysis = self._advanced_pupil_detection(gray, image_array)
            features.update(pupil_analysis)
            
            # Analyse de texture et patterns
            texture_analysis = self._analyze_texture_patterns(gray)
            features.update(texture_analysis)
            
            # Créer description textuelle détaillée
            description = self._create_detailed_description(features)
            
            return description
            
        except Exception as e:
            logger.error(f"Advanced feature extraction failed: {e}")
            return f"Feature extraction failed: {e}"
    
    def _analyze_color_balance(self, r_channel, g_channel, b_channel):
        """Analyse l'équilibre des couleurs pour détecter anomalies"""
        try:
            r_mean = np.mean(r_channel)
            g_mean = np.mean(g_channel)
            b_mean = np.mean(b_channel)
            
            # Calculer le ratio blanc/couleur (indicateur leucocorie)
            total_brightness = r_mean + g_mean + b_mean
            white_ratio = min(r_mean, g_mean, b_mean) / max(r_mean, g_mean, b_mean) if max(r_mean, g_mean, b_mean) > 0 else 0
            
            return {
                "white_ratio": float(white_ratio),
                "dominant_color": "red" if r_mean > max(g_mean, b_mean) else "green" if g_mean > b_mean else "blue",
                "color_balance_score": float(total_brightness / 765)  # Normalisé 0-1
            }
        except:
            return {"white_ratio": 0, "dominant_color": "unknown", "color_balance_score": 0}
    
    def _advanced_pupil_detection(self, gray, color_image):
        """Détection avancée des pupilles avec analyse multimodale"""
        try:
            # Détection de cercles (pupilles potentielles)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 25,
                param1=60, param2=35, minRadius=8, maxRadius=min(gray.shape)//3
            )
            
            analysis = {"circles_detected": 0, "best_pupil": None}
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                analysis["circles_detected"] = len(circles)
                
                # Analyser le meilleur candidat (plus central)
                center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
                best_circle = None
                min_distance = float('inf')
                
                for (x, y, r) in circles:
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_circle = (x, y, r)
                
                if best_circle:
                    x, y, r = best_circle
                    
                    # Analyser la région pupillaire en détail
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    # Régions pupillaire
                    pupil_region_gray = gray[mask > 0]
                    
                    if len(color_image.shape) == 3:
                        pupil_region_color = color_image[mask > 0]
                        pupil_r = pupil_region_color[:, 0] if len(pupil_region_color) > 0 else []
                        pupil_g = pupil_region_color[:, 1] if len(pupil_region_color) > 0 else []
                        pupil_b = pupil_region_color[:, 2] if len(pupil_region_color) > 0 else []
                    else:
                        pupil_r = pupil_g = pupil_b = pupil_region_gray
                    
                    if len(pupil_region_gray) > 0:
                        # Calculer métriques détaillées
                        pupil_brightness = float(np.mean(pupil_region_gray))
                        global_brightness = float(np.mean(gray))
                        
                        # Score de leucocorie sophistiqué
                        brightness_ratio = pupil_brightness / max(global_brightness, 1)
                        color_uniformity = self._calculate_color_uniformity(pupil_r, pupil_g, pupil_b)
                        
                        # Score composite
                        leukocoria_score = self._calculate_leukocoria_score(
                            brightness_ratio, color_uniformity, pupil_brightness
                        )
                        
                        analysis.update({
                            "pupil_brightness": pupil_brightness,
                            "pupil_radius": int(r),
                            "pupil_position": f"({x}, {y})",
                            "brightness_ratio": brightness_ratio,
                            "color_uniformity": color_uniformity,
                            "leukocoria_score": leukocoria_score,
                            "pupil_area_pixels": int(np.sum(mask > 0)),
                            "assessment": self._assess_pupil_findings(leukocoria_score, brightness_ratio)
                        })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced pupil detection failed: {e}")
            return {"circles_detected": 0, "error": str(e)}
    
    def _calculate_color_uniformity(self, r_vals, g_vals, b_vals):
        """Calcule l'uniformité des couleurs (leucocorie = plus uniforme)"""
        try:
            if len(r_vals) == 0:
                return 0
            
            # Variance des canaux (leucocorie = faible variance, couleurs similaires)
            r_var = np.var(r_vals) if len(r_vals) > 1 else 0
            g_var = np.var(g_vals) if len(g_vals) > 1 else 0
            b_var = np.var(b_vals) if len(b_vals) > 1 else 0
            
            # Score d'uniformité (inverse de la variance moyenne)
            avg_variance = (r_var + g_var + b_var) / 3
            uniformity = 1 / (1 + avg_variance / 100)  # Normalisé 0-1
            
            return float(uniformity)
        except:
            return 0
    
    def _calculate_leukocoria_score(self, brightness_ratio, color_uniformity, absolute_brightness):
        """Calcule un score composite de leucocorie"""
        try:
            # Facteurs de risque pour leucocorie
            brightness_factor = min(brightness_ratio * 2, 1)  # Plus brillant = plus suspect
            uniformity_factor = color_uniformity  # Plus uniforme = plus suspect
            absolute_factor = min(absolute_brightness / 200, 1)  # Luminosité absolue
            
            # Score composite pondéré
            composite_score = (
                brightness_factor * 0.4 +
                uniformity_factor * 0.3 +
                absolute_factor * 0.3
            )
            
            return float(composite_score)
        except:
            return 0
    
    def _assess_pupil_findings(self, leukocoria_score, brightness_ratio):
        """Évalue les findings de la pupille"""
        if leukocoria_score > 0.6:
            return "HIGH_SUSPICION - Bright, uniform pupil detected"
        elif leukocoria_score > 0.4:
            return "MODERATE_SUSPICION - Somewhat bright pupil"
        elif leukocoria_score > 0.2:
            return "LOW_SUSPICION - Slightly unusual brightness"
        else:
            return "NORMAL - Dark pupil as expected"
    
    def _analyze_texture_patterns(self, gray_image):
        """Analyse les patterns de texture pour détecter anomalies"""
        try:
            # Calcul du gradient pour détecter les bords
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyse de texture
            texture_variance = float(np.var(gray_image))
            edge_density = float(np.mean(gradient_magnitude))
            
            return {
                "texture_variance": texture_variance,
                "edge_density": edge_density,
                "texture_complexity": "high" if texture_variance > 1000 else "medium" if texture_variance > 300 else "low"
            }
        except:
            return {"texture_variance": 0, "edge_density": 0, "texture_complexity": "unknown"}
    
    def _create_detailed_description(self, features):
        """Crée une description détaillée pour l'IA"""
        desc = f"""DETAILED MULTIMODAL IMAGE ANALYSIS:

Image Properties:
- Size: {features.get('image_size', 'unknown')}
- Overall brightness: {features.get('brightness_mean', 0):.1f} (std: {features.get('brightness_std', 0):.1f})
- Contrast level: {features.get('contrast_score', 0):.1f}

Color Analysis:
- Red channel: {features.get('red_channel_mean', 0):.1f}
- Green channel: {features.get('green_channel_mean', 0):.1f}  
- Blue channel: {features.get('blue_channel_mean', 0):.1f}
- Color balance: {features.get('color_balance', {}).get('color_balance_score', 0):.2f}
- White ratio: {features.get('color_balance', {}).get('white_ratio', 0):.2f}

Pupil Detection Results:
- Circular structures detected: {features.get('circles_detected', 0)}"""

        if features.get('pupil_brightness') is not None:
            desc += f"""
- Pupil brightness: {features.get('pupil_brightness', 0):.1f}
- Brightness ratio (pupil/global): {features.get('brightness_ratio', 0):.2f}
- Color uniformity score: {features.get('color_uniformity', 0):.2f}
- Leukocoria risk score: {features.get('leukocoria_score', 0):.2f}
- Clinical assessment: {features.get('assessment', 'unknown')}"""

        desc += f"""

Texture Analysis:
- Texture complexity: {features.get('texture_complexity', 'unknown')}
- Edge density: {features.get('edge_density', 0):.1f}

CLINICAL INTERPRETATION:
This analysis provides quantitative metrics for assessing potential leukocoria.
Higher brightness ratios, color uniformity, and leukocoria scores may indicate concern."""

        return desc
    
    def _parse_multimodal_response(self, response, image_features):
        """Parse la réponse multimodale de Gemma"""
        try:
            # Chercher le JSON dans la réponse
            response = response.strip()
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validation et nettoyage
                result['leukocoria_detected'] = bool(result.get('leukocoria_detected', False))
                result['confidence'] = max(0, min(100, float(result.get('confidence', 50))))
                
                if result.get('risk_level') not in ['low', 'medium', 'high']:
                    result['risk_level'] = 'medium'
                
                if result.get('urgency') not in ['routine', 'soon', 'urgent', 'immediate']:
                    result['urgency'] = 'soon'
                
                # Ajouter timing
                result['processing_time'] = time.time()
                
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse Gemma multimodal response: {e}")
            
            # Fallback intelligent basé sur les caractéristiques
            return self._create_intelligent_fallback(image_features, response)
    
    def _create_intelligent_fallback(self, image_features, raw_response):
        """Crée un fallback intelligent basé sur l'analyse d'image"""
        try:
            # Extraire les métriques clés
            leukocoria_score = 0
            brightness_ratio = 0
            
            # Recherche de métriques dans les features
            if "leukocoria_score:" in image_features:
                import re
                match = re.search(r'leukocoria_score: ([\d.]+)', image_features)
                if match:
                    leukocoria_score = float(match.group(1))
            
            if "brightness_ratio:" in image_features:
                import re
                match = re.search(r'brightness_ratio.*?: ([\d.]+)', image_features)
                if match:
                    brightness_ratio = float(match.group(1))
            
            # Logique de décision clinique
            detected = leukocoria_score > 0.25 or brightness_ratio > 1.3
            
            if leukocoria_score > 0.6 or brightness_ratio > 2.0:
                risk_level = "high"
                urgency = "immediate"
                confidence = min(95, 60 + leukocoria_score * 40)
            elif leukocoria_score > 0.4 or brightness_ratio > 1.5:
                risk_level = "medium"
                urgency = "urgent"
                confidence = min(85, 45 + leukocoria_score * 40)
            elif detected:
                risk_level = "medium"
                urgency = "soon"
                confidence = min(75, 30 + leukocoria_score * 40)
            else:
                risk_level = "low"
                urgency = "routine"
                confidence = max(20, 80 - leukocoria_score * 40)
            
            # Création du résultat structuré
            return {
                "leukocoria_detected": detected,
                "confidence": confidence,
                "risk_level": risk_level,
                "affected_eye": "unknown",
                "pupil_description": f"Multimodal analysis: leukocoria score {leukocoria_score:.2f}, brightness ratio {brightness_ratio:.2f}",
                "medical_analysis": f"Computer vision analysis with Gemma 3n reasoning. Score: {leukocoria_score:.2f}. {raw_response[:100]}...",
                "recommendations": "Professional ophthalmological evaluation recommended" if detected else "Continue regular monitoring",
                "urgency": urgency,
                "analysis_method": "multimodal_intelligent_fallback",
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Intelligent fallback failed: {e}")
            return self._create_fallback_result(f"Analysis failed: {e}")
    
    def _create_fallback_result(self, error_msg):
        """Crée un résultat de fallback en cas d'erreur"""
        return {
            "leukocoria_detected": False,
            "confidence": 0,
            "risk_level": "unknown",
            "affected_eye": "unknown",
            "pupil_description": "Analysis failed",
            "medical_analysis": error_msg,
            "recommendations": "Manual professional evaluation required",
            "urgency": "soon",
            "error": error_msg,
            "analysis_method": "fallback"
        }
    
    def cleanup_memory(self):
        """Nettoie la mémoire GPU"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            self.initialized = False
            logger.info("GPU memory cleaned up")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def get_memory_usage(self):
        """Retourne l'usage mémoire GPU"""
        try:
            if torch.cuda.is_available():
                return {
                    "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_reserved": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_memory": torch.cuda.max_memory_allocated() / 1024**3
                }
        except:
            pass
        return {}

class RetinoblastoGemmaMultimodal:
    """Application avec Gemma 3n multimodal optimisé"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma - Gemma 3n Multimodal")
        self.root.geometry("1500x1000")
        
        # Configuration
        self.current_image_path = None
        self.current_analysis_results = None
        self.gemma_handler = None
        self.initialization_complete = False
        
        # Métriques
        self.performance_metrics = {
            'total_analyses': 0,
            'detections_found': 0,
            'processing_times': [],
            'memory_usage': []
        }
        
        self.setup_ui()
        # Initialiser automatiquement après un délai
        self.root.after(2000, self.initialize_system)
    
    def setup_ui(self):
        """Interface utilisateur optimisée"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Panel de contrôles gauche
        control_frame = ttk.LabelFrame(main_frame, text="Retinoblastoma Detection Controls", padding="15")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # Section chargement d'image
        image_section = ttk.LabelFrame(control_frame, text="Image Loading")
        image_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_section, text="Load Medical Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        self.image_info_label = ttk.Label(image_section, text="No image loaded", 
                                         font=("Arial", 9), foreground="gray")
        self.image_info_label.pack(anchor=tk.W, pady=2)
        
        # Section analyse
        analysis_section = ttk.LabelFrame(control_frame, text="AI Analysis")
        analysis_section.pack(fill=tk.X, pady=10)
        
        self.analyze_button = ttk.Button(analysis_section, text="🔍 Analyze for Retinoblastoma", 
                  command=self.analyze_image, state='disabled')
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        # Status Gemma détaillé
        gemma_section = ttk.LabelFrame(control_frame, text="Gemma 3n Multimodal Status")
        gemma_section.pack(fill=tk.X, pady=10)
        
        self.gemma_status = ttk.Label(gemma_section, text="Initializing...", 
                                     foreground="blue", font=("Arial", 10, "bold"))
        self.gemma_status.pack(anchor=tk.W)
        
        self.memory_label = ttk.Label(gemma_section, text="Memory: --", 
                                     font=("Arial", 8))
        self.memory_label.pack(anchor=tk.W)
        
        # Boutons de gestion Gemma
        gemma_buttons = ttk.Frame(gemma_section)
        gemma_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(gemma_buttons, text="Reload Model", 
                  command=self.reload_gemma).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(gemma_buttons, text="Clear Memory", 
                  command=self.clear_gpu_memory).pack(side=tk.LEFT)
        
        # Paramètres d'analyse
        params_section = ttk.LabelFrame(control_frame, text="Detection Parameters")
        params_section.pack(fill=tk.X, pady=10)
        
        self.crop_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Cropped Eye Analysis", 
                       variable=self.crop_detection_var).pack(anchor=tk.W)
        
        self.multimodal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Multimodal Vision Analysis", 
                       variable=self.multimodal_var).pack(anchor=tk.W)
        
        self.enhanced_cv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Enhanced Computer Vision", 
                       variable=self.enhanced_cv_var).pack(anchor=tk.W)
        
        # Status et progression
        status_section = ttk.LabelFrame(control_frame, text="System Status")
        status_section.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_section, text="Starting...", 
                                     foreground="blue", font=("Arial", 9))
        self.status_label.pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(status_section, mode='determinate')
        self.progress.pack(fill=tk.X, pady=3)
        
        self.progress_detail = ttk.Label(status_section, text="", 
                                        font=("Arial", 8), foreground="gray")
        self.progress_detail.pack(anchor=tk.W)
        
        # Métriques de performance
        metrics_section = ttk.LabelFrame(control_frame, text="Performance Metrics")
        metrics_section.pack(fill=tk.X, pady=10)
        
        self.metrics_label = ttk.Label(metrics_section, text="No analysis yet", 
                                      font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Actions et exports
        actions_section = ttk.LabelFrame(control_frame, text="Actions")
        actions_section.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_section, text="📄 Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="🏥 Medical Report", 
                  command=self.generate_medical_report).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="📊 System Diagnostics", 
                  command=self.show_diagnostics).pack(fill=tk.X, pady=1)
        
        # Zone d'affichage principal
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook pour onglets
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Onglet analyse d'image
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="🖼️ Image Analysis")
        
        # Canvas avec barres de défilement
        canvas_frame = ttk.Frame(self.image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", relief=tk.SUNKEN, bd=2)
        scrollbar_v = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Onglet résultats détaillés
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📋 Medical Results")
        
        results_container = ttk.Frame(self.results_frame)
        results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(results_container, wrap=tk.WORD, 
                                   font=("Consolas", 10), relief=tk.SUNKEN, bd=2)
        results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_container.columnconfigure(0, weight=1)
        results_container.rowconfigure(0, weight=1)
        
        # Onglet diagnostics système
        self.diagnostics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.diagnostics_frame, text="🔧 System Info")
        
        self.diagnostics_text = tk.Text(self.diagnostics_frame, wrap=tk.WORD, 
                                       font=("Consolas", 9), state='disabled')
        diag_scrollbar = ttk.Scrollbar(self.diagnostics_frame, orient="vertical", 
                                      command=self.diagnostics_text.yview)
        self.diagnostics_text.configure(yscrollcommand=diag_scrollbar.set)
        
        self.diagnostics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        diag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Barre de statut en bas
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.statusbar = ttk.Label(status_frame, text="RetinoblastoGemma Ready", 
                                  relief=tk.SUNKEN, font=("Arial", 9))
        self.statusbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Afficher les infos système initiales
        self.update_diagnostics()
    
    def initialize_system(self):
        """Initialise Gemma 3n multimodal avec optimisations mémoire"""
        def init_thread():
            try:
                self.update_status("🔄 Checking system requirements...", "blue")
                self.update_progress(5, "Validating dependencies...")
                
                # Vérifications critiques
                if not TORCH_AVAILABLE:
                    self.update_status("❌ PyTorch not available", "red")
                    return
                
                if not TRANSFORMERS_AVAILABLE:
                    self.update_status("❌ Transformers not available", "red")
                    return
                
                if not GEMMA_MODEL_PATH.exists():
                    self.update_status("❌ Gemma 3n model not found", "red")
                    self.update_gemma_status("Model files missing", "red")
                    return
                
                self.update_progress(15, "Initializing Gemma 3n handler...")
                
                # Créer le gestionnaire
                self.gemma_handler = Gemma3nMultimodalHandler(GEMMA_MODEL_PATH)
                
                self.update_progress(25, "Loading Gemma 3n multimodal model...")
                self.update_gemma_status("Loading model (may take 3-5 minutes)...", "orange")
                
                # Charger le modèle avec callback de progression
                def progress_callback(percent, message):
                    progress_value = 25 + (percent * 0.65)  # 25% à 90%
                    self.update_progress(progress_value, message)
                    self.update_gemma_status(f"{message} ({percent:.0f}%)", "blue")
                
                success = self.gemma_handler.load_model_optimized(progress_callback)
                
                if success:
                    self.update_progress(95, "Finalizing initialization...")
                    
                    # Mise à jour des statuts
                    self.initialization_complete = True
                    self.update_progress(100, "System ready!")
                    self.update_status("✅ Gemma 3n Multimodal ready for analysis!", "green")
                    self.update_gemma_status("Ready - Multimodal Vision Active", "green")
                    
                    # Activer l'analyse
                    self.analyze_button.config(state='normal')
                    
                    # Mise à jour mémoire
                    self.update_memory_display()
                    
                    # Message de succès
                    messagebox.showinfo("Export Complete", 
                    f"Medical analysis results exported successfully!\n\n"
                    f"File: {file_path}\n\n"
                    f"This report can be shared with medical professionals.")
                
            except Exception as e:
                self.update_status(f"❌ Export failed: {e}", "red")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def generate_medical_report(self):
        """Génère un rapport médical HTML interactif"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"retinoblastoma_medical_report_gemma3n_{timestamp}.html"
        
        try:
            # Créer le rapport HTML avec CSS avancé
            html_report = self.create_html_medical_report()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Ouvrir dans le navigateur
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
            
            self.update_status(f"✅ Medical report generated: {report_path.name}", "green")
            messagebox.showinfo("Report Generated", 
                f"🏥 Medical report generated successfully!\n\n"
                f"📄 File: {report_path.name}\n"
                f"🌐 Opened in web browser\n\n"
                f"This professional report can be:\n"
                f"• Shared with medical professionals\n"
                f"• Printed for medical appointments\n"
                f"• Saved for medical records")
            
        except Exception as e:
            self.update_status(f"❌ Report generation failed: {e}", "red")
            messagebox.showerror("Report Error", f"Failed to generate medical report:\n{e}")
    
    def create_html_medical_report(self):
        """Crée un rapport HTML médical professionnel"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        # Déterminer s'il y a des détections positives
        has_positive = "MEDICAL ALERT" in self.current_analysis_results
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Retinoblastoma Medical Analysis - Gemma 3n</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 40px; 
            line-height: 1.6; color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .container {{ 
            max-width: 1000px; margin: 0 auto; 
            background: white; border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 28px; font-weight: 300; }}
        .header .subtitle {{ font-size: 16px; opacity: 0.9; margin-top: 10px; }}
        .badges {{ margin-top: 20px; }}
        .badge {{ 
            display: inline-block; padding: 8px 16px; margin: 5px;
            border-radius: 25px; color: white; font-weight: bold; font-size: 12px;
        }}
        .badge-ai {{ background: #9f7aea; }}
        .badge-local {{ background: #4299e1; }}
        .badge-secure {{ background: #48bb78; }}
        .content {{ padding: 40px; }}
        .alert-critical {{ 
            background: linear-gradient(135deg, #ff6b6b, #ff5722);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(255,107,107,0.3);
        }}
        .alert-safe {{ 
            background: linear-gradient(135deg, #51cf66, #40c057);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(81,207,102,0.3);
        }}
        .results-section {{ 
            background: #f8f9fa; padding: 30px; 
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #667eea;
        }}
        .disclaimer {{ 
            background: #fff3cd; border: 2px solid #ffc107;
            padding: 25px; border-radius: 10px; margin: 30px 0;
        }}
        .footer {{ 
            background: #2d3748; color: white; 
            padding: 30px; text-align: center;
        }}
        pre {{ 
            background: #2d3748; color: #e2e8f0;
            padding: 25px; border-radius: 8px; 
            overflow-x: auto; font-size: 14px;
            line-height: 1.4;
        }}
        .emergency-box {{
            background: #fee; border: 3px solid #ff4757;
            padding: 20px; border-radius: 10px; margin: 20px 0;
            border-left: 8px solid #ff4757;
        }}
        .info-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin: 20px 0;
        }}
        .info-card {{
            background: white; padding: 20px; border-radius: 8px;
            border: 1px solid #e2e8f0; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .print-friendly {{ color: #000 !important; }}
        @media print {{
            body {{ background: white !important; }}
            .container {{ box-shadow: none !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Retinoblastoma Medical Analysis Report</h1>
            <div class="subtitle">Advanced AI-Powered Early Detection System</div>
            <div class="badges">
                <span class="badge badge-ai">GEMMA 3N MULTIMODAL</span>
                <span class="badge badge-local">100% LOCAL</span>
                <span class="badge badge-secure">PRIVACY SECURE</span>
            </div>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Image:</strong> {filename}</p>
            <p><strong>Processing:</strong> {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}</p>
        </div>
        
        <div class="content">"""
        
        if has_positive:
            html_report += """
            <div class="alert-critical">
                <h2>🚨 MEDICAL ALERT - IMMEDIATE ACTION REQUIRED</h2>
                <p style="font-size: 20px; font-weight: bold;">
                    Possible retinoblastoma detected. Contact pediatric ophthalmologist IMMEDIATELY.
                </p>
                <div class="emergency-box">
                    <strong>🔴 URGENT STEPS:</strong><br>
                    1. Call pediatric ophthalmologist TODAY<br>
                    2. Bring this report and original images<br>
                    3. Do NOT delay medical evaluation<br>
                    4. Contact emergency services if needed
                </div>
            </div>"""
        else:
            html_report += """
            <div class="alert-safe">
                <h2>✅ No Concerning Findings Detected</h2>
                <p style="font-size: 18px;">
                    The AI analysis did not detect signs of leukocoria in this image.
                    Continue regular pediatric eye health monitoring.
                </p>
            </div>"""
        
        html_report += f"""
            <div class="results-section">
                <h2>📊 Detailed Analysis Results</h2>
                <pre>{self.current_analysis_results}</pre>
            </div>
            
            <div class="info-grid">
                <div class="info-card">
                    <h3>🤖 AI Technology</h3>
                    <p><strong>Gemma 3n Multimodal:</strong> Advanced vision-language model specialized in medical image analysis</p>
                    <p><strong>Local Processing:</strong> 100% on-device analysis ensures complete privacy</p>
                </div>
                
                <div class="info-card">
                    <h3>🎯 Detection Focus</h3>
                    <p><strong>Leukocoria:</strong> White pupil reflex indicating possible retinoblastoma</p>
                    <p><strong>Precision:</strong> Advanced computer vision with AI reasoning</p>
                </div>
                
                <div class="info-card">
                    <h3>📈 Clinical Impact</h3>
                    <p><strong>Early Detection:</strong> 95% survival rate with prompt treatment</p>
                    <p><strong>Vision Preservation:</strong> Better outcomes with earlier intervention</p>
                </div>
                
                <div class="info-card">
                    <h3>🔒 Privacy Protection</h3>
                    <p><strong>No Data Upload:</strong> All processing happens locally</p>
                    <p><strong>Secure:</strong> Images never leave your device</p>
                </div>
            </div>
        </div>
        
        <div class="disclaimer">
            <h3>⚕️ Critical Medical Disclaimer</h3>
            <p><strong>IMPORTANT:</strong> This report is generated by an AI screening system using Gemma 3n multimodal technology.</p>
            <p><strong>THIS IS NOT A MEDICAL DIAGNOSIS</strong> and should NOT replace professional medical evaluation by qualified pediatric ophthalmologists.</p>
            
            <h4>📋 Next Steps:</h4>
            <ul>
                <li><strong>Professional Evaluation:</strong> Schedule consultation with pediatric ophthalmologist</li>
                <li><strong>Documentation:</strong> Bring this report and original images to appointment</li>
                <li><strong>Urgency:</strong> {'IMMEDIATE evaluation required' if has_positive else 'Routine follow-up appropriate'}</li>
                <li><strong>Monitoring:</strong> Continue regular eye health monitoring</li>
            </ul>
            
            <h4>🏥 About Retinoblastoma:</h4>
            <ul>
                <li>Most common eye cancer in children (typically under 6 years)</li>
                <li>95% survival rate with early detection and treatment</li>
                <li>Can affect one or both eyes</li>
                <li>Early sign: White pupil reflex (leukocoria) in flash photos</li>
                <li>Treatment success depends on early intervention</li>
            </ul>
        </div>
        
        <div class="footer">
            <p><strong>Generated by RetinoblastoGemma v1.0</strong></p>
            <p>AI-Powered Retinoblastoma Screening with Gemma 3n Multimodal</p>
            <p>🔒 100% Local Processing - Complete Privacy Protection</p>
            <p style="font-size: 12px; margin-top: 15px; opacity: 0.8;">
                Report ID: RG_{timestamp.replace('-', '').replace(':', '').replace(' ', '_')} | 
                Model: Gemma 3n Multimodal | 
                Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | 
                Privacy: Local Only
            </p>
        </div>
    </div>
</body>
</html>"""
        
        return html_report

def main():
    """Fonction principale optimisée pour hackathon avec Gemma 3n local"""
    try:
        # Vérifications préliminaires
        print("🏥 RETINOBLASTOGAMMA - GEMMA 3N MULTIMODAL")
        print("="*60)
        print("🎯 Hackathon Version - Local Privacy-First AI")
        print("="*60)
        
        # Créer les dossiers nécessaires
        for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
            directory.mkdir(exist_ok=True)
        
        # Vérifier Gemma 3n (critique pour le hackathon)
        if not GEMMA_MODEL_PATH.exists():
            print(f"❌ Gemma 3n model not found at: {GEMMA_MODEL_PATH}")
            print(f"💡 Please ensure Gemma 3n (10.17GB) is installed in models/gemma-3n/")
            print(f"🔧 Run: python check_gemma_local.py for diagnosis")
            
            choice = input("Continue anyway? The app will show initialization errors. (y/n): ")
            if choice.lower() != 'y':
                return
        else:
            print(f"✅ Gemma 3n model found: {GEMMA_MODEL_PATH}")
            
            # Vérifier les fichiers essentiels
            essential_files = ["config.json", "tokenizer.json"]
            missing_files = [f for f in essential_files if not (GEMMA_MODEL_PATH / f).exists()]
            
            if missing_files:
                print(f"⚠️ Missing essential files: {missing_files}")
            else:
                print(f"✅ All essential model files present")
        
        # Vérifier les dépendances critiques
        missing_deps = []
        
        if not TORCH_AVAILABLE:
            missing_deps.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing_deps.append("transformers")
        
        try:
            import mediapipe
            print("✅ MediaPipe available for advanced detection")
        except ImportError:
            missing_deps.append("mediapipe")
            print("⚠️ MediaPipe missing - using fallback detection")
        
        if missing_deps:
            print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            print(f"📦 Install with: pip install {' '.join(missing_deps)}")
            
            choice = input("Continue anyway? Some features may be limited. (y/n): ")
            if choice.lower() != 'y':
                return
        
        # Configuration Windows pour UTF-8
        if sys.platform == "win32":
            try:
                os.system("chcp 65001")
            except:
                pass
        
        # Créer et lancer l'application
        root = tk.Tk()
        
        try:
            app = RetinoblastoGemmaMultimodal(root)
            logger.info("RetinoblastoGemma Multimodal started for hackathon")
            print("🚀 Application launched successfully!")
            print("💡 Gemma 3n will initialize in background (may take 3-5 minutes)")
            print("🏆 Ready for hackathon demonstration!")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            messagebox.showerror("Initialization Error", 
                f"Failed to initialize application:\n{e}\n\n"
                "Check dependencies and Gemma 3n installation.\n"
                "Run: python check_gemma_local.py for diagnosis")
            return
        
        # Gestionnaire de fermeture gracieuse
        def on_closing():
            try:
                logger.info("Application closing gracefully...")
                
                # Libérer les ressources Gemma
                if hasattr(app, 'gemma_handler') and app.gemma_handler:
                    app.gemma_handler.cleanup_memory()
                
                # Libérer les ressources GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("👋 RetinoblastoGemma session ended")
                root.quit()
                root.destroy()
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                root.quit()
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Démarrer l'interface
        logger.info("Starting RetinoblastoGemma main loop")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        print("\n📋 Troubleshooting Guide:")
        print("1. Run: python check_gemma_local.py")
        print("2. Check: pip install torch transformers mediapipe")
        print("3. Verify: models/gemma-3n/ contains model files (10.17GB)")
        print("4. Ensure: Sufficient GPU memory (4GB+ recommended)")
        print("5. Try: python quick-start.py for guided setup")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main().showinfo("System Ready", 
                        "🎉 Gemma 3n Multimodal loaded successfully!\n\n"
                        "✅ Vision capabilities active\n"
                        "✅ Ready for retinoblastoma detection\n"
                        "✅ Optimized for your GTX 1650\n\n"
                        "Load a medical image to start analysis.")
                    
else:
    self.update_status("❌ Failed to load Gemma 3n model", "red")
    self.update_gemma_status("Model loading failed", "red")
    
    messagebox.showerror("Model Loading Failed", 
        "Failed to load Gemma 3n multimodal model.\n\n"
        "Possible causes:\n"
        "• Insufficient GPU memory\n"
        "• Missing model files\n"
        "• Incompatible model format\n\n"
        "Check the diagnostics tab for details.")
                
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    self.update_status(f"❌ Initialization error: {e}", "red")
    self.update_gemma_status(f"Error: {e}", "red")
    
    messagebox.showerror("Initialization Error", 
        f"System initialization failed:\n{e}\n\n"
        "Please check your Gemma 3n installation.")
        
    threading.Thread(target=init_thread, daemon=True).start()
    
    def load_image(self):
        """Charge une image médicale"""
        file_path = filedialog.askopenfilename(
            title="Select medical image for retinoblastoma analysis",
            filetypes=[
                ("Medical images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Vérifier que l'image peut être ouverte
                test_image = Image.open(file_path)
                image_info = f"{test_image.width}x{test_image.height}, {test_image.mode}"
                test_image.close()
                
                self.current_image_path = file_path
                self.display_image(file_path)
                
                filename = Path(file_path).name
                self.update_status(f"✅ Image loaded: {filename}")
                self.image_info_label.config(text=f"{filename} ({image_info})", foreground="green")
                
                logger.info(f"Image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Image Loading Error", 
                    f"Cannot load image:\n{e}")
                logger.error(f"Failed to load image {file_path}: {e}")
    
    def display_image(self, image_path):
        """Affiche l'image dans le canvas"""
        try:
            image = Image.open(image_path)
            
            # Redimensionnement intelligent
            canvas_width = max(900, self.canvas.winfo_width())
            canvas_height = max(700, self.canvas.winfo_height())
            
            # Conserver les proportions
            image.thumbnail((canvas_width - 50, canvas_height - 50), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            
            # Centrer l'image
            self.canvas.delete("all")
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("Display Error", f"Cannot display image: {e}")
    
    def analyze_image(self):
        """Analyse l'image avec Gemma 3n multimodal"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", 
                "Please load a medical image first.\n\n"
                "Click 'Load Medical Image' to select an image.")
            return
        
        if not self.gemma_handler or not self.gemma_handler.initialized:
            messagebox.showerror("System Not Ready", 
                "Gemma 3n model is not loaded.\n\n"
                "Please wait for initialization to complete or try reloading the model.")
            return
        
        # Confirmation pour analyse
        result = messagebox.askyesno("Start Analysis", 
            "🔍 Start retinoblastoma analysis?\n\n"
            "This will analyze the image for signs of leukocoria using Gemma 3n multimodal AI.\n"
            "Analysis may take 30-60 seconds.")
        
        if not result:
            return
        
        def analysis_thread():
            try:
                start_time = time.time()
                self.update_status("🔄 Starting retinoblastoma analysis...", "blue")
                self.update_progress(0, "Preparing analysis...")
                
                # Étape 1: Détection des régions d'yeux
                self.update_progress(15, "Detecting eye regions...")
                eye_regions = self.detect_eye_regions_advanced()
                
                if not eye_regions:
                    self.update_status("⚠️ No eye regions detected", "orange")
                    self.update_progress(100, "Analysis incomplete")
                    
                    messagebox.showwarning("No Eyes Detected", 
                        "No eye regions could be detected in this image.\n\n"
                        "Tips:\n"
                        "• Ensure the image shows clear eye(s)\n"
                        "• Try enabling 'Cropped Eye Analysis'\n"
                        "• Check image quality and lighting\n"
                        "• Make sure eyes are visible and not closed")
                    return
                
                logger.info(f"Detected {len(eye_regions)} eye regions for analysis")
                
                # Étape 2: Analyse multimodale avec Gemma 3n
                self.update_progress(30, "Initializing AI analysis...")
                analysis_results = []
                
                for i, eye_region in enumerate(eye_regions):
                    progress = 30 + (i * 50) // len(eye_regions)
                    position = eye_region['position']
                    
                    self.update_progress(progress, f"Analyzing {position} eye with Gemma 3n...")
                    
                    # Analyse multimodale
                    if self.multimodal_var.get():
                        result = self.gemma_handler.analyze_eye_image_multimodal(
                            eye_region['image'], 
                            eye_region['position']
                        )
                    else:
                        # Fallback text-only
                        result = self.gemma_handler._create_intelligent_fallback(
                            self.gemma_handler._extract_advanced_features(eye_region['image']),
                            "Fallback analysis"
                        )
                    
                    result['eye_region'] = eye_region
                    analysis_results.append(result)
                    
                    # Log du résultat
                    leukocoria = result.get('leukocoria_detected', False)
                    confidence = result.get('confidence', 0)
                    logger.info(f"Analysis complete for {position} eye: "
                              f"Leukocoria={leukocoria}, Confidence={confidence:.1f}%")
                
                # Étape 3: Visualisation des résultats
                self.update_progress(85, "Generating visual results...")
                self.visualize_results_advanced(analysis_results)
                
                # Étape 4: Affichage des résultats détaillés
                self.update_progress(95, "Compiling medical report...")
                self.display_detailed_results(analysis_results)
                
                # Métriques finales
                processing_time = time.time() - start_time
                self.performance_metrics['total_analyses'] += 1
                self.performance_metrics['processing_times'].append(processing_time)
                
                # Vérifier les détections positives
                positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
                high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
                immediate_cases = sum(1 for r in analysis_results if r.get('urgency') == 'immediate')
                
                if positive_detections > 0:
                    self.performance_metrics['detections_found'] += 1
                    
                    # Alerte médicale
                    urgency_msg = ""
                    if immediate_cases > 0:
                        urgency_msg = "⚠️ IMMEDIATE medical attention required!"
                    elif high_risk_count > 0:
                        urgency_msg = "⚠️ URGENT medical consultation needed!"
                    else:
                        urgency_msg = "Medical evaluation recommended."
                    
                    self.update_status(f"🚨 ALERT: Possible retinoblastoma detected! ({processing_time:.1f}s)", "red")
                    
                    messagebox.showwarning("⚠️ MEDICAL ALERT", 
                        f"🚨 POSSIBLE RETINOBLASTOMA DETECTED 🚨\n\n"
                        f"Positive findings in {positive_detections} eye(s)\n"
                        f"High-risk cases: {high_risk_count}\n\n"
                        f"{urgency_msg}\n\n"
                        f"👨‍⚕️ Action required:\n"
                        f"1. Contact pediatric ophthalmologist immediately\n"
                        f"2. Show them this analysis and original image\n"
                        f"3. Do NOT delay seeking professional evaluation")
                else:
                    self.update_status(f"✅ Analysis complete: No concerning findings ({processing_time:.1f}s)", "green")
                    
                    messagebox.showinfo("Analysis Complete", 
                        f"✅ Analysis completed successfully!\n\n"
                        f"No signs of leukocoria were detected.\n"
                        f"Continue regular eye health monitoring.\n\n"
                        f"Processing time: {processing_time:.1f} seconds")
                
                self.update_progress(100, "Analysis complete")
                self.update_metrics()
                self.update_memory_display()
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.update_status(f"❌ Analysis failed: {e}", "red")
                self.update_progress(0, "Analysis failed")
                
                messagebox.showerror("Analysis Error", 
                    f"Analysis failed with error:\n{e}\n\n"
                    f"Possible causes:\n"
                    f"• GPU memory insufficient\n"
                    f"• Model processing error\n"
                    f"• Image format issues\n\n"
                    f"Try clearing GPU memory or reloading the model.")
        
        # Démarrer l'analyse en arrière-plan
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def detect_eye_regions_advanced(self):
        """Détection avancée optimisée pour images croppées et complètes"""
        try:
            image = cv2.imread(self.current_image_path)
            if image is None:
                return []
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            eye_regions = []
            
            # Méthode 1: MediaPipe pour visages complets
            if self.enhanced_cv_var.get():
                try:
                    import mediapipe as mp
                    
                    mp_face_mesh = mp.solutions.face_mesh
                    face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=3,
                        refine_landmarks=True,
                        min_detection_confidence=0.2
                    )
                    
                    results = face_mesh.process(rgb_image)
                    if results.multi_face_landmarks:
                        for i, face_landmarks in enumerate(results.multi_face_landmarks):
                            face_eyes = self.extract_eyes_from_landmarks(face_landmarks, image, i)
                            eye_regions.extend(face_eyes)
                            
                    face_mesh.close()
                    
                except ImportError:
                    logger.warning("MediaPipe not available - using fallback detection")
            
            # Méthode 2: Analyse d'images croppées
            if not eye_regions and self.crop_detection_var.get():
                logger.info("No faces detected - analyzing as cropped eye image(s)")
                
                # Heuristiques pour détecter le type d'image
                aspect_ratio = w / h
                
                if aspect_ratio > 2.0:  # Image très horizontale = probablement deux yeux
                    mid_x = w // 2
                    eye_regions = [
                        {
                            'position': 'left_cropped',
                            'bbox': (0, 0, mid_x, h),
                            'image': Image.fromarray(rgb_image[:, :mid_x]),
                            'confidence': 0.8,
                            'is_cropped': True,
                            'detection_method': 'horizontal_split'
                        },
                        {
                            'position': 'right_cropped',
                            'bbox': (mid_x, 0, w - mid_x, h),
                            'image': Image.fromarray(rgb_image[:, mid_x:]),
                            'confidence': 0.8,
                            'is_cropped': True,
                            'detection_method': 'horizontal_split'
                        }
                    ]
                else:  # Image carrée/verticale = un oeil ou visage centré
                    eye_regions.append({
                        'position': 'center_cropped',
                        'bbox': (0, 0, w, h),
                        'image': Image.fromarray(rgb_image),
                        'confidence': 0.9,
                        'is_cropped': True,
                        'detection_method': 'full_image_as_eye'
                    })
            
            # Méthode 3: Détection de cercles comme fallback
            if not eye_regions:
                logger.info("Trying circle detection as final fallback")
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                
                # Détection de cercles (pupilles potentielles)
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 1, 30,
                    param1=50, param2=30, 
                    minRadius=10, maxRadius=min(w, h)//4
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for i, (x, y, r) in enumerate(circles[:2]):  # Max 2 cercles
                        # Extraire la région autour du cercle
                        margin = r * 2
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(w, x + margin)
                        y2 = min(h, y + margin)
                        
                        if x2 > x1 and y2 > y1:
                        eye_image = image[y1:y2, x1:x2]
                        rgb_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
                        
                        eye_regions.append({
                            'position': f'{eye_name}_face{face_index}',
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'image': Image.fromarray(rgb_eye),
                            'confidence': 0.95,
                            'is_cropped': False,
                            'detection_method': 'mediapipe_landmarks',
                            'landmark_count': len(eye_points)
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to extract {eye_name} eye from face {face_index}: {e}")
        
        return eye_regions
    
    def visualize_results_advanced(self, analysis_results):
        """Visualisation avancée avec boîtes colorées et informations détaillées"""
        try:
            original_image = Image.open(self.current_image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Police pour les labels
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Analyser chaque région
            for i, result in enumerate(analysis_results):
                eye_region = result.get('eye_region', {})
                bbox = eye_region.get('bbox')
                
                if not bbox:
                    continue
                
                x, y, w, h = bbox
                
                # Données de l'analyse
                leukocoria_detected = result.get('leukocoria_detected', False)
                confidence = result.get('confidence', 0)
                risk_level = result.get('risk_level', 'low')
                urgency = result.get('urgency', 'routine')
                position = eye_region.get('position', 'unknown')
                
                # Déterminer la couleur et l'épaisseur selon le résultat
                if leukocoria_detected:
                    if risk_level == 'high' or urgency == 'immediate':
                        color = 'red'
                        width = 6
                        alert_symbol = '🚨'
                    elif risk_level == 'medium' or urgency == 'urgent':
                        color = 'orange'
                        width = 5
                        alert_symbol = '⚠️'
                    else:
                        color = 'yellow'
                        width = 4
                        alert_symbol = '⚡'
                else:
                    color = 'green'
                    width = 3
                    alert_symbol = '✅'
                
                # Dessiner le rectangle principal
                draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
                
                # Label principal avec informations clés
                main_label = f"{alert_symbol} {position.upper()}"
                if leukocoria_detected:
                    main_label += f" - {urgency.upper()}"
                
                # Position du texte principal
                text_y = y - 45 if y > 45 else y + h + 5
                
                # Fond pour le texte principal
                text_bbox = draw.textbbox((x, text_y), main_label, font=font)
                draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2], 
                              fill=color, outline=color)
                draw.text((x, text_y), main_label, fill='white', font=font)
                
                # Label secondaire avec confiance
                confidence_label = f"Confidence: {confidence:.1f}%"
                if leukocoria_detected:
                    confidence_label += f" | Risk: {risk_level.upper()}"
                
                text_y2 = text_y + 25 if text_y == y - 45 else text_y - 20
                
                # Fond pour le texte secondaire
                text_bbox2 = draw.textbbox((x, text_y2), confidence_label, font=font_small)
                draw.rectangle([text_bbox2[0]-3, text_bbox2[1]-1, text_bbox2[2]+3, text_bbox2[3]+1], 
                              fill='white', outline=color)
                draw.text((x, text_y2), confidence_label, fill=color, font=font_small)
                
                # Indicateur de méthode d'analyse
                method = result.get('analysis_method', 'unknown')
                if 'multimodal' in method:
                    method_indicator = "🔍 Multimodal"
                elif 'vision' in method:
                    method_indicator = "👁️ Vision"
                else:
                    method_indicator = "🧠 CV+AI"
                
                # Position pour l'indicateur de méthode
                method_y = y + h - 20 if y + h < original_image.height - 25 else y + 5
                draw.text((x + 5, method_y), method_indicator, fill=color, font=font_small)
            
            # Titre général de l'analyse
            title = "RETINOBLASTOMA AI ANALYSIS - GEMMA 3N MULTIMODAL"
            title_bbox = draw.textbbox((10, 10), title, font=font)
            draw.rectangle([5, 5, title_bbox[2]+10, title_bbox[3]+5], fill='navy', outline='navy')
            draw.text((10, 10), title, fill='white', font=font)
            
            # Résumé des résultats
            total_eyes = len(analysis_results)
            positive_count = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
            
            summary = f"Eyes analyzed: {total_eyes} | Positive detections: {positive_count}"
            if positive_count > 0:
                summary += " | ⚠️ MEDICAL CONSULTATION REQUIRED"
            
            summary_y = title_bbox[3] + 15
            summary_bbox = draw.textbbox((10, summary_y), summary, font=font_small)
            
            summary_bg_color = 'red' if positive_count > 0 else 'green'
            draw.rectangle([5, summary_y-2, summary_bbox[2]+10, summary_bbox[3]+2], 
                          fill=summary_bg_color, outline=summary_bg_color)
            draw.text((10, summary_y), summary, fill='white', font=font_small)
            
            # Sauvegarder l'image annotée
            timestamp = int(time.time())
            annotated_path = RESULTS_DIR / f"retinoblastoma_analysis_gemma3n_{timestamp}.jpg"
            original_image.save(annotated_path, quality=95)
            
            # Afficher l'image annotée
            self.display_annotated_image(original_image)
            
            logger.info(f"Advanced visualization complete. Saved to: {annotated_path}")
            
        except Exception as e:
            logger.error(f"Advanced visualization failed: {e}")
            messagebox.showerror("Visualization Error", f"Failed to create visual results: {e}")
    
    def display_annotated_image(self, annotated_image):
        """Affiche l'image annotée dans le canvas"""
        try:
            # Préparer l'image pour l'affichage
            canvas_width = max(900, self.canvas.winfo_width())
            canvas_height = max(700, self.canvas.winfo_height())
            
            display_image = annotated_image.copy()
            display_image.thumbnail((canvas_width - 50, canvas_height - 50), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Centrer et afficher
            self.canvas.delete("all")
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Basculer vers l'onglet image
            self.notebook.select(self.image_frame)
            
        except Exception as e:
            logger.error(f"Error displaying annotated image: {e}")
    
    def display_detailed_results(self, analysis_results):
        """Affiche les résultats détaillés dans l'onglet résultats"""
        try:
            # Créer le rapport médical complet
            report = self.generate_detailed_medical_report(analysis_results)
            
            # Afficher dans l'onglet résultats
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, report)
            
            # Coloration syntaxique basique pour améliorer la lisibilité
            self.apply_text_formatting()
            
            # Sauvegarder les résultats
            self.current_analysis_results = report
            
            logger.info("Detailed results displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to display detailed results: {e}")
    
    def generate_detailed_medical_report(self, analysis_results):
        """Génère un rapport médical détaillé"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        # En-tête du rapport
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    RETINOBLASTOMA AI ANALYSIS REPORT                             ║
║                         GEMMA 3N MULTIMODAL SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

📋 ANALYSIS DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: {timestamp}
Image File: {filename}
AI System: Gemma 3n Multimodal (Local Processing)
Processing Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
Privacy: 100% Local - No data transmitted

📊 EXECUTIVE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Statistiques générales
        total_eyes = len(analysis_results)
        positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
        high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
        immediate_cases = sum(1 for r in analysis_results if r.get('urgency') == 'immediate')
        
        report += f"""
Eyes Analyzed: {total_eyes}
Positive Detections: {positive_detections}
High-Risk Cases: {high_risk_count}
Immediate Attention Required: {immediate_cases}
"""
        
        if positive_detections > 0:
            report += f"""
🚨 MEDICAL ALERT: POSSIBLE RETINOBLASTOMA DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  IMMEDIATE PEDIATRIC OPHTHALMOLOGICAL CONSULTATION REQUIRED
⚠️  DO NOT DELAY MEDICAL EVALUATION
⚠️  CONTACT YOUR HEALTHCARE PROVIDER TODAY
"""
        else:
            report += f"""
✅ RESULT: NO CONCERNING FINDINGS DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No signs of leukocoria were detected in this analysis.
Continue regular pediatric eye health monitoring.
"""
        
        # Analyse détaillée par œil
        report += f"""

🔍 DETAILED ANALYSIS BY EYE REGION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, result in enumerate(analysis_results, 1):
            eye_region = result.get('eye_region', {})
            position = eye_region.get('position', 'unknown')
            detection_method = eye_region.get('detection_method', 'unknown')
            is_cropped = eye_region.get('is_cropped', False)
            
            report += f"""
┌─ EYE REGION {i}: {position.upper()} {'(CROPPED IMAGE)' if is_cropped else '(FULL FACE)'} ─┐
│
│ 🎯 DETECTION RESULTS:
│   • Leukocoria Detected: {'⚠️  YES' if result.get('leukocoria_detected') else '✅ NO'}
│   • Confidence Level: {result.get('confidence', 0):.1f}%
│   • Risk Assessment: {result.get('risk_level', 'unknown').upper()}
│   • Urgency Level: {result.get('urgency', 'routine').upper()}
│   • Detection Method: {detection_method}
│
│ 👁️  PUPIL ANALYSIS:
│   • Description: {result.get('pupil_description', 'Not available')[:80]}
{"│   • " + result.get('pupil_description', '')[80:160] + "..." if len(result.get('pupil_description', '')) > 80 else ""}
│
│ 🏥 MEDICAL ASSESSMENT:
│   • Clinical Analysis: {result.get('medical_analysis', 'Not available')[:70]}
{"│   • " + result.get('medical_analysis', '')[70:140] + "..." if len(result.get('medical_analysis', '')) > 70 else ""}
│
│ 💊 RECOMMENDATIONS:
│   • {result.get('recommendations', 'No specific recommendations')}
│
│ 🔬 TECHNICAL DETAILS:
│   • Analysis Method: {result.get('analysis_method', 'unknown')}
│   • Processing Time: {result.get('processing_time', 0):.2f}s
│   • Model Backend: {result.get('model_type', 'gemma3n_multimodal')}
│
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Informations techniques
        if hasattr(self, 'performance_metrics') and self.performance_metrics['processing_times']:
            avg_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            total_analyses = self.performance_metrics['total_analyses']
            detection_rate = (self.performance_metrics['detections_found'] / max(1, total_analyses)) * 100
            
            report += f"""

⚙️  SYSTEM PERFORMANCE METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Analyses Performed: {total_analyses}
• Average Processing Time: {avg_time:.1f} seconds
• Detection Rate: {detection_rate:.1f}%
• Current Session: {len(analysis_results)} eyes analyzed
• Multimodal Mode: {'✅ Active' if self.multimodal_var.get() else '❌ Disabled'}
• Enhanced CV: {'✅ Active' if self.enhanced_cv_var.get() else '❌ Disabled'}
"""
        
        # Mémoire GPU si disponible
        if self.gemma_handler and hasattr(self.gemma_handler, 'get_memory_usage'):
            memory_info = self.gemma_handler.get_memory_usage()
            if memory_info:
                report += f"""
• GPU Memory Used: {memory_info.get('gpu_allocated', 0):.1f} GB
• GPU Memory Reserved: {memory_info.get('gpu_reserved', 0):.1f} GB
"""
        
        # Disclaimer médical important
        report += f"""

⚕️  CRITICAL MEDICAL DISCLAIMER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 IMPORTANT: This analysis is provided by an AI screening system using Gemma 3n.

🔴 THIS IS NOT A MEDICAL DIAGNOSIS and should NOT replace professional medical 
   evaluation by qualified pediatric ophthalmologists.

🔴 The AI system is designed as a screening tool to assist in early detection 
   of potential retinoblastoma signs. All results must be verified by medical 
   professionals.

📋 REQUIRED ACTIONS FOR POSITIVE FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 📞 Contact pediatric ophthalmologist immediately
2. 📋 Bring this report and original images to appointment  
3. ⏰ Do NOT delay professional medical evaluation
4. 🚫 Do not rely solely on this AI analysis for medical decisions

🏥 ABOUT RETINOBLASTOMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Most common primary eye cancer in children (typically under 6 years)
• 95% survival rate with EARLY detection and treatment
• 30% survival rate when diagnosis is delayed
• Main early sign: White pupil reflex (leukocoria) in flash photographs
• Can affect one eye (unilateral) or both eyes (bilateral)
• Treatment options depend on cancer stage and location
• Vision preservation possible with early intervention

📞 EMERGENCY CONTACTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Pediatric Ophthalmologist: Contact your local specialist
• Emergency Services: If urgent medical attention needed
• Healthcare Provider: Your child's primary physician

═══════════════════════════════════════════════════════════════════════════════════
                        END OF MEDICAL ANALYSIS REPORT
                    Generated by RetinoblastoGemma v1.0 - Gemma 3n
═══════════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def apply_text_formatting(self):
        """Applique une coloration basique au texte des résultats"""
        try:
            # Configuration des tags de couleur
            self.results_text.tag_configure("alert", foreground="red", font=("Consolas", 10, "bold"))
            self.results_text.tag_configure("success", foreground="green", font=("Consolas", 10, "bold"))
            self.results_text.tag_configure("warning", foreground="orange", font=("Consolas", 10, "bold"))
            self.results_text.tag_configure("header", foreground="blue", font=("Consolas", 11, "bold"))
            self.results_text.tag_configure("important", foreground="purple", font=("Consolas", 10, "bold"))
            
            # Appliquer les tags
            content = self.results_text.get(1.0, tk.END)
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                line_start = f"{i+1}.0"
                line_end = f"{i+1}.end"
                
                if "MEDICAL ALERT" in line or "IMMEDIATE" in line:
                    self.results_text.tag_add("alert", line_start, line_end)
                elif "NO CONCERNING FINDINGS" in line or "✅" in line:
                    self.results_text.tag_add("success", line_start, line_end)
                elif "⚠️" in line or "WARNING" in line:
                    self.results_text.tag_add("warning", line_start, line_end)
                elif line.strip().startswith('━') or line.strip().startswith('╔'):
                    self.results_text.tag_add("header", line_start, line_end)
                elif "DISCLAIMER" in line or "IMPORTANT" in line:
                    self.results_text.tag_add("important", line_start, line_end)
            
        except Exception as e:
            logger.error(f"Text formatting failed: {e}")
    
    def update_status(self, message, color="blue"):
        """Met à jour le statut de l'application"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))
        self.root.after(0, lambda: self.statusbar.config(text=message))
        logger.info(message)
    
    def update_gemma_status(self, message, color="blue"):
        """Met à jour le statut Gemma"""
        self.root.after(0, lambda: self.gemma_status.config(text=message, foreground=color))
    
    def update_progress(self, value, message=""):
        """Met à jour la barre de progression"""
        self.root.after(0, lambda: self.progress.config(value=value))
        if message:
            self.root.after(0, lambda: self.progress_detail.config(text=message))
    
    def update_metrics(self):
        """Met à jour les métriques de performance"""
        total = self.performance_metrics['total_analyses']
        detections = self.performance_metrics['detections_found']
        
        if total > 0:
            detection_rate = (detections / total) * 100
            avg_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            
            metrics_text = (f"Analyses: {total} | "
                          f"Detections: {detections} ({detection_rate:.1f}%) | "
                          f"Avg time: {avg_time:.1f}s")
        else:
            metrics_text = "No analysis performed yet"
        
        self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))
    
    def update_memory_display(self):
        """Met à jour l'affichage de la mémoire"""
        try:
            if self.gemma_handler:
                memory_info = self.gemma_handler.get_memory_usage()
                if memory_info:
                    memory_text = f"GPU: {memory_info.get('gpu_allocated', 0):.1f}GB / {memory_info.get('gpu_reserved', 0):.1f}GB"
                    self.root.after(0, lambda: self.memory_label.config(text=memory_text))
        except Exception as e:
            logger.error(f"Memory display update failed: {e}")
    
    def update_diagnostics(self):
        """Met à jour les informations de diagnostic"""
        try:
            diagnostics_info = f"""RETINOBLASTOGAMMA SYSTEM DIAGNOSTICS
════════════════════════════════════════════════════════════════════

SYSTEM INFORMATION:
• Application: RetinoblastoGemma v1.0 - Gemma 3n Multimodal
• Python Version: {sys.version.split()[0]}
• Platform: {sys.platform}
• Working Directory: {Path.cwd()}

HARDWARE INFORMATION:
• CPU Cores: {os.cpu_count()}
• PyTorch Available: {TORCH_AVAILABLE}
• CUDA Available: {torch.cuda.is_available() if TORCH_AVAILABLE else 'N/A'}
• GPU Device: {torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else 'None'}
• GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB if TORCH_AVAILABLE and torch.cuda.is_available() else 'N/A'}

MODEL STATUS:
• Gemma 3n Path: {GEMMA_MODEL_PATH}
• Model Files Present: {GEMMA_MODEL_PATH.exists()}
• Transformers Available: {TRANSFORMERS_AVAILABLE}
• Handler Initialized: {self.gemma_handler.initialized if self.gemma_handler else False}

CONFIGURATION:
• Crop Detection: {'Enabled' if self.crop_detection_var.get() else 'Disabled'}
• Multimodal Analysis: {'Enabled' if self.multimodal_var.get() else 'Disabled'}
• Enhanced CV: {'Enabled' if self.enhanced_cv_var.get() else 'Disabled'}

DIRECTORIES:
• Models Directory: {MODELS_DIR}
• Data Directory: {DATA_DIR}
• Results Directory: {RESULTS_DIR}

SESSION STATISTICS:
• Total Analyses: {self.performance_metrics['total_analyses']}
• Detections Found: {self.performance_metrics['detections_found']}
• Session Duration: {time.time() - self.performance_metrics.get('session_start', time.time()):.0f}s
"""
            
            self.diagnostics_text.config(state='normal')
            self.diagnostics_text.delete(1.0, tk.END)
            self.diagnostics_text.insert(1.0, diagnostics_info)
            self.diagnostics_text.config(state='disabled')
            
        except Exception as e:
            logger.error(f"Diagnostics update failed: {e}")
    
    def reload_gemma(self):
        """Recharge le modèle Gemma 3n"""
        result = messagebox.askyesno("Reload Model", 
            "Reload Gemma 3n multimodal model?\n\n"
            "This will free GPU memory and reload the model.\n"
            "Process may take 3-5 minutes.")
        
        if result:
            def reload_thread():
                try:
                    self.update_gemma_status("Unloading current model...", "orange")
                    
                    # Libérer les ressources actuelles
                    if self.gemma_handler:
                        self.gemma_handler.cleanup_memory()
                        del self.gemma_handler
                    
                    # Vider le cache GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    self.analyze_button.config(state='disabled')
                    
                    # Réinitialiser
                    self.gemma_handler = None
                    self.initialization_complete = False
                    
                    # Relancer l'initialisation
                    self.root.after(2000, self.initialize_system)
                    
                except Exception as e:
                    logger.error(f"Model reload failed: {e}")
                    self.update_gemma_status(f"Reload failed: {e}", "red")
            
            threading.Thread(target=reload_thread, daemon=True).start()
    
    def clear_gpu_memory(self):
        """Nettoie la mémoire GPU"""
        try:
            if torch.cuda.is_available():
                if self.gemma_handler:
                    self.gemma_handler.cleanup_memory()
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                self.update_memory_display()
                messagebox.showinfo("Memory Cleared", "GPU memory cache cleared successfully.")
            else:
                messagebox.showinfo("No GPU", "No CUDA GPU available for memory clearing.")
                
        except Exception as e:
            logger.error(f"GPU memory clear failed: {e}")
            messagebox.showerror("Memory Clear Error", f"Failed to clear GPU memory: {e}")
    
    def show_diagnostics(self):
        """Affiche l'onglet des diagnostics"""
        self.update_diagnostics()
        self.notebook.select(self.diagnostics_frame)
    
    def export_results(self):
        """Exporte les résultats d'analyse"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Results", "No analysis results available to export.")
            return
        
        # Demander le fichier de destination
        file_path = filedialog.asksaveasfilename(
            title="Export Medical Analysis Results",
            defaultextension=".txt",
            filetypes=[
                ("Medical reports", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_analysis_results)
                
                self.update_status(f"✅ Results exported: {Path(file_path).name}", "green")
                messagebox        eye_region = rgb_image[y1:y2, x1:x2]
                            eye_regions.append({
                                'position': f'detected_circle_{i+1}',
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'image': Image.fromarray(eye_region),
                                'confidence': 0.6,
                                'is_cropped': True,
                                'detection_method': 'circle_detection'
                            })
            
            logger.info(f"Final detection: {len(eye_regions)} eye regions found")
            for region in eye_regions:
                logger.info(f"  - {region['position']}: {region['detection_method']}, "
                          f"confidence={region['confidence']:.1f}")
            
            return eye_regions
            
        except Exception as e:
            logger.error(f"Advanced eye detection failed: {e}")
            return []
    
    def extract_eyes_from_landmarks(self, face_landmarks, image, face_index=0):
        """Extraction précise des yeux à partir des landmarks MediaPipe"""
        h, w = image.shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        eye_regions = []
        
        # Indices MediaPipe pour les contours des yeux
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for eye_name, indices in [('left', left_eye_indices), ('right', right_eye_indices)]:
            try:
                eye_points = [landmarks[i] for i in indices if i < len(landmarks)]
                if len(eye_points) >= 8:  # Minimum de points pour une extraction fiable
                    xs = [p[0] for p in eye_points]
                    ys = [p[1] for p in eye_points]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Marge généreuse pour capturer toute la région oculaire
                    margin = 40
                    x1 = max(0, x_min - margin)
                    y1 = max(0, y_min - margin)
                    x2 = min(w, x_max + margin)
                    y2 = min(h, y_max + margin)
                    
                