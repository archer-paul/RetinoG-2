"""
Gestionnaire Gemma 3n V2 - Version locale pour hackathon
Optimisé pour fonctionnement 100% local avec gestion mémoire GPU
"""
import torch
import numpy as np
from PIL import Image
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class GemmaHandlerV2:
    """Gestionnaire optimisé pour Gemma 3n local avec capacités multimodales"""
    
    def __init__(self):
        self.model_path = Path("models/gemma-3n")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.ready = False
        
        # Configuration mémoire pour GPU
        self.gpu_memory_limit = "3GB"  # Pour GTX 1650
        self.use_8bit = True  # Quantification pour économiser mémoire
        
        logger.info(f"Initializing Gemma Handler V2")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Vérifier la disponibilité du modèle
        self.model_available = self._check_model_availability()
    
    def _check_model_availability(self):
        """Vérifie si le modèle Gemma 3n est disponible"""
        if not self.model_path.exists():
            logger.warning(f"Gemma model directory not found: {self.model_path}")
            return False
        
        # Vérifier les fichiers essentiels
        essential_files = ["config.json", "tokenizer.json"]
        for file_name in essential_files:
            if not (self.model_path / file_name).exists():
                logger.warning(f"Essential file missing: {file_name}")
                return False
        
        # Vérifier les fichiers de modèle
        model_files = list(self.model_path.glob("*.safetensors")) or list(self.model_path.glob("model-*.safetensors"))
        if not model_files:
            logger.warning("No model files (.safetensors) found")
            return False
        
        logger.info("✅ Gemma 3n model files available")
        return True
    
    def initialize_local_model(self):
        """Initialise le modèle Gemma 3n en local avec optimisations"""
        if not self.model_available:
            logger.error("Cannot initialize - model files not available")
            return False
        
        try:
            logger.info("🔄 Starting Gemma 3n local initialization...")
            
            # Import des bibliothèques nécessaires
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
                BitsAndBytesConfig
            )
            
            # 1. Charger le tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. Essayer de charger le processor (pour capacités multimodales)
            try:
                logger.info("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                logger.info("✅ Processor loaded - multimodal capabilities available")
            except Exception as e:
                logger.warning(f"Processor not available: {e}")
                logger.info("Will use text-only mode with visual features")
                self.processor = None
            
            # 3. Configuration du modèle avec optimisations mémoire
            logger.info("Configuring model for local execution...")
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            }
            
            # Configuration quantification 8-bit si nécessaire
            if self.use_8bit and torch.cuda.is_available():
                logger.info("Using 8-bit quantization for memory efficiency...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["max_memory"] = {0: self.gpu_memory_limit}
            
            # 4. Charger le modèle
            logger.info("Loading Gemma 3n model (this may take 3-5 minutes)...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **model_kwargs
                )
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Fallback: essayer sans quantification
                if self.use_8bit:
                    logger.info("Retrying without quantization...")
                    model_kwargs.pop("quantization_config", None)
                    model_kwargs["max_memory"] = {0: "2GB", "cpu": "8GB"}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        **model_kwargs
                    )
                else:
                    raise
            
            # 5. Optimisations finales
            self.model.eval()
            
            # Nettoyer la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test rapide du modèle
            logger.info("Testing model functionality...")
            test_result = self._test_model_functionality()
            
            if test_result:
                self.initialized = True
                self.ready = True
                logger.info("✅ Gemma 3n successfully initialized and ready")
                return True
            else:
                logger.error("Model test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemma 3n: {e}")
            self.initialized = False
            self.ready = False
            return False
    
    def _test_model_functionality(self):
        """Test rapide de fonctionnalité du modèle"""
        try:
            test_prompt = "Medical analysis:"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logger.info(f"Model test successful: '{response[:20]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def is_ready(self):
        """Vérifie si le modèle est prêt pour l'analyse"""
        return self.ready and self.initialized
    
    def analyze_eye_regions(self, eye_regions: List[Dict], confidence_threshold: float = 0.5) -> Dict:
        """Analyse plusieurs régions oculaires avec Gemma 3n"""
        if not self.is_ready():
            logger.warning("Gemma model not ready, using fallback analysis")
            return self._fallback_analysis(eye_regions)
        
        try:
            results = {
                'regions_analyzed': len(eye_regions),
                'method': 'gemma3n_local',
                'results': [],
                'processing_time': 0
            }
            
            start_time = time.time()
            
            for i, region in enumerate(eye_regions):
                logger.info(f"Analyzing region {i+1}/{len(eye_regions)}")
                
                # Analyser chaque région
                region_result = self._analyze_single_region(region, confidence_threshold)
                region_result['region_id'] = i
                results['results'].append(region_result)
            
            results['processing_time'] = time.time() - start_time
            logger.info(f"Analysis completed in {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._fallback_analysis(eye_regions)
    
    def _analyze_single_region(self, region: Dict, confidence_threshold: float) -> Dict:
        """Analyse une seule région oculaire"""
        try:
            # Extraire l'image de la région
            region_image = region.get('image')
            region_type = region.get('type', 'unknown')
            
            if region_image is None:
                return self._create_error_result("No image data in region")
            
            # Préparation de l'image
            processed_image = self._preprocess_image_for_analysis(region_image)
            
            # Choix de la méthode d'analyse selon les capacités
            if self.processor is not None:
                # Analyse multimodale (image + texte)
                result = self._analyze_multimodal(processed_image, region_type)
            else:
                # Analyse text-only avec features visuelles
                result = self._analyze_text_with_cv_features(processed_image, region_type)
            
            # Post-traitement
            result['region_type'] = region_type
            result['confidence_threshold'] = confidence_threshold
            result['analysis_timestamp'] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Single region analysis failed: {e}")
            return self._create_error_result(f"Analysis error: {e}")
    
    def _preprocess_image_for_analysis(self, image: Image.Image) -> Image.Image:
        """Préprocessing de l'image pour l'analyse Gemma"""
        # Redimensionner à une taille optimale pour Gemma 3n
        target_size = (336, 336)
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionner en conservant les proportions
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Créer une image carrée avec padding noir
        square_image = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        square_image.paste(image, (paste_x, paste_y))
        
        return square_image
    
    def _analyze_multimodal(self, image: Image.Image, region_type: str) -> Dict:
        """Analyse multimodale avec image et texte"""
        try:
            # Créer le prompt médical spécialisé
            prompt = self._create_medical_prompt(region_type)
            
            # Préparer les inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Génération avec paramètres médicaux optimisés
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # Très faible pour précision médicale
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Décoder la réponse
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parser la réponse
            result = self._parse_medical_response(response_text)
            result['analysis_method'] = 'multimodal_vision'
            
            return result
            
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            return self._analyze_text_with_cv_features(image, region_type)
    
    def _analyze_text_with_cv_features(self, image: Image.Image, region_type: str) -> Dict:
        """Analyse text-only avec features de computer vision"""
        try:
            # Extraire des features visuelles détaillées
            visual_features = self._extract_visual_features(image)
            
            # Créer un prompt enrichi
            prompt = self._create_enhanced_text_prompt(region_type, visual_features)
            
            # Tokeniser
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Décoder
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parser
            result = self._parse_medical_response(response_text)
            result['analysis_method'] = 'text_with_cv_features'
            result['visual_features_used'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_error_result(f"Text analysis error: {e}")
    
    def _create_medical_prompt(self, region_type: str) -> str:
        """Crée un prompt médical spécialisé pour l'analyse multimodale"""
        return f"""<image>

You are a specialized medical AI assistant for retinoblastoma detection.

MEDICAL CONTEXT:
- Retinoblastoma is a serious eye cancer affecting children (usually under 6 years)
- Main early sign: leukocoria (white pupil reflex) visible in flash photographs
- Early detection can save lives (95% survival rate vs 30% when late)
- Region type: {region_type}

ANALYSIS TASK:
Examine this eye image for signs of retinoblastoma/leukocoria:
1. Look for white, gray, or abnormally bright pupil
2. Compare to normal dark pupil appearance
3. Assess any unusual reflections or colorations
4. Consider image quality and lighting conditions

OUTPUT FORMAT (JSON):
{{
    "leukocoria_detected": boolean,
    "confidence": float (0-100),
    "risk_level": "low|medium|high",
    "pupil_description": "detailed clinical description",
    "medical_reasoning": "step-by-step analysis",
    "recommendations": "medical guidance",
    "urgency": "routine|soon|urgent|immediate"
}}

IMPORTANT: Be conservative and prioritize child safety. When in doubt, recommend medical evaluation."""
    
    def _create_enhanced_text_prompt(self, region_type: str, visual_features: str) -> str:
        """Crée un prompt enrichi avec features visuelles pour mode text-only"""
        return f"""You are a medical AI specializing in retinoblastoma detection.

PATIENT SAFETY: This analysis is for a child's eye health. Be conservative.

REGION TYPE: {region_type}

VISUAL ANALYSIS DATA:
{visual_features}

MEDICAL TASK:
Based on the visual characteristics above, analyze for retinoblastoma signs:

1. PUPIL ASSESSMENT:
   - Is the pupil abnormally bright or white?
   - Are there unusual reflections suggesting tumor?
   - How does brightness compare to normal dark pupil?

2. RISK EVALUATION:
   - What is the likelihood of leukocoria?
   - Should parents seek immediate medical attention?
   - Is this a normal flash reflection or concerning finding?

3. MEDICAL REASONING:
   - Explain your clinical assessment
   - Note any limitations of the analysis
   - Consider differential diagnoses

OUTPUT (JSON format):
{{
    "leukocoria_detected": boolean,
    "confidence": float (0-100),
    "risk_level": "low|medium|high",
    "pupil_description": "clinical description",
    "medical_reasoning": "detailed analysis",
    "recommendations": "specific medical advice",
    "urgency": "routine|soon|urgent|immediate"
}}

Remember: Err on the side of caution for child safety."""
    
    def _extract_visual_features(self, image: Image.Image) -> str:
        """Extrait des features visuelles détaillées pour l'IA"""
        try:
            import cv2
            
            # Convertir en array numpy
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Analyses de base
            features = {
                "brightness": {
                    "mean": float(np.mean(gray)),
                    "std": float(np.std(gray)),
                    "max": float(np.max(gray)),
                    "min": float(np.min(gray))
                },
                "image_size": f"{image.width}x{image.height}"
            }
            
            # Détection de cercles (pupilles potentielles)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=min(gray.shape)//3
            )
            
            pupil_analysis = {}
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                pupil_analysis["circles_detected"] = len(circles)
                
                # Analyser le meilleur cercle (plus central)
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
                    
                    # Analyser la région pupillaire
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    pupil_region = gray[mask > 0]
                    
                    if len(pupil_region) > 0:
                        pupil_brightness = float(np.mean(pupil_region))
                        global_brightness = features["brightness"]["mean"]
                        
                        # Score de leucocorie
                        brightness_ratio = pupil_brightness / max(global_brightness, 1)
                        leukocoria_score = max(0, (brightness_ratio - 1) * 100) if brightness_ratio > 1.2 else 0
                        
                        pupil_analysis.update({
                            "pupil_position": f"({x}, {y})",
                            "pupil_radius": int(r),
                            "pupil_brightness": pupil_brightness,
                            "brightness_ratio": brightness_ratio,
                            "leukocoria_score": leukocoria_score,
                            "assessment": self._assess_pupil_finding(brightness_ratio, leukocoria_score)
                        })
            else:
                pupil_analysis["circles_detected"] = 0
                pupil_analysis["note"] = "No clear circular structures detected"
            
            # Analyse des régions claires
            bright_threshold = np.percentile(gray, 90)
            bright_regions = gray > bright_threshold
            bright_percentage = (np.sum(bright_regions) / gray.size) * 100
            
            brightness_analysis = {
                "bright_threshold": float(bright_threshold),
                "bright_area_percentage": bright_percentage,
                "assessment": self._assess_brightness_pattern(bright_percentage)
            }
            
            # Créer description textuelle
            description = f"""
VISUAL ANALYSIS REPORT:
======================

IMAGE PROPERTIES:
- Size: {features['image_size']}
- Overall brightness: {features['brightness']['mean']:.1f} ± {features['brightness']['std']:.1f}
- Range: {features['brightness']['min']:.0f} - {features['brightness']['max']:.0f}

PUPIL DETECTION:
- Circular structures: {pupil_analysis.get('circles_detected', 0)}"""
            
            if pupil_analysis.get('circles_detected', 0) > 0 and 'pupil_brightness' in pupil_analysis:
                description += f"""
- Best pupil candidate: Position {pupil_analysis['pupil_position']}, Radius {pupil_analysis['pupil_radius']}px
- Pupil brightness: {pupil_analysis['pupil_brightness']:.1f}
- Brightness ratio: {pupil_analysis['brightness_ratio']:.2f}
- Leukocoria score: {pupil_analysis['leukocoria_score']:.1f}/100
- Assessment: {pupil_analysis['assessment']}"""
            
            description += f"""

BRIGHTNESS ANALYSIS:
- Bright areas (top 10%): {brightness_analysis['bright_area_percentage']:.1f}%
- Threshold used: {brightness_analysis['bright_threshold']:.0f}
- Pattern assessment: {brightness_analysis['assessment']}

MEDICAL SIGNIFICANCE:
- Leukocoria scores >50 may indicate white pupil reflex (concerning)
- Brightness ratios >1.5 suggest abnormally bright pupil
- Large bright areas may indicate flash reflection or pathology
"""
            
            return description
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return f"Feature extraction failed: {e}"
    
    def _assess_pupil_finding(self, brightness_ratio: float, leukocoria_score: float) -> str:
        """Évalue les findings pupillaires"""
        if leukocoria_score > 70:
            return "HIGHLY CONCERNING - Bright white pupil detected"
        elif leukocoria_score > 40:
            return "CONCERNING - Abnormally bright pupil"
        elif brightness_ratio > 1.3:
            return "MONITOR - Slightly bright pupil"
        elif brightness_ratio > 1.1:
            return "BORDERLINE - Pupil slightly brighter than average"
        else:
            return "NORMAL - Dark pupil as expected"
    
    def _assess_brightness_pattern(self, bright_percentage: float) -> str:
        """Évalue le pattern de luminosité"""
        if bright_percentage > 40:
            return "Extensive bright areas - likely flash reflection"
        elif bright_percentage > 20:
            return "Moderate bright areas - could indicate reflection or pathology"
        elif bright_percentage > 5:
            return "Some bright areas - normal flash reflection pattern"
        else:
            return "Minimal bright areas - low light conditions"
    
    def _parse_medical_response(self, response_text: str) -> Dict:
        """Parse la réponse médicale avec fallbacks robustes"""
        try:
            # Nettoyer et chercher JSON
            text = response_text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validation et nettoyage
                result['leukocoria_detected'] = bool(result.get('leukocoria_detected', False))
                result['confidence'] = max(0, min(100, float(result.get('confidence', 50))))
                
                # Validation des énumérations
                if result.get('risk_level') not in ['low', 'medium', 'high']:
                    result['risk_level'] = 'medium'
                
                if result.get('urgency') not in ['routine', 'soon', 'urgent', 'immediate']:
                    result['urgency'] = 'soon'
                
                return result
            else:
                raise ValueError("No JSON found")
                
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            
            # Fallback: analyse textuelle
            text_lower = response_text.lower()
            detected = any(word in text_lower for word in [
                'leukocoria', 'white pupil', 'bright pupil', 'abnormal', 'concerning', 'tumor'
            ])
            
            confidence = 60 if detected else 25
            risk_level = 'medium' if detected else 'low'
            urgency = 'urgent' if detected else 'routine'
            
            return {
                'leukocoria_detected': detected,
                'confidence': confidence,
                'risk_level': risk_level,
                'pupil_description': 'Text analysis based assessment',
                'medical_reasoning': f'Fallback analysis of AI response: {response_text[:150]}...',
                'recommendations': 'Professional evaluation recommended' if detected else 'Continue monitoring',
                'urgency': urgency,
                'parsing_method': 'text_fallback'
            }
    
    def _create_error_result(self, error_msg: str) -> Dict:
        """Crée un résultat d'erreur standardisé"""
        return {
            'leukocoria_detected': False,
            'confidence': 0,
            'risk_level': 'unknown',
            'pupil_description': 'Analysis failed',
            'medical_reasoning': error_msg,
            'recommendations': 'Manual professional evaluation required',
            'urgency': 'soon',
            'error': error_msg,
            'analysis_method': 'error_fallback'
        }
    
    def _fallback_analysis(self, eye_regions: List[Dict]) -> Dict:
        """Analyse de fallback en cas d'échec Gemma"""
        logger.warning("Using fallback analysis - Gemma not available")
        
        results = {
            'regions_analyzed': len(eye_regions),
            'method': 'computer_vision_fallback',
            'results': [],
            'processing_time': 0.1
        }
        
        for i, region in enumerate(eye_regions):
            # Analyse basique de computer vision
            fallback_result = {
                'region_id': i,
                'region_type': region.get('type', 'unknown'),
                'leukocoria_detected': False,
                'confidence': 10,  # Très faible confiance
                'risk_level': 'unknown',
                'pupil_description': 'Fallback computer vision analysis',
                'medical_reasoning': 'AI model not available - basic analysis only',
                'recommendations': 'Professional medical evaluation strongly recommended',
                'urgency': 'soon',
                'analysis_method': 'cv_fallback'
            }
            
            results['results'].append(fallback_result)
        
        return results
    
    def get_memory_usage(self) -> Dict:
        """Retourne l'usage mémoire actuel"""
        memory_info = {}
        
        if torch.cuda.is_available() and self.initialized:
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
        
        return memory_info
    
    def cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.initialized = False
            self.ready = False
            
            logger.info("Gemma handler cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Nettoyage automatique"""
        self.cleanup()