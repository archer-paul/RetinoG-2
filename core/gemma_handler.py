"""
Gestionnaire spécialisé pour Gemma 3n multimodal
Résout les problèmes de mémoire GPU et utilise les capacités vision
"""
import torch
import numpy as np
from PIL import Image
import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class Gemma3nMultimodalHandler:
    """Gestionnaire optimisé pour Gemma 3n multimodal avec vision"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        logger.info(f"Initializing Gemma 3n Multimodal Handler")
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    def load_model_optimized(self, progress_callback=None):
        """Charge le modèle avec optimisations mémoire"""
        try:
            if progress_callback:
                progress_callback(10, "Importing libraries...")
            
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
                BitsAndBytesConfig
            )
            import accelerate
            
            if progress_callback:
                progress_callback(20, "Loading tokenizer...")
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if progress_callback:
                progress_callback(40, "Loading processor...")
            
            # Charger le processor pour les images
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                logger.info("✅ Processor loaded for multimodal capabilities")
            except Exception as e:
                logger.warning(f"No processor found, using tokenizer only: {e}")
                self.processor = None
            
            if progress_callback:
                progress_callback(60, "Configuring model loading...")
            
            # Configuration optimisée pour votre GTX 1650 (4GB VRAM)
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,  # Réduire l'usage mémoire
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "max_memory": {0: "3GB"},  # Limiter à 3GB pour votre GTX 1650
                "offload_folder": "offload_temp",  # Dossier temporaire pour offload
            }
            
            # Alternative: Quantification 8-bit si problème de mémoire
            try:
                if progress_callback:
                    progress_callback(70, "Loading model (may take several minutes)...")
                
                # Essayer d'abord le chargement normal
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **model_kwargs
                )
                
            except Exception as e:
                logger.warning(f"Normal loading failed: {e}")
                logger.info("Trying with 8-bit quantization...")
                
                if progress_callback:
                    progress_callback(75, "Loading with 8-bit quantization...")
                
                # Configuration 8-bit pour économiser la mémoire
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                model_kwargs.update({
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "max_memory": {0: "3GB", "cpu": "8GB"}
                })
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **model_kwargs
                )
            
            if progress_callback:
                progress_callback(90, "Optimizing model...")
            
            # Optimisations finales
            self.model.eval()
            
            # Vider le cache GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if progress_callback:
                progress_callback(100, "Model ready!")
            
            self.initialized = True
            logger.info("✅ Gemma 3n Multimodal loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n: {e}")
            return False
    
    def analyze_eye_image_multimodal(self, image_pil: Image.Image, eye_position: str = "unknown") -> Dict:
        """Analyse multimodale avec image + texte"""
        if not self.initialized:
            return self._create_fallback_result("Model not initialized")
        
        try:
            start_time = time.time()
            
            # Préparer l'image pour Gemma 3n
            processed_image = self._preprocess_image_for_gemma(image_pil)
            
            # Créer le prompt médical spécialisé
            medical_prompt = self._create_multimodal_prompt(eye_position)
            
            # Préparer les inputs selon les capacités du modèle
            if self.processor is not None:
                # Mode multimodal complet
                result = self._analyze_with_vision_model(processed_image, medical_prompt)
            else:
                # Mode text-only avec description d'image
                result = self._analyze_text_only_with_features(processed_image, medical_prompt)
            
            # Post-traitement
            result['processing_time'] = time.time() - start_time
            result['model_type'] = 'gemma3n_multimodal'
            result['analysis_mode'] = 'vision' if self.processor else 'text_with_cv'
            
            return result
            
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            return self._create_fallback_result(f"Analysis error: {e}")
    
    def _preprocess_image_for_gemma(self, image_pil: Image.Image) -> Image.Image:
        """Préprocessing optimisé pour Gemma 3n vision"""
        # Redimensionner à une taille compatible
        # Gemma 3n vision attend généralement 224x224 ou 336x336
        target_size = (336, 336)
        
        # Conserver les proportions avec padding si nécessaire
        image_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Créer une image carrée avec padding
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - image_pil.width) // 2
        paste_y = (target_size[1] - image_pil.height) // 2
        new_image.paste(image_pil, (paste_x, paste_y))
        
        return new_image
    
    def _create_multimodal_prompt(self, eye_position: str) -> str:
        """Prompt optimisé pour Gemma 3n multimodal"""
        return f"""<image>

Analyze this eye image for signs of retinoblastoma (leukocoria).

MEDICAL CONTEXT:
- Retinoblastoma is a serious eye cancer in children
- Main sign: White pupil reflex (leukocoria) in photos
- Eye position: {eye_position}
- Critical for early detection

ANALYSIS REQUIRED:
1. Examine pupil for white/abnormal coloration
2. Compare to normal dark pupil appearance
3. Assess risk level for retinoblastoma
4. Provide medical recommendations

OUTPUT FORMAT (JSON):
{{
    "leukocoria_detected": boolean,
    "confidence": float (0-100),
    "risk_level": "low|medium|high", 
    "pupil_description": "detailed description",
    "medical_analysis": "clinical reasoning",
    "recommendations": "medical advice",
    "urgency": "routine|soon|urgent|immediate"
}}

Focus on medical accuracy and child safety. Be conservative in assessment."""
    
    def _analyze_with_vision_model(self, image: Image.Image, prompt: str) -> Dict:
        """Analyse avec les capacités vision complètes"""
        try:
            # Préparer les inputs multimodaux
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Génération avec paramètres optimisés pour médical
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
            
            # Parser la réponse JSON
            result = self._parse_medical_response(response_text)
            result['analysis_method'] = 'multimodal_vision'
            
            return result
            
        except Exception as e:
            logger.error(f"Vision model analysis failed: {e}")
            return self._analyze_text_only_with_features(image, prompt)
    
    def _analyze_text_only_with_features(self, image: Image.Image, prompt: str) -> Dict:
        """Analyse text-only avec features CV en input"""
        try:
            # Extraire des features visuelles détaillées
            visual_features = self._extract_detailed_visual_features(image)
            
            # Créer un prompt enrichi avec les features
            enhanced_prompt = f"""{prompt}

VISUAL ANALYSIS DATA:
{visual_features}

Based on these visual characteristics and medical knowledge, provide the JSON analysis."""
            
            # Tokeniser
            inputs = self.tokenizer(
                enhanced_prompt,
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
            result['visual_features'] = visual_features
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_fallback_result(f"Text analysis error: {e}")
    
    def _extract_detailed_visual_features(self, image: Image.Image) -> str:
        """Extraction de features visuelles détaillées pour l'IA"""
        try:
            import cv2
            
            # Convertir en array numpy
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Analyses avancées
            features = {
                "brightness_stats": {
                    "mean": float(np.mean(gray)),
                    "std": float(np.std(gray)),
                    "min": float(np.min(gray)),
                    "max": float(np.max(gray))
                },
                "image_properties": {
                    "width": image.width,
                    "height": image.height,
                    "aspect_ratio": image.width / image.height
                }
            }
            
            # Détection de cercles (pupilles)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=min(gray.shape)//3
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                features["pupil_detection"] = {
                    "circles_found": len(circles),
                    "analysis": []
                }
                
                # Analyser chaque cercle détecté
                for i, (x, y, r) in enumerate(circles[:3]):  # Max 3 cercles
                    # Région pupillaire
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    pupil_region = gray[mask > 0]
                    
                    if len(pupil_region) > 0:
                        pupil_brightness = float(np.mean(pupil_region))
                        pupil_contrast = float(np.std(pupil_region))
                        
                        # Score de leucocorie
                        global_brightness = features["brightness_stats"]["mean"]
                        brightness_ratio = pupil_brightness / max(global_brightness, 1)
                        
                        leukocoria_score = 0
                        if brightness_ratio > 1.3:  # Pupille plus claire que moyenne
                            leukocoria_score = min(100, (brightness_ratio - 1) * 100)
                        
                        circle_analysis = {
                            "position": f"({x}, {y})",
                            "radius": int(r),
                            "brightness": pupil_brightness,
                            "contrast": pupil_contrast,
                            "brightness_ratio": brightness_ratio,
                            "leukocoria_score": leukocoria_score,
                            "assessment": self._assess_pupil_brightness(pupil_brightness, global_brightness)
                        }
                        
                        features["pupil_detection"]["analysis"].append(circle_analysis)
            else:
                features["pupil_detection"] = {
                    "circles_found": 0,
                    "note": "No circular structures detected"
                }
            
            # Analyse de texture
            # Gradient pour détecter les contours
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features["texture_analysis"] = {
                "edge_density": float(np.mean(gradient_magnitude)),
                "edge_variation": float(np.std(gradient_magnitude))
            }
            
            # Analyse de régions claires (potentielle leucocorie)
            bright_threshold = np.percentile(gray, 85)  # 15% des pixels les plus clairs
            bright_regions = gray > bright_threshold
            bright_area_ratio = np.sum(bright_regions) / gray.size
            
            features["brightness_analysis"] = {
                "bright_threshold": float(bright_threshold),
                "bright_area_percentage": float(bright_area_ratio * 100),
                "max_bright_value": float(np.max(gray)),
                "bright_regions_assessment": self._assess_brightness_pattern(bright_area_ratio, bright_threshold)
            }
            
            # Créer une description textuelle structurée
            description = f"""
VISUAL ANALYSIS REPORT:
=======================

IMAGE PROPERTIES:
- Dimensions: {features['image_properties']['width']}x{features['image_properties']['height']}
- Aspect ratio: {features['image_properties']['aspect_ratio']:.2f}

BRIGHTNESS ANALYSIS:
- Overall brightness: {features['brightness_stats']['mean']:.1f} (std: {features['brightness_stats']['std']:.1f})
- Range: {features['brightness_stats']['min']:.0f} - {features['brightness_stats']['max']:.0f}
- Bright regions: {features['brightness_analysis']['bright_area_percentage']:.1f}%
- Assessment: {features['brightness_analysis']['bright_regions_assessment']}

PUPIL DETECTION:
- Circular structures found: {features['pupil_detection']['circles_found']}"""
            
            if features['pupil_detection']['circles_found'] > 0:
                for i, analysis in enumerate(features['pupil_detection']['analysis']):
                    description += f"""
- Pupil {i+1}: Position {analysis['position']}, Radius {analysis['radius']}px
  - Brightness: {analysis['brightness']:.1f} (ratio: {analysis['brightness_ratio']:.2f})
  - Leukocoria score: {analysis['leukocoria_score']:.1f}/100
  - Assessment: {analysis['assessment']}"""
            else:
                description += "\n- No clear pupil structures detected"
            
            description += f"""

TEXTURE ANALYSIS:
- Edge density: {features['texture_analysis']['edge_density']:.1f}
- Edge variation: {features['texture_analysis']['edge_variation']:.1f}

MEDICAL SIGNIFICANCE:
- High leukocoria scores (>50) may indicate white pupil reflex
- Bright pupils compared to surrounding areas are concerning
- Multiple circular structures may indicate both eyes visible"""
            
            return description
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return f"Feature extraction failed: {e}"
    
    def _assess_pupil_brightness(self, pupil_brightness: float, global_brightness: float) -> str:
        """Évalue la luminosité pupillaire"""
        ratio = pupil_brightness / max(global_brightness, 1)
        
        if ratio > 1.5:
            return "VERY BRIGHT - High concern for leukocoria"
        elif ratio > 1.3:
            return "BRIGHT - Moderate concern for leukocoria"
        elif ratio > 1.1:
            return "SLIGHTLY BRIGHT - Monitor for changes"
        else:
            return "NORMAL - Dark pupil as expected"
    
    def _assess_brightness_pattern(self, bright_ratio: float, threshold: float) -> str:
        """Évalue le pattern de luminosité globale"""
        if bright_ratio > 0.3:
            return "Large bright areas detected - possible flash reflection or leukocoria"
        elif bright_ratio > 0.15:
            return "Moderate bright areas - could indicate abnormal reflection"
        elif bright_ratio > 0.05:
            return "Small bright areas - normal flash reflection likely"
        else:
            return "Minimal bright areas - low lighting conditions"
    
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
            detected = any(word in response_text.lower() 
                         for word in ['leukocoria', 'white', 'bright', 'abnormal', 'concerning'])
            
            confidence = 70 if detected else 30
            
            return {
                'leukocoria_detected': detected,
                'confidence': confidence,
                'risk_level': 'medium' if detected else 'low',
                'pupil_description': 'AI text analysis based fallback',
                'medical_analysis': f'Text analysis: {response_text[:200]}...',
                'recommendations': 'Professional evaluation recommended' if detected else 'Continue monitoring',
                'urgency': 'urgent' if detected else 'routine',
                'parsing_method': 'text_fallback'
            }
    
    def _create_fallback_result(self, error_msg: str) -> Dict:
        """Résultat de fallback en cas d'erreur"""
        return {
            'leukocoria_detected': False,
            'confidence': 0,
            'risk_level': 'unknown',
            'pupil_description': 'Analysis failed',
            'medical_analysis': error_msg,
            'recommendations': 'Manual professional evaluation required',
            'urgency': 'soon',
            'error': error_msg,
            'analysis_method': 'fallback'
        }
    
    def get_memory_usage(self) -> Dict:
        """Retourne l'usage mémoire actuel"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
        
        return memory_info
    
    def cleanup_memory(self):
        """Nettoie la mémoire GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("GPU memory cleaned")
