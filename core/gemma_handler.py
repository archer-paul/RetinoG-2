"""
Gestionnaire Gemma 3n optimisé avec Google AI Edge
Architecture hybride pour performance maximale sur device
"""
import torch
import numpy as np
from PIL import Image
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from dataclasses import dataclass
from enum import Enum

# Google AI Edge imports (adaptés selon disponibilité)
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logging.warning("Google AI non disponible, utilisation du mode local")

# Transformers pour Gemma local
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, AutoProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config.settings import *

logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Types de backend disponibles"""
    GOOGLE_AI_EDGE = "google_ai_edge"
    LOCAL_TRANSFORMERS = "local_transformers"
    HYBRID = "hybrid"
    SIMULATION = "simulation"

@dataclass
class AnalysisResult:
    """Structure standardisée des résultats d'analyse"""
    leukocoria_detected: bool
    confidence: float
    risk_level: str
    pupil_color: str
    description: str
    recommendations: str
    processing_time: float
    model_backend: str
    technical_details: Dict = None

class GemmaHandler:
    """Gestionnaire Gemma 3n optimisé pour Google AI Edge Prize"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = self._determine_best_backend()
        
        # Modèles et configurations
        self.local_model = None
        self.local_tokenizer = None
        self.google_model = None
        
        # Cache et optimisations
        self.analysis_cache = {}
        self.performance_metrics = {
            'total_analyses': 0,
            'average_processing_time': 0,
            'cache_hits': 0,
            'backend_usage': {backend.value: 0 for backend in ModelBackend}
        }
        
        # Initialisation
        self._initialize_backend()
        self._setup_medical_prompts()
    
    def _determine_best_backend(self) -> ModelBackend:
        """Détermine le meilleur backend selon l'environnement"""
        if GOOGLE_API_KEY and GOOGLE_AI_AVAILABLE:
            logger.info("🌐 Google AI Edge détecté - Mode hybride activé")
            return ModelBackend.HYBRID
        elif TRANSFORMERS_AVAILABLE and GEMMA_LOCAL_PATH.exists():
            logger.info("💻 Modèle local détecté - Mode transformers")
            return ModelBackend.LOCAL_TRANSFORMERS
        else:
            logger.warning("⚠️ Mode simulation - Fonctionnalité limitée")
            return ModelBackend.SIMULATION
    
    def _initialize_backend(self):
        """Initialise le(s) backend(s) disponible(s)"""
        try:
            if self.backend in [ModelBackend.GOOGLE_AI_EDGE, ModelBackend.HYBRID]:
                self._setup_google_ai_edge()
            
            if self.backend in [ModelBackend.LOCAL_TRANSFORMERS, ModelBackend.HYBRID]:
                self._setup_local_transformers()
                
            logger.info(f"✅ Backend initialisé: {self.backend.value}")
            
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            self.backend = ModelBackend.SIMULATION
    
    def _setup_google_ai_edge(self):
        """Configure Google AI Edge pour performances optimales"""
        if not GOOGLE_AI_AVAILABLE:
            return
        
        try:
            # Configuration Google AI avec paramètres optimisés
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Configuration du modèle pour usage médical
            generation_config = {
                "temperature": 0.1,  # Très faible pour cohérence médicale
                "top_p": 0.8,
                "top_k": 20,
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
            }
            
            # Paramètres de sécurité pour contenu médical
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Initialiser le modèle
            self.google_model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",  # Sera remplacé par gemma-3n
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self._get_medical_system_prompt()
            )
            
            logger.info("✅ Google AI Edge configuré")
            
        except Exception as e:
            logger.error(f"❌ Erreur Google AI Edge: {e}")
            raise
    
    def _setup_local_transformers(self):
        """Configure le modèle local Transformers"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            model_path = str(GEMMA_LOCAL_PATH)
            
            logger.info(f"🔄 Chargement du modèle local: {model_path}")
            
            # Configuration optimisée pour inférence
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Chargement du tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Chargement du modèle
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Optimisations pour inférence
            if torch.cuda.is_available():
                self.local_model = self.local_model.half()
            
            self.local_model.eval()
            
            # Configuration du pad token si nécessaire
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            logger.info("✅ Modèle local chargé et optimisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur modèle local: {e}")
            raise
    
    def _setup_medical_prompts(self):
        """Configure les prompts spécialisés pour l'analyse médicale"""
        self.medical_prompts = {
            "system": self._get_medical_system_prompt(),
            "analysis": self._get_analysis_prompt_template(),
            "batch": self._get_batch_analysis_prompt()
        }
    
    def _get_medical_system_prompt(self) -> str:
        """Prompt système optimisé pour l'analyse médicale"""
        return """You are a specialized AI assistant for retinoblastoma screening through leukocoria detection.

MEDICAL CONTEXT:
- Retinoblastoma is a rare but serious eye cancer in children
- Leukocoria (white pupil reflex) is the most common early sign
- Early detection saves lives and preserves vision
- 95% survival rate with early detection vs 30% when late

ANALYSIS GUIDELINES:
1. Focus specifically on white or abnormal pupil reflexes
2. Consider lighting conditions and image quality
3. Distinguish pathological from physiological leukocoria
4. Provide confidence levels and clear recommendations
5. Always emphasize need for professional medical evaluation

RESPONSE FORMAT:
Always respond in valid JSON with these exact fields:
{
  "leukocoria_detected": boolean,
  "confidence": float (0-100),
  "risk_level": "low|medium|high",
  "pupil_color": "description",
  "description": "detailed observation",
  "recommendations": "medical guidance",
  "technical_details": {
    "image_quality": "assessment",
    "lighting_conditions": "assessment",
    "pupil_visibility": "assessment"
  }
}"""
    
    def _get_analysis_prompt_template(self) -> str:
        """Template pour l'analyse d'une région oculaire"""
        return """Analyze this eye region image for signs of leukocoria (white pupil reflex) that may indicate retinoblastoma.

EYE POSITION: {eye_position}
IMAGE QUALITY: {image_quality}

SPECIFIC ANALYSIS REQUIRED:
1. Examine the pupil for any white, gray, or abnormal coloration
2. Assess if the reflection appears pathological vs physiological
3. Consider the image lighting and angle
4. Evaluate the overall visibility and image quality

CLINICAL CONTEXT:
- This is a screening tool for early detection
- False positives are preferable to false negatives
- Any concerning findings warrant immediate medical evaluation
- Consider the child's age and typical presentation patterns

Please provide a detailed analysis in the specified JSON format."""
    
    def _get_batch_analysis_prompt(self) -> str:
        """Prompt pour l'analyse en lot"""
        return """Analyze multiple eye regions from the same individual for consistency in leukocoria detection.

INSTRUCTIONS:
1. Analyze each eye region individually
2. Look for consistent patterns across images
3. Consider temporal progression if timestamps available
4. Provide overall risk assessment
5. Recommend urgency level for medical consultation

Focus on:
- Consistency of findings across multiple images
- Progressive changes over time
- Bilateral vs unilateral presentation
- Image quality variations"""
    
    def analyze_eye_region(self, eye_image: Image.Image, 
                          eye_position: str = "unknown",
                          use_cache: bool = True,
                          force_backend: Optional[ModelBackend] = None) -> AnalysisResult:
        """
        Analyse avancée d'une région oculaire avec optimisations Edge
        
        Args:
            eye_image: Image PIL de la région oculaire
            eye_position: Position de l'œil ("left", "right", "unknown")
            use_cache: Utiliser le cache pour accélérer
            force_backend: Forcer un backend spécifique
            
        Returns:
            AnalysisResult avec tous les détails
        """
        start_time = time.time()
        
        try:
            # Vérification du cache
            if use_cache:
                cache_key = self._generate_cache_key(eye_image, eye_position)
                if cache_key in self.analysis_cache:
                    self.performance_metrics['cache_hits'] += 1
                    cached_result = self.analysis_cache[cache_key]
                    cached_result.processing_time = time.time() - start_time
                    return cached_result
            
            # Préparation de l'image
            processed_image = self._preprocess_image_for_analysis(eye_image)
            image_quality = self._assess_image_quality(processed_image)
            
            # Sélection du backend
            backend = force_backend or self._select_optimal_backend(image_quality)
            
            # Analyse selon le backend
            if backend == ModelBackend.GOOGLE_AI_EDGE:
                result = self._analyze_with_google_ai(processed_image, eye_position, image_quality)
            elif backend == ModelBackend.LOCAL_TRANSFORMERS:
                result = self._analyze_with_local_model(processed_image, eye_position, image_quality)
            else:
                result = self._analyze_with_simulation(processed_image, eye_position, image_quality)
            
            # Post-traitement et optimisations
            result = self._post_process_result(result, backend, time.time() - start_time)
            
            # Mise en cache
            if use_cache:
                self.analysis_cache[cache_key] = result
            
            # Métriques
            self._update_metrics(backend, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur d'analyse: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _preprocess_image_for_analysis(self, image: Image.Image) -> Image.Image:
        """Préprocessing avancé pour optimiser l'analyse"""
        # Conversion en format standard
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionnement optimal pour Gemma
        target_size = (224, 224)  # Optimisé pour vision models
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Amélioration de la qualité
        image_array = np.array(image)
        
        # Correction gamma pour améliorer le contraste des pupilles
        gamma = 1.2
        image_array = np.power(image_array / 255.0, 1/gamma) * 255
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Réduction du bruit tout en préservant les détails
        image_array = cv2.bilateralFilter(image_array, 9, 75, 75)
        
        return Image.fromarray(image_array)
    
    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """Évalue la qualité de l'image pour optimiser l'analyse"""
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Métriques de qualité
        quality_metrics = {
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'overall_quality': 'good'
        }
        
        # Classification de la qualité
        if quality_metrics['sharpness'] < 100:
            quality_metrics['overall_quality'] = 'poor'
        elif quality_metrics['sharpness'] < 500:
            quality_metrics['overall_quality'] = 'fair'
        
        return quality_metrics
    
    def _select_optimal_backend(self, image_quality: Dict) -> ModelBackend:
        """Sélectionne le backend optimal selon la qualité de l'image"""
        if self.backend == ModelBackend.SIMULATION:
            return ModelBackend.SIMULATION
        
        # Pour images de haute qualité, utiliser Google AI si disponible
        if (image_quality['overall_quality'] == 'good' and 
            self.backend in [ModelBackend.GOOGLE_AI_EDGE, ModelBackend.HYBRID] and
            self.google_model is not None):
            return ModelBackend.GOOGLE_AI_EDGE
        
        # Sinon, utiliser le modèle local
        if self.local_model is not None:
            return ModelBackend.LOCAL_TRANSFORMERS
        
        return ModelBackend.SIMULATION
    
    def _analyze_with_google_ai(self, image: Image.Image, eye_position: str, 
                               image_quality: Dict) -> AnalysisResult:
        """Analyse avec Google AI Edge"""
        try:
            # Conversion de l'image pour Google AI
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Prompt contextualisé
            prompt = self.medical_prompts["analysis"].format(
                eye_position=eye_position,
                image_quality=image_quality['overall_quality']
            )
            
            # Requête à Google AI
            response = self.google_model.generate_content([
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": img_byte_arr
                }
            ])
            
            # Parsing de la réponse JSON
            result_dict = json.loads(response.text)
            
            return AnalysisResult(
                leukocoria_detected=result_dict['leukocoria_detected'],
                confidence=result_dict['confidence'],
                risk_level=result_dict['risk_level'],
                pupil_color=result_dict['pupil_color'],
                description=result_dict['description'],
                recommendations=result_dict['recommendations'],
                processing_time=0,  # Sera mis à jour
                model_backend="google_ai_edge",
                technical_details=result_dict.get('technical_details', {})
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur Google AI: {e}")
            # Fallback vers le modèle local
            return self._analyze_with_local_model(image, eye_position, image_quality)
    
    def _analyze_with_local_model(self, image: Image.Image, eye_position: str,
                                 image_quality: Dict) -> AnalysisResult:
        """Analyse avec le modèle local Transformers"""
        try:
            # Prompt pour le modèle local
            prompt = f"""<|system|>
{self.medical_prompts['system']}

<|user|>
{self.medical_prompts['analysis'].format(
    eye_position=eye_position,
    image_quality=image_quality['overall_quality']
)}

<|assistant|>"""

            # Tokenisation
            inputs = self.local_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Génération avec paramètres optimisés
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self.local_tokenizer.eos_token_id,
                    eos_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Décodage de la réponse
            response = self.local_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Extraction du JSON
            result_dict = self._extract_json_from_response(response)
            
            return AnalysisResult(
                leukocoria_detected=result_dict.get('leukocoria_detected', False),
                confidence=result_dict.get('confidence', 50.0),
                risk_level=result_dict.get('risk_level', 'medium'),
                pupil_color=result_dict.get('pupil_color', 'uncertain'),
                description=result_dict.get('description', 'Local model analysis'),
                recommendations=result_dict.get('recommendations', 'Consult ophthalmologist'),
                processing_time=0,
                model_backend="local_transformers",
                technical_details=result_dict.get('technical_details', {})
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur modèle local: {e}")
            return self._analyze_with_simulation(image, eye_position, image_quality)
    
    def _analyze_with_simulation(self, image: Image.Image, eye_position: str,
                                image_quality: Dict) -> AnalysisResult:
        """Analyse de simulation avancée basée sur computer vision"""
        # Analyse basique mais sophistiquée
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Détection de la région pupillaire
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        confidence = 40.0
        detected = False
        pupil_color = "dark"
        description = f"Computer vision analysis of {eye_position} eye"
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles[:1]:  # Premier cercle détecté
                # Analyse de la région pupillaire
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                pupil_region = cv2.bitwise_and(gray, gray, mask=mask)
                
                # Calculs de luminosité et contraste
                brightness = np.mean(pupil_region[pupil_region > 0])
                std_brightness = np.std(pupil_region[pupil_region > 0])
                
                # Heuristique sophistiquée
                brightness_score = min(100, max(0, (brightness - 80) / 175 * 100))
                contrast_score = min(100, max(0, std_brightness / 50 * 100))
                
                # Score composite
                composite_score = (brightness_score * 0.7 + contrast_score * 0.3)
                
                # Ajustements selon la qualité de l'image
                quality_modifier = {
                    'poor': 0.6,
                    'fair': 0.8,
                    'good': 1.0
                }.get(image_quality['overall_quality'], 0.8)
                
                confidence = min(95, composite_score * quality_modifier)
                detected = confidence > 50
                
                if detected:
                    pupil_color = "bright/white" if brightness > 120 else "grayish"
                    description = f"Potential leukocoria detected in {eye_position} eye (brightness: {brightness:.1f})"
                else:
                    pupil_color = "normal/dark"
                    description = f"Normal pupil appearance in {eye_position} eye"
                
                break
        
        # Détermination du niveau de risque
        if confidence > 75:
            risk_level = "high"
            recommendations = "URGENT: Immediate ophthalmological consultation required"
        elif confidence > 50:
            risk_level = "medium"
            recommendations = "Recommended: Schedule ophthalmologist appointment within 1-2 weeks"
        else:
            risk_level = "low"
            recommendations = "Continue regular monitoring, routine eye exams as scheduled"
        
        return AnalysisResult(
            leukocoria_detected=detected,
            confidence=confidence,
            risk_level=risk_level,
            pupil_color=pupil_color,
            description=description,
            recommendations=recommendations,
            processing_time=0,
            model_backend="simulation_cv",
            technical_details={
                "brightness": float(brightness) if 'brightness' in locals() else 0,
                "image_quality": image_quality,
                "detection_method": "hough_circles",
                "circles_detected": len(circles[0]) if circles is not None else 0
            }
        )
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extrait le JSON de la réponse du modèle"""
        try:
            # Chercher le JSON dans la réponse
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: parser la réponse texte
                return self._parse_text_response(response)
                
        except json.JSONDecodeError:
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict:
        """Parse une réponse texte quand le JSON échoue"""
        # Analyse basique du texte pour extraire les informations
        response_lower = response.lower()
        
        detected = any(word in response_lower for word in ['white', 'bright', 'leukocoria', 'abnormal'])
        confidence = 60.0 if detected else 40.0
        
        return {
            'leukocoria_detected': detected,
            'confidence': confidence,
            'risk_level': 'medium' if detected else 'low',
            'pupil_color': 'uncertain',
            'description': 'Text-based analysis from model response',
            'recommendations': 'Professional evaluation recommended',
            'technical_details': {'parsing_method': 'text_fallback'}
        }
    
    def _post_process_result(self, result: AnalysisResult, backend: ModelBackend, 
                           processing_time: float) -> AnalysisResult:
        """Post-traitement des résultats avec optimisations"""
        # Mise à jour du temps de traitement
        result.processing_time = processing_time
        result.model_backend = backend.value
        
        # Validation et correction des valeurs
        result.confidence = max(0, min(100, result.confidence))
        
        if result.risk_level not in ['low', 'medium', 'high']:
            result.risk_level = 'medium'
        
        # Enrichissement avec des métadonnées
        if result.technical_details is None:
            result.technical_details = {}
        
        result.technical_details.update({
            'backend_used': backend.value,
            'processing_time_ms': round(processing_time * 1000, 2),
            'device': str(self.device),
            'timestamp': time.time()
        })
        
        return result
    
    def _create_error_result(self, error_message: str, processing_time: float) -> AnalysisResult:
        """Crée un résultat d'erreur standardisé"""
        return AnalysisResult(
            leukocoria_detected=False,
            confidence=0.0,
            risk_level='unknown',
            pupil_color='error',
            description=f"Analysis error: {error_message}",
            recommendations="Unable to analyze - please retry or consult professional",
            processing_time=processing_time,
            model_backend='error',
            technical_details={'error': error_message}
        )
    
    def _generate_cache_key(self, image: Image.Image, eye_position: str) -> str:
        """Génère une clé de cache basée sur l'image"""
        # Hash de l'image pour le cache
        import hashlib
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
        return f"{img_hash}_{eye_position}"
    
    def _update_metrics(self, backend: ModelBackend, processing_time: float):
        """Met à jour les métriques de performance"""
        self.performance_metrics['total_analyses'] += 1
        self.performance_metrics['backend_usage'][backend.value] += 1
        
        # Moyenne mobile du temps de traitement
        total = self.performance_metrics['total_analyses']
        current_avg = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def batch_analyze_individual(self, eye_regions: List[Tuple[Image.Image, str, str]], 
                               individual_id: str = None) -> Dict:
        """
        Analyse en lot pour un individu avec détection de cohérence
        
        Args:
            eye_regions: Liste de (image, position, timestamp)
            individual_id: ID de l'individu pour suivi
            
        Returns:
            Analyse complète avec cohérence temporelle
        """
        start_time = time.time()
        logger.info(f"🔄 Analyse en lot: {len(eye_regions)} images pour individu {individual_id}")
        
        results = []
        consistency_scores = {'left': [], 'right': []}
        
        # Analyser chaque région
        for i, (eye_image, position, timestamp) in enumerate(eye_regions):
            logger.info(f"  📊 Analyse {i+1}/{len(eye_regions)}: {position} eye")
            
            result = self.analyze_eye_region(
                eye_image, position, use_cache=True
            )
            
            results.append({
                'timestamp': timestamp,
                'position': position,
                'result': result
            })
            
            # Collecter pour analyse de cohérence
            if result.leukocoria_detected:
                consistency_scores[position].append(result.confidence)
        
        # Analyse de cohérence
        coherence_analysis = self._analyze_temporal_coherence(results)
        
        # Synthèse finale
        batch_summary = {
            'individual_id': individual_id,
            'total_images': len(eye_regions),
            'processing_time': time.time() - start_time,
            'results': results,
            'coherence_analysis': coherence_analysis,
            'final_recommendation': self._generate_batch_recommendation(coherence_analysis)
        }
        
        logger.info(f"✅ Analyse en lot terminée: {batch_summary['final_recommendation']['urgency']}")
        return batch_summary
    
    def _analyze_temporal_coherence(self, results: List[Dict]) -> Dict:
        """Analyse la cohérence temporelle des détections"""
        coherence = {
            'left_eye': {'detections': 0, 'total': 0, 'avg_confidence': 0},
            'right_eye': {'detections': 0, 'total': 0, 'avg_confidence': 0},
            'overall_consistency': 0,
            'trend_analysis': 'stable'
        }
        
        # Analyser par position
        for eye_position in ['left', 'right']:
            eye_results = [r for r in results if r['position'] == eye_position]
            
            if eye_results:
                detections = sum(1 for r in eye_results if r['result'].leukocoria_detected)
                confidences = [r['result'].confidence for r in eye_results if r['result'].leukocoria_detected]
                
                coherence[f'{eye_position}_eye'] = {
                    'detections': detections,
                    'total': len(eye_results),
                    'detection_rate': detections / len(eye_results),
                    'avg_confidence': np.mean(confidences) if confidences else 0,
                    'consistency_score': (detections / len(eye_results)) * 100
                }
        
        # Score de cohérence global
        left_consistency = coherence['left_eye'].get('consistency_score', 0)
        right_consistency = coherence['right_eye'].get('consistency_score', 0)
        coherence['overall_consistency'] = max(left_consistency, right_consistency)
        
        # Analyse de tendance (simple)
        if coherence['overall_consistency'] > 60:
            coherence['trend_analysis'] = 'concerning_persistent'
        elif coherence['overall_consistency'] > 30:
            coherence['trend_analysis'] = 'intermittent_findings'
        else:
            coherence['trend_analysis'] = 'minimal_findings'
        
        return coherence
    
    def _generate_batch_recommendation(self, coherence: Dict) -> Dict:
        """Génère une recommandation basée sur l'analyse de cohérence"""
        consistency = coherence['overall_consistency']
        trend = coherence['trend_analysis']
        
        if consistency > 70:
            urgency = "URGENT"
            timeframe = "immediate (within 24-48 hours)"
            reason = "Highly consistent leukocoria detection across multiple images"
        elif consistency > 40:
            urgency = "HIGH"
            timeframe = "within 1 week"
            reason = "Moderately consistent findings requiring professional evaluation"
        elif consistency > 15:
            urgency = "MODERATE"
            timeframe = "within 2-4 weeks"
            reason = "Some concerning findings detected"
        else:
            urgency = "LOW"
            timeframe = "routine follow-up"
            reason = "Minimal or no concerning findings"
        
        return {
            'urgency': urgency,
            'timeframe': timeframe,
            'reason': reason,
            'consistency_score': consistency,
            'medical_action': f"Schedule ophthalmological consultation {timeframe}",
            'additional_notes': self._get_additional_medical_notes(coherence)
        }
    
    def _get_additional_medical_notes(self, coherence: Dict) -> List[str]:
        """Génère des notes médicales supplémentaires"""
        notes = []
        
        left_rate = coherence['left_eye'].get('detection_rate', 0)
        right_rate = coherence['right_eye'].get('detection_rate', 0)
        
        if left_rate > 0.5 and right_rate > 0.5:
            notes.append("Bilateral findings detected - higher priority for evaluation")
        elif left_rate > 0.5 or right_rate > 0.5:
            notes.append("Unilateral findings - document which eye is affected")
        
        if coherence['trend_analysis'] == 'concerning_persistent':
            notes.append("Persistent findings across multiple timepoints")
        
        notes.append("Continue photo documentation until professional evaluation")
        notes.append("This is a screening tool - professional diagnosis required")
        
        return notes
    
    def get_performance_report(self) -> Dict:
        """Retourne un rapport de performance détaillé"""
        return {
            'system_info': {
                'backend': self.backend.value,
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'google_ai_available': GOOGLE_AI_AVAILABLE and self.google_model is not None,
                'local_model_available': self.local_model is not None
            },
            'performance_metrics': self.performance_metrics.copy(),
            'cache_info': {
                'cache_size': len(self.analysis_cache),
                'hit_rate': (self.performance_metrics['cache_hits'] / 
                           max(1, self.performance_metrics['total_analyses'])) * 100
            },
            'optimization_suggestions': self._get_optimization_suggestions()
        }
    
    def _get_optimization_suggestions(self) -> List[str]:
        """Génère des suggestions d'optimisation"""
        suggestions = []
        
        if self.performance_metrics['average_processing_time'] > 5:
            suggestions.append("Consider enabling CUDA acceleration if available")
        
        if self.performance_metrics['cache_hits'] < self.performance_metrics['total_analyses'] * 0.2:
            suggestions.append("Low cache hit rate - consider analyzing similar images in batches")
        
        if self.backend == ModelBackend.SIMULATION:
            suggestions.append("Install Google AI or local Gemma model for improved accuracy")
        
        return suggestions
    
    def clear_cache(self):
        """Vide le cache d'analyse"""
        cache_size = len(self.analysis_cache)
        self.analysis_cache.clear()
        logger.info(f"🗑️ Cache vidé: {cache_size} entrées supprimées")
    
    def optimize_for_mobile(self):
        """Optimisations spécifiques pour mobile/edge"""
        if self.local_model is not None:
            # Quantification du modèle pour réduire la taille
            try:
                from torch.quantization import quantize_dynamic
                self.local_model = quantize_dynamic(
                    self.local_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✅ Modèle quantifié pour optimisation mobile")
            except Exception as e:
                logger.warning(f"⚠️ Quantification échouée: {e}")
        
        # Réduire la taille du cache
        if len(self.analysis_cache) > 100:
            # Garder seulement les 50 entrées les plus récentes
            sorted_cache = sorted(
                self.analysis_cache.items(),
                key=lambda x: x[1].technical_details.get('timestamp', 0),
                reverse=True
            )
            self.analysis_cache = dict(sorted_cache[:50])
            logger.info("🔧 Cache optimisé pour mobile")
    
    def export_analysis_data(self, filepath: str):
        """Exporte les données d'analyse pour recherche"""
        export_data = {
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'backend': self.backend.value,
                'device': str(self.device),
                'timestamp': time.time()
            },
            'cache_summary': {
                'total_entries': len(self.analysis_cache),
                'hit_rate': (self.performance_metrics['cache_hits'] / 
                           max(1, self.performance_metrics['total_analyses']))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"📊 Données exportées: {filepath}")
    
    def __del__(self):
        """Nettoyage lors de la destruction"""
        if hasattr(self, 'local_model') and self.local_model is not None:
            del self.local_model
        if hasattr(self, 'local_tokenizer') and self.local_tokenizer is not None:
            del self.local_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None