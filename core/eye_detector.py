"""
Détecteur d'yeux ultra-optimisé avec MediaPipe Advanced
Architecture hybride pour Google AI Edge Prize
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from config.settings import *

logger = logging.getLogger(__name__)

class DetectionQuality(Enum):
    """Niveaux de qualité de détection"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class EyeRegion:
    """Structure optimisée pour une région d'œil"""
    position: str  # 'left' or 'right'
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    image: Image.Image
    landmarks: List[Tuple[int, int]]
    center: Tuple[int, int]
    confidence: float
    quality_metrics: Dict
    enhanced_image: Optional[Image.Image] = None
    pupil_region: Optional[Dict] = None

@dataclass
class FaceDetection:
    """Structure complète pour une détection de visage"""
    face_id: int
    face_bbox: Tuple[int, int, int, int]
    landmarks: List[Tuple[int, int]]
    eyes: List[EyeRegion]
    confidence: float
    quality: DetectionQuality
    processing_time: float

class AdvancedEyeDetector:
    """Détecteur d'yeux de niveau production avec optimisations Edge"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configuration optimisée pour détection médicale
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,  # Augmenté pour photos de groupe
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Plus sensible pour enfants
            min_tracking_confidence=0.3
        )
        
        # Indices des landmarks optimisés pour analyse médicale
        self._setup_landmark_indices()
        
        # Paramètres d'amélioration d'image
        self._setup_enhancement_parameters()
        
        # Cache et optimisations
        self.detection_cache = {}
        self.performance_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_processing_time': 0,
            'quality_distribution': {q.value: 0 for q in DetectionQuality}
        }
        
        # Thread pool pour traitement parallèle
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("✅ AdvancedEyeDetector initialisé avec optimisations Edge")
    
    def _setup_landmark_indices(self):
        """Configure les indices de landmarks optimisés pour la détection médicale"""
        # Landmarks pour contour des yeux (plus précis)
        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Landmarks pour l'iris et la pupille
        self.LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]  # Centres approximatifs
        self.RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
        
        # Points critiques pour la qualité
        self.LEFT_EYE_CORNERS = [33, 133]  # Coins interne et externe
        self.RIGHT_EYE_CORNERS = [362, 263]
        
        # Landmarks pour évaluation de l'orientation du visage
        self.FACE_ORIENTATION_POINTS = [10, 151, 9, 175]  # Nez, menton
    
    def _setup_enhancement_parameters(self):
        """Configure les paramètres d'amélioration d'image"""
        self.enhancement_config = {
            'contrast_boost': 1.3,
            'brightness_adjustment': 1.1,
            'sharpness_factor': 1.5,
            'gamma_correction': 0.8,
            'noise_reduction_strength': 5,
            'bilateral_filter_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75
        }
    
    def detect_faces_and_eyes(self, image_path: str, 
                             enhance_quality: bool = True,
                             parallel_processing: bool = True) -> Dict:
        """
        Détection avancée avec optimisations pour Google AI Edge
        
        Args:
            image_path: Chemin vers l'image
            enhance_quality: Activer l'amélioration de qualité
            parallel_processing: Utiliser le traitement parallèle
            
        Returns:
            Résultats de détection enrichis
        """
        start_time = time.time()
        
        try:
            # Chargement et préparation de l'image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Préprocessing avancé
            processed_image = self._advanced_preprocessing(original_image, enhance_quality)
            
            # Détection MediaPipe
            detection_results = self._perform_mediapipe_detection(processed_image)
            
            # Traitement des résultats
            if parallel_processing:
                faces = self._process_faces_parallel(detection_results, original_image)
            else:
                faces = self._process_faces_sequential(detection_results, original_image)
            
            # Métriques de qualité globale
            overall_quality = self._assess_overall_quality(faces, processed_image)
            
            # Compilation des résultats
            results = {
                'faces': faces,
                'total_faces_detected': len(faces),
                'total_eyes_detected': sum(len(face.eyes) for face in faces),
                'image_shape': original_image.shape,
                'processing_time': time.time() - start_time,
                'overall_quality': overall_quality,
                'enhancement_applied': enhance_quality,
                'detection_metadata': {
                    'image_path': image_path,
                    'preprocessing_applied': True,
                    'parallel_processing': parallel_processing
                }
            }
            
            # Mise à jour des métriques
            self._update_performance_metrics(results)
            
            logger.info(f"✅ Détection terminée: {len(faces)} visages, "
                       f"{sum(len(face.eyes) for face in faces)} yeux "
                       f"({time.time() - start_time:.2f}s)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur de détection: {e}")
            return {
                'faces': [],
                'total_faces_detected': 0,
                'total_eyes_detected': 0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _advanced_preprocessing(self, image: np.ndarray, enhance: bool) -> np.ndarray:
        """Préprocessing avancé optimisé pour la détection médicale"""
        processed = image.copy()
        
        if enhance:
            # Conversion en PIL pour amélioration
            pil_image = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            
            # Amélioration du contraste (important pour leucocorie)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(self.enhancement_config['contrast_boost'])
            
            # Amélioration de la luminosité
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(self.enhancement_config['brightness_adjustment'])
            
            # Amélioration de la netteté
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(self.enhancement_config['sharpness_factor'])
            
            # Reconversion en OpenCV
            processed = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Correction gamma pour améliorer les détails sombres/clairs
            gamma = self.enhancement_config['gamma_correction']
            processed = np.power(processed / 255.0, gamma) * 255
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Réduction du bruit tout en préservant les contours
            processed = cv2.bilateralFilter(
                processed,
                self.enhancement_config['bilateral_filter_d'],
                self.enhancement_config['bilateral_sigma_color'],
                self.enhancement_config['bilateral_sigma_space']
            )
        
        return processed
    
    def _perform_mediapipe_detection(self, image: np.ndarray) -> Optional[object]:
        """Effectue la détection MediaPipe avec gestion d'erreurs"""
        try:
            # Conversion pour MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # Détection
            results = self.face_mesh.process(rgb_image)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur MediaPipe: {e}")
            return None
    
    def _process_faces_parallel(self, detection_results, original_image: np.ndarray) -> List[FaceDetection]:
        """Traitement parallèle des visages détectés"""
        if not detection_results or not detection_results.multi_face_landmarks:
            return []
        
        # Créer les tâches pour chaque visage
        futures = []
        for face_idx, face_landmarks in enumerate(detection_results.multi_face_landmarks):
            future = self.thread_pool.submit(
                self._process_single_face_complete,
                face_landmarks, original_image, face_idx
            )
            futures.append(future)
        
        # Collecter les résultats
        faces = []
        for future in futures:
            try:
                face_detection = future.result(timeout=10)  # Timeout de sécurité
                if face_detection:
                    faces.append(face_detection)
            except Exception as e:
                logger.warning(f"⚠️ Erreur traitement parallèle: {e}")
        
        return faces
    
    def _process_faces_sequential(self, detection_results, original_image: np.ndarray) -> List[FaceDetection]:
        """Traitement séquentiel des visages détectés"""
        if not detection_results or not detection_results.multi_face_landmarks:
            return []
        
        faces = []
        for face_idx, face_landmarks in enumerate(detection_results.multi_face_landmarks):
            face_detection = self._process_single_face_complete(
                face_landmarks, original_image, face_idx
            )
            if face_detection:
                faces.append(face_detection)
        
        return faces
    
    def _process_single_face_complete(self, face_landmarks, image: np.ndarray, 
                                    face_idx: int) -> Optional[FaceDetection]:
        """Traitement complet d'un seul visage avec toutes les optimisations"""
        start_time = time.time()
        
        try:
            h, w, _ = image.shape
            
            # Extraction des coordonnées des landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Évaluation de la qualité du visage
            face_quality = self._assess_face_quality(landmarks, image.shape)
            
            # Boîte englobante du visage avec marge adaptative
            face_bbox = self._compute_adaptive_face_bbox(landmarks, image.shape)
            
            # Extraction des régions oculaires
            left_eye = self._extract_advanced_eye_region(
                landmarks, self.LEFT_EYE_CONTOUR, image, "left", face_idx
            )
            right_eye = self._extract_advanced_eye_region(
                landmarks, self.RIGHT_EYE_CONTOUR, image, "right", face_idx
            )
            
            # Filtrer les yeux de qualité insuffisante
            eyes = []
            for eye in [left_eye, right_eye]:
                if eye and eye.confidence > EYE_DETECTION_THRESHOLD:
                    eyes.append(eye)
            
            # Calcul de la confiance globale du visage
            face_confidence = self._calculate_face_confidence(eyes, face_quality, landmarks)
            
            return FaceDetection(
                face_id=face_idx,
                face_bbox=face_bbox,
                landmarks=landmarks,
                eyes=eyes,
                confidence=face_confidence,
                quality=face_quality,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement visage {face_idx}: {e}")
            return None
    
    def _assess_face_quality(self, landmarks: List[Tuple[int, int]], 
                           image_shape: Tuple[int, int, int]) -> DetectionQuality:
        """Évalue la qualité de détection du visage"""
        h, w, _ = image_shape
        
        # Vérifier la complétude des landmarks
        if len(landmarks) < 400:  # MediaPipe doit détecter ~468 landmarks
            return DetectionQuality.POOR
        
        # Analyser la distribution des landmarks
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        
        # Taille relative du visage
        face_width = max(xs) - min(xs)
        face_height = max(ys) - min(ys)
        relative_size = (face_width * face_height) / (w * h)
        
        # Score de qualité basé sur plusieurs critères
        quality_score = 0
        
        # Critère 1: Taille du visage (optimal entre 5% et 60% de l'image)
        if 0.05 < relative_size < 0.6:
            quality_score += 30
        elif 0.02 < relative_size < 0.8:
            quality_score += 20
        else:
            quality_score += 10
        
        # Critère 2: Position du visage (centré est mieux)
        center_x, center_y = np.mean(xs), np.mean(ys)
        center_deviation = math.sqrt(
            ((center_x - w/2) / w)**2 + ((center_y - h/2) / h)**2
        )
        if center_deviation < 0.2:
            quality_score += 25
        elif center_deviation < 0.4:
            quality_score += 15
        else:
            quality_score += 5
        
        # Critère 3: Complétude des landmarks des yeux
        left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE_CONTOUR if i < len(landmarks)]
        right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE_CONTOUR if i < len(landmarks)]
        
        if len(left_eye_landmarks) >= 12 and len(right_eye_landmarks) >= 12:
            quality_score += 25
        elif len(left_eye_landmarks) >= 8 and len(right_eye_landmarks) >= 8:
            quality_score += 15
        else:
            quality_score += 5
        
        # Critère 4: Orientation du visage (frontal est optimal)
        face_orientation_score = self._assess_face_orientation(landmarks)
        quality_score += face_orientation_score * 20
        
        # Conversion en classe de qualité
        if quality_score >= 85:
            return DetectionQuality.EXCELLENT
        elif quality_score >= 65:
            return DetectionQuality.GOOD
        elif quality_score >= 45:
            return DetectionQuality.FAIR
        else:
            return DetectionQuality.POOR
    
    def _assess_face_orientation(self, landmarks: List[Tuple[int, int]]) -> float:
        """Évalue l'orientation du visage (0=profil, 1=frontal)"""
        try:
            # Utiliser des points de référence pour calculer l'orientation
            if len(landmarks) <= max(self.FACE_ORIENTATION_POINTS):
                return 0.5  # Score neutre si landmarks insuffisants
            
            # Points de référence: nez, coins des yeux
            nose_tip = landmarks[1] if len(landmarks) > 1 else (0, 0)
            left_eye_corner = landmarks[33] if len(landmarks) > 33 else (0, 0)
            right_eye_corner = landmarks[263] if len(landmarks) > 263 else (0, 0)
            
            # Calculer l'asymétrie horizontale
            eye_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            nose_offset = abs(nose_tip[0] - eye_center_x)
            eye_distance = abs(right_eye_corner[0] - left_eye_corner[0])
            
            if eye_distance > 0:
                asymmetry_ratio = nose_offset / eye_distance
                # Plus le ratio est faible, plus le visage est frontal
                orientation_score = max(0, 1 - asymmetry_ratio * 3)
                return min(1, orientation_score)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _compute_adaptive_face_bbox(self, landmarks: List[Tuple[int, int]], 
                                  image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Calcule une boîte englobante adaptative pour le visage"""
        h, w, _ = image_shape
        
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Marge adaptative basée sur la taille du visage
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        # Marge proportionnelle (15% de la taille du visage)
        margin_x = int(face_width * 0.15)
        margin_y = int(face_height * 0.15)
        
        # Application des marges avec contraintes d'image
        x1 = max(0, x_min - margin_x)
        y1 = max(0, y_min - margin_y)
        x2 = min(w, x_max + margin_x)
        y2 = min(h, y_max + margin_y)
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def _extract_advanced_eye_region(self, landmarks: List[Tuple[int, int]], 
                                   eye_indices: List[int], image: np.ndarray,
                                   eye_position: str, face_id: int) -> Optional[EyeRegion]:
        """Extraction avancée d'une région oculaire avec amélioration"""
        try:
            # Obtenir les points de l'œil
            eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
            
            if len(eye_points) < 8:  # Minimum de points requis
                return None
            
            # Calculer la boîte englobante optimale
            bbox = self._compute_optimal_eye_bbox(eye_points, image.shape)
            if bbox is None:
                return None
            
            x, y, w, h = bbox
            
            # Extraire la région avec vérifications de limites
            eye_region = image[y:y+h, x:x+w]
            if eye_region.size == 0:
                return None
            
            # Conversion en PIL Image
            eye_pil = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
            
            # Amélioration spécialisée pour l'analyse oculaire
            enhanced_eye = self._enhance_eye_region(eye_pil)
            
            # Calcul du centre et qualité
            center = self._calculate_eye_center(eye_points)
            quality_metrics = self._assess_eye_quality(eye_region, eye_points)
            
            # Analyse de la région pupillaire
            pupil_analysis = self._analyze_pupil_region(enhanced_eye)
            
            # Calcul de la confiance
            confidence = self._calculate_eye_confidence(quality_metrics, pupil_analysis)
            
            return EyeRegion(
                position=eye_position,
                bbox=bbox,
                image=eye_pil,
                landmarks=eye_points,
                center=center,
                confidence=confidence,
                quality_metrics=quality_metrics,
                enhanced_image=enhanced_eye,
                pupil_region=pupil_analysis
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction œil {eye_position}: {e}")
            return None
    
    def _compute_optimal_eye_bbox(self, eye_points: List[Tuple[int, int]], 
                                image_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Calcule la boîte englobante optimale pour un œil"""
        if not eye_points:
            return None
        
        h, w, _ = image_shape
        
        xs = [point[0] for point in eye_points]
        ys = [point[1] for point in eye_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Calculer le centre et la taille
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Taille basée sur la distance entre les points extrêmes + marge
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        # Taille finale (carré pour cohérence, avec marge de 40%)
        size = int(max(eye_width, eye_height) * 1.4)
        
        # Assurer une taille minimale pour l'analyse
        size = max(size, 64)
        
        # Coordonnées finales
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        
        # Ajuster si nécessaire pour maintenir la taille
        if x2 - x1 < size:
            x1 = max(0, x2 - size)
        if y2 - y1 < size:
            y1 = max(0, y2 - size)
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def _enhance_eye_region(self, eye_image: Image.Image) -> Image.Image:
        """Amélioration spécialisée pour l'analyse de leucocorie"""
        # Redimensionner à une taille standard pour analyse
        enhanced = eye_image.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Amélioration du contraste (crucial pour leucocorie)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        # Amélioration de la netteté
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.3)
        
        # Filtre pour réduire le bruit tout en préservant les détails
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return enhanced
    
    def _calculate_eye_center(self, eye_points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calcule le centre géométrique de l'œil"""
        if not eye_points:
            return (0, 0)
        
        center_x = sum(point[0] for point in eye_points) // len(eye_points)
        center_y = sum(point[1] for point in eye_points) // len(eye_points)
        
        return (center_x, center_y)
    
    def _assess_eye_quality(self, eye_region: np.ndarray, 
                          eye_points: List[Tuple[int, int]]) -> Dict:
        """Évalue la qualité de la région oculaire"""
        if eye_region.size == 0:
            return {'overall_quality': 0, 'sharpness': 0, 'contrast': 0, 'size': 0}
        
        # Conversion en niveaux de gris pour analyse
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Métriques de qualité
        quality_metrics = {
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'contrast': float(np.std(gray)),
            'brightness': float(np.mean(gray)),
            'size': eye_region.shape[0] * eye_region.shape[1],
            'landmark_count': len(eye_points)
        }
        
        # Score global de qualité (0-100)
        quality_score = 0
        
        # Netteté (0-30 points)
        if quality_metrics['sharpness'] > 500:
            quality_score += 30
        elif quality_metrics['sharpness'] > 200:
            quality_score += 20
        elif quality_metrics['sharpness'] > 50:
            quality_score += 10
        
        # Contraste (0-25 points)
        if quality_metrics['contrast'] > 50:
            quality_score += 25
        elif quality_metrics['contrast'] > 30:
            quality_score += 15
        elif quality_metrics['contrast'] > 15:
            quality_score += 10
        
        # Taille (0-25 points)
        if quality_metrics['size'] > 8000:
            quality_score += 25
        elif quality_metrics['size'] > 4000:
            quality_score += 15
        elif quality_metrics['size'] > 1000:
            quality_score += 10
        
        # Landmarks (0-20 points)
        if quality_metrics['landmark_count'] >= 14:
            quality_score += 20
        elif quality_metrics['landmark_count'] >= 10:
            quality_score += 15
        elif quality_metrics['landmark_count'] >= 6:
            quality_score += 10
        
        quality_metrics['overall_quality'] = quality_score
        
        return quality_metrics
    
    def _analyze_pupil_region(self, enhanced_eye: Image.Image) -> Dict:
        """Analyse avancée de la région pupillaire pour détecter la leucocorie"""
        # Conversion en OpenCV
        cv_image = cv2.cvtColor(np.array(enhanced_eye), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Détection de cercles (pupilles potentielles)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=8, maxRadius=40
        )
        
        pupil_analysis = {
            'circles_detected': 0,
            'best_circle': None,
            'brightness_score': 0,
            'contrast_score': 0,
            'potential_leukocoria': False,
            'confidence': 0
        }
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            pupil_analysis['circles_detected'] = len(circles)
            
            # Analyser le meilleur cercle (le plus central et approprié)
            best_circle = self._select_best_pupil_circle(circles, gray.shape)
            
            if best_circle is not None:
                x, y, r = best_circle
                pupil_analysis['best_circle'] = {'x': int(x), 'y': int(y), 'radius': int(r)}
                
                # Analyser la région pupillaire
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                pupil_region = cv2.bitwise_and(gray, gray, mask=mask)
                
                # Métriques de la pupille
                pupil_pixels = pupil_region[pupil_region > 0]
                if len(pupil_pixels) > 0:
                    brightness = float(np.mean(pupil_pixels))
                    contrast = float(np.std(pupil_pixels))
                    
                    pupil_analysis['brightness_score'] = brightness
                    pupil_analysis['contrast_score'] = contrast
                    
                    # Détection de leucocorie basée sur la luminosité
                    # Seuil adaptatif basé sur l'image globale
                    global_brightness = np.mean(gray)
                    leucocoria_threshold = global_brightness + 40  # Seuil adaptatif
                    
                    pupil_analysis['potential_leukocoria'] = brightness > leucocoria_threshold
                    
                    # Score de confiance pour la détection
                    if pupil_analysis['potential_leukocoria']:
                        # Plus la pupille est claire par rapport au fond, plus la confiance est élevée
                        relative_brightness = (brightness - global_brightness) / (255 - global_brightness)
                        pupil_analysis['confidence'] = min(95, relative_brightness * 100)
                    else:
                        pupil_analysis['confidence'] = 20  # Confiance faible pour pupille normale
        
        return pupil_analysis
    
    def _select_best_pupil_circle(self, circles: np.ndarray, 
                                image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
        """Sélectionne le meilleur cercle représentant la pupille"""
        if len(circles) == 0:
            return None
        
        h, w = image_shape
        center_x, center_y = w // 2, h // 2
        
        best_circle = None
        best_score = -1
        
        for (x, y, r) in circles:
            # Score basé sur la position (plus central = mieux)
            distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            position_score = max(0, 1 - distance_from_center / (min(w, h) / 2))
            
            # Score basé sur la taille (ni trop petit ni trop grand)
            size_score = 1.0
            if r < 5 or r > min(w, h) / 3:
                size_score = 0.5
            
            # Score composite
            total_score = position_score * 0.7 + size_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_circle = (x, y, r)
        
        return best_circle
    
    def _calculate_eye_confidence(self, quality_metrics: Dict, 
                                pupil_analysis: Dict) -> float:
        """Calcule la confiance globale pour la détection d'œil"""
        # Confiance basée sur la qualité (0-70 points)
        quality_confidence = (quality_metrics['overall_quality'] / 100) * 70
        
        # Confiance basée sur la détection de pupille (0-30 points)
        pupil_confidence = 0
        if pupil_analysis['circles_detected'] > 0:
            pupil_confidence = 20
            if pupil_analysis['best_circle'] is not None:
                pupil_confidence = 30
        
        total_confidence = quality_confidence + pupil_confidence
        
        return min(100, max(0, total_confidence))
    
    def _calculate_face_confidence(self, eyes: List[EyeRegion], 
                                 face_quality: DetectionQuality,
                                 landmarks: List[Tuple[int, int]]) -> float:
        """Calcule la confiance globale pour la détection de visage"""
        base_confidence = {
            DetectionQuality.EXCELLENT: 85,
            DetectionQuality.GOOD: 70,
            DetectionQuality.FAIR: 55,
            DetectionQuality.POOR: 30
        }.get(face_quality, 30)
        
        # Bonus pour les yeux détectés
        eye_bonus = len(eyes) * 7.5  # +7.5 points par œil détecté
        
        # Bonus pour qualité des yeux
        eye_quality_bonus = sum(eye.confidence for eye in eyes) / max(1, len(eyes)) * 0.1
        
        # Penalty si pas assez de landmarks
        landmark_penalty = 0 if len(landmarks) > 400 else -10
        
        final_confidence = base_confidence + eye_bonus + eye_quality_bonus + landmark_penalty
        
        return min(100, max(0, final_confidence))
    
    def _assess_overall_quality(self, faces: List[FaceDetection], 
                              processed_image: np.ndarray) -> Dict:
        """Évalue la qualité globale de la détection"""
        if not faces:
            return {
                'level': DetectionQuality.POOR.value,
                'score': 0,
                'recommendations': ['No faces detected', 'Check image quality and lighting']
            }
        
        # Métriques globales
        avg_face_confidence = np.mean([face.confidence for face in faces])
        avg_eyes_per_face = np.mean([len(face.eyes) for face in faces])
        quality_distribution = [face.quality.value for face in faces]
        
        # Score global
        overall_score = (avg_face_confidence * 0.6 + 
                        avg_eyes_per_face * 25 * 0.4)  # Max 2 yeux * 25 = 50
        
        # Niveau de qualité global
        if overall_score >= 80:
            level = DetectionQuality.EXCELLENT
        elif overall_score >= 65:
            level = DetectionQuality.GOOD
        elif overall_score >= 45:
            level = DetectionQuality.FAIR
        else:
            level = DetectionQuality.POOR
        
        # Recommandations
        recommendations = []
        if overall_score < 60:
            recommendations.append("Consider improving image quality")
        if avg_eyes_per_face < 1.5:
            recommendations.append("Some eyes not detected - check lighting and angles")
        if len(faces) > 5:
            recommendations.append("Multiple faces detected - consider individual analysis")
        
        return {
            'level': level.value,
            'score': overall_score,
            'avg_face_confidence': avg_face_confidence,
            'avg_eyes_per_face': avg_eyes_per_face,
            'quality_distribution': quality_distribution,
            'recommendations': recommendations
        }
    
    def _update_performance_metrics(self, results: Dict):
        """Met à jour les métriques de performance"""
        self.performance_metrics['total_detections'] += 1
        
        if results['total_faces_detected'] > 0:
            self.performance_metrics['successful_detections'] += 1
        
        # Moyenne mobile du temps de traitement
        processing_time = results['processing_time']
        total = self.performance_metrics['total_detections']
        current_avg = self.performance_metrics['average_processing_time']
        
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Distribution de qualité
        if 'overall_quality' in results:
            quality_level = results['overall_quality']['level']
            self.performance_metrics['quality_distribution'][quality_level] += 1
    
    def get_performance_report(self) -> Dict:
        """Retourne un rapport de performance détaillé"""
        total = self.performance_metrics['total_detections']
        successful = self.performance_metrics['successful_detections']
        
        return {
            'detection_statistics': {
                'total_detections': total,
                'successful_detections': successful,
                'success_rate': (successful / max(1, total)) * 100,
                'average_processing_time': self.performance_metrics['average_processing_time']
            },
            'quality_distribution': self.performance_metrics['quality_distribution'],
            'cache_statistics': {
                'cache_size': len(self.detection_cache),
                'cache_efficiency': 'Not implemented'  # À implémenter si nécessaire
            },
            'optimization_recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        success_rate = (self.performance_metrics['successful_detections'] / 
                       max(1, self.performance_metrics['total_detections']))
        
        if success_rate < 0.8:
            recommendations.append("Low success rate - consider image preprocessing")
        
        if self.performance_metrics['average_processing_time'] > 3:
            recommendations.append("High processing time - consider parallel processing")
        
        poor_quality_count = self.performance_metrics['quality_distribution'].get('poor', 0)
        if poor_quality_count > self.performance_metrics['total_detections'] * 0.3:
            recommendations.append("Many poor quality detections - improve input images")
        
        return recommendations
    
    def optimize_for_batch_processing(self):
        """Optimise les paramètres pour le traitement en lot"""
        # Ajuster les paramètres MediaPipe pour le lot
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=15,  # Augmenté pour photos de groupe
            refine_landmarks=True,
            min_detection_confidence=0.4,  # Légèrement plus strict
            min_tracking_confidence=0.4
        )
        
        logger.info("🔧 Optimisé pour traitement en lot")
    
    def cleanup_resources(self):
        """Nettoie les ressources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        self.detection_cache.clear()
        
        logger.info("🧹 Ressources nettoyées")
    
    def __del__(self):
        """Nettoyage automatique"""
        self.cleanup_resources()

# Alias pour compatibilité avec le code existant
EyeDetector = AdvancedEyeDetector