"""
Eye Detector V2 - Détecteur d'yeux modulaire
Gère les images complètes ET les images croppées autour des yeux
Optimisé pour le hackathon Google Gemma
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class EyeDetectorV2:
    """Détecteur d'yeux optimisé pour images complètes et croppées"""
    
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        self.initialized = False
        
        # Configuration pour différents types d'images
        self.detection_modes = {
            'full_face': 'Détection sur visage complet',
            'cropped_eye': 'Analyse d\'image croppée autour de l\'œil',
            'mixed': 'Détection hybride'
        }
        
        # Paramètres de détection
        self.config = {
            'min_detection_confidence': 0.2,
            'max_num_faces': 5,
            'enable_image_enhancement': True,
            'cropped_threshold_ratio': 2.0,  # Si width/height < 2, considérer comme image croppée
        }
        
        # Métriques
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'full_face_detections': 0,
            'cropped_detections': 0,
            'processing_times': []
        }
        
        # Initialiser MediaPipe
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Initialise MediaPipe avec gestion d'erreurs"""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.config['max_num_faces'],
                refine_landmarks=True,
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_detection_confidence']
            )
            
            self.initialized = True
            logger.info("✅ MediaPipe Face Mesh initialized")
            
        except ImportError:
            logger.warning("MediaPipe not available - using fallback detection")
            self.initialized = False
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
            self.initialized = False
    
    def detect_eyes_and_faces(self, image_path: str, enhanced_mode: bool = True) -> Dict:
        """
        Détection principale qui gère automatiquement les images complètes et croppées
        
        Args:
            image_path: Chemin vers l'image
            enhanced_mode: Activer les améliorations d'image
            
        Returns:
            Dictionnaire avec les résultats de détection
        """
        start_time = time.time()
        
        try:
            # Charger et analyser l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            h, w, _ = image.shape
            aspect_ratio = w / h
            
            # Déterminer le mode de détection
            detection_mode = self._determine_detection_mode(w, h, aspect_ratio)
            
            logger.info(f"Image {w}x{h}, ratio {aspect_ratio:.2f}, mode: {detection_mode}")
            
            # Préprocessing si activé
            if enhanced_mode and self.config['enable_image_enhancement']:
                image = self._enhance_image_quality(image)
            
            # Détection selon le mode
            if detection_mode == 'full_face':
                results = self._detect_full_face_mode(image, image_path)
            elif detection_mode == 'cropped_eye':
                results = self._detect_cropped_mode(image, image_path)
            else:  # mixed
                results = self._detect_mixed_mode(image, image_path)
            
            # Finaliser les résultats
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            results['detection_mode'] = detection_mode
            results['image_dimensions'] = (w, h)
            results['enhanced'] = enhanced_mode
            
            # Mettre à jour les statistiques
            self._update_stats(results, detection_mode, processing_time)
            
            logger.info(f"Detection completed: {results['total_regions']} regions in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._create_empty_result(str(e))
    
    def _determine_detection_mode(self, width: int, height: int, aspect_ratio: float) -> str:
        """Détermine le mode de détection optimal selon l'image"""
        
        # Image très horizontale = probablement deux yeux côte à côte
        if aspect_ratio > 2.5:
            return 'cropped_eye'
        
        # Image carrée ou légèrement rectangulaire = probablement un œil seul
        if 0.5 <= aspect_ratio <= 2.0 and max(width, height) < 800:
            return 'cropped_eye'
        
        # Grande image ou aspect ratio normal = visage complet
        if aspect_ratio > 0.3 and min(width, height) > 200:
            return 'full_face'
        
        # Par défaut, essayer les deux
        return 'mixed'
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Améliore la qualité de l'image pour une meilleure détection"""
        try:
            # Conversion en PIL pour amélioration
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Amélioration du contraste (important pour leucocorie)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # Amélioration de la netteté
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Reconversion
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Réduction du bruit tout en préservant les détails
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _detect_full_face_mode(self, image: np.ndarray, image_path: str) -> Dict:
        """Détection sur visage complet avec MediaPipe"""
        if not self.initialized:
            return self._fallback_detection(image, 'full_face')
        
        try:
            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Détection MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            regions = []
            if results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extraire les yeux de ce visage
                    face_eyes = self._extract_eyes_from_face(
                        face_landmarks, image, face_idx
                    )
                    regions.extend(face_eyes)
            
            return {
                'total_regions': len(regions),
                'regions': regions,
                'faces_detected': len(results.multi_face_landmarks) if results.multi_face_landmarks else 0,
                'method': 'mediapipe_full_face',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Full face detection failed: {e}")
            return self._fallback_detection(image, 'full_face')
    
    def _detect_cropped_mode(self, image: np.ndarray, image_path: str) -> Dict:
        """Détection sur image croppée (un ou deux yeux)"""
        try:
            h, w, _ = image.shape
            
            regions = []
            
            # Analyser l'aspect ratio pour déterminer le layout
            aspect_ratio = w / h
            
            if aspect_ratio > 2.0:
                # Image horizontale = probablement deux yeux côte à côte
                regions = self._split_horizontal_eyes(image)
            else:
                # Image carrée/verticale = probablement un seul œil
                regions = self._analyze_single_eye_region(image)
            
            return {
                'total_regions': len(regions),
                'regions': regions,
                'faces_detected': 0,  # Pas de détection de visage en mode cropped
                'method': 'cropped_analysis',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Cropped detection failed: {e}")
            return self._fallback_detection(image, 'cropped')
    
    def _detect_mixed_mode(self, image: np.ndarray, image_path: str) -> Dict:
        """Mode mixte: essaye d'abord full face, puis cropped"""
        try:
            # Essayer d'abord le mode full face
            full_face_result = self._detect_full_face_mode(image, image_path)
            
            if full_face_result['total_regions'] > 0:
                full_face_result['method'] = 'mixed_full_face_success'
                return full_face_result
            
            # Si échec, essayer le mode cropped
            logger.info("Full face detection failed, trying cropped mode")
            cropped_result = self._detect_cropped_mode(image, image_path)
            cropped_result['method'] = 'mixed_cropped_fallback'
            
            return cropped_result
            
        except Exception as e:
            logger.error(f"Mixed mode detection failed: {e}")
            return self._fallback_detection(image, 'mixed')
    
    def _extract_eyes_from_face(self, face_landmarks, image: np.ndarray, face_idx: int) -> List[Dict]:
        """Extrait les yeux d'un visage détecté par MediaPipe"""
        h, w, _ = image.shape
        
        # Convertir les landmarks en coordonnées
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        eyes = []
        
        # Indices MediaPipe pour les yeux
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Extraire œil gauche
        left_eye = self._extract_single_eye(
            landmarks, left_eye_indices, image, f'left_face_{face_idx}'
        )
        if left_eye:
            eyes.append(left_eye)
        
        # Extraire œil droit
        right_eye = self._extract_single_eye(
            landmarks, right_eye_indices, image, f'right_face_{face_idx}'
        )
        if right_eye:
            eyes.append(right_eye)
        
        return eyes
    
    def _extract_single_eye(self, landmarks: List[Tuple[int, int]], 
                          eye_indices: List[int], image: np.ndarray, 
                          eye_id: str) -> Optional[Dict]:
        """Extrait une région oculaire spécifique"""
        try:
            h, w, _ = image.shape
            
            # Obtenir les points de l'œil
            eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
            
            if len(eye_points) < 8:
                return None
            
            # Calculer la boîte englobante
            xs = [p[0] for p in eye_points]
            ys = [p[1] for p in eye_points]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Ajouter une marge
            margin = 30
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extraire la région
            eye_region = image[y1:y2, x1:x2]
            eye_pil = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
            
            return {
                'id': eye_id,
                'type': eye_id.split('_')[0],  # 'left' ou 'right'
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'image': eye_pil,
                'landmarks': eye_points,
                'confidence': 0.9,
                'source': 'mediapipe_face_detection'
            }
            
        except Exception as e:
            logger.error(f"Single eye extraction failed: {e}")
            return None
    
    def _split_horizontal_eyes(self, image: np.ndarray) -> List[Dict]:
        """Divise une image horizontale en deux régions oculaires"""
        try:
            h, w, _ = image.shape
            mid_x = w // 2
            
            regions = []
            
            # Œil gauche (première moitié)
            left_region = image[:, :mid_x]
            if left_region.size > 0:
                left_pil = Image.fromarray(cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB))
                regions.append({
                    'id': 'left_cropped',
                    'type': 'left',
                    'bbox': (0, 0, mid_x, h),
                    'image': left_pil,
                    'landmarks': [],
                    'confidence': 0.7,
                    'source': 'horizontal_split'
                })
            
            # Œil droit (seconde moitié)
            right_region = image[:, mid_x:]
            if right_region.size > 0:
                right_pil = Image.fromarray(cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB))
                regions.append({
                    'id': 'right_cropped',
                    'type': 'right',
                    'bbox': (mid_x, 0, w - mid_x, h),
                    'image': right_pil,
                    'landmarks': [],
                    'confidence': 0.7,
                    'source': 'horizontal_split'
                })
            
            return regions
            
        except Exception as e:
            logger.error(f"Horizontal split failed: {e}")
            return []
    
    def _analyze_single_eye_region(self, image: np.ndarray) -> List[Dict]:
        """Analyse une image contenant un seul œil"""
        try:
            h, w, _ = image.shape
            
            # Utiliser l'image entière comme région oculaire
            eye_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Essayer de détecter quel œil c'est (optionnel)
            eye_type = self._guess_eye_type(image)
            
            region = {
                'id': f'{eye_type}_single',
                'type': eye_type,
                'bbox': (0, 0, w, h),
                'image': eye_pil,
                'landmarks': [],
                'confidence': 0.8,
                'source': 'single_eye_analysis'
            }
            
            return [region]
            
        except Exception as e:
            logger.error(f"Single eye analysis failed: {e}")
            return []
    
    def _guess_eye_type(self, image: np.ndarray) -> str:
        """Essaie de deviner si c'est un œil gauche ou droit"""
        # Analyse basique - peut être améliorée
        try:
            h, w, _ = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Chercher la position probable de la pupille
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=min(w, h)//4
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    # Prendre le cercle le plus central
                    center_x = circles[0][0]
                    
                    # Si la pupille est vers la gauche de l'image, c'est probablement l'œil droit
                    # (car on regarde depuis le point de vue de la personne)
                    if center_x < w * 0.4:
                        return 'right'
                    elif center_x > w * 0.6:
                        return 'left'
            
            return 'center'  # Par défaut
            
        except Exception:
            return 'center'
    
    def _fallback_detection(self, image: np.ndarray, mode: str) -> Dict:
        """Détection de fallback sans MediaPipe"""
        try:
            logger.warning(f"Using fallback detection for mode: {mode}")
            
            h, w, _ = image.shape
            
            # Créer une région basique couvrant l'image entière
            eye_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            fallback_region = {
                'id': 'fallback_region',
                'type': 'unknown',
                'bbox': (0, 0, w, h),
                'image': eye_pil,
                'landmarks': [],
                'confidence': 0.3,
                'source': f'fallback_{mode}'
            }
            
            return {
                'total_regions': 1,
                'regions': [fallback_region],
                'faces_detected': 0,
                'method': f'fallback_{mode}',
                'success': False
            }
            
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return self._create_empty_result(f"Fallback failed: {e}")
    
    def _create_empty_result(self, error_msg: str = "") -> Dict:
        """Crée un résultat vide en cas d'échec total"""
        return {
            'total_regions': 0,
            'regions': [],
            'faces_detected': 0,
            'method': 'failed',
            'success': False,
            'error': error_msg
        }
    
    def _update_stats(self, results: Dict, detection_mode: str, processing_time: float):
        """Met à jour les statistiques de détection"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['processing_times'].append(processing_time)
        
        if results['success'] and results['total_regions'] > 0:
            self.detection_stats['successful_detections'] += 1
            
            if detection_mode == 'full_face':
                self.detection_stats['full_face_detections'] += 1
            elif detection_mode == 'cropped_eye':
                self.detection_stats['cropped_detections'] += 1
    
    def get_detection_stats(self) -> Dict:
        """Retourne les statistiques de détection"""
        total = self.detection_stats['total_detections']
        if total == 0:
            return self.detection_stats
        
        success_rate = (self.detection_stats['successful_detections'] / total) * 100
        avg_time = sum(self.detection_stats['processing_times']) / len(self.detection_stats['processing_times'])
        
        return {
            **self.detection_stats,
            'success_rate': success_rate,
            'average_processing_time': avg_time
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'full_face_detections': 0,
            'cropped_detections': 0,
            'processing_times': []
        }
    
    def enhance_eye_region(self, eye_image: Image.Image) -> Image.Image:
        """Améliore une région oculaire pour l'analyse"""
        try:
            # Redimensionner à une taille standard
            enhanced = eye_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Amélioration du contraste pour la leucocorie
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            # Amélioration de la netteté
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Eye region enhancement failed: {e}")
            return eye_image
    
    def validate_eye_region(self, region: Dict) -> bool:
        """Valide qu'une région oculaire est acceptable pour l'analyse"""
        try:
            # Vérifier la présence des champs requis
            required_fields = ['id', 'type', 'image', 'bbox']
            if not all(field in region for field in required_fields):
                return False
            
            # Vérifier la taille de l'image
            image = region['image']
            if image.width < 32 or image.height < 32:
                return False
            
            # Vérifier le bbox
            bbox = region['bbox']
            if len(bbox) != 4 or any(v < 0 for v in bbox):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Region validation failed: {e}")
            return False
    
    def cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None
            
            self.initialized = False
            logger.info("Eye detector cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Nettoyage automatique"""
        self.cleanup()