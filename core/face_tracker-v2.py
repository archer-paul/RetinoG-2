"""
Traqueur de visages pour RetinoblastoGemma
Permet le suivi longitudinal et l'ajustement de confiance basé sur l'historique
"""
import numpy as np
import cv2
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("Face recognition library available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("Face recognition library not available - using fallback")

class FaceTracker:
    """Traqueur de visages avec historique médical pour ajustement de confiance"""
    
    def __init__(self, database_path=None):
        self.database_path = Path(database_path) if database_path else Path("data/face_database.pkl")
        self.medical_history_path = Path("data/medical_history.json")
        
        # Base de données des visages encodés
        self.face_database = {}
        
        # Historique médical par individu
        self.medical_history = defaultdict(list)
        
        # Configuration de tracking
        self.confidence_threshold = 0.6
        self.max_history_days = 365  # Garder l'historique 1 an
        self.confidence_adjustment_factor = 0.1  # Facteur d'ajustement
        
        # Charger les données existantes
        self.load_database()
        self.load_medical_history()
        
        logger.info(f"Face tracker initialized with {len(self.face_database)} known faces")
    
    def load_database(self):
        """Charge la base de données des visages"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info(f"Loaded {len(self.face_database)} faces from database")
            else:
                self.face_database = {}
                logger.info("No existing face database found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load face database: {e}")
            self.face_database = {}
    
    def save_database(self):
        """Sauvegarde la base de données des visages"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Saved face database with {len(self.face_database)} faces")
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
    
    def load_medical_history(self):
        """Charge l'historique médical"""
        try:
            if self.medical_history_path.exists():
                with open(self.medical_history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.medical_history = defaultdict(list, data)
                logger.info(f"Loaded medical history for {len(self.medical_history)} individuals")
            else:
                self.medical_history = defaultdict(list)
                logger.info("No existing medical history found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load medical history: {e}")
            self.medical_history = defaultdict(list)
    
    def save_medical_history(self):
        """Sauvegarde l'historique médical"""
        try:
            self.medical_history_path.parent.mkdir(parents=True, exist_ok=True)
            # Nettoyer l'historique ancien
            self._cleanup_old_history()
            
            with open(self.medical_history_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.medical_history), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved medical history for {len(self.medical_history)} individuals")
        except Exception as e:
            logger.error(f"Failed to save medical history: {e}")
    
    def _cleanup_old_history(self):
        """Nettoie l'historique médical ancien"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            for person_id in list(self.medical_history.keys()):
                # Filtrer les entrées récentes
                recent_entries = [
                    entry for entry in self.medical_history[person_id]
                    if entry.get('timestamp', 0) > cutoff_timestamp
                ]
                
                if recent_entries:
                    self.medical_history[person_id] = recent_entries
                else:
                    # Supprimer si plus d'historique récent
                    del self.medical_history[person_id]
                    if person_id in self.face_database:
                        del self.face_database[person_id]
            
            logger.info(f"Cleaned up old history, kept {len(self.medical_history)} active individuals")
        except Exception as e:
            logger.error(f"History cleanup failed: {e}")
    
    def encode_face(self, image_array):
        """Encode un visage pour la reconnaissance"""
        if not FACE_RECOGNITION_AVAILABLE:
            return self._fallback_face_encoding(image_array)
        
        try:
            # Convertir en RGB si nécessaire
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            
            # Détecter les visages
            face_locations = face_recognition.face_locations(rgb_image)
            
            if face_locations:
                # Encoder le premier visage trouvé
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if face_encodings:
                    return face_encodings[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return None
    
    def _fallback_face_encoding(self, image_array):
        """Encoding de fallback sans face_recognition"""
        try:
            # Utiliser des caractéristiques simples comme fallback
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Redimensionner à une taille standard
            resized = cv2.resize(gray, (64, 64))
            
            # Calculer des features simples
            features = np.array([
                np.mean(resized),
                np.std(resized),
                np.mean(resized[:32, :]),  # Partie supérieure
                np.mean(resized[32:, :]),  # Partie inférieure
                np.mean(resized[:, :32]),  # Partie gauche
                np.mean(resized[:, 32:]),  # Partie droite
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Fallback encoding failed: {e}")
            return None
    
    def identify_or_register_face(self, image_array, analysis_results=None):
        """Identifie un visage existant ou en enregistre un nouveau"""
        try:
            # Encoder le visage
            face_encoding = self.encode_face(image_array)
            if face_encoding is None:
                return None
            
            # Chercher dans la base de données
            person_id = self._find_matching_face(face_encoding)
            
            if person_id is None:
                # Nouveau visage - créer un ID
                person_id = self._generate_person_id()
                self.face_database[person_id] = {
                    'encoding': face_encoding,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'encounter_count': 1
                }
                logger.info(f"Registered new face: {person_id}")
            else:
                # Visage existant - mettre à jour
                self.face_database[person_id]['last_seen'] = time.time()
                self.face_database[person_id]['encounter_count'] += 1
                logger.info(f"Recognized existing face: {person_id}")
            
            # Enregistrer l'analyse actuelle
            if analysis_results:
                self._record_medical_analysis(person_id, analysis_results)
            
            return person_id
            
        except Exception as e:
            logger.error(f"Face identification failed: {e}")
            return None
    
    def _find_matching_face(self, face_encoding):
        """Trouve un visage correspondant dans la base de données"""
        if not self.face_database:
            return None
        
        try:
            if FACE_RECOGNITION_AVAILABLE:
                # Comparaison avec face_recognition
                known_encodings = [data['encoding'] for data in self.face_database.values()]
                known_ids = list(self.face_database.keys())
                
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                
                if any(matches):
                    match_index = matches.index(True)
                    return known_ids[match_index]
            else:
                # Comparaison par distance euclidienne (fallback)
                min_distance = float('inf')
                best_match = None
                
                for person_id, data in self.face_database.items():
                    distance = np.linalg.norm(face_encoding - data['encoding'])
                    if distance < min_distance and distance < 0.8:  # Seuil pour fallback
                        min_distance = distance
                        best_match = person_id
                
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            return None
    
    def _generate_person_id(self):
        """Génère un ID unique pour une nouvelle personne"""
        timestamp = int(time.time())
        counter = len(self.face_database)
        return f"person_{timestamp}_{counter:03d}"
    
    def _record_medical_analysis(self, person_id, analysis_results):
        """Enregistre une analyse médicale pour une personne"""
        try:
            medical_record = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'analysis': self._extract_medical_summary(analysis_results),
                'confidence_adjustments': []
            }
            
            self.medical_history[person_id].append(medical_record)
            
            # Limiter l'historique à 50 entrées par personne
            if len(self.medical_history[person_id]) > 50:
                self.medical_history[person_id] = self.medical_history[person_id][-50:]
            
            logger.info(f"Recorded medical analysis for {person_id}")
            
        except Exception as e:
            logger.error(f"Failed to record medical analysis: {e}")
    
    def _extract_medical_summary(self, analysis_results):
        """Extrait un résumé médical des résultats d'analyse"""
        try:
            summary = {
                'eyes_analyzed': len(analysis_results),
                'positive_detections': 0,
                'high_risk_count': 0,
                'average_confidence': 0,
                'eyes_data': []
            }
            
            total_confidence = 0
            
            for result in analysis_results:
                eye_data = {
                    'position': result.get('eye_region', {}).get('position', 'unknown'),
                    'leukocoria_detected': result.get('leukocoria_detected', False),
                    'confidence': result.get('confidence', 0),
                    'risk_level': result.get('risk_level', 'low'),
                    'urgency': result.get('urgency', 'routine')
                }
                
                summary['eyes_data'].append(eye_data)
                total_confidence += eye_data['confidence']
                
                if eye_data['leukocoria_detected']:
                    summary['positive_detections'] += 1
                
                if eye_data['risk_level'] == 'high':
                    summary['high_risk_count'] += 1
            
            if summary['eyes_analyzed'] > 0:
                summary['average_confidence'] = total_confidence / summary['eyes_analyzed']
            
            return summary
            
        except Exception as e:
            logger.error(f"Medical summary extraction failed: {e}")
            return {'error': str(e)}
    
    def adjust_confidence_with_history(self, person_id, current_analysis):
        """Ajuste la confiance basée sur l'historique médical"""
        if person_id not in self.medical_history:
            return current_analysis  # Pas d'historique
        
        try:
            history = self.medical_history[person_id]
            if len(history) < 2:
                return current_analysis  # Pas assez d'historique
            
            # Analyser l'historique récent (30 derniers jours)
            recent_cutoff = time.time() - (30 * 24 * 3600)
            recent_analyses = [
                entry for entry in history 
                if entry.get('timestamp', 0) > recent_cutoff
            ]
            
            if not recent_analyses:
                return current_analysis
            
            # Calculer les tendances
            adjustments = self._calculate_confidence_adjustments(recent_analyses, current_analysis)
            
            # Appliquer les ajustements
            adjusted_analysis = self._apply_confidence_adjustments(current_analysis, adjustments)
            
            # Enregistrer les ajustements
            self._record_adjustments(person_id, adjustments)
            
            return adjusted_analysis
            
        except Exception as e:
            logger.error(f"Confidence adjustment failed: {e}")
            return current_analysis
    
    def _calculate_confidence_adjustments(self, history, current_analysis):
        """Calcule les ajustements de confiance basés sur l'historique"""
        try:
            adjustments = []
            
            # Analyser la consistance des détections
            historical_positives = []
            historical_confidences = []
            
            for entry in history:
                analysis = entry.get('analysis', {})
                if analysis:
                    historical_positives.append(analysis.get('positive_detections', 0))
                    historical_confidences.append(analysis.get('average_confidence', 0))
            
            if not historical_positives:
                return adjustments
            
            # Tendance des détections positives
            avg_historical_positives = np.mean(historical_positives)
            current_positives = sum(1 for r in current_analysis if r.get('leukocoria_detected', False))
            
            # Tendance des niveaux de confiance
            avg_historical_confidence = np.mean(historical_confidences)
            current_avg_confidence = np.mean([r.get('confidence', 0) for r in current_analysis])
            
            # Ajustement basé sur la consistance
            for i, result in enumerate(current_analysis):
                adjustment = {
                    'eye_index': i,
                    'original_confidence': result.get('confidence', 0),
                    'adjustment_factor': 0,
                    'reasoning': []
                }
                
                # Si détection positive consistante dans l'historique
                if avg_historical_positives > 0.5 and result.get('leukocoria_detected', False):
                    adjustment['adjustment_factor'] += self.confidence_adjustment_factor
                    adjustment['reasoning'].append("Consistent positive detection history")
                
                # Si aucune détection dans l'historique mais détection actuelle
                elif avg_historical_positives < 0.2 and result.get('leukocoria_detected', False):
                    adjustment['adjustment_factor'] -= self.confidence_adjustment_factor
                    adjustment['reasoning'].append("Inconsistent with negative history")
                
                # Ajustement basé sur les niveaux de confiance
                confidence_diff = current_avg_confidence - avg_historical_confidence
                if abs(confidence_diff) > 20:  # Différence significative
                    if confidence_diff > 0:
                        adjustment['adjustment_factor'] += 0.05
                        adjustment['reasoning'].append("Higher confidence than historical average")
                    else:
                        adjustment['adjustment_factor'] -= 0.05
                        adjustment['reasoning'].append("Lower confidence than historical average")
                
                adjustments.append(adjustment)
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Confidence adjustment calculation failed: {e}")
            return []
    
    def _apply_confidence_adjustments(self, analysis_results, adjustments):
        """Applique les ajustements de confiance"""
        try:
            adjusted_results = []
            
            for i, result in enumerate(analysis_results):
                adjusted_result = result.copy()
                
                # Trouver l'ajustement correspondant
                adjustment = next((adj for adj in adjustments if adj['eye_index'] == i), None)
                
                if adjustment and adjustment['adjustment_factor'] != 0:
                    original_confidence = result.get('confidence', 0)
                    adjustment_factor = adjustment['adjustment_factor']
                    
                    # Appliquer l'ajustement (en pourcentage)
                    new_confidence = original_confidence * (1 + adjustment_factor)
                    new_confidence = max(0, min(100, new_confidence))  # Borner entre 0 et 100
                    
                    adjusted_result['confidence'] = new_confidence
                    adjusted_result['confidence_adjusted'] = True
                    adjusted_result['original_confidence'] = original_confidence
                    adjusted_result['adjustment_factor'] = adjustment_factor
                    adjusted_result['adjustment_reasoning'] = adjustment['reasoning']
                    
                    # Réévaluer le niveau de risque si nécessaire
                    if abs(new_confidence - original_confidence) > 10:
                        adjusted_result = self._reevaluate_risk_level(adjusted_result)
                
                adjusted_results.append(adjusted_result)
            
            return adjusted_results
            
        except Exception as e:
            logger.error(f"Confidence adjustment application failed: {e}")
            return analysis_results
    
    def _reevaluate_risk_level(self, result):
        """Réévalue le niveau de risque basé sur la nouvelle confiance"""
        try:
            confidence = result.get('confidence', 0)
            
            if confidence >= 80:
                result['risk_level'] = 'high'
                result['urgency'] = 'immediate'
            elif confidence >= 60:
                result['risk_level'] = 'medium'
                result['urgency'] = 'urgent'
            elif confidence >= 40:
                result['risk_level'] = 'medium'
                result['urgency'] = 'soon'
            else:
                result['risk_level'] = 'low'
                result['urgency'] = 'routine'
            
            return result
            
        except Exception as e:
            logger.error(f"Risk level reevaluation failed: {e}")
            return result
    
    def _record_adjustments(self, person_id, adjustments):
        """Enregistre les ajustements effectués"""
        try:
            if self.medical_history[person_id]:
                last_entry = self.medical_history[person_id][-1]
                last_entry['confidence_adjustments'] = adjustments
                
        except Exception as e:
            logger.error(f"Failed to record adjustments: {e}")
    
    def get_person_summary(self, person_id):
        """Obtient un résumé médical pour une personne"""
        try:
            if person_id not in self.face_database:
                return None
            
            face_data = self.face_database[person_id]
            history = self.medical_history.get(person_id, [])
            
            summary = {
                'person_id': person_id,
                'first_seen': datetime.fromtimestamp(face_data['first_seen']).isoformat(),
                'last_seen': datetime.fromtimestamp(face_data['last_seen']).isoformat(),
                'encounter_count': face_data['encounter_count'],
                'total_analyses': len(history),
                'recent_analyses': [],
                'trends': {}
            }
            
            # Analyses récentes (30 derniers jours)
            recent_cutoff = time.time() - (30 * 24 * 3600)
            recent_analyses = [
                entry for entry in history 
                if entry.get('timestamp', 0) > recent_cutoff
            ]
            
            summary['recent_analyses'] = recent_analyses[-5:]  # 5 plus récentes
            
            # Calculer les tendances
            if len(recent_analyses) >= 2:
                summary['trends'] = self._calculate_trends(recent_analyses)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get person summary: {e}")
            return None
    
    def _calculate_trends(self, analyses):
        """Calcule les tendances médicales"""
        try:
            trends = {
                'detection_trend': 'stable',
                'confidence_trend': 'stable',
                'risk_trend': 'stable'
            }
            
            if len(analyses) < 2:
                return trends
            
            # Analyser les détections positives
            positive_counts = [analysis.get('analysis', {}).get('positive_detections', 0) for analysis in analyses]
            if len(positive_counts) >= 2:
                if positive_counts[-1] > positive_counts[0]:
                    trends['detection_trend'] = 'increasing'
                elif positive_counts[-1] < positive_counts[0]:
                    trends['detection_trend'] = 'decreasing'
            
            # Analyser les niveaux de confiance
            confidences = [analysis.get('analysis', {}).get('average_confidence', 0) for analysis in analyses]
            if len(confidences) >= 2:
                confidence_diff = confidences[-1] - confidences[0]
                if confidence_diff > 10:
                    trends['confidence_trend'] = 'increasing'
                elif confidence_diff < -10:
                    trends['confidence_trend'] = 'decreasing'
            
            return trends
            
        except Exception as e:
            logger.error(f"Trends calculation failed: {e}")
            return {}
    
    def get_all_individuals_summary(self):
        """Obtient un résumé de tous les individus suivis"""
        try:
            summaries = []
            
            for person_id in self.face_database.keys():
                summary = self.get_person_summary(person_id)
                if summary:
                    summaries.append(summary)
            
            # Trier par dernière vue (plus récent en premier)
            summaries.sort(key=lambda x: x['last_seen'], reverse=True)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to get all summaries: {e}")
            return []
    
    def cleanup_and_save(self):
        """Nettoie et sauvegarde toutes les données"""
        try:
            self._cleanup_old_history()
            self.save_database()
            self.save_medical_history()
            logger.info("Face tracker data cleaned up and saved")
        except Exception as e:
            logger.error(f"Cleanup and save failed: {e}")
    
    def __del__(self):
        """Destructeur pour sauvegarder automatiquement"""
        try:
            self.save_database()
            self.save_medical_history()
        except:
            pass