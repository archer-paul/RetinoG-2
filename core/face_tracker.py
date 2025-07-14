"""
Système de reconnaissance faciale pour le suivi des enfants
Permet de détecter l'évolution du rétinoblastome chez le même individu
"""
import face_recognition
import numpy as np
import cv2
from PIL import Image
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from config.settings import *

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self):
        self.known_faces = {}  # face_id -> {'encoding': array, 'metadata': dict}
        self.face_history = {}  # face_id -> [{'date': str, 'analysis': dict, 'image_path': str}]
        self.next_face_id = 0
        self.database_path = RESULTS_DIR / "face_database.pkl"
        self.history_path = RESULTS_DIR / "face_history.json"
        self.load_database()
    
    def load_database(self):
        """Charge la base de données des visages connus"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('known_faces', {})
                    self.next_face_id = data.get('next_face_id', 0)
                logger.info(f"Base de données chargée: {len(self.known_faces)} visages connus")
            
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    self.face_history = json.load(f)
                logger.info(f"Historique chargé: {len(self.face_history)} individus suivis")
                    
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la base: {e}")
            self.known_faces = {}
            self.face_history = {}
            self.next_face_id = 0
    
    def save_database(self):
        """Sauvegarde la base de données"""
        try:
            # Sauvegarder les encodages de visages
            data = {
                'known_faces': self.known_faces,
                'next_face_id': self.next_face_id
            }
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Sauvegarder l'historique (JSON pour lisibilité)
            with open(self.history_path, 'w') as f:
                json.dump(self.face_history, f, indent=2, default=str)
                
            logger.info("Base de données sauvegardée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def extract_face_encoding(self, image_path: str, face_box: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Extrait l'encodage facial d'une image
        
        Args:
            image_path: Chemin vers l'image
            face_box: Boîte englobante du visage (x, y, w, h)
            
        Returns:
            Encodage facial ou None si échec
        """
        try:
            # Charger l'image
            image = face_recognition.load_image_file(image_path)
            
            if face_box:
                # Utiliser la boîte englobante fournie
                x, y, w, h = face_box
                face_locations = [(y, x + w, y + h, x)]  # Format (top, right, bottom, left)
            else:
                # Détecter automatiquement les visages
                face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"Aucun visage détecté dans {image_path}")
                return None
            
            # Utiliser le premier visage détecté
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if face_encodings:
                return face_encodings[0]
            else:
                logger.warning(f"Impossible d'encoder le visage dans {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction d'encodage: {e}")
            return None
    
    def identify_or_register_face(self, image_path: str, face_box: Tuple[int, int, int, int] = None, 
                                 metadata: Dict = None) -> Tuple[str, bool]:
        """
        Identifie un visage ou l'enregistre s'il est nouveau
        
        Args:
            image_path: Chemin vers l'image
            face_box: Boîte englobante du visage
            metadata: Métadonnées sur le visage (âge estimé, etc.)
            
        Returns:
            Tuple (face_id, is_new_face)
        """
        encoding = self.extract_face_encoding(image_path, face_box)
        
        if encoding is None:
            return None, False
        
        # Chercher des correspondances avec les visages connus
        face_id = self._find_matching_face(encoding)
        
        if face_id is None:
            # Nouveau visage
            face_id = f"child_{self.next_face_id:04d}"
            self.known_faces[face_id] = {
                'encoding': encoding,
                'metadata': metadata or {},
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }
            self.face_history[face_id] = []
            self.next_face_id += 1
            logger.info(f"Nouveau visage enregistré: {face_id}")
            return face_id, True
        else:
            # Visage connu, mettre à jour la dernière vue
            self.known_faces[face_id]['last_seen'] = datetime.now().isoformat()
            if metadata:
                self.known_faces[face_id]['metadata'].update(metadata)
            logger.info(f"Visage reconnu: {face_id}")
            return face_id, False
    
    def _find_matching_face(self, encoding: np.ndarray) -> Optional[str]:
        """Trouve un visage correspondant dans la base"""
        if not self.known_faces:
            return None
        
        known_encodings = []
        known_ids = []
        
        for face_id, face_data in self.known_faces.items():
            known_encodings.append(face_data['encoding'])
            known_ids.append(face_id)
        
        # Comparer avec tous les visages connus
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=FACE_SIMILARITY_THRESHOLD
        )
        
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        
        # Trouver la meilleure correspondance
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return known_ids[best_match_index]
        
        return None
    
    def add_analysis_to_history(self, face_id: str, analysis_results: Dict, 
                               image_path: str = None):
        """Ajoute une analyse à l'historique d'un individu"""
        if face_id not in self.face_history:
            self.face_history[face_id] = []
        
        history_entry = {
            'date': datetime.now().isoformat(),
            'analysis': analysis_results,
            'image_path': str(image_path) if image_path else None,
            'risk_assessment': self._assess_risk_level(analysis_results)
        }
        
        self.face_history[face_id].append(history_entry)
        logger.info(f"Analyse ajoutée à l'historique de {face_id}")
    
    def _assess_risk_level(self, analysis_results: Dict) -> Dict:
        """Évalue le niveau de risque basé sur l'analyse"""
        risk_score = 0
        risk_factors = []
        
        # Analyser chaque œil
        for eye_data in analysis_results.get('eyes', []):
            if eye_data.get('leukocoria_detected', False):
                risk_score += eye_data.get('confidence', 0)
                risk_factors.append(f"{eye_data.get('position')} eye leukocoria")
        
        # Déterminer le niveau de risque
        if risk_score > 80:
            level = "HIGH"
        elif risk_score > 50:
            level = "MEDIUM"
        elif risk_score > 20:
            level = "LOW"
        else:
            level = "MINIMAL"
        
        return {
            'level': level,
            'score': risk_score,
            'factors': risk_factors
        }
    
    def get_individual_progression(self, face_id: str) -> Dict:
        """Analyse la progression du rétinoblastome pour un individu"""
        if face_id not in self.face_history:
            return {'error': 'Individual not found'}
        
        history = self.face_history[face_id]
        if not history:
            return {'error': 'No analysis history'}
        
        # Analyser la progression
        progression = {
            'face_id': face_id,
            'total_analyses': len(history),
            'first_analysis': history[0]['date'],
            'last_analysis': history[-1]['date'],
            'risk_progression': [],
            'consistency_score': 0,
            'recommendation': ''
        }
        
        # Calculer la progression du risque
        for entry in history:
            risk = entry.get('risk_assessment', {})
            progression['risk_progression'].append({
                'date': entry['date'],
                'level': risk.get('level', 'UNKNOWN'),
                'score': risk.get('score', 0),
                'factors': risk.get('factors', [])
            })
        
        # Calculer la cohérence (détections répétées = plus fiable)
        high_risk_count = sum(1 for p in progression['risk_progression'] 
                             if p['level'] in ['HIGH', 'MEDIUM'])
        
        progression['consistency_score'] = (high_risk_count / len(history)) * 100
        
        # Générer une recommandation
        if progression['consistency_score'] > 60:
            progression['recommendation'] = "URGENT: Multiple consistent detections - seek immediate medical attention"
        elif progression['consistency_score'] > 30:
            progression['recommendation'] = "MODERATE: Some concerning findings - schedule ophthalmologist consultation"
        else:
            progression['recommendation'] = "MONITORING: Continue regular photo monitoring"
        
        return progression
    
    def get_all_individuals_summary(self) -> List[Dict]:
        """Retourne un résumé de tous les individus suivis"""
        summary = []
        
        for face_id in self.face_history:
            progression = self.get_individual_progression(face_id)
            if 'error' not in progression:
                metadata = self.known_faces.get(face_id, {}).get('metadata', {})
                summary.append({
                    'face_id': face_id,
                    'metadata': metadata,
                    'analyses_count': progression['total_analyses'],
                    'last_risk_level': progression['risk_progression'][-1]['level'] if progression['risk_progression'] else 'UNKNOWN',
                    'consistency_score': progression['consistency_score'],
                    'recommendation': progression['recommendation']
                })
        
        # Trier par score de cohérence (plus préoccupant en premier)
        summary.sort(key=lambda x: x['consistency_score'], reverse=True)
        return summary
    
    def export_individual_report(self, face_id: str, output_path: str = None) -> str:
        """Exporte un rapport détaillé pour un individu"""
        progression = self.get_individual_progression(face_id)
        
        if 'error' in progression:
            return f"Error: {progression['error']}"
        
        if output_path is None:
            output_path = RESULTS_DIR / f"{face_id}_report.json"
        
        # Ajouter les métadonnées
        full_report = {
            'individual_info': self.known_faces.get(face_id, {}),
            'progression_analysis': progression,
            'detailed_history': self.face_history[face_id],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        logger.info(f"Rapport exporté: {output_path}")
        return str(output_path)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Nettoie les anciennes données (optionnel)"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for face_id in list(self.face_history.keys()):
            # Filtrer l'historique ancien
            filtered_history = []
            for entry in self.face_history[face_id]:
                entry_date = datetime.fromisoformat(entry['date'])
                if entry_date > cutoff_date:
                    filtered_history.append(entry)
            
            if filtered_history:
                self.face_history[face_id] = filtered_history
            else:
                # Supprimer l'individu si plus d'historique récent
                del self.face_history[face_id]
                if face_id in self.known_faces:
                    del self.known_faces[face_id]
        
        logger.info(f"Nettoyage effectué: données antérieures à {cutoff_date}")
        self.save_database()