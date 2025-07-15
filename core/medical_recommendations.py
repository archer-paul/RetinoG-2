"""
Medical Recommendations Module - Recommandations médicales intelligentes
Génère des recommandations personnalisées basées sur l'analyse et l'historique patient
"""
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MedicalRecommendation:
    """Structure pour une recommandation médicale"""
    urgency_level: str  # immediate, urgent, soon, routine
    primary_action: str
    timeframe: str
    reasoning: str
    additional_actions: List[str]
    follow_up: str
    risk_assessment: str

class MedicalRecommendationsEngine:
    """Moteur de recommandations médicales intelligentes"""
    
    def __init__(self):
        # Base de connaissances médicales
        self.medical_knowledge = {
            'retinoblastoma_facts': {
                'incidence': '1 sur 15 000 à 20 000 naissances',
                'age_peak': 'Moins de 6 ans (90% des cas)',
                'survival_early': 95,  # %
                'survival_late': 30,   # %
                'doubling_time': '2-4 semaines',
                'critical_window': '3-6 mois'
            },
            
            'urgency_thresholds': {
                'immediate': {
                    'positive_detections': 1,
                    'confidence_min': 70,
                    'consistency_rate': 50,
                    'recent_positive': 1
                },
                'urgent': {
                    'positive_detections': 1,
                    'confidence_min': 50,
                    'consistency_rate': 30,
                    'recent_positive': 0
                },
                'soon': {
                    'positive_detections': 0,
                    'confidence_min': 30,
                    'consistency_rate': 20,
                    'recent_positive': 0
                }
            },
            
            'risk_factors': {
                'family_history': 2.0,      # Multiplicateur de risque
                'bilateral': 1.8,
                'early_age': 1.5,
                'genetic_syndrome': 2.5
            }
        }
        
        # Templates de recommandations
        self.recommendation_templates = self._initialize_templates()
        
        logger.info("Medical recommendations engine initialized")
    
    def _initialize_templates(self) -> Dict:
        """Initialise les templates de recommandations"""
        return {
            'immediate': {
                'primary_action': 'Contacter un ophtalmologiste pédiatrique IMMÉDIATEMENT',
                'timeframe': 'AUJOURD\'HUI - Ne pas attendre',
                'follow_up': 'Rendez-vous d\'urgence sous 24-48h',
                'additional_actions': [
                    'Appeler directement le service d\'ophtalmologie pédiatrique',
                    'Se rendre aux urgences si aucun spécialiste disponible',
                    'Apporter toutes les photos et analyses précédentes',
                    'Préparer la liste des antécédents familiaux',
                    'Éviter toute exposition intense à la lumière en attendant'
                ]
            },
            'urgent': {
                'primary_action': 'Programmer une consultation d\'ophtalmologie pédiatrique',
                'timeframe': 'Dans les 1-2 semaines',
                'follow_up': 'Suivi rapproché selon recommandations spécialiste',
                'additional_actions': [
                    'Prendre des photos quotidiennes jusqu\'à la consultation',
                    'Noter tout changement dans l\'apparence des yeux',
                    'Rassembler l\'historique médical familial',
                    'Surveiller d\'autres symptômes (strabisme, douleur)',
                    'Préparer questions pour le spécialiste'
                ]
            },
            'soon': {
                'primary_action': 'Consulter un ophtalmologiste pédiatrique',
                'timeframe': 'Dans le mois',
                'follow_up': 'Surveillance continue avec photos mensuelles',
                'additional_actions': [
                    'Maintenir surveillance photographique régulière',
                    'Programmer examen oculaire de routine',
                    'Éduquer l\'entourage sur les signes à surveiller',
                    'Continuer dépistage IA mensuel',
                    'Documenter toute observation inhabituelle'
                ]
            },
            'routine': {
                'primary_action': 'Continuer surveillance oculaire pédiatrique régulière',
                'timeframe': 'Selon calendrier de suivi habituel',
                'follow_up': 'Dépistage IA trimestriel recommandé',
                'additional_actions': [
                    'Photos mensuelles sous bon éclairage',
                    'Examens oculaires pédiatriques annuels',
                    'Éducation continue sur la leucocorie',
                    'Maintenir dossier photographique',
                    'Répéter dépistage si inquiétudes'
                ]
            }
        }
    
    def generate_recommendations(
        self,
        analysis_results: Dict,
        patient_history: Optional[Dict] = None,
        risk_