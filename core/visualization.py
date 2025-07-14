"""
Module de visualisation pour RetinoblastoGemma
Affiche les résultats de détection avec des boîtes colorées et informations
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from config.settings import *

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self):
        self.colors = COLORS
        self.box_thickness = BOX_THICKNESS
        self.font_scale = FONT_SCALE
        self.font_thickness = FONT_THICKNESS
        
        # Essayer de charger une police personnalisée
        try:
            self.pil_font = ImageFont.truetype("arial.ttf", 16)
            self.pil_font_large = ImageFont.truetype("arial.ttf", 20)
        except:
            self.pil_font = ImageFont.load_default()
            self.pil_font_large = ImageFont.load_default()
    
    def draw_detection_results(self, image_path: str, detection_results: Dict, 
                              analysis_results: Dict = None, 
                              face_tracking_info: Dict = None) -> np.ndarray:
        """
        Dessine les résultats de détection sur l'image
        
        Args:
            image_path: Chemin vers l'image originale
            detection_results: Résultats de la détection d'yeux
            analysis_results: Résultats de l'analyse Gemma
            face_tracking_info: Informations de suivi facial
            
        Returns:
            Image annotée (format OpenCV)
        """
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Impossible de charger l'image: {image_path}")
            return None
        
        annotated_image = image.copy()
        
        # Traiter chaque visage détecté
        for face_idx, face_data in enumerate(detection_results.get('faces', [])):
            annotated_image = self._draw_single_face(
                annotated_image, face_data, analysis_results, 
                face_tracking_info, face_idx
            )
        
        # Ajouter des informations générales
        annotated_image = self._add_general_info(
            annotated_image, detection_results, analysis_results
        )
        
        return annotated_image
    
    def _draw_single_face(self, image: np.ndarray, face_data: Dict, 
                         analysis_results: Dict, face_tracking_info: Dict, 
                         face_idx: int) -> np.ndarray:
        """Dessine les annotations pour un seul visage"""
        
        # Dessiner la boîte du visage
        face_box = face_data.get('face_box')
        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(image, (x, y), (x + w, y + h), 
                         self.colors['face_box'], self.box_thickness)
            
            # Ajouter l'ID du visage si disponible
            if face_tracking_info and 'face_ids' in face_tracking_info:
                face_id = face_tracking_info['face_ids'].get(str(face_idx), 'Unknown')
                cv2.putText(image, f"ID: {face_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                           self.colors['face_box'], self.font_thickness)
        
        # Dessiner les yeux
        for eye_data in face_data.get('eyes', []):
            self._draw_single_eye(image, eye_data, analysis_results, face_idx)
        
        return image
    
    def _draw_single_eye(self, image: np.ndarray, eye_data: Dict, 
                        analysis_results: Dict, face_idx: int):
        """Dessine les annotations pour un seul œil"""
        eye_bbox = eye_data.get('bbox')
        if not eye_bbox:
            return
        
        x, y, w, h = eye_bbox
        position = eye_data.get('position', 'unknown')
        
        # Déterminer la couleur basée sur l'analyse
        color = self.colors['normal']  # Par défaut vert
        confidence = 0
        risk_level = 'low'
        
        if analysis_results:
            # Chercher l'analyse correspondante
            eye_analysis = self._find_eye_analysis(analysis_results, face_idx, position)
            if eye_analysis:
                confidence = eye_analysis.get('confidence', 0)
                if eye_analysis.get('leukocoria_detected', False):
                    risk_level = eye_analysis.get('risk_level', 'medium')
                    if risk_level == 'high':
                        color = self.colors['abnormal']  # Rouge
                    else:
                        color = self.colors['uncertain']  # Jaune
        
        # Dessiner la boîte de l'œil
        cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_thickness)
        
        # Étiquette avec position et confiance
        label = f"{position.upper()}"
        if confidence > 0:
            label += f" {confidence:.1f}%"
        
        # Position du texte
        text_y = y - 10 if y > 30 else y + h + 25
        cv2.putText(image, label, (x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   color, self.font_thickness)
        
        # Indicateur de risque
        if risk_level != 'low':
            risk_text = f"RISK: {risk_level.upper()}"
            text_y2 = text_y + 20 if text_y == y - 10 else text_y - 20
            cv2.putText(image, risk_text, (x, text_y2),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8,
                       color, self.font_thickness)
    
    def _find_eye_analysis(self, analysis_results: Dict, face_idx: int, 
                          eye_position: str) -> Dict:
        """Trouve l'analyse correspondant à un œil spécifique"""
        if not analysis_results or 'faces' not in analysis_results:
            return None
        
        face_analyses = analysis_results.get('faces', [])
        if face_idx >= len(face_analyses):
            return None
        
        face_analysis = face_analyses[face_idx]
        for eye_analysis in face_analysis.get('eyes', []):
            if eye_analysis.get('position') == eye_position:
                return eye_analysis
        
        return None
    
    def _add_general_info(self, image: np.ndarray, detection_results: Dict, 
                         analysis_results: Dict) -> np.ndarray:
        """Ajoute des informations générales sur l'image"""
        h, w = image.shape[:2]
        
        # Informations de base
        total_faces = len(detection_results.get('faces', []))
        total_eyes = detection_results.get('total_eyes_detected', 0)
        
        # Compter les détections positives
        positive_detections = 0
        if analysis_results:
            for face_analysis in analysis_results.get('faces', []):
                for eye_analysis in face_analysis.get('eyes', []):
                    if eye_analysis.get('leukocoria_detected', False):
                        positive_detections += 1
        
        # Texte d'information
        info_lines = [
            f"Faces: {total_faces} | Eyes: {total_eyes}",
            f"Positive detections: {positive_detections}"
        ]
        
        if positive_detections > 0:
            info_lines.append("⚠️ MEDICAL CONSULTATION ADVISED")
        
        # Position en haut à gauche
        y_start = 30
        for i, line in enumerate(info_lines):
            color = self.colors['abnormal'] if 'MEDICAL' in line else (255, 255, 255)
            cv2.putText(image, line, (10, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       color, self.font_thickness)
        
        return image
    
    def create_analysis_summary(self, analysis_results: Dict, 
                               face_tracking_info: Dict = None) -> plt.Figure:
        """Crée un résumé graphique de l'analyse"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('RetinoblastoGemma - Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Distribution des niveaux de confiance
        confidences = []
        risk_levels = []
        
        for face_analysis in analysis_results.get('faces', []):
            for eye_analysis in face_analysis.get('eyes', []):
                confidences.append(eye_analysis.get('confidence', 0))
                risk_levels.append(eye_analysis.get('risk_level', 'low'))
        
        if confidences:
            axes[0, 0].hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Confidence Distribution')
            axes[0, 0].set_xlabel('Confidence (%)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].axvline(x=CONFIDENCE_THRESHOLD*100, color='red', linestyle='--', 
                              label=f'Threshold ({CONFIDENCE_THRESHOLD*100}%)')
            axes[0, 0].legend()
        
        # 2. Distribution des niveaux de risque
        if risk_levels:
            risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
            colors_map = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            
            bars = axes[0, 1].bar(risk_counts.keys(), risk_counts.values(),
                                 color=[colors_map.get(k, 'gray') for k in risk_counts.keys()])
            axes[0, 1].set_title('Risk Level Distribution')
            axes[0, 1].set_ylabel('Count')
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
        
        # 3. Détections par position d'œil
        positions = []
        detections = []
        
        for face_analysis in analysis_results.get('faces', []):
            for eye_analysis in face_analysis.get('eyes', []):
                pos = eye_analysis.get('position', 'unknown')
                detected = eye_analysis.get('leukocoria_detected', False)
                positions.append(pos)
                detections.append(detected)
        
        if positions:
            pos_detection = {}
            for pos in set(positions):
                pos_indices = [i for i, p in enumerate(positions) if p == pos]
                pos_detections = [detections[i] for i in pos_indices]
                pos_detection[pos] = {
                    'total': len(pos_detections),
                    'positive': sum(pos_detections)
                }
            
            pos_names = list(pos_detection.keys())
            total_counts = [pos_detection[pos]['total'] for pos in pos_names]
            positive_counts = [pos_detection[pos]['positive'] for pos in pos_names]
            
            x = np.arange(len(pos_names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, total_counts, width, label='Total', alpha=0.7)
            axes[1, 0].bar(x + width/2, positive_counts, width, label='Positive', alpha=0.7)
            axes[1, 0].set_title('Detections by Eye Position')
            axes[1, 0].set_xlabel('Eye Position')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(pos_names)
            axes[1, 0].legend()
        
        # 4. Résumé textuel
        axes[1, 1].axis('off')
        summary_text = self._generate_text_summary(analysis_results, face_tracking_info)
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Summary')
        
        plt.tight_layout()
        return fig
    
    def _generate_text_summary(self, analysis_results: Dict, 
                              face_tracking_info: Dict = None) -> str:
        """Génère un résumé textuel de l'analyse"""
        total_faces = len(analysis_results.get('faces', []))
        total_eyes = sum(len(face.get('eyes', [])) for face in analysis_results.get('faces', []))
        
        positive_count = 0
        high_risk_count = 0
        
        for face_analysis in analysis_results.get('faces', []):
            for eye_analysis in face_analysis.get('eyes', []):
                if eye_analysis.get('leukocoria_detected', False):
                    positive_count += 1
                    if eye_analysis.get('risk_level') == 'high':
                        high_risk_count += 1
        
        summary = f"""Analysis Results:

• Faces detected: {total_faces}
• Eyes analyzed: {total_eyes}
• Positive detections: {positive_count}
• High-risk cases: {high_risk_count}

"""
        
        if positive_count > 0:
            summary += "⚠️ ATTENTION REQUIRED\n"
            summary += "Medical consultation recommended\n"
        else:
            summary += "✓ No concerning findings\n"
            summary += "Continue regular monitoring\n"
        
        if face_tracking_info and 'tracked_individuals' in face_tracking_info:
            tracked = len(face_tracking_info['tracked_individuals'])
            summary += f"\n• Tracked individuals: {tracked}"
        
        return summary
    
    def save_annotated_image(self, annotated_image: np.ndarray, 
                            output_path: str = None) -> str:
        """Sauvegarde l'image annotée"""
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"analysis_{timestamp}.jpg"
        
        success = cv2.imwrite(str(output_path), annotated_image)
        
        if success:
            logger.info(f"Image annotée sauvegardée: {output_path}")
            return str(output_path)
        else:
            logger.error(f"Erreur lors de la sauvegarde: {output_path}")
            return None
    
    def create_progression_chart(self, face_id: str, progression_data: Dict) -> plt.Figure:
        """Crée un graphique de progression pour un individu"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Progression Analysis - {face_id}', fontsize=14, fontweight='bold')
        
        risk_progression = progression_data.get('risk_progression', [])
        
        if not risk_progression:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        # Données pour les graphiques
        dates = [entry['date'][:10] for entry in risk_progression]  # YYYY-MM-DD
        scores = [entry['score'] for entry in risk_progression]
        levels = [entry['level'] for entry in risk_progression]
        
        # Graphique 1: Scores de risque dans le temps
        ax1.plot(dates, scores, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Risk Score Over Time')
        ax1.set_ylabel('Risk Score')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Seuils de risque
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='High Risk')
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Medium Risk')
        ax1.axhline(y=20, color='yellow', linestyle='--', alpha=0.7, label='Low Risk')
        ax1.legend()
        
        # Graphique 2: Distribution des niveaux de risque
        level_counts = {level: levels.count(level) for level in set(levels)}
        colors_map = {'MINIMAL': 'green', 'LOW': 'lightgreen', 
                     'MEDIUM': 'orange', 'HIGH': 'red'}
        
        bars = ax2.bar(level_counts.keys(), level_counts.values(),
                      color=[colors_map.get(k, 'gray') for k in level_counts.keys()])
        ax2.set_title('Risk Level Distribution')
        ax2.set_ylabel('Number of Analyses')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
