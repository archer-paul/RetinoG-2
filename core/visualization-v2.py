"""
Module de visualisation pour RetinoblastoGemma
Gère l'affichage, les annotations et la génération de rapports visuels
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class Visualizer:
    """Gestionnaire de visualisation et annotations médicales"""
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des couleurs et styles
        self.color_scheme = {
            'normal': '#2ECC71',      # Vert pour normal
            'low_risk': '#F39C12',    # Orange pour risque faible
            'medium_risk': '#E67E22', # Orange foncé pour risque moyen
            'high_risk': '#E74C3C',   # Rouge pour risque élevé
            'immediate': '#C0392B',   # Rouge foncé pour immédiat
            'background': '#ECF0F1',  # Gris clair pour fond
            'text': '#2C3E50',        # Gris foncé pour texte
            'accent': '#3498DB'       # Bleu pour accents
        }
        
        self.alert_symbols = {
            'normal': '✅',
            'low_risk': '⚠️',
            'medium_risk': '⚠️',
            'high_risk': '🚨',
            'immediate': '🚨'
        }
        
        # Configuration des polices
        self.fonts = self._load_fonts()
        
        logger.info("Visualizer initialized")
    
    def _load_fonts(self):
        """Charge les polices disponibles"""
        fonts = {}
        
        try:
            # Essayer différentes polices système
            font_names = [
                "arial.ttf", "Arial.ttf", 
                "calibri.ttf", "Calibri.ttf",
                "segoeui.ttf", "seguisb.ttf"
            ]
            
            for size_name, size in [('large', 20), ('medium', 16), ('small', 12), ('tiny', 10)]:
                for font_name in font_names:
                    try:
                        fonts[size_name] = ImageFont.truetype(font_name, size)
                        break
                    except (OSError, IOError):
                        continue
                
                # Fallback vers police par défaut
                if size_name not in fonts:
                    fonts[size_name] = ImageFont.load_default()
            
            logger.info("Fonts loaded successfully")
            
        except Exception as e:
            logger.warning(f"Font loading failed, using defaults: {e}")
            # Fallback complet
            for size_name in ['large', 'medium', 'small', 'tiny']:
                fonts[size_name] = ImageFont.load_default()
        
        return fonts
    
    def annotate_analysis_results(self, image_path, analysis_results, person_id=None, history_adjustments=None):
        """Annote une image avec les résultats d'analyse"""
        try:
            # Charger l'image originale
            original_image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(original_image)
            
            # Ajouter le titre général
            self._draw_main_title(draw, original_image.size)
            
            # Annoter chaque région d'œil
            for i, result in enumerate(analysis_results):
                self._annotate_eye_region(draw, result, i)
            
            # Ajouter le résumé global
            self._draw_global_summary(draw, original_image.size, analysis_results)
            
            # Ajouter les informations de tracking si disponibles
            if person_id:
                self._draw_tracking_info(draw, original_image.size, person_id, history_adjustments)
            
            # Ajouter l'horodatage
            self._draw_timestamp(draw, original_image.size)
            
            # Sauvegarder l'image annotée
            annotated_path = self._save_annotated_image(original_image, "analysis")
            
            logger.info(f"Image annotated and saved to: {annotated_path}")
            return annotated_path, original_image
            
        except Exception as e:
            logger.error(f"Image annotation failed: {e}")
            return None, None
    
    def _draw_main_title(self, draw, image_size):
        """Dessine le titre principal"""
        try:
            title = "RETINOBLASTOMA AI ANALYSIS - GEMMA 3N MULTIMODAL"
            
            # Position et style du titre
            title_y = 10
            
            # Fond du titre
            title_bbox = self._get_text_bbox(draw, (10, title_y), title, self.fonts['medium'])
            if title_bbox:
                padding = 10
                draw.rectangle([
                    title_bbox[0] - padding, title_bbox[1] - padding,
                    min(title_bbox[2] + padding, image_size[0] - 10), title_bbox[3] + padding
                ], fill=self.color_scheme['accent'], outline=self.color_scheme['accent'])
            
            # Texte du titre
            draw.text((10, title_y), title, fill='white', font=self.fonts['medium'])
            
        except Exception as e:
            logger.error(f"Main title drawing failed: {e}")
    
    def _annotate_eye_region(self, draw, result, index):
        """Annote une région d'œil spécifique"""
        try:
            eye_region = result.get('eye_region', {})
            bbox = eye_region.get('bbox')
            
            if not bbox:
                return
            
            x, y, w, h = bbox
            
            # Déterminer le style basé sur les résultats
            style = self._determine_annotation_style(result)
            
            # Dessiner le rectangle principal
            draw.rectangle([x, y, x + w, y + h], 
                          outline=style['color'], width=style['width'])
            
            # Dessiner un rectangle intérieur pour l'effet de profondeur
            if style['width'] > 3:
                inner_margin = 2
                draw.rectangle([x + inner_margin, y + inner_margin, 
                               x + w - inner_margin, y + h - inner_margin], 
                              outline=style['color'], width=1)
            
            # Label principal avec symbole
            position = eye_region.get('position', 'unknown')
            confidence = result.get('confidence', 0)
            
            main_label = f"{style['symbol']} {position.upper()}"
            confidence_label = f"Confidence: {confidence:.1f}%"
            
            # Position des labels
            label_y = y - 60 if y > 70 else y + h + 10
            
            # Dessiner le label principal
            self._draw_label_with_background(draw, (x, label_y), main_label, 
                                           style['color'], 'white', self.fonts['medium'])
            
            # Dessiner le label de confiance
            self._draw_label_with_background(draw, (x, label_y + 25), confidence_label,
                                           'white', style['color'], self.fonts['small'])
            
            # Ajouter des détails supplémentaires si ajustement d'historique
            if result.get('confidence_adjusted'):
                adjustment_label = f"Adjusted from {result.get('original_confidence', 0):.1f}%"
                self._draw_label_with_background(draw, (x, label_y + 45), adjustment_label,
                                               self.color_scheme['accent'], 'white', self.fonts['tiny'])
            
            # Dessiner l'indicateur de méthode
            method = result.get('analysis_method', 'unknown')
            method_indicator = self._get_method_indicator(method)
            
            method_y = y + h - 25 if y + h < 500 else y + 5
            self._draw_label_with_background(draw, (x + 5, method_y), method_indicator,
                                           self.color_scheme['background'], style['color'], self.fonts['tiny'])
            
        except Exception as e:
            logger.error(f"Eye region annotation failed: {e}")
    
    def _determine_annotation_style(self, result):
        """Détermine le style d'annotation basé sur les résultats"""
        try:
            leukocoria_detected = result.get('leukocoria_detected', False)
            risk_level = result.get('risk_level', 'low')
            urgency = result.get('urgency', 'routine')
            
            if not leukocoria_detected:
                return {
                    'color': self.color_scheme['normal'],
                    'width': 3,
                    'symbol': self.alert_symbols['normal']
                }
            
            # Détection positive - déterminer la sévérité
            if urgency == 'immediate' or risk_level == 'high':
                return {
                    'color': self.color_scheme['immediate'],
                    'width': 6,
                    'symbol': self.alert_symbols['immediate']
                }
            elif urgency == 'urgent' or risk_level == 'medium':
                return {
                    'color': self.color_scheme['high_risk'],
                    'width': 5,
                    'symbol': self.alert_symbols['high_risk']
                }
            else:
                return {
                    'color': self.color_scheme['medium_risk'],
                    'width': 4,
                    'symbol': self.alert_symbols['medium_risk']
                }
                
        except Exception as e:
            logger.error(f"Style determination failed: {e}")
            return {
                'color': self.color_scheme['normal'],
                'width': 3,
                'symbol': '?'
            }
    
    def _get_method_indicator(self, method):
        """Retourne l'indicateur de méthode d'analyse"""
        method_indicators = {
            'multimodal_vision': '🔍 Multimodal',
            'multimodal_intelligent_fallback': '🧠 AI+CV',
            'feature_based_fallback': '👁️ Vision',
            'fallback': '⚙️ Basic'
        }
        
        return method_indicators.get(method, '❓ Unknown')
    
    def _draw_global_summary(self, draw, image_size, analysis_results):
        """Dessine le résumé global de l'analyse"""
        try:
            total_eyes = len(analysis_results)
            positive_count = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
            high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
            
            # Créer le texte de résumé
            summary_parts = [f"Eyes: {total_eyes}"]
            
            if positive_count > 0:
                summary_parts.append(f"Positive: {positive_count}")
                summary_parts.append("⚠️ MEDICAL CONSULTATION REQUIRED")
                summary_color = self.color_scheme['high_risk']
            else:
                summary_parts.append("All Normal")
                summary_color = self.color_scheme['normal']
            
            summary_text = " | ".join(summary_parts)
            
            # Position du résumé (en bas de l'image)
            summary_y = image_size[1] - 60
            
            # Dessiner le résumé avec fond
            self._draw_label_with_background(draw, (10, summary_y), summary_text,
                                           summary_color, 'white', self.fonts['medium'])
            
        except Exception as e:
            logger.error(f"Global summary drawing failed: {e}")
    
    def _draw_tracking_info(self, draw, image_size, person_id, history_adjustments):
        """Dessine les informations de tracking"""
        try:
            if not person_id:
                return
            
            # Informations de tracking
            tracking_text = f"Person ID: {person_id}"
            
            if history_adjustments:
                tracking_text += " | History-Adjusted"
            
            # Position (coin supérieur droit)
            text_width = len(tracking_text) * 8  # Estimation approximative
            tracking_x = image_size[0] - text_width - 20
            tracking_y = 10
            
            self._draw_label_with_background(draw, (tracking_x, tracking_y), tracking_text,
                                           self.color_scheme['accent'], 'white', self.fonts['small'])
            
        except Exception as e:
            logger.error(f"Tracking info drawing failed: {e}")
    
    def _draw_timestamp(self, draw, image_size):
        """Dessine l'horodatage"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_text = f"Generated: {timestamp}"
            
            # Position (coin inférieur droit)
            text_width = len(timestamp_text) * 6
            timestamp_x = image_size[0] - text_width - 20
            timestamp_y = image_size[1] - 30
            
            self._draw_label_with_background(draw, (timestamp_x, timestamp_y), timestamp_text,
                                           self.color_scheme['background'], self.color_scheme['text'], 
                                           self.fonts['tiny'])
            
        except Exception as e:
            logger.error(f"Timestamp drawing failed: {e}")
    
    def _draw_label_with_background(self, draw, position, text, bg_color, text_color, font):
        """Dessine un label avec fond coloré"""
        try:
            x, y = position
            
            # Calculer la taille du texte
            text_bbox = self._get_text_bbox(draw, position, text, font)
            
            if text_bbox:
                padding = 5
                # Dessiner le fond
                draw.rectangle([
                    text_bbox[0] - padding, text_bbox[1] - padding,
                    text_bbox[2] + padding, text_bbox[3] + padding
                ], fill=bg_color, outline=bg_color)
            
            # Dessiner le texte
            draw.text(position, text, fill=text_color, font=font)
            
        except Exception as e:
            logger.error(f"Label drawing failed: {e}")
    
    def _get_text_bbox(self, draw, position, text, font):
        """Obtient la bounding box d'un texte"""
        try:
            # Méthode moderne si disponible
            if hasattr(draw, 'textbbox'):
                return draw.textbbox(position, text, font=font)
            else:
                # Fallback pour anciennes versions de Pillow
                text_size = draw.textsize(text, font=font)
                x, y = position
                return (x, y, x + text_size[0], y + text_size[1])
        except:
            # Estimation approximative en dernier recours
            x, y = position
            estimated_width = len(text) * 8
            estimated_height = 16
            return (x, y, x + estimated_width, y + estimated_height)
    
    def _save_annotated_image(self, image, prefix="analysis"):
        """Sauvegarde l'image annotée"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_annotated_{timestamp}.jpg"
            file_path = self.results_dir / filename
            
            # Sauvegarder avec haute qualité
            image.save(file_path, 'JPEG', quality=95, optimize=True)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Image saving failed: {e}")
            return None
    
    def create_comparison_view(self, image_paths, analysis_results_list, person_ids=None):
        """Crée une vue de comparaison de plusieurs analyses"""
        try:
            if not image_paths or len(image_paths) != len(analysis_results_list):
                raise ValueError("Mismatch between images and analysis results")
            
            # Charger toutes les images
            images = []
            max_width = 0
            total_height = 0
            
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                # Redimensionner si nécessaire
                if img.width > 800:
                    ratio = 800 / img.width
                    new_size = (800, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                images.append(img)
                max_width = max(max_width, img.width)
                total_height += img.height + 50  # 50px d'espace entre images
            
            # Créer l'image composite
            composite_width = max_width + 40  # Marges
            composite_height = total_height + 100  # Espace pour titre et résumé
            
            composite = Image.new('RGB', (composite_width, composite_height), 
                                 color=self.color_scheme['background'])
            draw = ImageDraw.Draw(composite)
            
            # Titre de comparaison
            title = f"COMPARISON VIEW - {len(images)} ANALYSES"
            self._draw_label_with_background(draw, (20, 20), title,
                                           self.color_scheme['accent'], 'white', self.fonts['large'])
            
            # Ajouter chaque image avec ses annotations
            current_y = 80
            for i, (img, analysis_results) in enumerate(zip(images, analysis_results_list)):
                # Annoter l'image
                img_annotated = img.copy()
                img_draw = ImageDraw.Draw(img_annotated)
                
                # Ajouter les annotations
                for result in analysis_results:
                    self._annotate_eye_region(img_draw, result, i)
                
                # Ajouter à l'image composite
                composite.paste(img_annotated, (20, current_y))
                
                # Ajouter un résumé à droite de l'image
                summary_x = img.width + 40
                summary_y = current_y + 20
                
                positive_count = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
                summary_text = f"Analysis {i+1}: {positive_count} positive detection(s)"
                
                if person_ids and i < len(person_ids):
                    summary_text += f"\nPerson: {person_ids[i]}"
                
                draw.text((summary_x, summary_y), summary_text, 
                         fill=self.color_scheme['text'], font=self.fonts['small'])
                
                current_y += img.height + 50
            
            # Sauvegarder la vue de comparaison
            comparison_path = self._save_annotated_image(composite, "comparison")
            
            logger.info(f"Comparison view created: {comparison_path}")
            return comparison_path, composite
            
        except Exception as e:
            logger.error(f"Comparison view creation failed: {e}")
            return None, None
    
    def create_progress_visualization(self, person_id, history_data):
        """Crée une visualisation de progression pour un individu"""
        try:
            # Créer un graphique de progression
            fig_width, fig_height = 800, 600
            
            # Créer l'image de base
            progress_img = Image.new('RGB', (fig_width, fig_height), 
                                   color=self.color_scheme['background'])
            draw = ImageDraw.Draw(progress_img)
            
            # Titre
            title = f"MEDICAL PROGRESS - {person_id}"
            self._draw_label_with_background(draw, (50, 30), title,
                                           self.color_scheme['accent'], 'white', self.fonts['large'])
            
            # Dessiner le graphique simple
            if history_data and len(history_data) > 1:
                self._draw_simple_progress_chart(draw, fig_width, fig_height, history_data)
            else:
                # Message si pas assez de données
                no_data_msg = "Insufficient data for progress visualization"
                draw.text((50, 150), no_data_msg, 
                         fill=self.color_scheme['text'], font=self.fonts['medium'])
            
            # Sauvegarder la visualisation de progression
            progress_path = self._save_annotated_image(progress_img, f"progress_{person_id}")
            
            logger.info(f"Progress visualization created: {progress_path}")
            return progress_path, progress_img
            
        except Exception as e:
            logger.error(f"Progress visualization creation failed: {e}")
            return None, None
    
    def _draw_simple_progress_chart(self, draw, width, height, history_data):
        """Dessine un graphique de progression simple"""
        try:
            # Zone de graphique
            chart_x = 80
            chart_y = 100
            chart_width = width - 160
            chart_height = height - 200
            
            # Fond du graphique
            draw.rectangle([chart_x, chart_y, chart_x + chart_width, chart_y + chart_height],
                          outline=self.color_scheme['text'], width=2)
            
            # Extraire les données de confiance
            dates = []
            confidences = []
            
            for entry in history_data[-10:]:  # 10 dernières entrées
                if 'analysis' in entry:
                    analysis = entry['analysis']
                    avg_confidence = analysis.get('average_confidence', 0)
                    confidences.append(avg_confidence)
                    
                    # Date approximative
                    date_str = entry.get('datetime', '')[:10]  # YYYY-MM-DD
                    dates.append(date_str)
            
            if len(confidences) < 2:
                return
            
            # Dessiner les points et lignes
            max_confidence = max(confidences) if confidences else 100
            min_confidence = min(confidences) if confidences else 0
            confidence_range = max_confidence - min_confidence or 1
            
            points = []
            for i, confidence in enumerate(confidences):
                x = chart_x + (i * chart_width // (len(confidences) - 1))
                y = chart_y + chart_height - ((confidence - min_confidence) / confidence_range * chart_height)
                points.append((x, y))
            
            # Dessiner les lignes de connexion
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=self.color_scheme['accent'], width=2)
            
            # Dessiner les points
            for i, (x, y) in enumerate(points):
                radius = 4
                draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                           fill=self.color_scheme['high_risk'] if confidences[i] > 70 else self.color_scheme['normal'])
                
                # Label de confiance
                conf_label = f"{confidences[i]:.0f}%"
                draw.text((x - 15, y - 20), conf_label, 
                         fill=self.color_scheme['text'], font=self.fonts['tiny'])
            
            # Labels des axes
            draw.text((chart_x, chart_y + chart_height + 10), "Time →", 
                     fill=self.color_scheme['text'], font=self.fonts['small'])
            
            draw.text((10, chart_y + chart_height // 2), "Confidence", 
                     fill=self.color_scheme['text'], font=self.fonts['small'])
            
        except Exception as e:
            logger.error(f"Progress chart drawing failed: {e}")
    
    def enhance_image_for_analysis(self, image_path, enhancement_level='medium'):
        """Améliore une image pour l'analyse"""
        try:
            image = Image.open(image_path)
            
            # Paramètres d'amélioration
            enhancements = {
                'low': {'contrast': 1.1, 'brightness': 1.0, 'sharpness': 1.1},
                'medium': {'contrast': 1.2, 'brightness': 1.05, 'sharpness': 1.2},
                'high': {'contrast': 1.3, 'brightness': 1.1, 'sharpness': 1.3}
            }
            
            params = enhancements.get(enhancement_level, enhancements['medium'])
            
            # Appliquer les améliorations
            if params['contrast'] != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(params['contrast'])
            
            if params['brightness'] != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(params['brightness'])
            
            if params['sharpness'] != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(params['sharpness'])
            
            # Sauvegarder l'image améliorée
            enhanced_path = self._save_annotated_image(image, "enhanced")
            
            logger.info(f"Image enhanced and saved: {enhanced_path}")
            return enhanced_path, image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return None, None
    
    def create_medical_summary_card(self, analysis_results, person_id=None, history_summary=None):
        """Crée une carte de résumé médical"""
        try:
            card_width, card_height = 600, 400
            card = Image.new('RGB', (card_width, card_height), color='white')
            draw = ImageDraw.Draw(card)
            
            # Bordure
            draw.rectangle([0, 0, card_width - 1, card_height - 1], 
                          outline=self.color_scheme['accent'], width=3)
            
            # En-tête
            header_height = 60
            draw.rectangle([0, 0, card_width, header_height], 
                          fill=self.color_scheme['accent'])
            
            title = "MEDICAL SUMMARY CARD"
            draw.text((20, 20), title, fill='white', font=self.fonts['large'])
            
            # Contenu principal
            y_offset = header_height + 20
            
            # Statistiques de base
            total_eyes = len(analysis_results)
            positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
            
            stats_text = f"Eyes Analyzed: {total_eyes}\nPositive Detections: {positive_detections}"
            draw.text((20, y_offset), stats_text, fill=self.color_scheme['text'], font=self.fonts['medium'])
            
            y_offset += 60
            
            # Informations de tracking
            if person_id:
                tracking_text = f"Person ID: {person_id}"
                if history_summary:
                    tracking_text += f"\nTotal Encounters: {history_summary.get('encounter_count', 0)}"
                    tracking_text += f"\nLast Seen: {history_summary.get('last_seen', 'Unknown')[:10]}"
                
                draw.text((20, y_offset), tracking_text, fill=self.color_scheme['text'], font=self.fonts['small'])
            
            # Recommandation
            y_offset = card_height - 80
            if positive_detections > 0:
                recommendation = "⚠️ IMMEDIATE MEDICAL CONSULTATION REQUIRED"
                rec_color = self.color_scheme['high_risk']
            else:
                recommendation = "✅ Continue regular monitoring"
                rec_color = self.color_scheme['normal']
            
            draw.text((20, y_offset), recommendation, fill=rec_color, font=self.fonts['medium'])
            
            # Sauvegarder la carte
            card_path = self._save_annotated_image(card, "summary_card")
            
            logger.info(f"Medical summary card created: {card_path}")
            return card_path, card
            
        except Exception as e:
            logger.error(f"Medical summary card creation failed: {e}")
            return None, None