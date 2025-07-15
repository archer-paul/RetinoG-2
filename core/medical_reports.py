"""
Medical Reports Generator - Module de génération de rapports médicaux
Génère des rapports HTML professionnels pour RetinoblastoGemma v6
"""
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class MedicalReportsGenerator:
    """Générateur de rapports médicaux professionnels"""
    
    def __init__(self):
        self.report_templates = {
            'standard': 'Rapport médical standard',
            'emergency': 'Rapport d\'urgence avec alertes',
            'follow_up': 'Rapport de suivi patient',
            'comprehensive': 'Rapport complet avec historique'
        }
        
        # Styles CSS pour les rapports
        self.css_styles = self._load_css_styles()
        
        logger.info("Medical reports generator initialized")
    
    def _load_css_styles(self) -> str:
        """Charge les styles CSS pour les rapports"""
        return """
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 40px; line-height: 1.6; color: #2c3e50;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; 
            background: white; border-radius: 15px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; text-align: center;
            position: relative;
        }
        .header::before {
            content: '🏥'; font-size: 60px; 
            position: absolute; top: 20px; left: 40px;
            opacity: 0.3;
        }
        .header h1 { margin: 0; font-size: 32px; font-weight: 300; }
        .header .subtitle { font-size: 18px; opacity: 0.9; margin-top: 10px; }
        .badges { margin-top: 20px; }
        .badge { 
            display: inline-block; padding: 8px 16px; margin: 5px;
            border-radius: 25px; color: white; font-weight: bold; font-size: 12px;
        }
        .badge-hackathon { background: linear-gradient(45deg, #ff6b6b, #ffa500); }
        .badge-local { background: linear-gradient(45deg, #4299e1, #0066cc); }
        .badge-secure { background: linear-gradient(45deg, #48bb78, #38a169); }
        .content { padding: 40px; }
        .alert-critical { 
            background: linear-gradient(135deg, #ff6b6b, #ff5722);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 15px; text-align: center;
            box-shadow: 0 10px 25px rgba(255,107,107,0.3);
            animation: pulse 2s infinite;
        }
        .alert-safe { 
            background: linear-gradient(135deg, #51cf66, #40c057);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 15px; text-align: center;
            box-shadow: 0 10px 25px rgba(81,207,102,0.3);
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .patient-section {
            background: #f8f9fa; padding: 25px; 
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .patient-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 15px; margin-top: 15px;
        }
        .results-section { 
            background: #f8f9fa; padding: 30px; 
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .disclaimer { 
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 2px solid #ffc107;
            padding: 25px; border-radius: 15px; margin: 30px 0;
            box-shadow: 0 5px 15px rgba(255,193,7,0.2);
        }
        .footer { 
            background: linear-gradient(135deg, #2d3748, #1a202c);
            color: white; padding: 30px; text-align: center;
        }
        .tech-details {
            background: #e8f4fd; padding: 20px;
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #4299e1;
        }
        .emergency-actions {
            background: rgba(255,255,255,0.2); 
            padding: 20px; border-radius: 10px; 
            margin: 20px 0; font-weight: bold;
        }
        pre { 
            background: #2d3748; color: #e2e8f0;
            padding: 25px; border-radius: 10px; 
            overflow-x: auto; font-size: 14px;
            line-height: 1.4; white-space: pre-wrap;
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 20px 0;
        }
        .stat-card {
            background: white; padding: 20px; border-radius: 10px;
            text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-number { font-size: 24px; font-weight: bold; color: #667eea; }
        .qr-section {
            text-align: center; margin: 30px 0;
            padding: 20px; background: #f8f9fa; border-radius: 10px;
        }
        .timeline {
            position: relative; padding-left: 30px;
        }
        .timeline::before {
            content: ''; position: absolute; left: 15px; top: 0; bottom: 0;
            width: 2px; background: #667eea;
        }
        .timeline-item {
            position: relative; margin-bottom: 20px; padding: 15px;
            background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .timeline-item::before {
            content: ''; position: absolute; left: -37px; top: 20px;
            width: 12px; height: 12px; border-radius: 50%;
            background: #667eea; border: 3px solid white;
        }
        .comparison-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; margin: 20px 0;
        }
        .recommendation-box {
            padding: 20px; margin: 15px 0; border-radius: 10px;
            border-left: 5px solid;
        }
        .recommendation-immediate { 
            background: #ffebee; border-color: #f44336;
        }
        .recommendation-urgent { 
            background: #fff3e0; border-color: #ff9800;
        }
        .recommendation-routine { 
            background: #e8f5e8; border-color: #4caf50;
        }
        @media print {
            body { background: white !important; }
            .container { box-shadow: none !important; }
            .alert-critical { animation: none !important; }
        }
        """
    
    def generate_comprehensive_report(
        self, 
        analysis_results: str,
        image_path: str,
        patient_summary: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        face_tracking_results: Optional[Dict] = None
    ) -> str:
        """Génère un rapport médical complet"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(image_path).name if image_path else 'Unknown'
        
        # Déterminer s'il y a des détections positives
        has_positive = "MEDICAL ALERT" in analysis_results if analysis_results else False
        
        # Générer les sections du rapport
        header_section = self._generate_header_section(timestamp, filename)
        alert_section = self._generate_alert_section(has_positive)
        patient_section = self._generate_patient_section(patient_summary)
        metrics_section = self._generate_metrics_section(metrics)
        results_section = self._generate_results_section(analysis_results)
        tech_section = self._generate_technical_section(face_tracking_results)
        recommendations_section = self._generate_recommendations_section(has_positive, patient_summary)
        disclaimer_section = self._generate_disclaimer_section(has_positive)
        footer_section = self._generate_footer_section(timestamp)
        
        # Assembler le rapport complet
        html_report = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Médical Retinoblastoma - {timestamp}</title>
    <style>{self.css_styles}</style>
</head>
<body>
    <div class="container">
        {header_section}
        <div class="content">
            {alert_section}
            {patient_section}
            {metrics_section}
            {results_section}
            {tech_section}
            {recommendations_section}
            {disclaimer_section}
        </div>
        {footer_section}
    </div>
</body>
</html>"""
        
        return html_report
    
    def _generate_header_section(self, timestamp: str, filename: str) -> str:
        """Génère la section d'en-tête"""
        return f"""
        <div class="header">
            <h1>🏥 Rapport Médical d'Analyse Retinoblastoma</h1>
            <div class="subtitle">Système de Détection Précoce par Intelligence Artificielle</div>
            <div class="badges">
                <span class="badge badge-hackathon">🏆 HACKATHON GOOGLE GEMMA</span>
                <span class="badge badge-local">100% TRAITEMENT LOCAL</span>
                <span class="badge badge-secure">VIE PRIVÉE GARANTIE</span>
            </div>
            <p style="margin-top: 20px;"><strong>Généré le:</strong> {timestamp}</p>
            <p><strong>Image analysée:</strong> {filename}</p>
            <p><strong>Système:</strong> Gemma 3n Multimodal (Traitement Local)</p>
        </div>
        """
    
    def _generate_alert_section(self, has_positive: bool) -> str:
        """Génère la section d'alerte"""
        if has_positive:
            return """
            <div class="alert-critical">
                <h2>🚨 ALERTE MÉDICALE - ACTION IMMÉDIATE REQUISE</h2>
                <p style="font-size: 20px; font-weight: bold; margin: 15px 0;">
                    Possible rétinoblastome détecté. Contactez un ophtalmologiste pédiatrique IMMÉDIATEMENT.
                </p>
                <div class="emergency-actions">
                    <h3>🚨 PROTOCOLE D'URGENCE:</h3>
                    <p>1. 📞 Appelez un ophtalmologiste pédiatrique AUJOURD'HUI</p>
                    <p>2. 📋 Imprimez ce rapport et apportez-le au rendez-vous</p>
                    <p>3. 📸 Apportez les images originales sur téléphone/appareil</p>
                    <p>4. 🏥 Rendez-vous aux urgences si impossible de joindre un spécialiste</p>
                </div>
            </div>
            """
        else:
            return """
            <div class="alert-safe">
                <h2>✅ Aucun Signe Préoccupant Détecté</h2>
                <p style="font-size: 18px; margin: 15px 0;">
                    L'analyse par IA n'a pas détecté de signes de leucocorie dans cette image.
                    Continuez la surveillance oculaire pédiatrique régulière comme recommandé.
                </p>
            </div>
            """
    
    def _generate_patient_section(self, patient_summary: Optional[Dict]) -> str:
        """Génère la section patient"""
        if not patient_summary:
            return """
            <div class="patient-section">
                <h3>👤 Informations Patient</h3>
                <p>Aucun suivi patient actif. Le suivi facial peut être activé pour un monitoring longitudinal.</p>
            </div>
            """
        
        return f"""
        <div class="patient-section">
            <h3>👤 Informations Patient</h3>
            <div class="patient-grid">
                <div><strong>ID Patient:</strong> {patient_summary.get('face_id', 'Inconnu')}</div>
                <div><strong>Première Analyse:</strong> {patient_summary.get('first_seen', 'Inconnu')[:10]}</div>
                <div><strong>Nombre Total d'Analyses:</strong> {patient_summary.get('total_analyses', 0)}</div>
                <div><strong>Analyses Positives:</strong> {patient_summary.get('positive_analyses', 0)}</div>
                <div><strong>Dernière Visite:</strong> {patient_summary.get('last_seen', 'Inconnu')[:10]}</div>
                <div><strong>Taux de Cohérence:</strong> {patient_summary.get('consistency_rate', 0):.1f}%</div>
            </div>
            
            <div class="recommendation-box recommendation-{self._get_urgency_class(patient_summary.get('urgency', 'routine'))}">
                <h4>📋 Recommandation Médicale</h4>
                <p><strong>{patient_summary.get('recommendation', 'Aucune recommandation disponible')}</strong></p>
                <p><em>Niveau d'urgence: {patient_summary.get('urgency', 'routine').upper()}</em></p>
            </div>
        </div>
        """
    
    def _get_urgency_class(self, urgency: str) -> str:
        """Retourne la classe CSS pour l'urgence"""
        urgency_map = {
            'immediate': 'immediate',
            'urgent': 'urgent', 
            'soon': 'urgent',
            'routine': 'routine'
        }
        return urgency_map.get(urgency.lower(), 'routine')
    
    def _generate_metrics_section(self, metrics: Optional[Dict]) -> str:
        """Génère la section des métriques"""
        if not metrics:
            return ""
        
        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{metrics.get('modules_ready', 0)}/4</div>
                <div>Modules Actifs</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{metrics.get('total_analyses', 0)}</div>
                <div>Analyses Totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">100%</div>
                <div>Traitement Local</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">95%</div>
                <div>Taux de Survie*</div>
            </div>
        </div>
        """
    
    def _generate_results_section(self, analysis_results: str) -> str:
        """Génère la section des résultats"""
        return f"""
        <div class="results-section">
            <h2>📊 Résultats Détaillés de l'Analyse</h2>
            <pre>{analysis_results if analysis_results else 'Aucun résultat détaillé disponible'}</pre>
        </div>
        """
    
    def _generate_technical_section(self, face_tracking_results: Optional[Dict]) -> str:
        """Génère la section technique"""
        face_tracking_info = ""
        if face_tracking_results:
            tracked_faces = face_tracking_results.get('tracked_faces', 0)
            new_faces = face_tracking_results.get('new_faces', 0)
            recognized_faces = face_tracking_results.get('recognized_faces', 0)
            
            face_tracking_info = f"""
            <p><strong>Suivi Facial:</strong> {tracked_faces} visages traités</p>
            <p><strong>Nouveaux Visages:</strong> {new_faces}</p>
            <p><strong>Visages Reconnus:</strong> {recognized_faces}</p>
            """
        
        return f"""
        <div class="tech-details">
            <h3>🤖 Détails Techniques de l'Analyse</h3>
            <p><strong>Modèle IA:</strong> Gemma 3n Multimodal (Exécution Locale)</p>
            <p><strong>Traitement:</strong> 100% Hors ligne - Aucune donnée transmise</p>
            <p><strong>Vie Privée:</strong> Complète - Tout le traitement effectué sur votre appareil</p>
            <p><strong>Modules Utilisés:</strong> Détection Oculaire, Suivi Facial, Analyse IA, Visualisation</p>
            <p><strong>Méthode d'Analyse:</strong> Vision par Ordinateur + Modèle de Langage Large</p>
            {face_tracking_info}
        </div>
        """
    
    def _generate_recommendations_section(self, has_positive: bool, patient_summary: Optional[Dict]) -> str:
        """Génère la section des recommandations"""
        if has_positive:
            return """
            <div class="recommendation-box recommendation-immediate">
                <h3>🚨 RECOMMANDATIONS MÉDICALES URGENTES</h3>
                <h4>ACTION IMMÉDIATE REQUISE</h4>
                <ul>
                    <li><strong>📞 Urgence:</strong> Contactez un ophtalmologiste pédiatrique AUJOURD'HUI</li>
                    <li><strong>📋 Documentation:</strong> Apportez ce rapport et les images originales</li>
                    <li><strong>🚫 Ne pas attendre:</strong> Ne retardez pas l'évaluation médicale professionnelle</li>
                    <li><strong>🏥 Urgences:</strong> Rendez-vous aux urgences si impossible de joindre un spécialiste</li>
                </ul>
                
                <h4>⚠️ Pourquoi c'est urgent:</h4>
                <ul>
                    <li>Le rétinoblastome nécessite une attention médicale immédiate</li>
                    <li>La détection précoce peut sauver la vue et la vie</li>
                    <li>95% de taux de survie avec détection et traitement précoces</li>
                    <li>Chaque jour compte pour le pronostic</li>
                </ul>
            </div>
            """
        else:
            return """
            <div class="recommendation-box recommendation-routine">
                <h3>✅ RECOMMANDATIONS DE SURVEILLANCE ROUTINE</h3>
                <ul>
                    <li><strong>📅 Surveillance:</strong> Continuez les examens oculaires pédiatriques réguliers</li>
                    <li><strong>📸 Photos:</strong> Prenez des photos mensuelles sous bon éclairage</li>
                    <li><strong>👀 Observation:</strong> Surveillez tout changement dans l'apparence des pupilles</li>
                    <li><strong>🔄 Dépistage:</strong> Répétez le dépistage IA si des préoccupations surviennent</li>
                </ul>
                
                <h4>💡 Conseils de surveillance:</h4>
                <ul>
                    <li>La surveillance régulière est la clé de la détection précoce</li>
                    <li>Consultez un ophtalmologiste pédiatrique si des inquiétudes surviennent</li>
                    <li>Maintenez un dossier photographique pour le suivi longitudinal</li>
                </ul>
            </div>
            """
    
    def _generate_disclaimer_section(self, has_positive: bool) -> str:
        """Génère la section de disclaimer médical"""
        return f"""
        <div class="disclaimer">
            <h3>⚕️ Avertissement Médical Critique</h3>
            <p><strong>IMPORTANT:</strong> Ce rapport est généré par un système de dépistage IA utilisant Gemma 3n.</p>
            <p><strong>CECI N'EST PAS UN DIAGNOSTIC MÉDICAL</strong> et ne doit PAS remplacer une évaluation médicale professionnelle.</p>
            
            <h4>📋 Prochaines Étapes:</h4>
            <ul>
                <li><strong>Évaluation Professionnelle:</strong> Programmez une consultation avec un ophtalmologiste pédiatrique</li>
                <li><strong>Documentation:</strong> Apportez ce rapport et les images originales au rendez-vous</li>
                <li><strong>Urgence:</strong> {'Évaluation IMMÉDIATE requise' if has_positive else 'Suivi de routine approprié'}</li>
                <li><strong>Surveillance:</strong> Continuez la surveillance régulière de la santé oculaire</li>
            </ul>
            
            <h4>🏥 À Propos du Rétinoblastome:</h4>
            <ul>
                <li>Cancer de l'œil le plus fréquent chez les enfants (typiquement moins de 6 ans)</li>
                <li><strong>95% de taux de survie avec détection et traitement précoces*</strong></li>
                <li>Peut affecter un œil ou les deux yeux</li>
                <li>Signe précoce: Reflet pupillaire blanc (leucocorie) dans les photos</li>
                <li>Nécessite une attention médicale immédiate en cas de suspicion</li>
            </ul>
            <p style="font-size: 12px; margin-top: 10px;">*Avec détection précoce et traitement approprié</p>
        </div>
        """
    
    def _generate_footer_section(self, timestamp: str) -> str:
        """Génère la section de pied de page"""
        report_id = timestamp.replace('-', '').replace(':', '').replace(' ', '_')
        
        return f"""
        <div class="footer">
            <p><strong>Généré par RetinoblastoGemma v6</strong></p>
            <p>🏆 Participation au Hackathon Google Gemma Mondial</p>
            <p>🤖 Dépistage du Rétinoblastome Alimenté par IA avec Gemma 3n Local</p>
            <p>🔒 100% Traitement Local - Vie Privée Garantie</p>
            <div class="qr-section">
                <h4>📱 Partager ce Rapport</h4>
                <p>Ce rapport peut être sauvegardé, imprimé ou partagé avec des professionnels de santé.</p>
                <p><strong>ID du Rapport:</strong> RBG_{report_id}</p>
            </div>
            <p style="font-size: 12px; margin-top: 15px; opacity: 0.8;">
                Système: Gemma 3n Local | Hackathon: Google Gemma Mondial | 
                Générateur: RetinoblastoGemma v6
            </p>
        </div>
        """
    
    def generate_follow_up_report(
        self, 
        patient_history: List[Dict],
        patient_id: str,
        current_analysis: str
    ) -> str:
        """Génère un rapport de suivi avec historique patient"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Générer la timeline des analyses
        timeline_section = self._generate_timeline_section(patient_history)
        
        # Analyser les tendances
        trends_section = self._generate_trends_section(patient_history)
        
        html_report = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Suivi Patient - {patient_id}</title>
    <style>{self.css_styles}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Rapport de Suivi Patient</h1>
            <div class="subtitle">Analyse Longitudinale - RetinoblastoGemma v6</div>
            <p><strong>Patient ID:</strong> {patient_id}</p>
            <p><strong>Généré le:</strong> {timestamp}</p>
        </div>
        
        <div class="content">
            <div class="patient-section">
                <h3>📈 Évolution Temporelle</h3>
                {timeline_section}
            </div>
            
            <div class="results-section">
                <h3>📊 Analyse des Tendances</h3>
                {trends_section}
            </div>
            
            <div class="results-section">
                <h3>🔍 Analyse Actuelle</h3>
                <pre>{current_analysis}</pre>
            </div>
            
            {self._generate_disclaimer_section(False)}
        </div>
        
        {self._generate_footer_section(timestamp)}
    </div>
</body>
</html>"""
        
        return html_report
    
    def _generate_timeline_section(self, patient_history: List[Dict]) -> str:
        """Génère la section timeline"""
        if not patient_history:
            return "<p>Aucun historique disponible.</p>"
        
        timeline_html = '<div class="timeline">'
        
        for entry in patient_history[-10:]:  # 10 dernières analyses
            timestamp = entry.get('timestamp', 'Inconnu')
            has_positive = entry.get('has_positive_findings', False)
            analysis_summary = entry.get('analysis_summary', {})
            
            status_icon = '🚨' if has_positive else '✅'
            status_text = 'POSITIF' if has_positive else 'NÉGATIF'
            
            timeline_html += f"""
            <div class="timeline-item">
                <h4>{status_icon} Analyse du {timestamp[:10]}</h4>
                <p><strong>Résultat:</strong> {status_text}</p>
                <p><strong>Régions analysées:</strong> {analysis_summary.get('regions_analyzed', 0)}</p>
                <p><strong>Méthode:</strong> {analysis_summary.get('method', 'Inconnue')}</p>
            </div>
            """
        
        timeline_html += '</div>'
        return timeline_html
    
    def _generate_trends_section(self, patient_history: List[Dict]) -> str:
        """Génère la section d'analyse des tendances"""
        if len(patient_history) < 2:
            return "<p>Historique insuffisant pour l'analyse des tendances.</p>"
        
        # Analyser les tendances
        positive_count = sum(1 for entry in patient_history if entry.get('has_positive_findings', False))
        total_analyses = len(patient_history)
        consistency_rate = (positive_count / total_analyses) * 100
        
        # Tendance récente
        recent_analyses = patient_history[-5:]
        recent_positive = sum(1 for entry in recent_analyses if entry.get('has_positive_findings', False))
        
        if recent_positive > 0:
            trend_assessment = "🚨 PRÉOCCUPANT - Détections positives récentes"
            trend_class = "recommendation-immediate"
        elif consistency_rate > 20:
            trend_assessment = "⚠️ À SURVEILLER - Taux de détection élevé"
            trend_class = "recommendation-urgent"
        else:
            trend_assessment = "✅ STABLE - Pas de préoccupation majeure"
            trend_class = "recommendation-routine"
        
        return f"""
        <div class="comparison-grid">
            <div>
                <h4>📊 Statistiques Globales</h4>
                <p><strong>Analyses totales:</strong> {total_analyses}</p>
                <p><strong>Détections positives:</strong> {positive_count}</p>
                <p><strong>Taux de cohérence:</strong> {consistency_rate:.1f}%</p>
            </div>
            <div>
                <h4>📈 Tendance Récente</h4>
                <p><strong>5 dernières analyses:</strong> {len(recent_analyses)}</p>
                <p><strong>Positives récentes:</strong> {recent_positive}</p>
                <p><strong>Évaluation:</strong> {trend_assessment}</p>
            </div>
        </div>
        
        <div class="recommendation-box {trend_class}">
            <h4>🎯 Recommandation Basée sur l'Historique</h4>
            <p>{self._get_trend_recommendation(consistency_rate, recent_positive)}</p>
        </div>
        """
    
    def _get_trend_recommendation(self, consistency_rate: float, recent_positive: int) -> str:
        """Génère une recommandation basée sur les tendances"""
        if recent_positive >= 2:
            return ("URGENCE: Plusieurs détections positives récentes. Consultation ophtalmologique "
                   "pédiatrique IMMÉDIATE requise. Ne pas attendre.")
        elif recent_positive == 1 and consistency_rate > 30:
            return ("PRIORITÉ ÉLEVÉE: Pattern préoccupant détecté. Programmez une consultation "
                   "dans les 1-2 semaines.")
        elif consistency_rate > 20:
            return ("SURVEILLANCE RENFORCÉE: Taux de détection élevé. Consultation recommandée "
                   "dans le mois. Surveillez attentivement.")
        else:
            return ("SURVEILLANCE ROUTINE: Tendances stables. Continuez la surveillance régulière "
                   "et les photos mensuelles.")
    
    def save_report(self, html_content: str, report_type: str = "medical") -> Optional[str]:
        """Sauvegarde un rapport HTML"""
        try:
            # Créer le dossier de résultats
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retinoblastoma_{report_type}_report_{timestamp}.html"
            filepath = results_dir / filename
            
            # Sauvegarder le fichier
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Medical report saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return None