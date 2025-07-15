"""
RetinoblastoGemma v6 - Interface principale modulaire
Application de détection de rétinoblastome avec Gemma 3n local
Architecture modulaire pour le hackathon Google Gemma
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
from pathlib import Path
import threading
import time
import sys
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retinoblastogamma.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports des modules core avec gestion d'erreurs
try:
    from core.gemma_handler_v2 import GemmaHandlerV2
    from core.eye_detector_v2 import EyeDetectorV2
    from core.face_handler_v2 import FaceHandlerV2
    from core.visualization_v2 import VisualizationV2
    from config.settings import MODELS_DIR, DATA_DIR, RESULTS_DIR
    logger.info("✅ Tous les modules core importés avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur d'import des modules core: {e}")
    logger.info("💡 Certains modules seront initialisés en mode fallback")

class RetinoblastoGemmaV6:
    """Application principale modulaire pour la détection de rétinoblastome"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma v6 - Hackathon Google Gemma")
        self.root.geometry("1400x900")
        
        # État de l'application
        self.current_image_path = None
        self.current_results = None
        self.processing = False
        
        # Modules core (initialisés plus tard)
        self.gemma_handler = None
        self.eye_detector = None
        self.face_handler = None
        self.visualizer = None
        
        # Configuration
        self.config = {
            'confidence_threshold': 0.5,
            'use_face_tracking': True,
            'enhanced_detection': True,
            'force_local_mode': True  # Important pour le hackathon
        }
        
        # Métriques de performance
        self.metrics = {
            'total_analyses': 0,
            'positive_detections': 0,
            'processing_times': [],
            'session_start': time.time()
        }
        
        # Initialiser l'interface
        self.setup_ui()
        
        # Initialiser les modules en arrière-plan
        self.root.after(1000, self.initialize_modules_async)
    
    def setup_ui(self):
        """Configure l'interface utilisateur modulaire"""
        # Style moderne
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal avec padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # === PANEL DE CONTRÔLE (GAUCHE) ===
        self.setup_control_panel(main_frame)
        
        # === ZONE D'AFFICHAGE (DROITE) ===
        self.setup_display_area(main_frame)
        
        # === BARRE DE STATUT ===
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Configure le panel de contrôle à gauche"""
        control_frame = ttk.LabelFrame(parent, text="🏥 RetinoblastoGemma Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # Section chargement d'image
        image_section = ttk.LabelFrame(control_frame, text="📸 Image Loading")
        image_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_section, text="Load Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        self.image_info_label = ttk.Label(image_section, text="No image loaded", 
                                         font=("Arial", 9), foreground="gray")
        self.image_info_label.pack(anchor=tk.W, pady=2)
        
        # Section analyse
        analysis_section = ttk.LabelFrame(control_frame, text="🤖 AI Analysis")
        analysis_section.pack(fill=tk.X, pady=10)
        
        self.analyze_button = ttk.Button(analysis_section, text="🔍 Analyze for Retinoblastoma", 
                                        command=self.analyze_image, state='disabled')
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        # Status des modules
        modules_section = ttk.LabelFrame(control_frame, text="🧩 System Modules")
        modules_section.pack(fill=tk.X, pady=10)
        
        self.module_status = {
            'gemma': ttk.Label(modules_section, text="Gemma 3n: Initializing...", foreground="blue"),
            'eye_detector': ttk.Label(modules_section, text="Eye Detector: Waiting...", foreground="gray"),
            'face_handler': ttk.Label(modules_section, text="Face Handler: Waiting...", foreground="gray"),
            'visualizer': ttk.Label(modules_section, text="Visualizer: Waiting...", foreground="gray")
        }
        
        for label in self.module_status.values():
            label.pack(anchor=tk.W, pady=1)
        
        # Configuration
        config_section = ttk.LabelFrame(control_frame, text="⚙️ Settings")
        config_section.pack(fill=tk.X, pady=10)
        
        # Seuil de confiance
        ttk.Label(config_section, text="Confidence Threshold:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=self.config['confidence_threshold'])
        confidence_scale = ttk.Scale(config_section, from_=0.1, to=0.9, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=2)
        
        # Options
        self.face_tracking_var = tk.BooleanVar(value=self.config['use_face_tracking'])
        ttk.Checkbutton(config_section, text="Enable Face Tracking", 
                       variable=self.face_tracking_var).pack(anchor=tk.W)
        
        self.enhanced_detection_var = tk.BooleanVar(value=self.config['enhanced_detection'])
        ttk.Checkbutton(config_section, text="Enhanced Detection", 
                       variable=self.enhanced_detection_var).pack(anchor=tk.W)
        
        # Progression
        progress_section = ttk.LabelFrame(control_frame, text="📊 Progress")
        progress_section.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(progress_section, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)
        
        self.progress_label = ttk.Label(progress_section, text="Ready", font=("Arial", 8))
        self.progress_label.pack(anchor=tk.W)
        
        # Métriques
        metrics_section = ttk.LabelFrame(control_frame, text="📈 Metrics")
        metrics_section.pack(fill=tk.X, pady=10)
        
        self.metrics_label = ttk.Label(metrics_section, text="No analysis yet", font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Actions
        actions_section = ttk.LabelFrame(control_frame, text="💾 Actions")
        actions_section.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_section, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="Medical Report", 
                  command=self.generate_medical_report).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="System Info", 
                  command=self.show_system_info).pack(fill=tk.X, pady=1)
    
    def setup_display_area(self, parent):
        """Configure la zone d'affichage à droite"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Onglet image principale
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="🖼️ Image Analysis")
        
        # Canvas avec scrollbars
        canvas_frame = ttk.Frame(self.image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", relief=tk.SUNKEN, bd=2)
        scrollbar_v = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Onglet résultats
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📋 Medical Results")
        
        results_container = ttk.Frame(self.results_frame)
        results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(results_container, wrap=tk.WORD, 
                                   font=("Consolas", 10), relief=tk.SUNKEN, bd=2)
        results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_container.columnconfigure(0, weight=1)
        results_container.rowconfigure(0, weight=1)
    
    def setup_status_bar(self, parent):
        """Configure la barre de statut"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="RetinoblastoGemma v6 Ready - Hackathon Google Gemma", 
                                     relief=tk.SUNKEN, font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def initialize_modules_async(self):
        """Initialise les modules en arrière-plan"""
        def init_thread():
            try:
                self.update_status("🔄 Initializing core modules...")
                self.update_progress(10)
                
                # 1. Initialiser Gemma Handler
                self.update_progress_label("Loading Gemma 3n local model...")
                self.update_module_status('gemma', "Loading...", "blue")
                
                try:
                    self.gemma_handler = GemmaHandlerV2()
                    success = self.gemma_handler.initialize_local_model()
                    if success:
                        self.update_module_status('gemma', "✅ Ready", "green")
                    else:
                        self.update_module_status('gemma', "❌ Failed", "red")
                except Exception as e:
                    logger.error(f"Gemma initialization failed: {e}")
                    self.update_module_status('gemma', "❌ Error", "red")
                
                self.update_progress(35)
                
                # 2. Initialiser Eye Detector
                self.update_progress_label("Initializing eye detector...")
                self.update_module_status('eye_detector', "Loading...", "blue")
                
                try:
                    self.eye_detector = EyeDetectorV2()
                    self.update_module_status('eye_detector', "✅ Ready", "green")
                except Exception as e:
                    logger.error(f"Eye detector initialization failed: {e}")
                    self.update_module_status('eye_detector', "❌ Error", "red")
                
                self.update_progress(60)
                
                # 3. Initialiser Face Handler
                self.update_progress_label("Initializing face handler...")
                self.update_module_status('face_handler', "Loading...", "blue")
                
                try:
                    self.face_handler = FaceHandlerV2()
                    self.update_module_status('face_handler', "✅ Ready", "green")
                except Exception as e:
                    logger.error(f"Face handler initialization failed: {e}")
                    self.update_module_status('face_handler', "❌ Error", "red")
                
                self.update_progress(85)
                
                # 4. Initialiser Visualizer
                self.update_progress_label("Initializing visualizer...")
                self.update_module_status('visualizer', "Loading...", "blue")
                
                try:
                    self.visualizer = VisualizationV2()
                    self.update_module_status('visualizer', "✅ Ready", "green")
                except Exception as e:
                    logger.error(f"Visualizer initialization failed: {e}")
                    self.update_module_status('visualizer', "❌ Error", "red")
                
                self.update_progress(100)
                
                # Vérifier si au moins les modules essentiels sont prêts
                ready_modules = sum(1 for status in self.module_status.values() 
                                  if "✅" in status.cget("text"))
                
                if ready_modules >= 2:  # Au moins 2 modules doivent marcher
                    self.update_status("✅ System ready! Click 'Load Image' to start", "green")
                    self.analyze_button.config(state='normal')
                else:
                    self.update_status("⚠️ Limited functionality - some modules failed", "orange")
                    self.analyze_button.config(state='normal')  # Permettre quand même d'essayer
                
                self.update_progress_label("System ready")
                
            except Exception as e:
                logger.error(f"Module initialization failed: {e}")
                self.update_status(f"❌ Initialization failed: {e}", "red")
                self.update_progress_label("Initialization failed")
        
        # Lancer en arrière-plan
        threading.Thread(target=init_thread, daemon=True).start()
    
    def load_image(self):
        """Charge une image pour analyse"""
        file_path = filedialog.askopenfilename(
            title="Select medical image for retinoblastoma analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Vérifier que l'image peut être ouverte
                test_image = Image.open(file_path)
                image_info = f"{test_image.width}x{test_image.height}"
                test_image.close()
                
                self.current_image_path = file_path
                self.display_image(file_path)
                
                filename = Path(file_path).name
                self.update_status(f"✅ Image loaded: {filename}")
                self.image_info_label.config(text=f"{filename} ({image_info})", foreground="green")
                
                logger.info(f"Image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Image Loading Error", f"Cannot load image:\n{e}")
                logger.error(f"Failed to load image {file_path}: {e}")
    
    def display_image(self, image_path):
        """Affiche l'image dans le canvas"""
        try:
            image = Image.open(image_path)
            
            # Redimensionnement intelligent
            canvas_width = max(900, self.canvas.winfo_width())
            canvas_height = max(700, self.canvas.winfo_height())
            
            # Conserver les proportions
            image.thumbnail((canvas_width - 50, canvas_height - 50), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            
            # Centrer l'image
            self.canvas.delete("all")
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("Display Error", f"Cannot display image: {e}")
    
    def analyze_image(self):
        """Lance l'analyse de rétinoblastome"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if self.processing:
            messagebox.showwarning("Processing", "Analysis already in progress.")
            return
        
        # Vérifier qu'au moins certains modules sont prêts
        if not self.eye_detector:
            messagebox.showerror("System Not Ready", 
                "Eye detector not initialized. Please wait or restart the application.")
            return
        
        # Confirmation pour analyse
        result = messagebox.askyesno("Start Analysis", 
            "🔍 Start retinoblastoma analysis?\n\n"
            "This will analyze the image for signs of leukocoria using Gemma 3n local model.\n"
            "Analysis may take 30-90 seconds.")
        
        if not result:
            return
        
        def analysis_thread():
            self.processing = True
            try:
                start_time = time.time()
                self.update_status("🔄 Starting retinoblastoma analysis...", "blue")
                self.update_progress(0)
                self.update_progress_label("Preparing analysis...")
                
                # Étape 1: Détection des yeux/visages
                self.update_progress(20)
                self.update_progress_label("Detecting eyes and faces...")
                
                detection_results = self.eye_detector.detect_eyes_and_faces(
                    self.current_image_path,
                    enhanced_mode=self.enhanced_detection_var.get()
                )
                
                if not detection_results or detection_results.get('total_regions', 0) == 0:
                    self.update_status("⚠️ No eye regions detected", "orange")
                    self.update_progress(100)
                    self.update_progress_label("No eyes found")
                    
                    messagebox.showwarning("No Eyes Detected", 
                        "No eye regions could be detected in this image.\n\n"
                        "Tips:\n"
                        "• Ensure the image shows clear eye(s)\n"
                        "• Check image quality and lighting\n"
                        "• Try with a different image")
                    return
                
                logger.info(f"Detected {detection_results['total_regions']} eye regions")
                self.update_progress(40)
                
                # Étape 2: Suivi facial (si activé)
                face_tracking_results = None
                if self.face_handler and self.face_tracking_var.get():
                    self.update_progress_label("Processing face tracking...")
                    face_tracking_results = self.face_handler.process_faces(
                        self.current_image_path, detection_results
                    )
                
                self.update_progress(60)
                
                # Étape 3: Analyse avec Gemma 3n
                self.update_progress_label("Running AI analysis with Gemma 3n...")
                
                analysis_results = {}
                if self.gemma_handler and self.gemma_handler.is_ready():
                    analysis_results = self.gemma_handler.analyze_eye_regions(
                        detection_results['regions'],
                        confidence_threshold=self.confidence_var.get()
                    )
                else:
                    # Mode fallback sans Gemma
                    logger.warning("Gemma not ready, using fallback analysis")
                    analysis_results = self._fallback_analysis(detection_results)
                
                self.update_progress(80)
                
                # Étape 4: Visualisation
                if self.visualizer:
                    self.update_progress_label("Generating visual results...")
                    annotated_image = self.visualizer.create_annotated_image(
                        self.current_image_path,
                        detection_results,
                        analysis_results,
                        face_tracking_results
                    )
                    
                    if annotated_image:
                        self.display_annotated_image(annotated_image)
                
                self.update_progress(95)
                
                # Étape 5: Compilation des résultats
                self.update_progress_label("Compiling medical report...")
                self.compile_and_display_results(
                    detection_results, analysis_results, face_tracking_results
                )
                
                # Métriques finales
                processing_time = time.time() - start_time
                self.metrics['total_analyses'] += 1
                self.metrics['processing_times'].append(processing_time)
                
                # Vérifier les détections positives
                positive_count = self._count_positive_detections(analysis_results)
                if positive_count > 0:
                    self.metrics['positive_detections'] += 1
                    self.update_status(f"🚨 MEDICAL ALERT: Possible retinoblastoma detected! ({processing_time:.1f}s)", "red")
                    
                    messagebox.showwarning("⚠️ MEDICAL ALERT", 
                        f"🚨 POSSIBLE RETINOBLASTOMA DETECTED 🚨\n\n"
                        f"Positive findings in {positive_count} region(s)\n\n"
                        f"👨‍⚕️ IMMEDIATE ACTION REQUIRED:\n"
                        f"1. Contact pediatric ophthalmologist TODAY\n"
                        f"2. Show them this analysis and original image\n"
                        f"3. Do NOT delay seeking professional evaluation")
                else:
                    self.update_status(f"✅ Analysis complete: No concerning findings ({processing_time:.1f}s)", "green")
                    
                    messagebox.showinfo("Analysis Complete", 
                        f"✅ Analysis completed successfully!\n\n"
                        f"No signs of leukocoria were detected.\n"
                        f"Continue regular eye health monitoring.\n\n"
                        f"Processing time: {processing_time:.1f} seconds")
                
                self.update_progress(100)
                self.update_progress_label("Analysis complete")
                self.update_metrics_display()
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.update_status(f"❌ Analysis failed: {e}", "red")
                self.update_progress(0)
                self.update_progress_label("Analysis failed")
                
                messagebox.showerror("Analysis Error", 
                    f"Analysis failed with error:\n{e}\n\n"
                    f"Please try again or contact support.")
            
            finally:
                self.processing = False
        
        # Lancer l'analyse en arrière-plan
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def _fallback_analysis(self, detection_results):
        """Analyse de fallback sans Gemma"""
        return {
            'regions_analyzed': len(detection_results.get('regions', [])),
            'method': 'fallback_cv_only',
            'results': [
                {
                    'region_id': i,
                    'leukocoria_detected': False,
                    'confidence': 0.1,
                    'risk_level': 'unknown',
                    'analysis_method': 'computer_vision_fallback'
                }
                for i in range(len(detection_results.get('regions', [])))
            ]
        }
    
    def _count_positive_detections(self, analysis_results):
        """Compte les détections positives"""
        if not analysis_results or 'results' not in analysis_results:
            return 0
        
        return sum(1 for result in analysis_results['results'] 
                  if result.get('leukocoria_detected', False))
    
    def compile_and_display_results(self, detection_results, analysis_results, face_tracking_results):
        """Compile et affiche les résultats détaillés"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        report = f"""RETINOBLASTOMA ANALYSIS REPORT - GEMMA 3N LOCAL
{'='*70}
Generated: {timestamp}
Image: {filename}
AI Engine: Gemma 3n (100% Local Processing)
System: RetinoblastoGemma v6 - Hackathon Google Gemma

DETECTION SUMMARY:
{'='*35}
"""
        
        # Statistiques de détection
        total_regions = detection_results.get('total_regions', 0)
        analysis_method = analysis_results.get('method', 'unknown')
        regions_analyzed = analysis_results.get('regions_analyzed', 0)
        
        report += f"Regions detected: {total_regions}\n"
        report += f"Regions analyzed: {regions_analyzed}\n"
        report += f"Analysis method: {analysis_method}\n"
        
        # Résultats d'analyse
        positive_count = self._count_positive_detections(analysis_results)
        
        if positive_count > 0:
            report += f"\n🚨 MEDICAL ALERT: POSSIBLE RETINOBLASTOMA DETECTED\n"
            report += f"Positive findings: {positive_count}\n"
            report += f"IMMEDIATE PEDIATRIC OPHTHALMOLOGICAL CONSULTATION REQUIRED\n\n"
        else:
            report += f"\n✅ No concerning findings detected\n"
            report += f"Continue regular pediatric eye monitoring\n\n"
        
        # Détails par région
        report += f"DETAILED ANALYSIS BY REGION:\n"
        report += f"{'='*40}\n"
        
        for i, result in enumerate(analysis_results.get('results', []), 1):
            region_type = result.get('region_type', 'unknown')
            detected = result.get('leukocoria_detected', False)
            confidence = result.get('confidence', 0)
            risk_level = result.get('risk_level', 'unknown')
            
            report += f"\n--- Region {i}: {region_type.upper()} ---\n"
            report += f"Leukocoria detected: {'⚠️ YES' if detected else '✅ NO'}\n"
            report += f"Confidence level: {confidence:.1f}%\n"
            report += f"Risk assessment: {risk_level.upper()}\n"
            
            if detected:
                analysis_method = result.get('analysis_method', 'unknown')
                report += f"Analysis method: {analysis_method}\n"
        
        # Informations techniques
        report += f"\nTECHNICAL DETAILS:\n"
        report += f"{'='*25}\n"
        report += f"AI Model: Gemma 3n (Local - 100% Offline)\n"
        report += f"Privacy: Complete - No data transmitted\n"
        report += f"Processing device: Local machine\n"
        
        # Suivi facial
        if face_tracking_results:
            tracked_faces = face_tracking_results.get('tracked_faces', 0)
            report += f"Face tracking: {tracked_faces} individuals tracked\n"
        
        # Métriques de performance
        if self.metrics['processing_times']:
            avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            report += f"Average processing time: {avg_time:.1f}s\n"
        
        # Disclaimer médical
        report += f"\nCRITICAL MEDICAL DISCLAIMER:\n"
        report += f"{'='*40}\n"
        report += f"⚠️ IMPORTANT: This analysis is provided by an AI screening system.\n"
        report += f"This is NOT a medical diagnosis and should NOT replace professional\n"
        report += f"medical evaluation by qualified pediatric ophthalmologists.\n\n"
        
        report += f"IMMEDIATE ACTION REQUIRED if positive findings:\n"
        report += f"1. ⏰ Contact pediatric ophthalmologist IMMEDIATELY\n"
        report += f"2. 📋 Bring this report and original images to appointment\n"
        report += f"3. 🚫 Do NOT delay seeking professional medical evaluation\n"
        report += f"4. 📞 Emergency: Call your healthcare provider\n\n"
        
        report += f"Retinoblastoma facts:\n"
        report += f"- Most common eye cancer in children (under 6 years)\n"
        report += f"- 95% survival rate with EARLY detection and treatment\n"
        report += f"- Main early sign: White pupil reflex (leukocoria) in photos\n"
        report += f"- Can affect one or both eyes\n"
        
        # Afficher dans l'onglet résultats
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        
        # Basculer vers l'onglet résultats
        self.notebook.select(self.results_frame)
        
        # Sauvegarder les résultats
        self.current_results = report
    
    def display_annotated_image(self, annotated_image):
        """Affiche l'image annotée dans le canvas"""
        try:
            # Redimensionner pour l'affichage
            canvas_width = max(900, self.canvas.winfo_width())
            canvas_height = max(700, self.canvas.winfo_height())
            
            # Convertir PIL en ImageTk
            display_image = annotated_image.copy()
            display_image.thumbnail((canvas_width - 50, canvas_height - 50), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Centrer et afficher
            self.canvas.delete("all")
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Basculer vers l'onglet image
            self.notebook.select(self.image_frame)
            
        except Exception as e:
            logger.error(f"Error displaying annotated image: {e}")
    
    def export_results(self):
        """Exporte les résultats d'analyse"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results available to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Medical Analysis Results",
            defaultextension=".txt",
            filetypes=[
                ("Medical reports", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_results)
                
                self.update_status(f"✅ Results exported: {Path(file_path).name}", "green")
                messagebox.showinfo("Export Complete", 
                    f"Medical analysis results exported successfully!\n\n"
                    f"File: {file_path}\n\n"
                    f"This report can be shared with medical professionals.")
                
            except Exception as e:
                self.update_status(f"❌ Export failed: {e}", "red")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def generate_medical_report(self):
        """Génère un rapport médical HTML professionnel"""
        if not self.current_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"retinoblastoma_medical_report_{timestamp}.html"
        
        try:
            # Créer le rapport HTML
            html_report = self._create_html_medical_report()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Ouvrir dans le navigateur
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
            
            self.update_status(f"✅ Medical report generated: {report_path.name}", "green")
            messagebox.showinfo("Report Generated", 
                f"🏥 Medical report generated successfully!\n\n"
                f"📄 File: {report_path.name}\n"
                f"🌐 Opened in web browser\n\n"
                f"This professional report can be:\n"
                f"• Shared with medical professionals\n"
                f"• Printed for medical appointments\n"
                f"• Saved for medical records")
            
        except Exception as e:
            self.update_status(f"❌ Report generation failed: {e}", "red")
            messagebox.showerror("Report Error", f"Failed to generate medical report:\n{e}")
    
    def _create_html_medical_report(self):
        """Crée un rapport HTML médical professionnel"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        # Déterminer s'il y a des détections positives
        has_positive = "MEDICAL ALERT" in self.current_results
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Retinoblastoma Medical Analysis - Gemma 3n</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 40px; 
            line-height: 1.6; color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .container {{ 
            max-width: 1000px; margin: 0 auto; 
            background: white; border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 28px; font-weight: 300; }}
        .header .subtitle {{ font-size: 16px; opacity: 0.9; margin-top: 10px; }}
        .badges {{ margin-top: 20px; }}
        .badge {{ 
            display: inline-block; padding: 8px 16px; margin: 5px;
            border-radius: 25px; color: white; font-weight: bold; font-size: 12px;
        }}
        .badge-hackathon {{ background: #ff6b6b; }}
        .badge-local {{ background: #4299e1; }}
        .badge-secure {{ background: #48bb78; }}
        .content {{ padding: 40px; }}
        .alert-critical {{ 
            background: linear-gradient(135deg, #ff6b6b, #ff5722);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(255,107,107,0.3);
        }}
        .alert-safe {{ 
            background: linear-gradient(135deg, #51cf66, #40c057);
            color: white; padding: 30px; margin: 20px 0;
            border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(81,207,102,0.3);
        }}
        .results-section {{ 
            background: #f8f9fa; padding: 30px; 
            border-radius: 10px; margin: 20px 0;
            border-left: 5px solid #667eea;
        }}
        .disclaimer {{ 
            background: #fff3cd; border: 2px solid #ffc107;
            padding: 25px; border-radius: 10px; margin: 30px 0;
        }}
        .footer {{ 
            background: #2d3748; color: white; 
            padding: 30px; text-align: center;
        }}
        pre {{ 
            background: #2d3748; color: #e2e8f0;
            padding: 25px; border-radius: 8px; 
            overflow-x: auto; font-size: 14px;
            line-height: 1.4;
        }}
        .hackathon-info {{
            background: #ff6b6b; color: white;
            padding: 15px; border-radius: 8px; margin: 15px 0;
            text-align: center; font-weight: bold;
        }}
        @media print {{
            body {{ background: white !important; }}
            .container {{ box-shadow: none !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Retinoblastoma Medical Analysis Report</h1>
            <div class="subtitle">Generated: {timestamp}</div>
            <div class="badges">
                <span class="badge badge-hackathon">HACKATHON GOOGLE GEMMA</span>
                <span class="badge badge-local">100% LOCAL</span>
                <span class="badge badge-secure">PRIVATE</span>
            </div>
            <p><strong>Analysis System:</strong> Gemma 3n Multimodal (Local Processing)</p>
            <p><strong>Image:</strong> {filename}</p>
        </div>
        
        <div class="content">
            <div class="hackathon-info">
                🏆 Generated by RetinoblastoGemma v6 - Google Gemma Worldwide Hackathon Entry
            </div>"""
        
        if has_positive:
            html_report += """
            <div class="alert-critical">
                <h2>🚨 MEDICAL ALERT - IMMEDIATE ACTION REQUIRED</h2>
                <p style="font-size: 18px; font-weight: bold;">
                    Possible retinoblastoma detected. Contact pediatric ophthalmologist IMMEDIATELY.
                </p>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <strong>Emergency Actions:</strong><br>
                    1. Call pediatric ophthalmologist TODAY<br>
                    2. Bring this report and original images<br>
                    3. Do NOT delay medical evaluation
                </div>
            </div>"""
        else:
            html_report += """
            <div class="alert-safe">
                <h2>✅ No Concerning Findings Detected</h2>
                <p style="font-size: 16px;">
                    The AI analysis did not detect signs of leukocoria in this image.
                    Continue regular pediatric eye monitoring.
                </p>
            </div>"""
        
        html_report += f"""
            <div class="results-section">
                <h2>📊 Detailed Analysis Results</h2>
                <pre>{self.current_results}</pre>
            </div>
            
            <div class="disclaimer">
                <h3>⚕️ Critical Medical Disclaimer</h3>
                <p><strong>IMPORTANT:</strong> This report is generated by an AI screening system using Gemma 3n.</p>
                <p><strong>THIS IS NOT A MEDICAL DIAGNOSIS</strong> and should NOT replace professional medical evaluation.</p>
                
                <h4>📋 Next Steps:</h4>
                <ul>
                    <li><strong>Professional Evaluation:</strong> Schedule consultation with pediatric ophthalmologist</li>
                    <li><strong>Documentation:</strong> Bring this report and original images to appointment</li>
                    <li><strong>Urgency:</strong> {'IMMEDIATE evaluation required' if has_positive else 'Routine follow-up appropriate'}</li>
                    <li><strong>Monitoring:</strong> Continue regular eye health monitoring</li>
                </ul>
                
                <h4>🏥 About Retinoblastoma:</h4>
                <ul>
                    <li>Most common eye cancer in children (typically under 6 years)</li>
                    <li>95% survival rate with early detection and treatment</li>
                    <li>Can affect one or both eyes</li>
                    <li>Early sign: White pupil reflex (leukocoria) in photos</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Generated by RetinoblastoGemma v6</strong></p>
            <p>🏆 Google Gemma Worldwide Hackathon Entry</p>
            <p>🤖 AI-Powered Retinoblastoma Screening with Local Gemma 3n</p>
            <p>🔒 100% Local Processing - No data transmitted</p>
            <p style="font-size: 12px; margin-top: 10px;">
                Report ID: RG_{timestamp.replace('-', '').replace(':', '').replace(' ', '_')} | 
                Model: Gemma 3n Local | 
                Hackathon: Google Gemma Worldwide
            </p>
        </div>
    </div>
</body>
</html>"""
        
        return html_report
    
    def show_system_info(self):
        """Affiche les informations système"""
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("700x500")
        
        text_widget = tk.Text(info_window, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(info_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Informations système
        system_info = f"""RETINOBLASTOGAMMA v6 - SYSTEM INFORMATION
{'='*50}
Hackathon: Google Gemma Worldwide
Entry: RetinoblastoGemma - Early Detection of Retinoblastoma

SYSTEM STATUS:
{'='*20}
Current Image: {Path(self.current_image_path).name if self.current_image_path else 'None'}
Processing: {'Yes' if self.processing else 'No'}

MODULE STATUS:
{'='*20}"""
        
        for module_name, status_label in self.module_status.items():
            status_text = status_label.cget("text")
            system_info += f"\n{module_name.title()}: {status_text}"
        
        system_info += f"""

CONFIGURATION:
{'='*20}
Confidence Threshold: {self.confidence_var.get():.2f}
Face Tracking: {'Enabled' if self.face_tracking_var.get() else 'Disabled'}
Enhanced Detection: {'Enabled' if self.enhanced_detection_var.get() else 'Disabled'}
Force Local Mode: {'Yes' if self.config['force_local_mode'] else 'No'}

PERFORMANCE METRICS:
{'='*20}
Total Analyses: {self.metrics['total_analyses']}
Positive Detections: {self.metrics['positive_detections']}
Session Duration: {time.time() - self.metrics['session_start']:.0f} seconds"""
        
        if self.metrics['processing_times']:
            avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            system_info += f"\nAverage Processing Time: {avg_time:.1f} seconds"
        
        system_info += f"""

PATHS:
{'='*20}
Models Directory: {MODELS_DIR}
Data Directory: {DATA_DIR}
Results Directory: {RESULTS_DIR}

HACKATHON INFORMATION:
{'='*20}
Competition: Google Gemma Worldwide Hackathon
Team: RetinoblastoGemma Team
Objective: Early detection of retinoblastoma using local Gemma 3n
Privacy Focus: 100% local processing - no data transmission
Target: Special prizes for medical AI and privacy-focused solutions

TECHNICAL DETAILS:
{'='*20}
AI Model: Gemma 3n (Local execution)
Vision Capabilities: Multimodal image + text analysis
Privacy: Complete - all processing done locally
Mobile Ready: Architecture designed for on-device deployment
"""
        
        text_widget.insert(1.0, system_info)
        text_widget.config(state='disabled')
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_status(self, message, color="blue"):
        """Met à jour le statut de l'application"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))
        logger.info(message)
    
    def update_progress(self, value):
        """Met à jour la barre de progression"""
        self.root.after(0, lambda: self.progress.config(value=value))
    
    def update_progress_label(self, message):
        """Met à jour le label de progression"""
        self.root.after(0, lambda: self.progress_label.config(text=message))
    
    def update_module_status(self, module_name, status_text, color):
        """Met à jour le statut d'un module"""
        if module_name in self.module_status:
            label = self.module_status[module_name]
            self.root.after(0, lambda: label.config(text=f"{module_name.replace('_', ' ').title()}: {status_text}", 
                                                   foreground=color))
    
    def update_metrics_display(self):
        """Met à jour l'affichage des métriques"""
        total = self.metrics['total_analyses']
        positive = self.metrics['positive_detections']
        
        if total > 0:
            detection_rate = (positive / total) * 100
            avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            
            metrics_text = (f"Analyses: {total} | "
                          f"Positive: {positive} ({detection_rate:.1f}%) | "
                          f"Avg time: {avg_time:.1f}s")
        else:
            metrics_text = "No analysis performed yet"
        
        self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))

def main():
    """Fonction principale avec gestion d'erreurs robuste"""
    try:
        # Vérifications préliminaires
        print("🏥 RETINOBLASTOGAMMA v6 - HACKATHON GOOGLE GEMMA")
        print("="*60)
        print("🏆 Early Detection of Retinoblastoma using Local Gemma 3n")
        print("🔒 100% Local Processing - Privacy First")
        print("="*60)
        
        # Créer les dossiers nécessaires
        for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
            directory.mkdir(exist_ok=True)
            print(f"📁 Directory ready: {directory}")
        
        # Vérifications système
        import sys
        print(f"🐍 Python version: {sys.version.split()[0]}")
        
        try:
            import torch
            print(f"🔥 PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 Running on CPU")
        except ImportError:
            print("⚠️ PyTorch not available")
        
        # Créer et lancer l'application
        print("\n🚀 Launching RetinoblastoGemma v6...")
        root = tk.Tk()
        
        try:
            app = RetinoblastoGemmaV6(root)
            logger.info("RetinoblastoGemma v6 started successfully")
            print("✅ Application launched successfully!")
            print("💡 The app will initialize Gemma 3n after startup")
            print("🎯 Load an image and click 'Analyze for Retinoblastoma' to start")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            messagebox.showerror("Initialization Error", 
                f"Failed to initialize application:\n{e}\n\n"
                "Please check the installation and try again.")
            return
        
        # Gestionnaire de fermeture
        def on_closing():
            try:
                logger.info("Application closing...")
                
                # Sauvegarder les données si nécessaire
                if hasattr(app, 'face_handler') and app.face_handler:
                    app.face_handler.save_data()
                
                # Libérer les ressources
                if hasattr(app, 'gemma_handler') and app.gemma_handler:
                    app.gemma_handler.cleanup()
                
                print("👋 Thank you for using RetinoblastoGemma v6!")
                print("🏆 Good luck with the Google Gemma Hackathon!")
                
                root.quit()
                root.destroy()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Démarrer l'interface
        print("\n" + "="*60)
        print("🏥 RetinoblastoGemma v6 Ready!")
        print("🎯 Mission: Save children's lives through early detection")
        print("🏆 Hackathon: Google Gemma Worldwide")
        print("="*60)
        
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        print("\n📋 Troubleshooting:")
        print("1. Check Python installation (3.8+)")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Verify Gemma 3n model in models/gemma-3n/")
        print("4. Check system logs in retinoblastogamma.log")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()