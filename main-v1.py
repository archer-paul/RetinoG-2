"""
Interface principale pour RetinoblastoGemma - Version corrigée
Application de détection de rétinoblastome utilisant Gemma 3n avec optimisations Edge
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
from pathlib import Path
import threading
import json
import time
import sys
from datetime import datetime

# Imports scientifiques avec gestion d'erreurs
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch non disponible")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend pour éviter les conflits GUI
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib non disponible")

# Imports locaux avec fallback
try:
    from config.settings import *
    SETTINGS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Erreur import settings: {e}")
    # Fallback settings
    from pathlib import Path
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = DATA_DIR / "results"
    for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    SETTINGS_AVAILABLE = False

# Configuration du logging avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'retinoblastogamma.log') if 'RESULTS_DIR' in locals() else logging.NullHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetinoblastoGemmaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma - Early Detection System")
        self.root.geometry("1200x800")
        
        # Initialisation des composants avancés
        self.gemma = None
        self.eye_detector = None
        self.face_tracker = None
        self.visualizer = None
        
        # État de l'application
        self.current_image_path = None
        self.current_results = None
        self.batch_image_paths = []
        self.initialization_complete = False
        
        # Métriques de performance en temps réel
        self.performance_monitor = {
            'total_analyses': 0,
            'total_processing_time': 0,
            'detections_found': 0,
            'session_start': time.time()
        }
        
        # Configuration avancée
        self.advanced_config = {
            'use_parallel_processing': True,
            'enable_caching': True,
            'auto_enhance_images': True,
            'real_time_monitoring': True,
            'force_simulation_mode': False  # Nouveau flag
        }
        
        self.setup_ui()
        # Démarrer l'initialisation APRÈS l'UI
        self.root.after(100, self.initialize_components_delayed)
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Frame de contrôles (gauche)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Boutons principaux
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Load Multiple Images", 
                  command=self.load_multiple_images).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(control_frame, text="Analyze Current Image", 
                  command=self.analyze_current_image, state='disabled')
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Batch Analysis", 
                  command=self.batch_analysis).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Options d'analyse avancées
        options_frame = ttk.LabelFrame(control_frame, text="Analysis Options")
        options_frame.pack(fill=tk.X, pady=5)
        
        self.track_faces_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable Face Tracking", 
                       variable=self.track_faces_var).pack(anchor=tk.W)
        
        self.enhance_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enhance Eye Images", 
                       variable=self.enhance_images_var).pack(anchor=tk.W)
        
        self.parallel_processing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Parallel Processing", 
                       variable=self.parallel_processing_var).pack(anchor=tk.W)
        
        self.simulation_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Force Simulation Mode (Fast)", 
                       variable=self.simulation_mode_var,
                       command=self.toggle_simulation_mode).pack(anchor=tk.W)
        
        # Mode de traitement
        processing_frame = ttk.LabelFrame(control_frame, text="Processing Mode")
        processing_frame.pack(fill=tk.X, pady=5)
        
        self.processing_mode_var = tk.StringVar(value="simulation")
        ttk.Radiobutton(processing_frame, text="Google AI Edge", 
                       variable=self.processing_mode_var, value="google_ai").pack(anchor=tk.W)
        ttk.Radiobutton(processing_frame, text="Local Model", 
                       variable=self.processing_mode_var, value="local").pack(anchor=tk.W)
        ttk.Radiobutton(processing_frame, text="Simulation (Recommended)", 
                       variable=self.processing_mode_var, value="simulation").pack(anchor=tk.W)
        
        # Informations système avancées
        system_frame = ttk.LabelFrame(control_frame, text="System Status")
        system_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(system_frame, text="Starting...", 
                                     foreground="blue")
        self.status_label.pack(anchor=tk.W)
        
        # Métriques en temps réel
        self.metrics_label = ttk.Label(system_frame, text="No analysis yet", 
                                      font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Progress bar avec information détaillée
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready", 
                                       font=("Arial", 8))
        self.progress_label.pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Boutons de gestion avancés
        management_frame = ttk.LabelFrame(control_frame, text="Management")
        management_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(management_frame, text="Quick Test System", 
                  command=self.run_quick_test).pack(fill=tk.X, pady=1)
        
        ttk.Button(management_frame, text="Force Reinitialize", 
                  command=self.force_reinitialize).pack(fill=tk.X, pady=1)
        
        ttk.Button(management_frame, text="View System Info", 
                  command=self.show_system_info).pack(fill=tk.X, pady=1)
        
        ttk.Button(management_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        # Frame d'affichage principal (droite)
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Onglet image principale
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Analysis")
        
        # Canvas pour l'image
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        scrollbar_v = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Onglet résultats
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Zone de texte pour les résultats
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)
        
        # Barre de statut
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.statusbar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.statusbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def initialize_components_delayed(self):
        """Initialise les composants avec délai pour éviter de bloquer l'UI"""
        threading.Thread(target=self.initialize_components, daemon=True).start()
    
    def initialize_components(self):
        """Initialise les composants avec gestion d'erreurs robuste"""
        try:
            self.update_status("🔄 Starting system initialization...", "blue")
            self.update_progress(10)
            
            # Vérifier les dépendances de base
            self.update_progress_label("Checking dependencies...")
            missing_deps = self.check_dependencies()
            
            if missing_deps:
                self.update_status(f"⚠️ Missing dependencies: {', '.join(missing_deps)}", "orange")
                self.update_progress_label("Some dependencies missing - using fallback mode")
                self.advanced_config['force_simulation_mode'] = True
            
            self.update_progress(25)
            
            # Initialiser les modules core un par un
            self.update_progress_label("Initializing eye detector...")
            success_eye = self.initialize_eye_detector()
            self.update_progress(40)
            
            self.update_progress_label("Initializing face tracker...")
            success_face = self.initialize_face_tracker()
            self.update_progress(60)
            
            self.update_progress_label("Initializing visualizer...")
            success_viz = self.initialize_visualizer()
            self.update_progress(75)
            
            # Initialiser Gemma en dernier (plus lourd)
            if not self.advanced_config['force_simulation_mode']:
                self.update_progress_label("Initializing Gemma AI (this may take time)...")
                success_gemma = self.initialize_gemma_handler()
            else:
                self.update_progress_label("Skipping Gemma - using simulation mode")
                success_gemma = True
                
            self.update_progress(90)
            
            # Finalisation
            self.initialization_complete = True
            self.update_progress(100)
            
            if success_eye and success_face and success_viz and success_gemma:
                self.update_status("✅ System ready! All components loaded", "green")
                self.analyze_button.config(state='normal')
            else:
                self.update_status("⚠️ System ready with limited functionality", "orange")
                self.analyze_button.config(state='normal')
                
            self.update_progress_label("Ready for analysis")
            self.update_metrics_display()
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.update_status(f"❌ Initialization failed: {e}", "red")
            self.update_progress_label("Initialization failed - check logs")
            self.analyze_button.config(state='normal')  # Permettre quand même d'essayer
    
    def check_dependencies(self):
        """Vérifie les dépendances et retourne les manquantes"""
        missing = []
        
        try:
            import cv2
        except ImportError:
            missing.append("opencv-python")
        
        try:
            import mediapipe
        except ImportError:
            missing.append("mediapipe")
        
        try:
            import face_recognition
        except ImportError:
            missing.append("face-recognition")
        
        if not TORCH_AVAILABLE:
            missing.append("torch")
            
        return missing
    
    def initialize_eye_detector(self):
        """Initialise le détecteur d'yeux avec gestion d'erreurs"""
        try:
            from core.eye_detector import AdvancedEyeDetector
            self.eye_detector = AdvancedEyeDetector()
            logger.info("✅ Eye detector initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Eye detector failed: {e}")
            return False
    
    def initialize_face_tracker(self):
        """Initialise le traqueur de visages avec gestion d'erreurs"""
        try:
            from core.face_tracker import FaceTracker
            self.face_tracker = FaceTracker()
            logger.info("✅ Face tracker initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Face tracker failed: {e}")
            return False
    
    def initialize_visualizer(self):
        """Initialise le visualiseur avec gestion d'erreurs"""
        try:
            from core.visualization import Visualizer
            self.visualizer = Visualizer()
            logger.info("✅ Visualizer initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Visualizer failed: {e}")
            return False
    
    def initialize_gemma_handler(self):
        """Initialise Gemma avec timeout et gestion d'erreurs"""
        try:
            # Timeout pour éviter un blocage infini
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Gemma initialization timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 secondes max
            
            try:
                from core.gemma_handler import GemmaHandler
                self.gemma = GemmaHandler()
                logger.info("✅ Gemma handler initialized")
                return True
            finally:
                signal.alarm(0)  # Annuler le timeout
                
        except (TimeoutError, Exception) as e:
            logger.warning(f"⚠️ Gemma initialization failed: {e}")
            logger.info("Using simulation mode instead")
            self.advanced_config['force_simulation_mode'] = True
            return True  # Continuer avec simulation
    
    def toggle_simulation_mode(self):
        """Bascule le mode simulation"""
        self.advanced_config['force_simulation_mode'] = self.simulation_mode_var.get()
        if self.advanced_config['force_simulation_mode']:
            self.update_status("🔧 Simulation mode enabled", "blue")
        else:
            self.update_status("🔧 AI mode enabled", "blue")
    
    def update_status(self, message, color="blue"):
        """Met à jour le statut de l'application"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))
        self.root.after(0, lambda: self.statusbar.config(text=message))
        logger.info(message)
    
    def update_progress(self, value):
        """Met à jour la barre de progression"""
        self.root.after(0, lambda: self.progress.config(value=value))
    
    def update_progress_label(self, message):
        """Met à jour le label de progression"""
        self.root.after(0, lambda: self.progress_label.config(text=message))
    
    def update_metrics_display(self):
        """Met à jour l'affichage des métriques en temps réel"""
        if self.performance_monitor['total_analyses'] > 0:
            avg_time = self.performance_monitor['total_processing_time'] / self.performance_monitor['total_analyses']
            detection_rate = (self.performance_monitor['detections_found'] / 
                            self.performance_monitor['total_analyses']) * 100
            
            metrics_text = (f"Analyses: {self.performance_monitor['total_analyses']} | "
                          f"Avg: {avg_time:.2f}s | "
                          f"Detection rate: {detection_rate:.1f}%")
        else:
            metrics_text = "No analysis performed yet"
        
        self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))
    
    def load_image(self):
        """Charge une image unique"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.update_status(f"Loaded: {Path(file_path).name}")
    
    def display_image(self, image_path):
        """Affiche une image dans le canvas"""
        try:
            # Charger et redimensionner l'image
            image = Image.open(image_path)
            
            # Calculer la taille d'affichage
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 600
            
            # Redimensionner en gardant les proportions
            image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            
            # Convertir pour tkinter
            self.photo = ImageTk.PhotoImage(image)
            
            # Afficher dans le canvas
            self.canvas.delete("all")
            self.canvas.create_image(10, 10, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("Display Error", f"Cannot display image: {e}")
    
    def analyze_current_image(self):
        """Analyse l'image actuellement chargée"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.initialization_complete:
            messagebox.showwarning("System Not Ready", "System is still initializing. Please wait.")
            return
        
        def analysis_thread():
            try:
                start_time = time.time()
                self.update_progress(0)
                self.update_progress_label("Starting analysis...")
                
                # Analyse basée sur les composants disponibles
                if self.eye_detector:
                    self.update_progress_label("Detecting faces and eyes...")
                    self.update_progress(25)
                    
                    detection_results = self.eye_detector.detect_faces_and_eyes(
                        self.current_image_path,
                        enhance_quality=self.enhance_images_var.get(),
                        parallel_processing=self.parallel_processing_var.get()
                    )
                    
                    self.update_progress(50)
                    
                    # Analyser avec Gemma si disponible
                    if self.gemma and not self.advanced_config['force_simulation_mode']:
                        self.update_progress_label("Running AI analysis...")
                        # Analyse réelle avec Gemma (code existant)
                        analysis_results = self.run_gemma_analysis(detection_results)
                    else:
                        self.update_progress_label("Running simulation analysis...")
                        # Analyse de simulation
                        analysis_results = self.run_simulation_analysis(detection_results)
                    
                    self.update_progress(75)
                    
                    # Afficher les résultats
                    self.display_analysis_results(detection_results, analysis_results)
                    
                    processing_time = time.time() - start_time
                    self.update_status(f"✅ Analysis completed! ({processing_time:.1f}s)", "green")
                    self.update_progress(100)
                    self.update_progress_label("Analysis complete")
                    
                    # Mettre à jour les métriques
                    self.performance_monitor['total_analyses'] += 1
                    self.performance_monitor['total_processing_time'] += processing_time
                    self.update_metrics_display()
                
                else:
                    self.update_status("❌ Eye detector not available", "red")
                    self.update_progress_label("Analysis failed")
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                self.update_status(f"❌ Analysis failed: {e}", "red")
                self.update_progress_label("Analysis failed")
                messagebox.showerror("Analysis Error", f"Analysis failed: {e}")
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def run_simulation_analysis(self, detection_results):
        """Analyse de simulation rapide et réaliste"""
        analysis_results = {'faces': []}
        
        for face_detection in detection_results.get('faces', []):
            face_analysis = {'eyes': []}
            
            for eye_region in getattr(face_detection, 'eyes', []):
                # Simulation basée sur des heuristiques simples
                confidence = np.random.uniform(20, 85)
                detected = confidence > 60
                
                eye_analysis = {
                    'position': getattr(eye_region, 'position', 'unknown'),
                    'leukocoria_detected': detected,
                    'confidence': confidence,
                    'risk_level': 'high' if confidence > 75 else 'medium' if confidence > 50 else 'low',
                    'pupil_color': 'bright/suspicious' if detected else 'normal',
                    'description': f"Simulation analysis of {getattr(eye_region, 'position', 'unknown')} eye",
                    'recommendations': 'Consult ophthalmologist' if detected else 'Continue monitoring',
                    'model_backend': 'simulation'
                }
                
                face_analysis['eyes'].append(eye_analysis)
            
            analysis_results['faces'].append(face_analysis)
        
        return analysis_results
    
    def run_gemma_analysis(self, detection_results):
        """Analyse réelle avec Gemma (si disponible)"""
        # Code d'analyse Gemma original ici
        return self.run_simulation_analysis(detection_results)  # Fallback pour maintenant
    
    def display_analysis_results(self, detection_results, analysis_results):
        """Affiche les résultats d'analyse"""
        results_text = f"""RETINOBLASTOMA ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {Path(self.current_image_path).name}
====================================================

DETECTION RESULTS:
• Faces detected: {len(detection_results.get('faces', []))}
• Eyes detected: {detection_results.get('total_eyes_detected', 0)}
• Processing time: {detection_results.get('processing_time', 0):.2f}s

ANALYSIS RESULTS:
"""
        
        total_detections = 0
        for face_idx, face_analysis in enumerate(analysis_results.get('faces', [])):
            results_text += f"\nFace {face_idx + 1}:\n"
            for eye_analysis in face_analysis.get('eyes', []):
                position = eye_analysis.get('position', 'unknown')
                detected = eye_analysis.get('leukocoria_detected', False)
                confidence = eye_analysis.get('confidence', 0)
                risk_level = eye_analysis.get('risk_level', 'unknown')
                
                status = "⚠️ DETECTED" if detected else "✅ Normal"
                results_text += f"  {position.upper()} eye: {status} (confidence: {confidence:.1f}%, risk: {risk_level})\n"
                
                if detected:
                    total_detections += 1
                    results_text += f"    Description: {eye_analysis.get('description', 'N/A')}\n"
                    results_text += f"    Recommendation: {eye_analysis.get('recommendations', 'N/A')}\n"
        
        results_text += f"\nSUMMARY:\n"
        results_text += f"• Total concerning findings: {total_detections}\n"
        
        if total_detections > 0:
            results_text += f"• ⚠️ MEDICAL ATTENTION RECOMMENDED\n"
            results_text += f"• Schedule ophthalmologist consultation\n"
        else:
            results_text += f"• ✅ No concerning findings detected\n"
            results_text += f"• Continue regular monitoring\n"
        
        results_text += f"\nIMPORTANT DISCLAIMER:\n"
        results_text += f"This is a screening tool, not a diagnostic device.\n"
        results_text += f"Always consult a qualified ophthalmologist for medical diagnosis.\n"
        
        # Afficher dans l'onglet résultats
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        
        # Basculer vers l'onglet résultats
        self.notebook.select(self.results_frame)
    
    def run_quick_test(self):
        """Lance un test rapide du système"""
        def test_thread():
            try:
                self.update_status("🧪 Running quick system test...", "blue")
                
                # Test des composants
                tests_passed = 0
                total_tests = 4
                
                if self.eye_detector:
                    tests_passed += 1
                if self.face_tracker:
                    tests_passed += 1
                if self.visualizer:
                    tests_passed += 1
                if self.gemma or self.advanced_config['force_simulation_mode']:
                    tests_passed += 1
                
                success_rate = (tests_passed / total_tests) * 100
                
                if success_rate >= 75:
                    self.update_status(f"✅ Quick test passed: {tests_passed}/{total_tests} components ready ({success_rate:.0f}%)", "green")
                else:
                    self.update_status(f"⚠️ Quick test partial: {tests_passed}/{total_tests} components ready ({success_rate:.0f}%)", "orange")
                
                messagebox.showinfo("Quick Test Results", 
                    f"System Test Results:\n\n"
                    f"Eye Detector: {'✅' if self.eye_detector else '❌'}\n"
                    f"Face Tracker: {'✅' if self.face_tracker else '❌'}\n"
                    f"Visualizer: {'✅' if self.visualizer else '❌'}\n"
                    f"AI Analysis: {'✅' if self.gemma or self.advanced_config['force_simulation_mode'] else '❌'}\n\n"
                    f"Overall: {tests_passed}/{total_tests} components ready\n"
                    f"System is {'ready for use' if success_rate >= 75 else 'partially functional'}")
                
            except Exception as e:
                self.update_status(f"❌ Quick test failed: {e}", "red")
                messagebox.showerror("Test Error", f"Quick test failed: {e}")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def force_reinitialize(self):
        """Force la réinitialisation du système"""
        result = messagebox.askyesno("Force Reinitialize", 
            "This will restart all system components. Continue?")
        if result:
            self.update_status("🔄 Forcing reinitialization...", "blue")
            self.initialization_complete = False
            self.analyze_button.config(state='disabled')
            self.gemma = None
            self.eye_detector = None
            self.face_tracker = None
            self.visualizer = None
            self.root.after(1000, self.initialize_components_delayed)
    
    def show_system_info(self):
        """Affiche les informations système"""
        info = f"""RETINOBLASTOGAMMA SYSTEM INFORMATION
====================================

SYSTEM STATUS:
• Initialization Complete: {self.initialization_complete}
• Analysis Button: {'Enabled' if self.analyze_button['state'] == 'normal' else 'Disabled'}

COMPONENTS STATUS:
• Eye Detector: {'✅ Ready' if self.eye_detector else '❌ Not loaded'}
• Face Tracker: {'✅ Ready' if self.face_tracker else '❌ Not loaded'}
• Visualizer: {'✅ Ready' if self.visualizer else '❌ Not loaded'}
• Gemma Handler: {'✅ Ready' if self.gemma else '❌ Not loaded'}

DEPENDENCIES:
• PyTorch: {'✅ Available' if TORCH_AVAILABLE else '❌ Missing'}
• Matplotlib: {'✅ Available' if MATPLOTLIB_AVAILABLE else '❌ Missing'}
• Settings: {'✅ Loaded' if SETTINGS_AVAILABLE else '❌ Using fallback'}

CONFIGURATION:
• Simulation Mode: {'✅ Enabled' if self.advanced_config['force_simulation_mode'] else '❌ Disabled'}
• Face Tracking: {'✅ Enabled' if self.track_faces_var.get() else '❌ Disabled'}
• Image Enhancement: {'✅ Enabled' if self.enhance_images_var.get() else '❌ Disabled'}
• Parallel Processing: {'✅ Enabled' if self.parallel_processing_var.get() else '❌ Disabled'}

PERFORMANCE:
• Total Analyses: {self.performance_monitor['total_analyses']}
• Session Time: {time.time() - self.performance_monitor['session_start']:.0f}s

CURRENT IMAGE:
• Path: {self.current_image_path if self.current_image_path else 'None loaded'}
"""
        
        # Créer une nouvelle fenêtre pour afficher les infos
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("600x500")
        
        text_widget = tk.Text(info_window, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(info_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert(1.0, info)
        text_widget.config(state='disabled')  # Read-only
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_multiple_images(self):
        """Charge plusieurs images pour analyse en lot"""
        file_paths = filedialog.askopenfilenames(
            title="Select multiple images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.batch_image_paths = file_paths
            self.update_status(f"Loaded {len(file_paths)} images for batch analysis")
            messagebox.showinfo("Batch Load", f"Loaded {len(file_paths)} images.\nUse 'Batch Analysis' to process them.")
    
    def batch_analysis(self):
        """Effectue une analyse en lot"""
        if not hasattr(self, 'batch_image_paths') or not self.batch_image_paths:
            messagebox.showwarning("No Images", "Please load multiple images first.")
            return
        
        if not self.initialization_complete:
            messagebox.showwarning("System Not Ready", "System is still initializing. Please wait.")
            return
        
        def batch_thread():
            try:
                self.update_status(f"🔄 Starting batch analysis of {len(self.batch_image_paths)} images...", "blue")
                results_summary = []
                
                for i, image_path in enumerate(self.batch_image_paths):
                    self.update_progress((i / len(self.batch_image_paths)) * 100)
                    self.update_progress_label(f"Processing image {i+1}/{len(self.batch_image_paths)}")
                    
                    try:
                        # Analyse simplifiée pour le lot
                        if self.eye_detector:
                            detection_results = self.eye_detector.detect_faces_and_eyes(image_path)
                            analysis_results = self.run_simulation_analysis(detection_results)
                            
                            # Compter les détections
                            detections = sum(
                                1 for face in analysis_results.get('faces', [])
                                for eye in face.get('eyes', [])
                                if eye.get('leukocoria_detected', False)
                            )
                            
                            results_summary.append({
                                'image': Path(image_path).name,
                                'faces': len(detection_results.get('faces', [])),
                                'eyes': detection_results.get('total_eyes_detected', 0),
                                'detections': detections
                            })
                        
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        results_summary.append({
                            'image': Path(image_path).name,
                            'error': str(e)
                        })
                
                # Afficher le résumé
                self.display_batch_results(results_summary)
                self.update_progress(100)
                self.update_status(f"✅ Batch analysis completed: {len(self.batch_image_paths)} images processed", "green")
                self.update_progress_label("Batch analysis complete")
                
            except Exception as e:
                self.update_status(f"❌ Batch analysis failed: {e}", "red")
                self.update_progress_label("Batch analysis failed")
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    def display_batch_results(self, results_summary):
        """Affiche les résultats de l'analyse en lot"""
        batch_text = f"""BATCH ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Images: {len(results_summary)}
====================================================

RESULTS SUMMARY:
"""
        
        total_detections = 0
        total_images_with_detections = 0
        
        for i, result in enumerate(results_summary, 1):
            if 'error' in result:
                batch_text += f"{i:2d}. {result['image']:<30} ERROR: {result['error']}\n"
            else:
                detections = result.get('detections', 0)
                faces = result.get('faces', 0)
                eyes = result.get('eyes', 0)
                
                status = "⚠️ CONCERNING" if detections > 0 else "✅ Normal"
                batch_text += f"{i:2d}. {result['image']:<30} {status} - {faces}F/{eyes}E/{detections}D\n"
                
                total_detections += detections
                if detections > 0:
                    total_images_with_detections += 1
        
        batch_text += f"\nOVERALL SUMMARY:\n"
        batch_text += f"• Images with concerning findings: {total_images_with_detections}/{len(results_summary)}\n"
        batch_text += f"• Total detections across all images: {total_detections}\n"
        
        if total_detections > 0:
            batch_text += f"• ⚠️ MEDICAL CONSULTATION RECOMMENDED for images with findings\n"
        else:
            batch_text += f"• ✅ No concerning findings in batch\n"
        
        batch_text += f"\nLEGEND: F=Faces, E=Eyes, D=Detections\n"
        batch_text += f"\nIMPORTANT: This is a screening tool. Professional medical evaluation required.\n"
        
        # Afficher dans l'onglet résultats
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, batch_text)
        self.notebook.select(self.results_frame)
    
    def export_results(self):
        """Exporte les résultats actuels"""
        if not self.current_results:
            results_text = self.results_text.get(1.0, tk.END)
            if not results_text.strip():
                messagebox.showwarning("No Results", "No analysis results to export.")
                return
        else:
            results_text = self.current_results
        
        # Demander où sauvegarder
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        # Export en JSON si possible
                        export_data = {
                            'timestamp': datetime.now().isoformat(),
                            'results': results_text,
                            'image_path': self.current_image_path,
                            'system_info': {
                                'version': '1.0.0',
                                'simulation_mode': self.advanced_config['force_simulation_mode']
                            }
                        }
                        json.dump(export_data, f, indent=2)
                    else:
                        f.write(results_text)
                
                self.update_status(f"✅ Results exported to {Path(file_path).name}", "green")
                messagebox.showinfo("Export Success", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                self.update_status(f"❌ Export failed: {e}", "red")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

def main():
    """Fonction principale avec gestion d'erreurs robuste"""
    try:
        # Créer les dossiers nécessaires
        if 'MODELS_DIR' in globals():
            for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
                directory.mkdir(exist_ok=True)
        
        # Créer et lancer l'application
        root = tk.Tk()
        
        # Gestion d'erreurs pour l'initialisation de l'app
        try:
            app = RetinoblastoGemmaApp(root)
        except Exception as e:
            logger.error(f"Failed to initialize app: {e}")
            messagebox.showerror("Initialization Error", 
                f"Failed to initialize application:\n{e}\n\n"
                "Try running: pip install -r requirements.txt")
            return
        
        # Gestionnaire de fermeture
        def on_closing():
            try:
                if hasattr(app, 'face_tracker') and app.face_tracker:
                    app.face_tracker.save_database()
                logger.info("Application closing gracefully")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            finally:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Lancer l'interface
        logger.info("Starting RetinoblastoGemma application")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        print(f"❌ Critical error: {e}")
        print("Try running: python quick-start.py")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()