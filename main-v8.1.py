"""
RetinoblastoGemma v6 - Interface principale modulaire CORRIGÉE
Application de détection de rétinoblastome avec Gemma 3n local
Architecture modulaire optimisée pour le hackathon Google Gemma
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
from pathlib import Path
import threading
import time
import sys
import os
from datetime import datetime
import json
import traceback

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

# Imports des modules de configuration avec fallback
try:
    from config.settings import (
        MODELS_DIR, DATA_DIR, RESULTS_DIR, 
        get_config_for_environment, MESSAGE_TEMPLATES,
        ensure_directories, validate_environment, get_model_info
    )
    logger.info("✅ Configuration importée avec succès")
    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Erreur d'import configuration: {e}")
    # Fallback - créer les dossiers manuellement
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data") 
    RESULTS_DIR = Path("results")
    for d in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
        d.mkdir(exist_ok=True)
    CONFIG_AVAILABLE = False

# Imports des modules core avec suivi détaillé
modules_imported = {
    'gemma_handler': False,
    'eye_detector': False, 
    'face_handler': False,
    'visualizer': False
}

print("🔄 Importing core modules...")

try:
    from core.gemma_handler_v2 import GemmaHandlerV2
    modules_imported['gemma_handler'] = True
    print("✅ GemmaHandlerV2 imported")
    logger.info("✅ GemmaHandlerV2 importé")
except ImportError as e:
    print(f"❌ GemmaHandlerV2 import failed: {e}")
    logger.error(f"❌ Erreur import GemmaHandlerV2: {e}")

try:
    from core.eye_detector_v2 import EyeDetectorV2
    modules_imported['eye_detector'] = True
    print("✅ EyeDetectorV2 imported")
    logger.info("✅ EyeDetectorV2 importé")
except ImportError as e:
    print(f"❌ EyeDetectorV2 import failed: {e}")
    logger.error(f"❌ Erreur import EyeDetectorV2: {e}")

try:
    from core.face_handler_v2 import FaceHandlerV2
    modules_imported['face_handler'] = True
    print("✅ FaceHandlerV2 imported")
    logger.info("✅ FaceHandlerV2 importé")
except ImportError as e:
    print(f"❌ FaceHandlerV2 import failed: {e}")
    logger.error(f"❌ Erreur import FaceHandlerV2: {e}")

try:
    from core.visualization_v2 import VisualizationV2
    modules_imported['visualizer'] = True
    print("✅ VisualizationV2 imported")
    logger.info("✅ VisualizationV2 importé")
except ImportError as e:
    print(f"❌ VisualizationV2 import failed: {e}")
    logger.error(f"❌ Erreur import VisualizationV2: {e}")
try:
    from ui_modern import ModernUI
    MODERN_UI_AVAILABLE = True
    logger.info("✅ Interface moderne importée")
except ImportError as e:
    logger.warning(f"Interface moderne non disponible: {e}")
    MODERN_UI_AVAILABLE = False

print(f"📦 Modules imported: {sum(modules_imported.values())}/4")

class RetinoblastoGemmaV6:
    """Application principale modulaire pour la détection de rétinoblastome"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma v6 - Hackathon Google Gemma")
        self.root.geometry("1400x900")
        
        # Initialiser la configuration
        if CONFIG_AVAILABLE:
            self.config = get_config_for_environment('hackathon_demo')
        else:
            self.config = {
                'analysis': {
                    'default_confidence_threshold': 0.5,
                    'enable_face_tracking': True,
                    'enhanced_detection': True,
                    'force_local_mode': True
                }
            }
        
        # État de l'application
        self.current_image_path = None
        self.current_results = None
        self.current_face_id = None
        self.processing = False
        
        # Modules core (initialisés plus tard)
        self.gemma_handler = None
        self.eye_detector = None
        self.face_handler = None
        self.visualizer = None
        
        # Configuration interface
        self.ui_config = {
            'confidence_threshold': self.config['analysis']['default_confidence_threshold'],
            'use_face_tracking': self.config['analysis']['enable_face_tracking'],
            'enhanced_detection': self.config['analysis']['enhanced_detection'],
            'force_local_mode': self.config['analysis']['force_local_mode']
        }
        
        # Métriques de performance
        self.metrics = {
            'total_analyses': 0,
            'positive_detections': 0,
            'processing_times': [],
            'session_start': time.time(),
            'modules_ready': 0,
            'errors_count': 0,
            'gemma_load_time': 0,
            'gemma_load_success': False
        }
        
        # Statut des modules détaillé
        self.module_details = {
            'gemma': {'status': 'waiting', 'error': None, 'load_time': 0},
            'eye_detector': {'status': 'waiting', 'error': None, 'load_time': 0},
            'face_handler': {'status': 'waiting', 'error': None, 'load_time': 0},
            'visualizer': {'status': 'waiting', 'error': None, 'load_time': 0}
        }
        
        # Initialiser l'interface (moderne ou classique)
        if MODERN_UI_AVAILABLE:
            self.modern_ui = ModernUI(self)
            self.modern_ui.create_modern_interface()
            logger.info("✅ Interface moderne initialisée")
        else:
            self.setup_ui()  # Interface classique en fallback
        
        # Vérifier l'environnement
        self.root.after(500, self.check_environment)
        
        # Initialiser les modules en arrière-plan
        self.root.after(1000, self.initialize_modules_async)
    
    def check_environment(self):
        """Vérifie l'environnement système"""
        try:
            validation = validate_environment()
            
            env_status = "✅ Environment Ready" if all([
                validation.get('python_version_ok', False),
                validation.get('directories_exist', False)
            ]) else "⚠️ Environment Issues Detected"
            
            self.update_status(env_status)
            
            if not validation.get('gpu_available', False):
                logger.warning("GPU not available - will use CPU mode")
                
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            self.update_status("❌ Environment check failed")
    
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
        
        # Section hackathon info
        hackathon_section = ttk.LabelFrame(control_frame, text="🏆 Hackathon Info")
        hackathon_section.pack(fill=tk.X, pady=(0, 10))
        
        hackathon_info = ttk.Label(hackathon_section, 
                                  text="Google Gemma Worldwide\n100% Local AI Processing\nPrivacy-First Medical Analysis", 
                                  font=("Arial", 9), foreground="purple")
        hackathon_info.pack(anchor=tk.W, pady=5)
        
        # Section chargement d'image
        image_section = ttk.LabelFrame(control_frame, text="📸 Image Loading")
        image_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_section, text="Load Medical Image", 
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
        self.confidence_var = tk.DoubleVar(value=self.ui_config['confidence_threshold'])
        confidence_scale = ttk.Scale(config_section, from_=0.1, to=0.9, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(fill=tk.X, pady=2)
        
        # Options
        self.face_tracking_var = tk.BooleanVar(value=self.ui_config['use_face_tracking'])
        ttk.Checkbutton(config_section, text="Enable Face Tracking", 
                       variable=self.face_tracking_var).pack(anchor=tk.W)
        
        self.enhanced_detection_var = tk.BooleanVar(value=self.ui_config['enhanced_detection'])
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
        metrics_section = ttk.LabelFrame(control_frame, text="📈 Session Metrics")
        metrics_section.pack(fill=tk.X, pady=10)
        
        self.metrics_label = ttk.Label(metrics_section, text="No analysis yet", font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Actions
        actions_section = ttk.LabelFrame(control_frame, text="💾 Actions")
        actions_section.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_section, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="Medical Report (HTML)", 
                  command=self.generate_medical_report).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="🧠 Recommandations Intelligentes", 
                  command=self.generate_recommendations).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="Face Tracking Summary", 
                  command=self.show_face_tracking_summary).pack(fill=tk.X, pady=1)
        
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
        
        # Onglet historique patient
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="👤 Patient History")
        
        history_container = ttk.Frame(self.history_frame)
        history_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.history_text = tk.Text(history_container, wrap=tk.WORD, 
                                   font=("Consolas", 9), relief=tk.SUNKEN, bd=2)
        history_scrollbar = ttk.Scrollbar(history_container, orient="vertical", 
                                        command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        history_container.columnconfigure(0, weight=1)
        history_container.rowconfigure(0, weight=1)
    
    def setup_status_bar(self, parent):
        """Configure la barre de statut"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="RetinoblastoGemma v6 Ready - Hackathon Google Gemma", 
                                     relief=tk.SUNKEN, font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def initialize_modules_async(self):
        """Initialise les modules en arrière-plan avec feedback détaillé"""
        def init_thread():
            try:
                self.update_status("🔄 Initializing core modules...", "blue")
                self.update_progress(5)
                
                # 1. Initialiser Eye Detector (le plus fiable)
                self.update_progress_label("Initializing eye detector...")
                self.update_module_status('eye_detector', "Loading...", "blue")
                
                if modules_imported['eye_detector']:
                    eye_start = time.time()
                    try:
                        self.eye_detector = EyeDetectorV2()
                        self.module_details['eye_detector']['load_time'] = time.time() - eye_start
                        self.module_details['eye_detector']['status'] = 'ready'
                        self.update_module_status('eye_detector', "✅ Ready", "green")
                        self.metrics['modules_ready'] += 1
                        logger.info(f"Eye detector loaded in {self.module_details['eye_detector']['load_time']:.2f}s")
                    except Exception as e:
                        self.module_details['eye_detector']['status'] = 'error'
                        self.module_details['eye_detector']['error'] = str(e)
                        logger.error(f"Eye detector initialization failed: {e}")
                        self.update_module_status('eye_detector', "❌ Error", "red")
                        self.metrics['errors_count'] += 1
                else:
                    self.update_module_status('eye_detector', "❌ Not Available", "red")
                    self.module_details['eye_detector']['status'] = 'missing'
                
                self.update_progress(20)
                
                # 2. Initialiser Visualizer
                self.update_progress_label("Initializing visualizer...")
                self.update_module_status('visualizer', "Loading...", "blue")
                
                if modules_imported['visualizer']:
                    viz_start = time.time()
                    try:
                        self.visualizer = VisualizationV2()
                        self.module_details['visualizer']['load_time'] = time.time() - viz_start
                        self.module_details['visualizer']['status'] = 'ready'
                        self.update_module_status('visualizer', "✅ Ready", "green")
                        self.metrics['modules_ready'] += 1
                        logger.info(f"Visualizer loaded in {self.module_details['visualizer']['load_time']:.2f}s")
                    except Exception as e:
                        self.module_details['visualizer']['status'] = 'error'
                        self.module_details['visualizer']['error'] = str(e)
                        logger.error(f"Visualizer initialization failed: {e}")
                        self.update_module_status('visualizer', "❌ Error", "red")
                        self.metrics['errors_count'] += 1
                else:
                    self.update_module_status('visualizer', "❌ Not Available", "red")
                    self.module_details['visualizer']['status'] = 'missing'
                
                self.update_progress(35)
                
                # 3. Initialiser Face Handler
                self.update_progress_label("Initializing face handler...")
                self.update_module_status('face_handler', "Loading...", "blue")
                
                if modules_imported['face_handler']:
                    face_start = time.time()
                    try:
                        self.face_handler = FaceHandlerV2()
                        self.module_details['face_handler']['load_time'] = time.time() - face_start
                        self.module_details['face_handler']['status'] = 'ready'
                        self.update_module_status('face_handler', "✅ Ready", "green")
                        self.metrics['modules_ready'] += 1
                        logger.info(f"Face handler loaded in {self.module_details['face_handler']['load_time']:.2f}s")
                    except Exception as e:
                        self.module_details['face_handler']['status'] = 'error'
                        self.module_details['face_handler']['error'] = str(e)
                        logger.error(f"Face handler initialization failed: {e}")
                        self.update_module_status('face_handler', "❌ Error", "red")
                        self.metrics['errors_count'] += 1
                else:
                    self.update_module_status('face_handler', "❌ Not Available", "red")
                    self.module_details['face_handler']['status'] = 'missing'
                
                self.update_progress(50)
                
                # 4. Initialiser Gemma Handler (le plus critique et lourd)
                self.update_progress_label("🤖 Loading Gemma 3n local model...")
                self.update_module_status('gemma', "🔄 Loading Gemma 3n...", "orange")
                
                if modules_imported['gemma_handler']:
                    gemma_start = time.time()
                    try:
                        print("\n🤖 INITIALIZING GEMMA 3N LOCAL MODEL")
                        print("=" * 50)
                        print("⏳ This may take 3-5 minutes on first load...")
                        
                        # Créer l'instance
                        self.update_progress_label("Creating Gemma handler instance...")
                        self.gemma_handler = GemmaHandlerV2()
                        
                        self.update_progress(60)
                        
                        # Vérifier la disponibilité du modèle
                        self.update_progress_label("Checking Gemma model availability...")
                        if not self.gemma_handler.model_available:
                            raise Exception("Gemma 3n model files not found in models/gemma-3n/")
                        
                        self.update_progress(65)
                        
                        # Initialisation effective du modèle
                        self.update_progress_label("🚀 Loading Gemma 3n model weights...")
                        print("🔄 Loading model weights from disk...")
                        
                        success = self.gemma_handler.initialize_local_model()
                        
                        self.module_details['gemma']['load_time'] = time.time() - gemma_start
                        self.metrics['gemma_load_time'] = self.module_details['gemma']['load_time']
                        
                        if success:
                            self.module_details['gemma']['status'] = 'ready'
                            self.metrics['gemma_load_success'] = True
                            self.update_module_status('gemma', "✅ Gemma 3n Ready", "green")
                            self.metrics['modules_ready'] += 1
                            
                            print(f"✅ Gemma 3n loaded successfully in {self.module_details['gemma']['load_time']:.1f}s")
                            logger.info(f"Gemma 3n loaded successfully in {self.module_details['gemma']['load_time']:.1f}s")
                            
                            # Test rapide du modèle
                            self.update_progress_label("Testing Gemma 3n functionality...")
                            if self.gemma_handler.is_ready():
                                print("✅ Gemma 3n ready for medical analysis")
                                self.update_module_status('gemma', "✅ Gemma 3n Tested & Ready", "green")
                            else:
                                print("⚠️ Gemma 3n loaded but not fully ready")
                                self.update_module_status('gemma', "⚠️ Gemma 3n Partial", "orange")
                        else:
                            self.module_details['gemma']['status'] = 'failed'
                            self.module_details['gemma']['error'] = 'Model initialization returned False'
                            self.update_module_status('gemma', "❌ Gemma Failed", "red")
                            self.metrics['errors_count'] += 1
                            
                            print(f"❌ Gemma 3n initialization failed after {self.module_details['gemma']['load_time']:.1f}s")
                            logger.error("Gemma 3n initialization returned False")
                            
                    except Exception as e:
                        self.module_details['gemma']['status'] = 'error'
                        self.module_details['gemma']['error'] = str(e)
                        self.module_details['gemma']['load_time'] = time.time() - gemma_start
                        
                        print(f"❌ Gemma 3n initialization failed: {e}")
                        print(f"⏱️ Failed after {self.module_details['gemma']['load_time']:.1f}s")
                        logger.error(f"Gemma initialization failed: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        self.update_module_status('gemma', "❌ Gemma Error", "red")
                        self.metrics['errors_count'] += 1
                        
                        # Afficher des suggestions détaillées
                        print("\n🔧 TROUBLESHOOTING GEMMA 3N:")
                        print("1. Check if model files exist in models/gemma-3n/")
                        print("2. Verify sufficient RAM (8GB+) and GPU memory (4GB+)")
                        print("3. Try CPU-only mode if GPU memory insufficient")
                        print("4. Check logs in retinoblastogamma.log for details")
                        
                else:
                    self.update_module_status('gemma', "❌ Gemma Not Available", "red")
                    self.module_details['gemma']['status'] = 'missing'
                    print("❌ GemmaHandlerV2 not imported - module file missing")
                
                self.update_progress(85)
                print("\n" + "=" * 50)
                
                # Déterminer le statut final
                ready_modules = self.metrics['modules_ready']
                
                if ready_modules >= 3:  # Au moins 3 modules marchent
                    self.update_status("✅ System ready! Most modules operational", "green")
                    self.analyze_button.config(state='normal')
                    final_msg = "🚀 SYSTEM READY FOR ANALYSIS"
                elif ready_modules >= 2:  # Au moins 2 modules marchent
                    self.update_status("⚠️ Partial functionality - some modules failed", "orange")
                    self.analyze_button.config(state='normal')
                    final_msg = "⚠️ PARTIAL FUNCTIONALITY AVAILABLE"
                elif ready_modules >= 1:  # Au moins 1 module marche
                    self.update_status("⚠️ Limited functionality - many modules failed", "orange")
                    self.analyze_button.config(state='normal')
                    final_msg = "⚠️ LIMITED FUNCTIONALITY"
                else:
                    self.update_status("❌ System error - no modules available", "red")
                    final_msg = "❌ SYSTEM NOT OPERATIONAL"
                    messagebox.showerror("Critical Error", 
                        "No core modules could be initialized.\n\n"
                        "Please check:\n"
                        "• Python dependencies (pip install -r requirements.txt)\n"
                        "• Module files in core/ directory\n"
                        "• Gemma 3n model in models/gemma-3n/\n"
                        "• System logs for detailed errors")
                
                self.update_progress(100)
                self.update_progress_label("Initialization complete")
                self.update_metrics_display()
                
                print(f"\n{final_msg}")
                print(f"📊 Modules ready: {ready_modules}/4")
                print(f"⚠️ Errors: {self.metrics['errors_count']}")
                if self.metrics['gemma_load_success']:
                    print(f"🤖 Gemma 3n load time: {self.metrics['gemma_load_time']:.1f}s")
                print("=" * 50)
                
            except Exception as e:
                logger.error(f"Module initialization failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.update_status(f"❌ Initialization failed: {e}", "red")
                self.update_progress_label("Initialization failed")
                self.metrics['errors_count'] += 1
                
                print(f"❌ CRITICAL ERROR: {e}")
                print("Check retinoblastogamma.log for detailed error information")
        
        # Lancer en arrière-plan
        threading.Thread(target=init_thread, daemon=True).start()
    
    def update_status(self, message: str, color: str = 'black'):
        """Met à jour la barre de statut principale en bas de la fenêtre."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message, foreground=color)
            # Pas besoin de log ici pour ne pas surcharger, le statut est visuel
        
    def update_progress(self, value: int):
        """Met à jour la valeur de la barre de progression (0-100)."""
        if hasattr(self, 'progress'):
            self.progress['value'] = value
    
    def update_progress_label(self, text: str):
        """Met à jour le label de texte sous la barre de progression."""
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=text)

    def update_module_status(self, module_key: str, text: str, color: str):
        """Met à jour le label de statut pour un module spécifique."""
        if hasattr(self, 'module_status') and module_key in self.module_status:
            self.module_status[module_key].config(text=text, foreground=color)

    def update_metrics_display(self):
        """Met à jour l'affichage des métriques de la session."""
        if not hasattr(self, 'metrics_label'):
            return
        
        try:
            avg_time = 0
            if self.metrics['processing_times']:
                avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            
            metrics_text = (
                f"Analyses: {self.metrics['total_analyses']} | "
                f"Positives: {self.metrics['positive_detections']}\n"
                f"Temps moyen: {avg_time:.2f}s | "
                f"Erreurs: {self.metrics['errors_count']}"
            )
            self.metrics_label.config(text=metrics_text)
        except Exception as e:
            logger.warning(f"Impossible de mettre à jour l'affichage des métriques : {e}")
            self.metrics_label.config(text="Erreur de mise à jour des métriques.")
    
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
                
                # Réinitialiser les résultats précédents
                self.current_results = None
                self.current_face_id = None
                
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
        """Lance l'analyse de rétinoblastome avec tous les modules"""
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
            "This will analyze the image for signs of leukocoria using:\n"
            "• Local Gemma 3n AI model\n"
            "• Computer vision eye detection\n"
            "• Face tracking (if enabled)\n\n"
            "Analysis may take 30-90 seconds.")
        
        if not result:
            return
        
        def analysis_thread():
            self.processing = True
            analysis_success = False
            
            try:
                start_time = time.time()
                self.update_status("🔄 Starting comprehensive retinoblastoma analysis...", "blue")
                self.update_progress(0)
                self.update_progress_label("Preparing analysis...")
                
                # === ÉTAPE 1: DÉTECTION DES YEUX/VISAGES ===
                self.update_progress(15)
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
                        "Tips for better detection:\n"
                        "• Ensure the image shows clear eye(s)\n"
                        "• Check image quality and lighting\n"
                        "• Try with a frontal face image\n"
                        "• For cropped eye images, ensure good contrast")
                    return
                
                logger.info(f"Detected {detection_results['total_regions']} eye regions")
                self.update_progress(35)
                
                # === ÉTAPE 2: SUIVI FACIAL (SI ACTIVÉ) ===
                face_tracking_results = None
                if self.face_handler and self.face_tracking_var.get():
                    self.update_progress_label("Processing face tracking...")
                    try:
                        face_tracking_results = self.face_handler.process_faces(
                            self.current_image_path, detection_results
                        )
                        
                        # Extraire l'ID du visage principal
                        if face_tracking_results and face_tracking_results.get('face_mappings'):
                            self.current_face_id = list(face_tracking_results['face_mappings'].values())[0]
                            logger.info(f"Face tracking: {self.current_face_id}")
                        
                    except Exception as e:
                        logger.error(f"Face tracking failed: {e}")
                        face_tracking_results = None
                
                self.update_progress(55)
                
                # === ÉTAPE 3: ANALYSE AVEC GEMMA 3N ===
                self.update_progress_label("Running AI analysis with Gemma 3n...")
                
                analysis_results = {}
                if self.gemma_handler and self.gemma_handler.is_ready():
                    try:
                        analysis_results = self.gemma_handler.analyze_eye_regions(
                            detection_results['regions'],
                            confidence_threshold=self.confidence_var.get()
                        )
                        logger.info("Gemma 3n analysis completed successfully")
                    except Exception as e:
                        logger.error(f"Gemma analysis failed: {e}")
                        analysis_results = self._fallback_analysis(detection_results)
                else:
                    # Mode fallback sans Gemma
                    logger.warning("Gemma not ready, using fallback analysis")
                    analysis_results = self._fallback_analysis(detection_results)
                
                self.update_progress(75)
                
                # === ÉTAPE 4: AJUSTEMENT AVEC HISTORIQUE ===
                if self.face_handler and self.current_face_id and face_tracking_results:
                    self.update_progress_label("Adjusting confidence with patient history...")
                    try:
                        # Ajouter cette analyse à l'historique
                        self.face_handler.add_analysis_result(
                            self.current_face_id, analysis_results, self.current_image_path
                        )
                        
                        # Ajuster la confiance basée sur l'historique
                        adjusted_results = self.face_handler.adjust_confidence_with_history(
                            self.current_face_id, analysis_results.get('results', [])
                        )
                        
                        if adjusted_results != analysis_results.get('results', []):
                            analysis_results['results'] = adjusted_results
                            analysis_results['history_adjusted'] = True
                            logger.info("Confidence adjusted based on patient history")
                        
                    except Exception as e:
                        logger.error(f"History adjustment failed: {e}")
                
                self.update_progress(85)
                
                # === ÉTAPE 5: VISUALISATION ===
                annotated_image = None
                if self.visualizer:
                    self.update_progress_label("Generating visual results...")
                    try:
                        annotated_image = self.visualizer.create_annotated_image(
                            self.current_image_path,
                            detection_results,
                            analysis_results,
                            face_tracking_results
                        )
                        
                        if annotated_image:
                            self.display_annotated_image(annotated_image)
                            logger.info("Annotated image created successfully")
                        
                    except Exception as e:
                        logger.error(f"Visualization failed: {e}")
                
                self.update_progress(95)
                
                # === ÉTAPE 6: COMPILATION DES RÉSULTATS ===
                self.update_progress_label("Compiling medical report...")
                self.compile_and_display_results(
                    detection_results, analysis_results, face_tracking_results
                )
                
                # Mettre à jour l'historique patient
                self.update_patient_history()
                
                # === FINALISATION ===
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
                        f"3. Do NOT delay seeking professional evaluation\n"
                        f"4. Use 'Medical Report (HTML)' for complete documentation")
                else:
                    self.update_status(f"✅ Analysis complete: No concerning findings ({processing_time:.1f}s)", "green")
                    
                    messagebox.showinfo("Analysis Complete", 
                        f"✅ Analysis completed successfully!\n\n"
                        f"No signs of leukocoria were detected.\n"
                        f"Continue regular eye health monitoring.\n\n"
                        f"Processing time: {processing_time:.1f} seconds\n"
                        f"Modules used: {self.metrics['modules_ready']}/4")
                
                self.update_progress(100)
                self.update_progress_label("Analysis complete")
                self.update_metrics_display()
                analysis_success = True
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.update_status(f"❌ Analysis failed: {e}", "red")
                self.update_progress(0)
                self.update_progress_label("Analysis failed")
                self.metrics['errors_count'] += 1
                
                messagebox.showerror("Analysis Error", 
                    f"Analysis failed with error:\n{e}\n\n"
                    f"Possible solutions:\n"
                    f"• Check that modules are properly initialized\n"
                    f"• Verify image format and quality\n"
                    f"• Try restarting the application\n"
                    f"• Check system logs for detailed information")
            
            finally:
                self.processing = False
                if analysis_success:
                    logger.info("Analysis completed successfully")
        
        # Lancer l'analyse en arrière-plan
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def _fallback_analysis(self, detection_results):
        """Analyse de fallback sans Gemma"""
        logger.warning("Using fallback analysis - limited functionality")
        
        return {
            'regions_analyzed': len(detection_results.get('regions', [])),
            'method': 'computer_vision_fallback',
            'results': [
                {
                    'region_id': i,
                    'region_type': region.get('type', 'unknown'),
                    'leukocoria_detected': False,
                    'confidence': 15,  # Très faible confiance
                    'risk_level': 'unknown',
                    'medical_reasoning': 'Fallback analysis only - Gemma 3n not available',
                    'recommendations': 'Professional medical evaluation strongly recommended',
                    'urgency': 'soon',
                    'analysis_method': 'computer_vision_fallback'
                }
                for i, region in enumerate(detection_results.get('regions', []))
            ],
            'fallback_mode': True
        }
    
    def _count_positive_detections(self, analysis_results):
        """Compte les détections positives"""
        if not analysis_results or 'results' not in analysis_results:
            return 0
        
        return sum(1 for result in analysis_results['results'] 
                  if result.get('leukocoria_detected', False))
    
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

PATIENT INFORMATION:
{'='*25}"""
        
        if self.current_face_id:
            report += f"\nPatient ID: {self.current_face_id}"
            if self.face_handler:
                try:
                    face_summary = self.face_handler.get_face_analysis_summary(self.current_face_id)
                    if face_summary and 'error' not in face_summary:
                        report += f"\nTotal analyses: {face_summary.get('total_analyses', 0)}"
                        report += f"\nPositive analyses: {face_summary.get('positive_analyses', 0)}"
                        report += f"\nFirst seen: {face_summary.get('first_seen', 'Unknown')[:10]}"
                        report += f"\nRecommendation: {face_summary.get('recommendation', 'Unknown')}"
                except Exception as e:
                    logger.error(f"Error getting face summary: {e}")
        else:
            report += "\nPatient ID: Not tracked (face tracking disabled or unavailable)"
        
        # Statistiques de détection
        total_regions = detection_results.get('total_regions', 0)
        analysis_method = analysis_results.get('method', 'unknown')
        regions_analyzed = analysis_results.get('regions_analyzed', 0)
        
        report += f"\n\nDETECTION SUMMARY:\n{'='*25}"
        report += f"\nRegions detected: {total_regions}"
        report += f"\nRegions analyzed: {regions_analyzed}"
        report += f"\nAnalysis method: {analysis_method}"
        
        if analysis_results.get('history_adjusted'):
            report += f"\n🔄 Confidence adjusted based on patient history"
        
        # Résultats d'analyse
        positive_count = self._count_positive_detections(analysis_results)
        
        if positive_count > 0:
            report += f"\n\n🚨 MEDICAL ALERT: POSSIBLE RETINOBLASTOMA DETECTED\n"
            report += f"Positive findings: {positive_count}\n"
            report += f"IMMEDIATE PEDIATRIC OPHTHALMOLOGICAL CONSULTATION REQUIRED\n"
        else:
            report += f"\n\n✅ No concerning findings detected\n"
            report += f"Continue regular pediatric eye monitoring\n"
        
        # Détails par région
        report += f"\nDETAILED ANALYSIS BY REGION:\n{'='*40}"
        
        for i, result in enumerate(analysis_results.get('results', []), 1):
            region_type = result.get('region_type', 'unknown')
            detected = result.get('leukocoria_detected', False)
            confidence = result.get('confidence', 0)
            risk_level = result.get('risk_level', 'unknown')
            
            report += f"\n\n--- Region {i}: {region_type.upper()} ---"
            report += f"\nLeukocoria detected: {'⚠️ YES' if detected else '✅ NO'}"
            report += f"\nConfidence level: {confidence:.1f}%"
            report += f"\nRisk assessment: {risk_level.upper()}"
            
            if detected:
                analysis_method = result.get('analysis_method', 'unknown')
                report += f"\nAnalysis method: {analysis_method}"
                reasoning = result.get('medical_reasoning', 'No detailed reasoning available')
                report += f"\nMedical reasoning: {reasoning[:200]}..."
            
            if result.get('confidence_adjusted'):
                original_conf = result.get('original_confidence', confidence)
                report += f"\n🔄 Confidence adjusted from {original_conf:.1f}%"
        
        # Informations techniques
        report += f"\n\nTECHNICAL DETAILS:\n{'='*25}"
        report += f"\nAI Model: Gemma 3n (Local - 100% Offline)"
        report += f"\nPrivacy: Complete - No data transmitted"
        report += f"\nModules active: {self.metrics['modules_ready']}/4"
        
        # Suivi facial
        if face_tracking_results:
            tracked_faces = face_tracking_results.get('tracked_faces', 0)
            new_faces = face_tracking_results.get('new_faces', 0)
            recognized_faces = face_tracking_results.get('recognized_faces', 0)
            
            report += f"\nFace tracking: {tracked_faces} faces processed"
            report += f"\nNew faces: {new_faces}, Recognized: {recognized_faces}"
        
        # Métriques de performance
        if self.metrics['processing_times']:
            avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            report += f"\nAverage processing time: {avg_time:.1f}s"
        
        # Disclaimer médical complet
        report += f"\n\nCRITICAL MEDICAL DISCLAIMER:\n{'='*40}"
        report += f"\n⚠️ IMPORTANT: This analysis is provided by an AI screening system."
        report += f"\nThis is NOT a medical diagnosis and should NOT replace professional"
        report += f"\nmedical evaluation by qualified pediatric ophthalmologists."
        
        if positive_count > 0:
            report += f"\n\n🚨 IMMEDIATE ACTION REQUIRED FOR POSITIVE FINDINGS:"
            report += f"\n1. ⏰ Contact pediatric ophthalmologist IMMEDIATELY"
            report += f"\n2. 📋 Bring this report and original images to appointment"
            report += f"\n3. 🚫 Do NOT delay seeking professional medical evaluation"
            report += f"\n4. 📞 Emergency: Call your healthcare provider"
        else:
            report += f"\n\n✅ ROUTINE MONITORING FOR NEGATIVE FINDINGS:"
            report += f"\n1. 📅 Continue regular pediatric eye examinations"
            report += f"\n2. 📸 Take monthly photos under good lighting"
            report += f"\n3. 👀 Watch for any changes in pupil appearance"
            report += f"\n4. 🔄 Repeat screening if concerns arise"
        
        report += f"\n\nRetinoblastoma facts:"
        report += f"\n- Most common eye cancer in children (under 6 years)"
        report += f"\n- 95% survival rate with EARLY detection and treatment"
        report += f"\n- Main early sign: White pupil reflex (leukocoria) in photos"
        report += f"\n- Can affect one or both eyes"
        report += f"\n- Requires immediate medical attention when suspected"
        
        # Afficher dans l'onglet résultats
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        
        # Basculer vers l'onglet résultats
        self.notebook.select(self.results_frame)
        
        # Sauvegarder les résultats
        self.current_results = report
    
    def update_patient_history(self):
        """Met à jour l'affichage de l'historique patient"""
        try:
            if not self.face_handler or not self.current_face_id:
                self.history_text.delete(1.0, tk.END)
                self.history_text.insert(1.0, "No patient tracking available.\n\nFace tracking is disabled or not initialized.")
                return
            
            # Obtenir le résumé du patient
            face_summary = self.face_handler.get_face_analysis_summary(self.current_face_id)
            
            if not face_summary or 'error' in face_summary:
                self.history_text.delete(1.0, tk.END)
                self.history_text.insert(1.0, "No patient history available.")
                return
            history_report = f"""PATIENT HISTORY REPORT
{'='*50}
Patient ID: {face_summary.get('face_id', 'Unknown')}
First seen: {face_summary.get('first_seen', 'Unknown')[:19]}
Last seen: {face_summary.get('last_seen', 'Unknown')[:19]}
Total encounters: {face_summary.get('seen_count', 0)}

ANALYSIS SUMMARY:
{'='*25}
Total analyses: {face_summary.get('total_analyses', 0)}
Positive analyses: {face_summary.get('positive_analyses', 0)}
Consistency rate: {face_summary.get('consistency_rate', 0):.1f}%

MEDICAL RECOMMENDATION:
{'='*30}
{face_summary.get('recommendation', 'No recommendation available')}

Urgency level: {face_summary.get('urgency', 'routine').upper()}

RECENT ANALYSES:
{'='*20}"""
            
            recent_analyses = face_summary.get('recent_analyses', [])
            if recent_analyses:
                for i, analysis in enumerate(recent_analyses[-5:], 1):
                    timestamp = analysis.get('timestamp', 'Unknown')[:19]
                    has_positive = analysis.get('has_positive_findings', False)
                    analysis_summary = analysis.get('analysis_summary', {})
                    
                    history_report += f"\n\n{i}. Analysis on {timestamp}"
                    history_report += f"\n   Result: {'🚨 POSITIVE' if has_positive else '✅ NEGATIVE'}"
                    history_report += f"\n   Regions: {analysis_summary.get('regions_analyzed', 0)}"
                    history_report += f"\n   Method: {analysis_summary.get('method', 'unknown')}"
            else:
                history_report += "\n\nNo previous analyses recorded."
            
            # Afficher l'historique
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(1.0, history_report)
            
        except Exception as e:
            logger.error(f"Error updating patient history: {e}")
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(1.0, f"Error loading patient history: {e}")
            

    def generate_medical_report(self):
        """Génère un rapport médical HTML professionnel avec le nouveau module"""
        if not self.current_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        try:
            # Importer le module de rapports médicaux
            try:
                from core.medical_reports import MedicalReportsGenerator
                reports_generator = MedicalReportsGenerator()
            except ImportError as e:
                logger.error(f"Cannot import medical reports module: {e}")
                # Fallback vers la méthode originale
                self._generate_basic_html_report()
                return
            
            # Rassembler les informations pour le rapport
            patient_summary = None
            if self.current_face_id and self.face_handler:
                try:
                    patient_summary = self.face_handler.get_face_analysis_summary(self.current_face_id)
                    if 'error' in patient_summary:
                        patient_summary = None
                except Exception as e:
                    logger.error(f"Error getting patient summary: {e}")
                    patient_summary = None
            
            # Générer le rapport complet
            html_report = reports_generator.generate_comprehensive_report(
                analysis_results=self.current_results,
                image_path=self.current_image_path,
                patient_summary=patient_summary,
                metrics=self.metrics,
                face_tracking_results=getattr(self, 'last_face_tracking_results', None)
            )
            
            # Sauvegarder le rapport
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = RESULTS_DIR / f"retinoblastoma_comprehensive_report_{timestamp}.html"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Ouvrir dans le navigateur
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
            
            self.update_status(f"✅ Comprehensive medical report generated: {report_path.name}", "green")
            messagebox.showinfo("Report Generated", 
                f"🏥 Rapport médical complet généré avec succès!\n\n"
                f"📄 Fichier: {report_path.name}\n"
                f"🌐 Ouvert dans le navigateur web\n\n"
                f"Ce rapport professionnel inclut:\n"
                f"• Analyse détaillée avec visualisations\n"
                f"• Historique patient (si disponible)\n"
                f"• Recommandations médicales personnalisées\n"
                f"• Détails techniques et métriques\n"
                f"• Format professionnel pour usage médical")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            self.update_status(f"❌ Report generation failed: {e}", "red")
            messagebox.showerror("Report Error", f"Failed to generate medical report:\n{e}")
    
    def _generate_basic_html_report(self):
        """Génère un rapport HTML basique en cas de fallback"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = RESULTS_DIR / f"retinoblastoma_basic_report_{timestamp}.html"
            
            # Version simplifiée du rapport
            filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
            has_positive = "MEDICAL ALERT" in self.current_results if self.current_results else False
            
            html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Médical Retinoblastoma - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #667eea; color: white; padding: 20px; text-align: center; }}
        .alert {{ padding: 20px; margin: 20px 0; border-radius: 10px; }}
        .alert-critical {{ background: #ffebee; border-left: 5px solid #f44336; }}
        .alert-safe {{ background: #e8f5e8; border-left: 5px solid #4caf50; }}
        .content {{ padding: 20px; }}
        pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Rapport Médical d'Analyse Retinoblastoma</h1>
        <p>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Image: {filename}</p>
    </div>
    
    <div class="content">
        <div class="alert {'alert-critical' if has_positive else 'alert-safe'}">
            <h2>{'🚨 ALERTE MÉDICALE' if has_positive else '✅ AUCUN SIGNE PRÉOCCUPANT'}</h2>
            <p>{'Consultation ophtalmologique IMMÉDIATE recommandée' if has_positive else 'Surveillance de routine recommandée'}</p>
        </div>
        
        <h3>📊 Résultats Détaillés</h3>
        <pre>{self.current_results}</pre>
        
        <div class="alert alert-critical">
            <h3>⚕️ Avertissement Médical</h3>
            <p><strong>IMPORTANT:</strong> Ce rapport est généré par un système d'IA et ne constitue pas un diagnostic médical.</p>
            <p>Consultez toujours un professionnel de santé qualifié.</p>
        </div>
    </div>
</body>
</html>"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
            
            self.update_status(f"✅ Basic medical report generated: {report_path.name}", "green")
            
        except Exception as e:
            logger.error(f"Error generating basic report: {e}")
    
    def generate_recommendations(self):
        """Génère des recommandations médicales intelligentes"""
        if not self.current_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        try:
            # Importer le module de recommandations
            try:
                from core.medical_recommendations import MedicalRecommendationsEngine
                rec_engine = MedicalRecommendationsEngine()
            except ImportError as e:
                logger.error(f"Cannot import recommendations module: {e}")
                messagebox.showinfo("Recommendations", 
                    "Le module de recommandations n'est pas disponible.\n"
                    "Consultez le rapport médical pour les recommandations de base.")
                return
            
            # Rassembler les données pour les recommandations
            analysis_data = {
                'results': getattr(self, 'last_analysis_results', {}).get('results', []),
                'method': getattr(self, 'last_analysis_results', {}).get('method', 'unknown')
            }
            
            patient_history = None
            if self.current_face_id and self.face_handler:
                try:
                    patient_history = self.face_handler.get_face_analysis_summary(self.current_face_id)
                    if 'error' in patient_history:
                        patient_history = None
                except Exception as e:
                    logger.error(f"Error getting patient history: {e}")
            
            # Générer les recommandations
            recommendation = rec_engine.generate_recommendations(
                analysis_results=analysis_data,
                patient_history=patient_history
            )
            
            # Créer une fenêtre pour afficher les recommandations
            self._show_recommendations_window(recommendation, rec_engine)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            messagebox.showerror("Error", f"Failed to generate recommendations:\n{e}")
    
    def _show_recommendations_window(self, recommendation, rec_engine):
        """Affiche les recommandations dans une fenêtre dédiée"""
        try:
            rec_window = tk.Toplevel(self.root)
            rec_window.title("Recommandations Médicales Intelligentes")
            rec_window.geometry("900x700")
            rec_window.transient(self.root)
            
            # Frame principal
            main_frame = ttk.Frame(rec_window, padding="15")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Titre avec niveau d'urgence
            urgency_colors = {
                'immediate': 'red',
                'urgent': 'orange', 
                'soon': 'blue',
                'routine': 'green'
            }
            
            urgency_symbols = {
                'immediate': '🚨',
                'urgent': '⚠️',
                'soon': '💡', 
                'routine': '✅'
            }
            
            symbol = urgency_symbols.get(recommendation.urgency_level, '❓')
            color = urgency_colors.get(recommendation.urgency_level, 'black')
            
            title_label = ttk.Label(main_frame, 
                text=f"{symbol} Recommandations Médicales - Niveau {recommendation.urgency_level.upper()}", 
                font=("Arial", 16, "bold"))
            title_label.pack(pady=(0, 15))
            
            # Zone de texte avec scrollbar
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            rec_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 11))
            rec_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=rec_text.yview)
            rec_text.configure(yscrollcommand=rec_scrollbar.set)
            
            rec_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            rec_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Formater et afficher les recommandations
            formatted_rec = rec_engine.format_recommendation_for_display(recommendation)
            rec_text.insert(1.0, formatted_rec)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'affichage de la fenêtre des recommandations: {e}")
            messagebox.showerror("Erreur d'affichage", f"Impossible d'afficher les recommandations:\n{e}")

    def show_system_info(self):
        """Affiche les informations système détaillées dans une nouvelle fenêtre."""
        logger.info("Affichage des informations système demandé.")
        try:
            import sys
            import torch

            info_report = "📊 INFORMATIONS SYSTÈME\n"
            info_report += "="*45 + "\n\n"

            # --- Environnement ---
            info_report += "--- Environnement d'Exécution ---\n"
            info_report += f"Version Python : {sys.version.split()[0]}\n"
            info_report += f"Version PyTorch : {torch.__version__}\n"
            if torch.cuda.is_available():
                info_report += f"GPU : {torch.cuda.get_device_name(0)}\n"
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info_report += f"Mémoire GPU : {gpu_mem:.2f} GB\n"
            else:
                info_report += "GPU : Non disponible (Mode CPU)\n"
            info_report += "\n"

            # --- Statut des Modules ---
            info_report += "--- Statut des Modules ---\n"
            for name, details in self.module_details.items():
                status = details.get('status', 'inconnu').upper()
                load_time = details.get('load_time', 0)
                error = details.get('error', None)
                info_report += f"▶ Module : {name.replace('_', ' ').title()}\n"
                info_report += f"  - Statut : {status}\n"
                if load_time > 0:
                    info_report += f"  - Temps de chargement : {load_time:.2f}s\n"
                if error:
                    info_report += f"  - Erreur : {str(error)[:120]}...\n"
            info_report += "\n"

            # --- Informations sur le modèle Gemma ---
            info_report += "--- Modèle Gemma 3n ---\n"
            gemma_info = get_model_info()
            availability = gemma_info.get('availability', {})
            info_report += f"Fichiers du modèle : {availability.get('model_files_count', 0)} ({availability.get('total_model_size_gb', 0):.2f} GB)\n"
            info_report += f"Chargement réussi : {'Oui' if self.metrics.get('gemma_load_success') else 'Non'}\n"
            if self.metrics.get('gemma_load_time', 0) > 0:
                info_report += f"Temps de chargement Gemma : {self.metrics.get('gemma_load_time'):.2f}s\n"
            info_report += "\n"

            # --- Métriques de la session ---
            info_report += "--- Métriques de la Session ---\n"
            info_report += f"Analyses totales : {self.metrics.get('total_analyses', 0)}\n"
            info_report += f"Détections positives : {self.metrics.get('positive_detections', 0)}\n"
            if self.metrics.get('processing_times'):
                avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
                info_report += f"Temps de traitement moyen : {avg_time:.2f}s\n"

            # Création de la fenêtre d'information
            info_window = tk.Toplevel(self.root)
            info_window.title("Informations Système")
            info_window.geometry("750x600")
            info_window.transient(self.root)
            
            text_frame = ttk.Frame(info_window, padding="10")
            text_frame.pack(fill=tk.BOTH, expand=True)

            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), relief=tk.SUNKEN, bd=2)
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            
            text_frame.rowconfigure(0, weight=1)
            text_frame.columnconfigure(0, weight=1)

            text_widget.insert(tk.END, info_report)
            text_widget.config(state='disabled')

        except Exception as e:
            logger.error(f"Échec de l'affichage des informations système : {e}")
            messagebox.showerror("Erreur", f"Impossible de récupérer les informations système :\n{e}")

    def show_face_tracking_summary(self):
        """Affiche un résumé de tous les visages suivis dans une nouvelle fenêtre."""
        logger.info("Affichage du résumé du suivi facial demandé.")
        
        # Vérifier si le gestionnaire de visages est disponible
        if not hasattr(self, 'face_handler') or self.face_handler is None:
            messagebox.showinfo("Information", "Le module de suivi facial n'est pas encore initialisé.")
            return

        try:
            # Récupérer les données depuis le FaceHandlerV2
            all_faces = self.face_handler.get_all_tracked_faces()
            stats = self.face_handler.get_statistics()

            if not stats or stats.get('total_known_faces', 0) == 0:
                messagebox.showinfo("Résumé du Suivi Facial", "Aucun visage n'a encore été suivi.")
                return

            # Créer le rapport textuel
            summary_report = "📊 RÉSUMÉ DU SUIVI FACIAL\n"
            summary_report += "="*50 + "\n\n"
            
            summary_report += "STATISTIQUES GLOBALES :\n"
            summary_report += f"  - Total de visages connus : {stats.get('total_known_faces', 0)}\n"
            summary_report += f"  - Visages reconnus (session) : {stats.get('faces_recognized', 0)}\n"
            summary_report += f"  - Nouveaux visages (session) : {stats.get('new_faces_registered', 0)}\n"
            summary_report += f"  - Boosts de confiance appliqués : {stats.get('confidence_boosts_applied', 0)}\n\n"
            
            summary_report += "RÉSUMÉS INDIVIDUELS :\n"
            summary_report += "-"*50 + "\n\n"

            if not all_faces:
                summary_report += "Aucun résumé individuel à afficher.\n"
            else:
                # Trier les visages par urgence
                for i, face_summary in enumerate(all_faces):
                    summary_report += f"👤 Patient ID : {face_summary.get('face_id', 'N/A')}\n"
                    summary_report += f"   - Niveau d'urgence : {face_summary.get('urgency', 'routine').upper()}\n"
                    summary_report += f"   - Analyses totales : {face_summary.get('total_analyses', 0)}\n"
                    summary_report += f"   - Analyses positives : {face_summary.get('positive_analyses', 0)}\n"
                    summary_report += f"   - Recommandation : {face_summary.get('recommendation', 'N/A')}\n\n"

            # Créer une nouvelle fenêtre pour afficher le résumé
            summary_window = tk.Toplevel(self.root)
            summary_window.title("Résumé du Suivi Facial")
            summary_window.geometry("800x600")
            summary_window.transient(self.root)

            text_frame = ttk.Frame(summary_window, padding="10")
            text_frame.pack(fill=tk.BOTH, expand=True)

            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), relief=tk.SUNKEN, bd=2)
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            
            text_frame.rowconfigure(0, weight=1)
            text_frame.columnconfigure(0, weight=1)

            text_widget.insert(tk.END, summary_report)
            text_widget.config(state='disabled') # Lecture seule

        except Exception as e:
            logger.error(f"Échec de l'affichage du résumé du suivi facial : {e}")
            messagebox.showerror("Erreur", f"Impossible de récupérer le résumé du suivi facial :\n{e}")

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
                if file_path.endswith('.json'):
                    # Export JSON structuré
                    json_data = {
                        'timestamp': datetime.now().isoformat(),
                        'image_path': self.current_image_path,
                        'patient_id': self.current_face_id,
                        'report_text': self.current_results,
                        'system_info': {
                            'version': 'RetinoblastoGemma v6',
                            'modules_ready': self.metrics['modules_ready'],
                            'total_analyses': self.metrics['total_analyses']
                        }
                    }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                else:
                    # Export texte simple
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
            
def main():
    """Fonction principale avec gestion d'erreurs robuste et feedback détaillé"""
    try:
        # Banner de démarrage
        print("🏥 RETINOBLASTOGAMMA v6 - HACKATHON GOOGLE GEMMA")
        print("="*70)
        print("🏆 Détection Précoce du Rétinoblastome avec Gemma 3n Local")
        print("🔒 100% Traitement Local - Respect de la Vie Privée")
        print("🎯 Mission: Sauver des vies d'enfants par la détection précoce IA")
        print("="*70)
        
        # Vérifications préliminaires
        print("\n🔍 Vérifications Système:")
        
        # Créer les dossiers nécessaires
        try:
            if CONFIG_AVAILABLE:
                ensure_directories()
            else:
                for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
                    directory.mkdir(exist_ok=True)
            print("✅ Dossiers créés/vérifiés")
        except Exception as e:
            print(f"⚠️ Avertissement création dossiers: {e}")
            # Créer manuellement les dossiers essentiels
            for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
                try:
                    directory.mkdir(exist_ok=True)
                except:
                    pass
        
        # Vérifications système
        print(f"✅ Version Python: {sys.version.split()[0]}")
        
        # Vérification PyTorch/GPU
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                if gpu_memory < 4:
                    print("⚠️ GPU mémoire limitée - performance réduite possible")
            else:
                print("💻 Mode CPU (GPU non disponible - plus lent mais fonctionnel)")
        except ImportError:
            print("⚠️ PyTorch non disponible - certaines fonctionnalités limitées")
        
        # Vérification des modules importés
        imported_count = sum(modules_imported.values())
        print(f"📦 Modules core importés: {imported_count}/4")
        
        for module, imported in modules_imported.items():
            status = "✅" if imported else "❌"
            print(f"   {status} {module.replace('_', ' ').title()}")
        
        if imported_count == 0:
            print("\n❌ CRITIQUE: Aucun module core n'a pu être importé!")
            print("Veuillez vérifier:")
            print("• Le dossier core/ existe avec tous les fichiers de modules")
            print("• Les dépendances Python sont installées")
            print("• Le répertoire de travail est correct")
            
            # Essayer quand même de lancer l'interface
            print("\n🔄 Tentative de lancement de l'interface malgré tout...")
        elif imported_count < 4:
            print(f"\n⚠️ Fonctionnalité partielle: {imported_count}/4 modules disponibles")
            print("L'application fonctionnera avec les modules disponibles")
        
        # Vérification du modèle Gemma 3n
        gemma_model_dir = MODELS_DIR / "gemma-3n"
        if gemma_model_dir.exists():
            model_files = list(gemma_model_dir.glob("*.safetensors"))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
                print(f"🤖 Modèle Gemma 3n: {len(model_files)} fichiers ({total_size:.1f} GB)")
                
                if total_size < 8:
                    print("⚠️ Modèle potentiellement incomplet (< 8GB)")
                else:
                    print("✅ Modèle Gemma 3n semble complet")
            else:
                print("❌ Fichiers modèle Gemma 3n manquants dans models/gemma-3n/")
        else:
            print("❌ Dossier modèle Gemma 3n non trouvé")
            print("📥 Téléchargez le modèle Gemma 3n dans models/gemma-3n/")
        
        # Vérification de l'espace disque
        try:
            import shutil
            free_space = shutil.disk_usage(Path.cwd()).free / (1024**3)
            if free_space < 5:
                print(f"⚠️ Espace disque faible: {free_space:.1f}GB disponible")
            else:
                print(f"💾 Espace disque: {free_space:.1f}GB disponible")
        except:
            pass
        
        # Créer et lancer l'application
        print("\n🚀 Lancement de RetinoblastoGemma v6...")
        root = tk.Tk()
        
        try:
            app = RetinoblastoGemmaV6(root)
            logger.info("RetinoblastoGemma v6 démarré avec succès")
            print("✅ Application lancée avec succès!")
            print("💡 L'app va initialiser les modules en arrière-plan")
            print("🎯 Chargez une image et cliquez 'Analyze for Retinoblastoma' pour commencer")
            print("🤖 Surveillez l'initialisation de Gemma 3n dans les logs")
            
        except Exception as e:
            logger.error(f"Échec d'initialisation de l'application: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Erreur d'Initialisation", 
                f"Échec d'initialisation de l'application:\n{e}\n\n"
                "Solutions possibles:\n"
                "• Vérifiez que tous les modules core sont dans le dossier core/\n"
                "• Vérifiez que les dépendances Python sont installées\n"
                "• Exécutez: pip install -r requirements.txt\n"
                "• Consultez les logs système pour des erreurs détaillées")
            return
        
        # Gestionnaire de fermeture propre
        def on_closing():
            try:
                logger.info("Fermeture de l'application...")
                print("\n🔄 Fermeture en cours...")
                
                # Sauvegarder les données de suivi facial
                if hasattr(app, 'face_handler') and app.face_handler:
                    try:
                        app.face_handler.save_data()
                        print("💾 Données de suivi facial sauvegardées")
                    except Exception as e:
                        logger.error(f"Erreur sauvegarde données faciales: {e}")
                
                # Libérer les ressources Gemma
                if hasattr(app, 'gemma_handler') and app.gemma_handler:
                    try:
                        app.gemma_handler.cleanup()
                        print("🧹 Ressources Gemma nettoyées")
                    except Exception as e:
                        logger.error(f"Erreur nettoyage Gemma: {e}")
                
                # Nettoyer les autres modules
                for module_name in ['eye_detector', 'visualizer']:
                    module = getattr(app, module_name, None)
                    if module and hasattr(module, 'cleanup'):
                        try:
                            module.cleanup()
                            print(f"🧹 {module_name} nettoyé")
                        except Exception as e:
                            logger.error(f"Erreur nettoyage {module_name}: {e}")
                
                print("👋 Merci d'avoir utilisé RetinoblastoGemma v6!")
                print("🏆 Bonne chance pour le Hackathon Google Gemma!")
                print("🩺 Rappel: La détection précoce sauve des vies!")
                
                root.quit()
                root.destroy()
                
            except Exception as e:
                logger.error(f"Erreur durant la fermeture: {e}")
                # Forcer la fermeture
                try:
                    root.quit()
                except:
                    pass
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Message de lancement final
        print("\n" + "="*70)
        print("🏥 RetinoblastoGemma v6 Prêt!")
        print("🎯 Mission: Sauver des vies d'enfants par la détection précoce")
        print("🏆 Hackathon: Google Gemma Worldwide")
        print("🔒 Confidentialité: 100% Traitement Local")
        print("🤖 IA: Gemma 3n Multimodal (Chargement en cours...)")
        print("="*70)
        print("\n💡 INSTRUCTIONS D'UTILISATION:")
        print("1. Attendez l'initialisation des modules (voir statut dans l'app)")
        print("2. Cliquez 'Load Medical Image' pour charger une image")
        print("3. Cliquez 'Analyze for Retinoblastoma' pour lancer l'analyse")
        print("4. Consultez les résultats et recommandations")
        print("5. Générez un rapport médical HTML si nécessaire")
        print("\n⚠️ IMPORTANT: Gemma 3n peut prendre 3-5 minutes à charger")
        print("Surveillez les messages dans la console et l'app!")
        print("="*70)
        
        # Démarrer l'interface
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Application interrompue par l'utilisateur")
        logger.info("Application interrompue par l'utilisateur")
        
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        logger.error(f"Traceback complet: {traceback.format_exc()}")
        print(f"\n❌ Erreur critique: {e}")
        
        print("\n📋 Guide de Dépannage:")
        print("1. Vérifiez l'installation Python (3.8+)")
        print("2. Installez les dépendances: pip install -r requirements.txt")
        print("3. Vérifiez les modules core dans le dossier core/")
        print("4. Vérifiez le modèle Gemma 3n dans models/gemma-3n/")
        print("5. Consultez les logs système dans retinoblastogamma.log")
        print("6. Vérifiez les ressources système (8GB+ RAM recommandés)")
        print("7. Exécutez python check_system.py pour diagnostic complet")
        
        print(f"\n🔧 Commandes de diagnostic:")
        print(f"python check_system.py  # Vérification système complète")
        print(f"pip install -r requirements.txt  # Installation dépendances")
        print(f"dir core\\  # Vérifier modules (Windows)")
        print(f"ls core/   # Vérifier modules (Linux/Mac)")
        
        input("\nAppuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    # Exécuter la fonction principale
    main()    
    
    
