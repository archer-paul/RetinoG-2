"""
Interface moderne Gemini-like pour RetinoblastoGemma v6
Style futuriste en blanc et bleu avec accents Google
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import math

class ModernUI:
    """Interface utilisateur moderne avec style Gemini-like"""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.root = parent_app.root
        
        # Configuration des couleurs - Palette Gemini/Google
        self.colors = {
            'primary_blue': '#1a73e8',      # Bleu Google principal
            'light_blue': '#4285f4',        # Bleu clair
            'darker_blue': '#1557b0',       # Bleu foncé
            'white': '#ffffff',
            'light_gray': '#f8f9fa',        # Gris très clair
            'medium_gray': '#e8eaed',       # Gris moyen
            'dark_gray': '#5f6368',         # Gris foncé
            'text_primary': '#202124',      # Texte principal
            'text_secondary': '#5f6368',    # Texte secondaire
            'success': '#34a853',           # Vert Google
            'warning': '#fbbc04',           # Jaune Google
            'error': '#ea4335',             # Rouge Google
            'purple': '#9c27b0',            # Violet accent
            'gradient_start': '#667eea',    # Début dégradé
            'gradient_end': '#764ba2',      # Fin dégradé
            'card_bg': '#ffffff',
            'hover_blue': '#e8f0fe'
        }
        
        # Configuration des polices
        self.fonts = {
            'title_large': ('Google Sans', 32, 'normal'),
            'title': ('Google Sans', 24, 'normal'),
            'subtitle': ('Google Sans', 18, 'normal'),
            'body': ('Google Sans', 14, 'normal'),
            'small': ('Google Sans', 12, 'normal'),
            'tiny': ('Google Sans', 10, 'normal'),
            'mono': ('Roboto Mono', 11, 'normal')
        }
        
        # Style moderne
        self.style = ttk.Style()
        self.setup_modern_styles()
        
    def setup_modern_styles(self):
        """Configure les styles modernes TTK"""
        # Style pour les boutons principaux
        self.style.configure('Modern.TButton',
                           background=self.colors['primary_blue'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           relief='flat',
                           padding=(20, 12))
        
        self.style.map('Modern.TButton',
                      background=[('active', self.colors['light_blue']),
                                ('pressed', self.colors['darker_blue'])])
        
        # Style pour les boutons secondaires
        self.style.configure('Secondary.TButton',
                           background=self.colors['white'],
                           foreground=self.colors['primary_blue'],
                           borderwidth=1,
                           relief='solid',
                           padding=(16, 10))
        
        # Style pour les cartes
        self.style.configure('Card.TFrame',
                           background=self.colors['card_bg'],
                           relief='flat',
                           borderwidth=0)
        
        # Style pour les labels
        self.style.configure('Title.TLabel',
                           background=self.colors['white'],
                           foreground=self.colors['text_primary'],
                           font=self.fonts['title'])
        
        self.style.configure('Subtitle.TLabel',
                           background=self.colors['white'],
                           foreground=self.colors['text_secondary'],
                           font=self.fonts['subtitle'])
        
        self.style.configure('Body.TLabel',
                           background=self.colors['white'],
                           foreground=self.colors['text_primary'],
                           font=self.fonts['body'])
        
    def create_modern_interface(self):
        """Crée l'interface moderne complète"""
        # Configuration de la fenêtre principale
        self.root.configure(bg=self.colors['light_gray'])
        self.root.title("RetinoblastoGemma v6 - Google Gemma Hackathon")
        
        # Container principal avec padding
        main_container = tk.Frame(self.root, bg=self.colors['light_gray'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header moderne avec dégradé
        self.create_modern_header(main_container)
        
        # Corps principal avec layout en grille
        body_frame = tk.Frame(main_container, bg=self.colors['light_gray'])
        body_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Configuration de la grille
        body_frame.columnconfigure(0, weight=1, minsize=350)  # Panel gauche
        body_frame.columnconfigure(1, weight=2, minsize=600)  # Panel droit
        body_frame.rowconfigure(0, weight=1)
        
        # Panel de contrôle moderne (gauche)
        self.create_control_panel_modern(body_frame)
        
        # Zone d'affichage moderne (droite)
        self.create_display_area_modern(body_frame)
        
        # Footer moderne
        self.create_modern_footer(main_container)
        
    def create_modern_header(self, parent):
        """Crée un header moderne avec dégradé"""
        header_frame = tk.Frame(parent, bg=self.colors['white'], height=120)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Créer un canvas pour le dégradé
        header_canvas = tk.Canvas(header_frame, height=120, bg=self.colors['white'], highlightthickness=0)
        header_canvas.pack(fill=tk.BOTH)
        
        # Dégradé bleu
        self.create_gradient(header_canvas, 0, 0, 1400, 120, 
                           self.colors['primary_blue'], self.colors['light_blue'])
        
        # Logo et titre
        title_frame = tk.Frame(header_canvas, bg=self.colors['primary_blue'])
        title_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.6)
        
        # Icône Google Gemma
        icon_label = tk.Label(title_frame, text="🤖", font=('Arial', 32), 
                             bg=self.colors['primary_blue'], fg='white')
        icon_label.pack(side=tk.LEFT, padx=(20, 10))
        
        # Textes du titre
        text_frame = tk.Frame(title_frame, bg=self.colors['primary_blue'])
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(text_frame, text="RetinoblastoGemma v6", 
                              font=self.fonts['title_large'], 
                              bg=self.colors['primary_blue'], fg='white')
        title_label.pack(anchor=tk.W, pady=(0, 5))
        
        subtitle_label = tk.Label(text_frame, text="🏆 Google Gemma Hackathon • 100% Local AI • Privacy First", 
                                 font=self.fonts['body'], 
                                 bg=self.colors['primary_blue'], fg='white')
        subtitle_label.pack(anchor=tk.W)
        
        # Badge status (côté droit)
        status_frame = tk.Frame(title_frame, bg=self.colors['primary_blue'])
        status_frame.pack(side=tk.RIGHT, padx=(10, 20))
        
        self.status_badge = tk.Label(status_frame, text="🔄 INITIALIZING", 
                                    font=self.fonts['small'],
                                    bg=self.colors['warning'], fg='white',
                                    padx=12, pady=6)
        self.status_badge.pack()
        
    def create_control_panel_modern(self, parent):
        """Crée le panel de contrôle moderne"""
        # Container principal avec coins arrondis simulés
        control_container = tk.Frame(parent, bg=self.colors['light_gray'])
        control_container.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Hackathon Info Card
        self.create_info_card(control_container, "🏆 Hackathon Info", 
                             "Google Gemma Worldwide\n100% Local AI Processing\nPrivacy-First Medical Analysis",
                             self.colors['purple'])
        
        # Image Loading Card  
        self.create_image_loading_card(control_container)
        
        # AI Analysis Card
        self.create_analysis_card(control_container)
        
        # System Status Card
        self.create_system_status_card(control_container)
        
        # Settings Card
        self.create_settings_card(control_container)
        
        # Progress Card
        self.create_progress_card(control_container)
        
        # Actions Card
        self.create_actions_card(control_container)
        
    def create_info_card(self, parent, title, content, accent_color):
        """Crée une carte d'information moderne"""
        card_frame = self.create_card_frame(parent, title, accent_color)
        
        info_text = tk.Text(card_frame, height=3, font=self.fonts['small'],
                           bg=self.colors['card_bg'], fg=self.colors['text_secondary'],
                           borderwidth=0, wrap=tk.WORD, relief='flat')
        info_text.pack(fill=tk.X, padx=15, pady=(0, 15))
        info_text.insert(1.0, content)
        info_text.config(state='disabled')
        
    def create_image_loading_card(self, parent):
        """Crée la carte de chargement d'image"""
        card_frame = self.create_card_frame(parent, "📸 Image Loading", self.colors['primary_blue'])
        
        # Bouton moderne de chargement
        load_btn = tk.Button(card_frame, text="Load Medical Image", 
                           font=self.fonts['body'], 
                           bg=self.colors['primary_blue'], fg='white',
                           borderwidth=0, relief='flat', cursor='hand2',
                           command=self.app.load_image,
                           pady=12)
        load_btn.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # Info label
        self.app.image_info_label = tk.Label(card_frame, text="No image loaded",
                                           font=self.fonts['small'],
                                           bg=self.colors['card_bg'], 
                                           fg=self.colors['text_secondary'])
        self.app.image_info_label.pack(anchor=tk.W, padx=15, pady=(0, 15))
        
    def create_analysis_card(self, parent):
        """Crée la carte d'analyse AI"""
        card_frame = self.create_card_frame(parent, "🤖 AI Analysis", self.colors['success'])
        
        # Bouton d'analyse principal
        self.app.analyze_button = tk.Button(card_frame, text="🔍 Analyze for Retinoblastoma",
                                          font=self.fonts['body'],
                                          bg=self.colors['success'], fg='white',
                                          borderwidth=0, relief='flat', cursor='hand2',
                                          command=self.app.analyze_image,
                                          state='disabled', pady=12)
        self.app.analyze_button.pack(fill=tk.X, padx=15, pady=(0, 15))
        
    def create_system_status_card(self, parent):
        """Crée la carte de statut système"""
        card_frame = self.create_card_frame(parent, "🧩 System Modules", self.colors['primary_blue'])
        
        # Container pour les statuts
        status_container = tk.Frame(card_frame, bg=self.colors['card_bg'])
        status_container.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Créer les labels de statut avec style moderne
        self.app.module_status = {}
        modules = [
            ('gemma', 'Gemma 3n', 'Initializing...'),
            ('eye_detector', 'Eye Detector', 'Waiting...'),
            ('face_handler', 'Face Handler', 'Waiting...'),
            ('visualizer', 'Visualizer', 'Waiting...')
        ]
        
        for module_key, display_name, initial_status in modules:
            module_frame = tk.Frame(status_container, bg=self.colors['card_bg'])
            module_frame.pack(fill=tk.X, pady=2)
            
            # Indicateur de statut (cercle coloré)
            status_indicator = tk.Label(module_frame, text="●", font=('Arial', 12),
                                      bg=self.colors['card_bg'], fg=self.colors['warning'])
            status_indicator.pack(side=tk.LEFT, padx=(0, 8))
            
            # Texte du module
            status_text = tk.Label(module_frame, text=f"{display_name}: {initial_status}",
                                 font=self.fonts['small'],
                                 bg=self.colors['card_bg'], fg=self.colors['text_primary'])
            status_text.pack(side=tk.LEFT)
            
            # Stocker pour mise à jour
            self.app.module_status[module_key] = (status_indicator, status_text)
    
    def create_settings_card(self, parent):
        """Crée la carte des paramètres"""
        card_frame = self.create_card_frame(parent, "⚙️ Settings", self.colors['dark_gray'])
        
        settings_container = tk.Frame(card_frame, bg=self.colors['card_bg'])
        settings_container.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Seuil de confiance avec slider moderne
        conf_label = tk.Label(settings_container, text="Confidence Threshold:",
                            font=self.fonts['small'], bg=self.colors['card_bg'])
        conf_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.app.confidence_var = tk.DoubleVar(value=0.5)
        conf_scale = tk.Scale(settings_container, from_=0.1, to=0.9,
                            variable=self.app.confidence_var, orient=tk.HORIZONTAL,
                            bg=self.colors['card_bg'], highlightthickness=0,
                            font=self.fonts['tiny'])
        conf_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Checkboxes modernes
        self.app.face_tracking_var = tk.BooleanVar(value=True)
        face_check = tk.Checkbutton(settings_container, text="Enable Face Tracking",
                                  variable=self.app.face_tracking_var,
                                  bg=self.colors['card_bg'], font=self.fonts['small'])
        face_check.pack(anchor=tk.W, pady=2)
        
        self.app.enhanced_detection_var = tk.BooleanVar(value=True)
        enhanced_check = tk.Checkbutton(settings_container, text="Enhanced Detection",
                                      variable=self.app.enhanced_detection_var,
                                      bg=self.colors['card_bg'], font=self.fonts['small'])
        enhanced_check.pack(anchor=tk.W, pady=2)
        
    def create_progress_card(self, parent):
        """Crée la carte de progression"""
        card_frame = self.create_card_frame(parent, "📊 Progress", self.colors['warning'])
        
        progress_container = tk.Frame(card_frame, bg=self.colors['card_bg'])
        progress_container.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Barre de progression moderne
        self.app.progress = ttk.Progressbar(progress_container, mode='determinate',
                                          style='Modern.Horizontal.TProgressbar')
        self.app.progress.pack(fill=tk.X, pady=(0, 8))
        
        self.app.progress_label = tk.Label(progress_container, text="Ready",
                                         font=self.fonts['small'],
                                         bg=self.colors['card_bg'], fg=self.colors['text_secondary'])
        self.app.progress_label.pack(anchor=tk.W)
        
        # Métriques
        self.app.metrics_label = tk.Label(progress_container, text="No analysis yet",
                                        font=self.fonts['tiny'],
                                        bg=self.colors['card_bg'], fg=self.colors['text_secondary'])
        self.app.metrics_label.pack(anchor=tk.W, pady=(5, 0))
        
    def create_actions_card(self, parent):
        """Crée la carte des actions"""
        card_frame = self.create_card_frame(parent, "💾 Actions", self.colors['purple'])
        
        actions_container = tk.Frame(card_frame, bg=self.colors['card_bg'])
        actions_container.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Boutons d'action avec style moderne
        actions = [
            ("Export Results", self.app.export_results),
            ("Medical Report (HTML)", self.app.generate_medical_report),
            ("🧠 Smart Recommendations", self.app.generate_recommendations),
            ("Face Tracking Summary", self.app.show_face_tracking_summary),
            ("System Info", self.app.show_system_info)
        ]
        
        for text, command in actions:
            btn = tk.Button(actions_container, text=text, 
                          font=self.fonts['small'],
                          bg=self.colors['white'], fg=self.colors['primary_blue'],
                          borderwidth=1, relief='solid', cursor='hand2',
                          command=command, pady=8)
            btn.pack(fill=tk.X, pady=2)
            
    def create_display_area_modern(self, parent):
        """Crée la zone d'affichage moderne"""
        display_container = tk.Frame(parent, bg=self.colors['light_gray'])
        display_container.grid(row=0, column=1, sticky='nsew')
        display_container.rowconfigure(0, weight=1)
        display_container.columnconfigure(0, weight=1)
        
        # Notebook moderne
        style = ttk.Style()
        style.configure('Modern.TNotebook', background=self.colors['white'])
        style.configure('Modern.TNotebook.Tab', padding=[20, 12])
        
        self.app.notebook = ttk.Notebook(display_container, style='Modern.TNotebook')
        self.app.notebook.grid(row=0, column=0, sticky='nsew', padx=10)
        
        # Onglet image avec style moderne
        self.create_image_tab_modern()
        
        # Onglet résultats
        self.create_results_tab_modern()
        
        # Onglet historique
        self.create_history_tab_modern()
        
    def create_image_tab_modern(self):
        """Crée l'onglet image moderne"""
        self.app.image_frame = tk.Frame(self.app.notebook, bg=self.colors['white'])
        self.app.notebook.add(self.app.image_frame, text="🖼️ Image Analysis")
        
        # Canvas avec style moderne
        canvas_container = tk.Frame(self.app.image_frame, bg=self.colors['white'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.app.canvas = tk.Canvas(canvas_container, bg=self.colors['white'], 
                                  relief='flat', borderwidth=0)
        
        # Scrollbars modernes
        v_scroll = ttk.Scrollbar(canvas_container, orient="vertical", command=self.app.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.app.canvas.xview)
        
        self.app.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.app.canvas.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
    def create_results_tab_modern(self):
        """Crée l'onglet résultats moderne"""
        self.app.results_frame = tk.Frame(self.app.notebook, bg=self.colors['white'])
        self.app.notebook.add(self.app.results_frame, text="📋 Medical Results")
        
        results_container = tk.Frame(self.app.results_frame, bg=self.colors['white'])
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.app.results_text = tk.Text(results_container, wrap=tk.WORD,
                                      font=self.fonts['mono'], relief='flat',
                                      bg=self.colors['light_gray'], fg=self.colors['text_primary'],
                                      borderwidth=0, padx=15, pady=15)
        
        results_scroll = ttk.Scrollbar(results_container, orient="vertical",
                                     command=self.app.results_text.yview)
        self.app.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.app.results_text.grid(row=0, column=0, sticky='nsew')
        results_scroll.grid(row=0, column=1, sticky='ns')
        
        results_container.columnconfigure(0, weight=1)
        results_container.rowconfigure(0, weight=1)
        
    def create_history_tab_modern(self):
        """Crée l'onglet historique moderne"""
        self.app.history_frame = tk.Frame(self.app.notebook, bg=self.colors['white'])
        self.app.notebook.add(self.app.history_frame, text="👤 Patient History")
        
        history_container = tk.Frame(self.app.history_frame, bg=self.colors['white'])
        history_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.app.history_text = tk.Text(history_container, wrap=tk.WORD,
                                      font=self.fonts['mono'], relief='flat',
                                      bg=self.colors['light_gray'], fg=self.colors['text_primary'],
                                      borderwidth=0, padx=15, pady=15)
        
        history_scroll = ttk.Scrollbar(history_container, orient="vertical",
                                     command=self.app.history_text.yview)
        self.app.history_text.configure(yscrollcommand=history_scroll.set)
        
        self.app.history_text.grid(row=0, column=0, sticky='nsew')
        history_scroll.grid(row=0, column=1, sticky='ns')
        
        history_container.columnconfigure(0, weight=1)
        history_container.rowconfigure(0, weight=1)
        
    def create_modern_footer(self, parent):
        """Crée le footer moderne"""
        footer_frame = tk.Frame(parent, bg=self.colors['medium_gray'], height=50)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        footer_frame.pack_propagate(False)
        
        # Status principal
        self.app.status_label = tk.Label(footer_frame, 
                                       text="RetinoblastoGemma v6 Ready - Google Gemma Hackathon",
                                       font=self.fonts['small'], 
                                       bg=self.colors['medium_gray'], 
                                       fg=self.colors['text_secondary'])
        self.app.status_label.pack(expand=True)
        
    def create_card_frame(self, parent, title, accent_color):
        """Crée un frame de carte avec titre"""
        # Container avec ombre simulée
        shadow_frame = tk.Frame(parent, bg=self.colors['medium_gray'])
        shadow_frame.pack(fill=tk.X, pady=(0, 15), padx=2)
        
        # Carte principale
        card_frame = tk.Frame(shadow_frame, bg=self.colors['card_bg'])
        card_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Header coloré
        header_frame = tk.Frame(card_frame, bg=accent_color, height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text=title, font=self.fonts['subtitle'],
                              bg=accent_color, fg='white')
        title_label.pack(expand=True, pady=8)
        
        return card_frame
        
    def create_gradient(self, canvas, x1, y1, x2, y2, color1, color2):
        """Crée un dégradé sur un canvas"""
        # Convertir les couleurs hex en RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r1, g1, b1 = hex_to_rgb(color1)
        r2, g2, b2 = hex_to_rgb(color2)
        
        # Créer le dégradé
        height = y2 - y1
        for i in range(height):
            ratio = i / height
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(x1, y1 + i, x2, y1 + i, fill=color)
            
    def update_module_status_modern(self, module_key, text, color):
        """Met à jour le statut d'un module avec style moderne"""
        if hasattr(self.app, 'module_status') and module_key in self.app.module_status:
            indicator, label = self.app.module_status[module_key]
            
            # Couleurs d'indicateur
            indicator_colors = {
                'blue': self.colors['primary_blue'],
                'green': self.colors['success'],
                'red': self.colors['error'],
                'orange': self.colors['warning'],
                'gray': self.colors['dark_gray']
            }
            
            indicator.config(fg=indicator_colors.get(color, self.colors['dark_gray']))
            label.config(text=text)
            
    def update_status_badge(self, status, color='primary'):
        """Met à jour le badge de statut dans le header"""
        if hasattr(self, 'status_badge'):
            color_map = {
                'primary': self.colors['primary_blue'],
                'success': self.colors['success'],
                'warning': self.colors['warning'],
                'error': self.colors['error']
            }
            self.status_badge.config(text=status, bg=color_map.get(color, self.colors['primary_blue']))
            
    def show_loading_animation(self, text="Loading..."):
        """Affiche une animation de chargement moderne"""
        # Créer une popup de chargement moderne
        loading_window = tk.Toplevel(self.root)
        loading_window.title("")
        loading_window.geometry("300x150")
        loading_window.configure(bg=self.colors['white'])
        loading_window.transient(self.root)
        loading_window.grab_set()
        
        # Center the window
        loading_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        # Contenu de la popup
        content_frame = tk.Frame(loading_window, bg=self.colors['white'])
        content_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)
        
        # Animation spinner (simulé avec texte)
        spinner_label = tk.Label(content_frame, text="🔄", font=('Arial', 32),
                               bg=self.colors['white'], fg=self.colors['primary_blue'])
        spinner_label.pack(pady=(0, 15))
        
        # Texte de chargement
        text_label = tk.Label(content_frame, text=text, font=self.fonts['body'],
                            bg=self.colors['white'], fg=self.colors['text_primary'])
        text_label.pack()
        
        # Animation du spinner
        def animate_spinner():
            current_text = spinner_label.cget('text')
            if current_text == '🔄':
                spinner_label.config(text='⏳')
            else:
                spinner_label.config(text='🔄')
            loading_window.after(500, animate_spinner)
        
        animate_spinner()
        return loading_window
        
    def create_success_popup(self, title, message, details=None):
        """Crée une popup de succès moderne"""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("400x250")
        popup.configure(bg=self.colors['white'])
        popup.transient(self.root)
        popup.grab_set()
        
        # Center the popup
        popup.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 100,
            self.root.winfo_rooty() + 100
        ))
        
        # Header avec icône de succès
        header_frame = tk.Frame(popup, bg=self.colors['success'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        success_icon = tk.Label(header_frame, text="✅", font=('Arial', 24),
                              bg=self.colors['success'], fg='white')
        success_icon.pack(side=tk.LEFT, padx=20, pady=15)
        
        title_label = tk.Label(header_frame, text=title, font=self.fonts['subtitle'],
                             bg=self.colors['success'], fg='white')
        title_label.pack(side=tk.LEFT, pady=15)
        
        # Contenu
        content_frame = tk.Frame(popup, bg=self.colors['white'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        message_label = tk.Label(content_frame, text=message, font=self.fonts['body'],
                               bg=self.colors['white'], fg=self.colors['text_primary'],
                               wraplength=350, justify=tk.LEFT)
        message_label.pack(anchor=tk.W, pady=(0, 15))
        
        if details:
            details_label = tk.Label(content_frame, text=details, font=self.fonts['small'],
                                   bg=self.colors['white'], fg=self.colors['text_secondary'],
                                   wraplength=350, justify=tk.LEFT)
            details_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Bouton OK
        ok_button = tk.Button(content_frame, text="OK", font=self.fonts['body'],
                            bg=self.colors['primary_blue'], fg='white',
                            borderwidth=0, relief='flat', cursor='hand2',
                            command=popup.destroy, padx=30, pady=8)
        ok_button.pack(anchor=tk.E)
        
    def create_error_popup(self, title, message, details=None):
        """Crée une popup d'erreur moderne"""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("400x300")
        popup.configure(bg=self.colors['white'])
        popup.transient(self.root)
        popup.grab_set()
        
        # Header avec icône d'erreur
        header_frame = tk.Frame(popup, bg=self.colors['error'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        error_icon = tk.Label(header_frame, text="❌", font=('Arial', 24),
                            bg=self.colors['error'], fg='white')
        error_icon.pack(side=tk.LEFT, padx=20, pady=15)
        
        title_label = tk.Label(header_frame, text=title, font=self.fonts['subtitle'],
                             bg=self.colors['error'], fg='white')
        title_label.pack(side=tk.LEFT, pady=15)
        
        # Contenu
        content_frame = tk.Frame(popup, bg=self.colors['white'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        message_label = tk.Label(content_frame, text=message, font=self.fonts['body'],
                               bg=self.colors['white'], fg=self.colors['text_primary'],
                               wraplength=350, justify=tk.LEFT)
        message_label.pack(anchor=tk.W, pady=(0, 15))
        
        if details:
            # Zone de détails avec scrollbar
            details_frame = tk.Frame(content_frame, bg=self.colors['light_gray'])
            details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
            
            details_text = tk.Text(details_frame, height=6, font=self.fonts['tiny'],
                                 bg=self.colors['light_gray'], fg=self.colors['text_secondary'],
                                 borderwidth=0, wrap=tk.WORD)
            details_scroll = ttk.Scrollbar(details_frame, command=details_text.yview)
            details_text.configure(yscrollcommand=details_scroll.set)
            
            details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            details_text.insert(1.0, details)
            details_text.config(state='disabled')
        
        # Bouton OK
        ok_button = tk.Button(content_frame, text="OK", font=self.fonts['body'],
                            bg=self.colors['error'], fg='white',
                            borderwidth=0, relief='flat', cursor='hand2',
                            command=popup.destroy, padx=30, pady=8)
        ok_button.pack(anchor=tk.E)
        
    def show_analysis_confirmation(self, callback):
        """Affiche une confirmation d'analyse moderne"""
        popup = tk.Toplevel(self.root)
        popup.title("Start Analysis")
        popup.geometry("450x350")
        popup.configure(bg=self.colors['white'])
        popup.transient(self.root)
        popup.grab_set()
        
        # Header
        header_frame = tk.Frame(popup, bg=self.colors['primary_blue'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        icon_label = tk.Label(header_frame, text="🔍", font=('Arial', 32),
                            bg=self.colors['primary_blue'], fg='white')
        icon_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        title_label = tk.Label(header_frame, text="Start Retinoblastoma Analysis?",
                             font=self.fonts['subtitle'],
                             bg=self.colors['primary_blue'], fg='white')
        title_label.pack(side=tk.LEFT, pady=20)
        
        # Contenu
        content_frame = tk.Frame(popup, bg=self.colors['white'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        description = ("This will analyze the image for signs of leukocoria using:\n\n"
                      "• Local Gemma 3n AI model\n"
                      "• Computer vision eye detection\n"
                      "• Face tracking (if enabled)\n\n"
                      "Analysis may take 30-90 seconds.")
        
        desc_label = tk.Label(content_frame, text=description, font=self.fonts['body'],
                            bg=self.colors['white'], fg=self.colors['text_primary'],
                            justify=tk.LEFT, wraplength=400)
        desc_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Privacy notice
        privacy_frame = tk.Frame(content_frame, bg=self.colors['light_blue'], relief='flat')
        privacy_frame.pack(fill=tk.X, pady=(0, 20))
        
        privacy_label = tk.Label(privacy_frame, text="🔒 100% Local Processing - No data transmitted",
                               font=self.fonts['small'], bg=self.colors['light_blue'],
                               fg='white', pady=8)
        privacy_label.pack()
        
        # Boutons
        button_frame = tk.Frame(content_frame, bg=self.colors['white'])
        button_frame.pack(fill=tk.X)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", font=self.fonts['body'],
                             bg=self.colors['medium_gray'], fg=self.colors['text_primary'],
                             borderwidth=0, relief='flat', cursor='hand2',
                             command=popup.destroy, padx=20, pady=8)
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        start_btn = tk.Button(button_frame, text="Start Analysis", font=self.fonts['body'],
                            bg=self.colors['success'], fg='white',
                            borderwidth=0, relief='flat', cursor='hand2',
                            padx=20, pady=8,
                            command=lambda: [popup.destroy(), callback()])
        start_btn.pack(side=tk.RIGHT)