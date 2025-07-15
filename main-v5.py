                messagebox.showerror("Initialization Error", 
                    f"System initialization failed:\n{e}\n\n"
                    "Please check your Gemma 3n installation.")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def load_image(self):
        """Charge une image médicale"""
        file_path = filedialog.askopenfilename(
            title="Select medical image for retinoblastoma analysis",
            filetypes=[
                ("Medical images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Vérifier que l'image peut être ouverte
                test_image = Image.open(file_path)
                image_info = f"{test_image.width}x{test_image.height}, {test_image.mode}"
                test_image.close()
                
                self.current_image_path = file_path
                self.display_image(file_path)
                
                filename = Path(file_path).name
                self.update_status(f"✅ Image loaded: {filename}")
                self.image_info_label.config(text=f"{filename} ({image_info})", foreground="green")
                
                logger.info(f"Image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Image Loading Error", 
                    f"Cannot load image:\n{e}")
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
        """Analyse l'image avec Gemma 3n multimodal"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", 
                "Please load a medical image first.\n\n"
                "Click 'Load Medical Image' to select an image.")
            return
        
        if not self.gemma_handler or not self.gemma_handler.initialized:
            messagebox.showerror("System Not Ready", 
                "Gemma 3n model is not loaded.\n\n"
                "Please wait for initialization to complete or try reloading the model.")
            return
        
        # Confirmation pour analyse
        result = messagebox.askyesno("Start Analysis", 
            "🔍 Start retinoblastoma analysis?\n\n"
            "This will analyze the image for signs of leukocoria using Gemma 3n multimodal AI.\n"
            "Analysis may take 30-60 seconds.")
        
        if not result:
            return
        
        def analysis_thread():
            try:
                start_time = time.time()
                self.update_status("🔄 Starting retinoblastoma analysis...", "blue")
                self.update_progress(0, "Preparing analysis...")
                
                # Étape 1: Détection des régions d'yeux
                self.update_progress(15, "Detecting eye regions...")
                eye_regions = self.detect_eye_regions_advanced()
                
                if not eye_regions:
                    self.update_status("⚠️ No eye regions detected", "orange")
                    self.update_progress(100, "Analysis incomplete")
                    
                    messagebox.showwarning("No Eyes Detected", 
                        "No eye regions could be detected in this image.\n\n"
                        "Tips:\n"
                        "• Ensure the image shows clear eye(s)\n"
                        "• Try enabling 'Cropped Eye Analysis'\n"
                        "• Check image quality and lighting\n"
                        "• Make sure eyes are visible and not closed")
                    return
                
                logger.info(f"Detected {len(eye_regions)} eye regions for analysis")
                
                # Étape 2: Analyse multimodale avec Gemma 3n
                self.update_progress(30, "Initializing AI analysis...")
                analysis_results = []
                
                for i, eye_region in enumerate(eye_regions):
                    progress = 30 + (i * 50) // len(eye_regions)
                    position = eye_region['position']
                    
                    self.update_progress(progress, f"Analyzing {position} eye with Gemma 3n...")
                    
                    # Analyse multimodale
                    if self.multimodal_var.get():
                        result = self.gemma_handler.analyze_eye_image_multimodal(
                            eye_region['image'], 
                            eye_region['position']
                        )
                    else:
                        # Fallback intelligent
                        result = self.gemma_handler._create_intelligent_fallback(
                            self.gemma_handler._extract_advanced_features(eye_region['image']),
                            "Fallback analysis"
                        )
                    
                    result['eye_region'] = eye_region
                    analysis_results.append(result)
                    
                    # Log du résultat
                    leukocoria = result.get('leukocoria_detected', False)
                    confidence = result.get('confidence', 0)
                    logger.info(f"Analysis complete for {position} eye: "
                              f"Leukocoria={leukocoria}, Confidence={confidence:.1f}%")
                
                # Étape 3: Visualisation des résultats
                self.update_progress(85, "Generating visual results...")
                self.visualize_results_advanced(analysis_results)
                
                # Étape 4: Affichage des résultats détaillés
                self.update_progress(95, "Compiling medical report...")
                self.display_detailed_results(analysis_results)
                
                # Métriques finales
                processing_time = time.time() - start_time
                self.performance_metrics['total_analyses'] += 1
                self.performance_metrics['processing_times'].append(processing_time)
                
                # Vérifier les détections positives
                positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
                high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
                immediate_cases = sum(1 for r in analysis_results if r.get('urgency') == 'immediate')
                
                if positive_detections > 0:
                    self.performance_metrics['detections_found'] += 1
                    
                    # Alerte médicale
                    urgency_msg = ""
                    if immediate_cases > 0:
                        urgency_msg = "⚠️ IMMEDIATE medical attention required!"
                    elif high_risk_count > 0:
                        urgency_msg = "⚠️ URGENT medical consultation needed!"
                    else:
                        urgency_msg = "Medical evaluation recommended."
                    
                    self.update_status(f"🚨 ALERT: Possible retinoblastoma detected! ({processing_time:.1f}s)", "red")
                    
                    messagebox.showwarning("⚠️ MEDICAL ALERT", 
                        f"🚨 POSSIBLE RETINOBLASTOMA DETECTED 🚨\n\n"
                        f"Positive findings in {positive_detections} eye(s)\n"
                        f"High-risk cases: {high_risk_count}\n\n"
                        f"{urgency_msg}\n\n"
                        f"👨‍⚕️ Action required:\n"
                        f"1. Contact pediatric ophthalmologist immediately\n"
                        f"2. Show them this analysis and original image\n"
                        f"3. Do NOT delay seeking professional evaluation")
                else:
                    self.update_status(f"✅ Analysis complete: No concerning findings ({processing_time:.1f}s)", "green")
                    
                    messagebox.showinfo("Analysis Complete", 
                        f"✅ Analysis completed successfully!\n\n"
                        f"No signs of leukocoria were detected.\n"
                        f"Continue regular eye health monitoring.\n\n"
                        f"Processing time: {processing_time:.1f} seconds")
                
                self.update_progress(100, "Analysis complete")
                self.update_metrics()
                self.update_memory_display()
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.update_status(f"❌ Analysis failed: {e}", "red")
                self.update_progress(0, "Analysis failed")
                
                messagebox.showerror("Analysis Error", 
                    f"Analysis failed with error:\n{e}\n\n"
                    f"Possible causes:\n"
                    f"• GPU memory insufficient\n"
                    f"• Model processing error\n"
                    f"• Image format issues\n\n"
                    f"Try clearing GPU memory or reloading the model.")
        
        # Démarrer l'analyse en arrière-plan
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def detect_eye_regions_advanced(self):
        """Détection avancée des régions d'yeux"""
        try:
            if not self.eye_detector:
                logger.error("Eye detector not available")
                return []
            
            # Utiliser le détecteur d'yeux
            detection_results = self.eye_detector.detect_faces_and_eyes(
                self.current_image_path,
                enhance_quality=self.enhanced_cv_var.get(),
                parallel_processing=False
            )
            
            # Extraire les régions d'yeux
            eye_regions = []
            for face in detection_results.get('faces', []):
                eye_regions.extend(face.get('eyes', []))
            
            # Si pas de détection et mode cropped activé
            if not eye_regions and self.crop_detection_var.get():
                eye_regions = self._detect_cropped_fallback()
            
            return eye_regions
            
        except Exception as e:
            logger.error(f"Advanced eye detection failed: {e}")
            return []
    
    def _detect_cropped_fallback(self):
        """Fallback pour images croppées"""
        try:
            image = cv2.imread(self.current_image_path)
            if image is None:
                return []
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Analyser comme image croppée
            aspect_ratio = w / h
            
            if aspect_ratio > 2.0:  # Deux yeux côte à côte
                mid_x = w // 2
                return [
                    {
                        'position': 'left_cropped',
                        'bbox': (0, 0, mid_x, h),
                        'image': Image.fromarray(rgb_image[:, :mid_x]),
                        'confidence': 0.8,
                        'is_cropped': True,
                        'detection_method': 'fallback_split'
                    },
                    {
                        'position': 'right_cropped',
                        'bbox': (mid_x, 0, w - mid_x, h),
                        'image': Image.fromarray(rgb_image[:, mid_x:]),
                        'confidence': 0.8,
                        'is_cropped': True,
                        'detection_method': 'fallback_split'
                    }
                ]
            else:  # Un seul œil
                return [{
                    'position': 'center_cropped',
                    'bbox': (0, 0, w, h),
                    'image': Image.fromarray(rgb_image),
                    'confidence': 0.9,
                    'is_cropped': True,
                    'detection_method': 'fallback_full'
                }]
                
        except Exception as e:
            logger.error(f"Cropped fallback failed: {e}")
            return []
    
    def visualize_results_advanced(self, analysis_results):
        """Visualisation avancée avec annotations"""
        try:
            original_image = Image.open(self.current_image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Police pour les labels
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Analyser chaque région
            for i, result in enumerate(analysis_results):
                eye_region = result.get('eye_region', {})
                bbox = eye_region.get('bbox')
                
                if not bbox:
                    continue
                
                x, y, w, h = bbox
                
                # Données de l'analyse
                leukocoria_detected = result.get('leukocoria_detected', False)
                confidence = result.get('confidence', 0)
                risk_level = result.get('risk_level', 'low')
                urgency = result.get('urgency', 'routine')
                position = eye_region.get('position', 'unknown')
                
                # Déterminer la couleur selon le résultat
                if leukocoria_detected:
                    if risk_level == 'high' or urgency == 'immediate':
                        color = 'red'
                        width = 6
                        alert_symbol = '🚨'
                    elif risk_level == 'medium' or urgency == 'urgent':
                        color = 'orange'
                        width = 5
                        alert_symbol = '⚠️'
                    else:
                        color = 'yellow'
                        width = 4
                        alert_symbol = '⚡'
                else:
                    color = 'green'
                    width = 3
                    alert_symbol = '✅'
                
                # Dessiner le rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
                
                # Label principal
                main_label = f"{alert_symbol} {position.upper()}"
                if leukocoria_detected:
                    main_label += f" - {urgency.upper()}"
                
                # Position du texte
                text_y = y - 45 if y > 45 else y + h + 5
                
                # Fond pour le texte
                try:
                    text_bbox = draw.textbbox((x, text_y), main_label, font=font)
                    draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2], 
                                  fill=color, outline=color)
                except:
                    # Fallback si textbbox pas disponible
                    text_width = len(main_label) * 8
                    draw.rectangle([x-5, text_y-2, x+text_width+5, text_y+18], 
                                  fill=color, outline=color)
                
                draw.text((x, text_y), main_label, fill='white', font=font)
                
                # Label secondaire avec confiance
                confidence_label = f"Confidence: {confidence:.1f}%"
                if leukocoria_detected:
                    confidence_label += f" | Risk: {risk_level.upper()}"
                
                text_y2 = text_y + 25 if text_y == y - 45 else text_y - 20
                draw.text((x, text_y2), confidence_label, fill=color, font=font_small)
            
            # Titre général
            title = "RETINOBLASTOMA AI ANALYSIS - GEMMA 3N"
            try:
                title_bbox = draw.textbbox((10, 10), title, font=font)
                draw.rectangle([5, 5, title_bbox[2]+10, title_bbox[3]+5], fill='navy', outline='navy')
            except:
                draw.rectangle([5, 5, 400, 25], fill='navy', outline='navy')
            
            draw.text((10, 10), title, fill='white', font=font)
            
            # Sauvegarder l'image annotée
            timestamp = int(time.time())
            annotated_path = RESULTS_DIR / f"retinoblastoma_analysis_{timestamp}.jpg"
            original_image.save(annotated_path, quality=95)
            
            # Afficher l'image annotée
            self.display_annotated_image(original_image)
            
            logger.info(f"Visualization complete. Saved to: {annotated_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            messagebox.showerror("Visualization Error", f"Failed to create visual results: {e}")
    
    def display_annotated_image(self, annotated_image):
        """Affiche l'image annotée dans le canvas"""
        try:
            # Préparer l'image pour l'affichage
            canvas_width = max(900, self.canvas.winfo_width())
            canvas_height = max(700, self.canvas.winfo_height())
            
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
    
    def display_detailed_results(self, analysis_results):
        """Affiche les résultats détaillés"""
        try:
            # Créer le rapport médical complet
            report = self.generate_detailed_medical_report(analysis_results)
            
            # Afficher dans l'onglet résultats
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, report)
            
            # Sauvegarder les résultats
            self.current_analysis_results = report
            
            logger.info("Detailed results displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to display detailed results: {e}")
    
    def generate_detailed_medical_report(self, analysis_results):
        """Génère un rapport médical détaillé"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        # En-tête du rapport
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    RETINOBLASTOMA AI ANALYSIS REPORT                             ║
║                         GEMMA 3N MULTIMODAL SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

📋 ANALYSIS DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: {timestamp}
Image File: {filename}
AI System: Gemma 3n Multimodal (Local Processing)
Processing Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
Privacy: 100% Local - No data transmitted

📊 EXECUTIVE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Statistiques générales
        total_eyes = len(analysis_results)
        positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
        high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
        immediate_cases = sum(1 for r in analysis_results if r.get('urgency') == 'immediate')
        
        report += f"""
Eyes Analyzed: {total_eyes}
Positive Detections: {positive_detections}
High-Risk Cases: {high_risk_count}
Immediate Attention Required: {immediate_cases}
"""
        
        if positive_detections > 0:
            report += f"""
🚨 MEDICAL ALERT: POSSIBLE RETINOBLASTOMA DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  IMMEDIATE PEDIATRIC OPHTHALMOLOGICAL CONSULTATION REQUIRED
⚠️  DO NOT DELAY MEDICAL EVALUATION
⚠️  CONTACT YOUR HEALTHCARE PROVIDER TODAY
"""
        else:
            report += f"""
✅ RESULT: NO CONCERNING FINDINGS DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No signs of leukocoria were detected in this analysis.
Continue regular pediatric eye health monitoring.
"""
        
        # Analyse détaillée par œil
        report += f"""

🔍 DETAILED ANALYSIS BY EYE REGION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, result in enumerate(analysis_results, 1):
            eye_region = result.get('eye_region', {})
            position = eye_region.get('position', 'unknown')
            detection_method = eye_region.get('detection_method', 'unknown')
            is_cropped = eye_region.get('is_cropped', False)
            
            report += f"""
┌─ EYE REGION {i}: {position.upper()} {'(CROPPED IMAGE)' if is_cropped else '(FULL FACE)'} ─┐
│
│ 🎯 DETECTION RESULTS:
│   • Leukocoria Detected: {'⚠️  YES' if result.get('leukocoria_detected') else '✅ NO'}
│   • Confidence Level: {result.get('confidence', 0):.1f}%
│   • Risk Assessment: {result.get('risk_level', 'unknown').upper()}
│   • Urgency Level: {result.get('urgency', 'routine').upper()}
│   • Detection Method: {detection_method}
│
│ 👁️  PUPIL ANALYSIS:
│   • Description: {result.get('pupil_description', 'Not available')[:80]}
{"│   • " + result.get('pupil_description', '')[80:160] + "..." if len(result.get('pupil_description', '')) > 80 else ""}
│
│ 🏥 MEDICAL ASSESSMENT:
│   • Clinical Analysis: {result.get('medical_analysis', 'Not available')[:70]}
{"│   • " + result.get('medical_analysis', '')[70:140] + "..." if len(result.get('medical_analysis', '')) > 70 else ""}
│
│ 💊 RECOMMENDATIONS:
│   • {result.get('recommendations', 'No specific recommendations')}
│
└─────────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Disclaimer médical important
        report += f"""

⚕️  CRITICAL MEDICAL DISCLAIMER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 IMPORTANT: This analysis is provided by an AI screening system using Gemma 3n.

🔴 THIS IS NOT A MEDICAL DIAGNOSIS and should NOT replace professional medical 
   evaluation by qualified pediatric ophthalmologists.

📋 REQUIRED ACTIONS FOR POSITIVE FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 📞 Contact pediatric ophthalmologist immediately
2. 📋 Bring this report and original images to appointment  
3. ⏰ Do NOT delay professional medical evaluation
4. 🚫 Do not rely solely on this AI analysis for medical decisions

═══════════════════════════════════════════════════════════════════════════════════
                        END OF MEDICAL ANALYSIS REPORT
                    Generated by RetinoblastoGemma v1.0 - Gemma 3n
═══════════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def update_status(self, message, color="blue"):
        """Met à jour le statut de l'application"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))
        self.root.after(0, lambda: self.statusbar.config(text=message))
        logger.info(message)
    
    def update_gemma_status(self, message, color="blue"):
        """Met à jour le statut Gemma"""
        self.root.after(0, lambda: self.gemma_status.config(text=message, foreground=color))
    
    def update_progress(self, value, message=""):
        """Met à jour la barre de progression"""
        self.root.after(0, lambda: self.progress.config(value=value))
        if message:
            self.root.after(0, lambda: self.progress_detail.config(text=message))
    
    def update_metrics(self):
        """Met à jour les métriques de performance"""
        total = self.performance_metrics['total_analyses']
        detections = self.performance_metrics['detections_found']
        
        if total > 0:
            detection_rate = (detections / total) * 100
            avg_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            
            metrics_text = (f"Analyses: {total} | "
                          f"Detections: {detections} ({detection_rate:.1f}%) | "
                          f"Avg time: {avg_time:.1f}s")
        else:
            metrics_text = "No analysis performed yet"
        
        self.root.after(0, lambda: self.metrics_label.config(text=metrics_text))
    
    def update_memory_display(self):
        """Met à jour l'affichage de la mémoire"""
        try:
            if self.gemma_handler:
                memory_info = self.gemma_handler.get_memory_usage()
                if memory_info:
                    memory_text = f"GPU: {memory_info.get('gpu_allocated', 0):.1f}GB / {memory_info.get('gpu_reserved', 0):.1f}GB"
                    self.root.after(0, lambda: self.memory_label.config(text=memory_text))
        except Exception as e:
            logger.error(f"Memory display update failed: {e}")
    
    def update_diagnostics(self):
        """Met à jour les informations de diagnostic"""
        try:
            diagnostics_info = f"""RETINOBLASTOGAMMA SYSTEM DIAGNOSTICS
════════════════════════════════════════════════════════════════════

SYSTEM INFORMATION:
• Application: RetinoblastoGemma v1.0 - Gemma 3n Multimodal
• Python Version: {sys.version.split()[0]}
• Platform: {sys.platform}
• Working Directory: {Path.cwd()}

HARDWARE INFORMATION:
• CPU Cores: {os.cpu_count()}
• PyTorch Available: {TORCH_AVAILABLE}
• CUDA Available: {torch.cuda.is_available() if TORCH_AVAILABLE else 'N/A'}
• GPU Device: {torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else 'None'}

MODEL STATUS:
• Gemma 3n Path: {GEMMA_MODEL_PATH}
• Model Files Present: {GEMMA_MODEL_PATH.exists()}
• Core Modules Available: {CORE_MODULES_AVAILABLE}
• Handler Initialized: {self.gemma_handler.initialized if self.gemma_handler else False}

CONFIGURATION:
• Crop Detection: {'Enabled' if self.crop_detection_var.get() else 'Disabled'}
• Multimodal Analysis: {'Enabled' if self.multimodal_var.get() else 'Disabled'}
• Enhanced CV: {'Enabled' if self.enhanced_cv_var.get() else 'Disabled'}

DIRECTORIES:
• Models Directory: {MODELS_DIR}
• Data Directory: {DATA_DIR}
• Results Directory: {RESULTS_DIR}

SESSION STATISTICS:
• Total Analyses: {self.performance_metrics['total_analyses']}
• Detections Found: {self.performance_metrics['detections_found']}
• Session Duration: {time.time() - self.performance_metrics['session_start']:.0f}s
"""
            
            self.diagnostics_text.config(state='normal')
            self.diagnostics_text.delete(1.0, tk.END)
            self.diagnostics_text.insert(1.0, diagnostics_info)
            self.diagnostics_text.config(state='disabled')
            
        except Exception as e:
            logger.error(f"Diagnostics update failed: {e}")
    
    def reload_gemma(self):
        """Recharge le modèle Gemma 3n"""
        result = messagebox.askyesno("Reload Model", 
            "Reload Gemma 3n multimodal model?\n\n"
            "This will free GPU memory and reload the model.\n"
            "Process may take 3-5 minutes.")
        
        if result:
            def reload_thread():
                try:
                    self.update_gemma_status("Unloading current model...", "orange")
                    
                    # Libérer les ressources actuelles
                    if self.gemma_handler:
                        self.gemma_handler.cleanup_memory()
                        del self.gemma_handler
                    
                    # Vider le cache GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    self.analyze_button.config(state='disabled')
                    
                    # Réinitialiser
                    self.gemma_handler = None
                    self.initialization_complete = False
                    
                    # Relancer l'initialisation
                    self.root.after(2000, self.initialize_system)
                    
                except Exception as e:
                    logger.error(f"Model reload failed: {e}")
                    self.update_gemma_status(f"Reload failed: {e}", "red")
            
            threading.Thread(target=reload_thread, daemon=True).start()
    
    def clear_gpu_memory(self):
        """Nettoie la mémoire GPU"""
        try:
            if torch.cuda.is_available():
                if self.gemma_handler:
                    self.gemma_handler.cleanup_memory()
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                self.update_memory_display()
                messagebox.showinfo("Memory Cleared", "GPU memory cache cleared successfully.")
            else:
                messagebox.showinfo("No GPU", "No CUDA GPU available for memory clearing.")
                
        except Exception as e:
            logger.error(f"GPU memory clear failed: {e}")
            messagebox.showerror("Memory Clear Error", f"Failed to clear GPU memory: {e}")
    
    def show_diagnostics(self):
        """Affiche l'onglet des diagnostics"""
        self.update_diagnostics()
        self.notebook.select(self.diagnostics_frame)
    
    def export_results(self):
        """Exporte les résultats d'analyse"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Results", "No analysis results available to export.")
            return
        
        # Demander le fichier de destination
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
                    f.write(self.current_analysis_results)
                
                self.update_status(f"✅ Results exported: {Path(file_path).name}", "green")
                messagebox.showinfo("Export Complete", 
                    f"Medical analysis results exported successfully!\n\n"
                    f"File: {file_path}\n\n"
                    f"This report can be shared with medical professionals.")
                
            except Exception as e:
                self.update_status(f"❌ Export failed: {e}", "red")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def generate_medical_report(self):
        """Génère un rapport médical HTML interactif"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"retinoblastoma_medical_report_gemma3n_{timestamp}.html"
        
        try:
            # Créer le rapport HTML
            html_report = self.create_html_medical_report()
            
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
    
    def create_html_medical_report(self):
        """Crée un rapport HTML médical professionnel"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = Path(self.current_image_path).name if self.current_image_path else 'Unknown'
        
        # Déterminer s'il y a des détections positives
        has_positive = "MEDICAL ALERT" in self.current_analysis_results
        
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
        .badge-ai {{ background: #9f7aea; }}
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
        .emergency-box {{
            background: #fee; border: 3px solid #ff4757;
            padding: 20px; border-radius: 10px; margin: 20px 0;
            border-left: 8px solid #ff4757;
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
            <div class="subtitle">Advanced AI-Powered Early Detection System</div>
            <div class="badges">
                <span class="badge badge-ai">GEMMA 3N MULTIMODAL</span>
                <span class="badge badge-local">100% LOCAL</span>
                <span class="badge badge-secure">PRIVACY SECURE</span>
            </div>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Image:</strong> {filename}</p>
            <p><strong>Processing:</strong> {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}</p>
        </div>
        
        <div class="content">"""
        
        if has_positive:
            html_report += """
            <div class="alert-critical">
                <h2>🚨 MEDICAL ALERT - IMMEDIATE ACTION REQUIRED</h2>
                <p style="font-size: 20px; font-weight: bold;">
                    Possible retinoblastoma detected. Contact pediatric ophthalmologist IMMEDIATELY.
                </p>
                <div class="emergency-box">
                    <strong>🔴 URGENT STEPS:</strong><br>
                    1. Call pediatric ophthalmologist TODAY<br>
                    2. Bring this report and original images<br>
                    3. Do NOT delay medical evaluation<br>
                    4. Contact emergency services if needed
                </div>
            </div>"""
        else:
            html_report += """
            <div class="alert-safe">
                <h2>✅ No Concerning Findings Detected</h2>
                <p style="font-size: 18px;">
                    The AI analysis did not detect signs of leukocoria in this image.
                    Continue regular pediatric eye health monitoring.
                </p>
            </div>"""
        
        html_report += f"""
            <div class="results-section">
                <h2>📊 Detailed Analysis Results</h2>
                <pre>{self.current_analysis_results}</pre>
            </div>
        </div>
        
        <div class="disclaimer">
            <h3>⚕️ Critical Medical Disclaimer</h3>
            <p><strong>IMPORTANT:</strong> This report is generated by an AI screening system using Gemma 3n multimodal technology.</p>
            <p><strong>THIS IS NOT A MEDICAL DIAGNOSIS</strong> and should NOT replace professional medical evaluation by qualified pediatric ophthalmologists.</p>
            
            <h4>📋 Next Steps:</h4>
            <ul>
                <li><strong>Professional Evaluation:</strong> Schedule consultation with pediatric ophthalmologist</li>
                <li><strong>Documentation:</strong> Bring this report and original images to appointment</li>
                <li><strong>Urgency:</strong> {'IMMEDIATE evaluation required' if has_positive else 'Routine follow-up appropriate'}</li>
                <li><strong>Monitoring:</strong> Continue regular eye health monitoring</li>
            </ul>
        </div>
        
        <div class="footer">
            <p><strong>Generated by RetinoblastoGemma v1.0</strong></p>
            <p>AI-Powered Retinoblastoma Screening with Gemma 3n Multimodal</p>
            <p>🔒 100% Local Processing - Complete Privacy Protection</p>
            <p style="font-size: 12px; margin-top: 15px; opacity: 0.8;">
                Report ID: RG_{timestamp.replace('-', '').replace(':', '').replace(' ', '_')} | 
                Model: Gemma 3n Multimodal | 
                Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | 
                Privacy: Local Only
            </p>
        </div>
    </div>
</body>
</html>"""
        
        return html_report

def main():
    """Fonction principale optimisée pour hackathon avec Gemma 3n local"""
    try:
        # Vérifications préliminaires
        print("🏥 RETINOBLASTOGAMMA - GEMMA 3N MULTIMODAL")
        print("="*60)
        print("🎯 Hackathon Version - Local Privacy-First AI")
        print("="*60)
        
        # Créer les dossiers nécessaires
        for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
            directory.mkdir(exist_ok=True)
        
        # Vérifier Gemma 3n (critique pour le hackathon)
        if not GEMMA_MODEL_PATH.exists():
            print(f"❌ Gemma 3n model not found at: {GEMMA_MODEL_PATH}")
            print(f"💡 Please ensure Gemma 3n (10.17GB) is installed in models/gemma-3n/")
            print(f"🔧 Run: python check_gemma_local.py for diagnosis")
            
            choice = input("Continue anyway? The app will show initialization errors. (y/n): ")
            if choice.lower() != 'y':
                return
        else:
            print(f"✅ Gemma 3n model found: {GEMMA_MODEL_PATH}")
            
            # Vérifier les fichiers essentiels
            essential_files = ["config.json", "tokenizer.json"]
            missing_files = [f for f in essential_files if not (GEMMA_MODEL_PATH / f).exists()]
            
            if missing_files:
                print(f"⚠️ Missing essential files: {missing_files}")
            else:
                print(f"✅ All essential model files present")
        
        # Vérifier les dépendances critiques
        missing_deps = []
        
        if not TORCH_AVAILABLE:
            missing_deps.append("torch")
        
        try:
            from transformers import AutoTokenizer
            print("✅ Transformers available")
        except ImportError:
            missing_deps.append("transformers")
            print("❌ Transformers missing")
        
        try:
            import mediapipe
            print("✅ MediaPipe available for advanced detection")
        except ImportError:
            missing_deps.append("mediapipe")
            print("⚠️ MediaPipe missing - using fallback detection")
        
        if missing_deps:
            print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            print(f"📦 Install with: pip install {' '.join(missing_deps)}")
            
            choice = input("Continue anyway? Some features may be limited. (y/n): ")
            if choice.lower() != 'y':
                return
        
        # Configuration Windows pour UTF-8
        if sys.platform == "win32":
            try:
                os.system("chcp 65001")
            except:
                pass
        
        # Créer et lancer l'application
        root = tk.Tk()
        
        try:
            app = RetinoblastoGemmaApp(root)
            logger.info("RetinoblastoGemma Multimodal started for hackathon")
            print("🚀 Application launched successfully!")
            print("💡 Gemma 3n will initialize in background (may take 3-5 minutes)")
            print("🏆 Ready for hackathon demonstration!")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            messagebox.showerror("Initialization Error", 
                f"Failed to initialize application:\n{e}\n\n"
                "Check dependencies and Gemma 3n installation.\n"
                "Run: python check_gemma_local.py for diagnosis")
            return
        
        # Gestionnaire de fermeture gracieuse
        def on_closing():
            try:
                logger.info("Application closing gracefully...")
                
                # Libérer les ressources Gemma
                if hasattr(app, 'gemma_handler') and app.gemma_handler:
                    app.gemma_handler.cleanup_memory()
                
                # Libérer les ressources eye detector
                if hasattr(app, 'eye_detector') and app.eye_detector:
                    app.eye_detector.cleanup_resources()
                
                # Libérer les ressources GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("👋 RetinoblastoGemma session ended")
                root.quit()
                root.destroy()
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                root.quit()
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Démarrer l'interface
        logger.info("Starting RetinoblastoGemma main loop")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        print("\n📋 Troubleshooting Guide:")
        print("1. Run: python check_gemma_local.py")
        print("2. Check: pip install torch transformers mediapipe")
        print("3. Verify: models/gemma-3n/ contains model files (10.17GB)")
        print("4. Ensure: Sufficient GPU memory (4GB+ recommended)")
        print("5. Try: python quick-start.py for guided setup")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()"""
Interface principale pour RetinoblastoGemma - Version 3 Corrigée
Gemma 3n Multimodal avec architecture modulaire
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import logging
import threading
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retinoblastogamma.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports avec fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} available")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Configuration des chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
GEMMA_MODEL_PATH = MODELS_DIR / "gemma-3n"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"

# Créer les dossiers
for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Import des modules locaux
try:
    from core.gemma_handler import Gemma3nMultimodalHandler
    from core.eye_detector import AdvancedEyeDetector
    CORE_MODULES_AVAILABLE = True
    logger.info("Core modules imported successfully")
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logger.error(f"Core modules import failed: {e}")

class RetinoblastoGemmaApp:
    """Application principale avec Gemma 3n multimodal"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma - Gemma 3n Multimodal")
        self.root.geometry("1500x1000")
        
        # Configuration
        self.current_image_path = None
        self.current_analysis_results = None
        self.gemma_handler = None
        self.eye_detector = None
        self.initialization_complete = False
        
        # Métriques
        self.performance_metrics = {
            'total_analyses': 0,
            'detections_found': 0,
            'processing_times': [],
            'session_start': time.time()
        }
        
        self.setup_ui()
        # Initialiser automatiquement après un délai
        self.root.after(2000, self.initialize_system)
    
    def setup_ui(self):
        """Interface utilisateur optimisée"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Panel de contrôles gauche
        control_frame = ttk.LabelFrame(main_frame, text="Retinoblastoma Detection Controls", padding="15")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # Section chargement d'image
        image_section = ttk.LabelFrame(control_frame, text="Image Loading")
        image_section.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_section, text="Load Medical Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        self.image_info_label = ttk.Label(image_section, text="No image loaded", 
                                         font=("Arial", 9), foreground="gray")
        self.image_info_label.pack(anchor=tk.W, pady=2)
        
        # Section analyse
        analysis_section = ttk.LabelFrame(control_frame, text="AI Analysis")
        analysis_section.pack(fill=tk.X, pady=10)
        
        self.analyze_button = ttk.Button(analysis_section, text="🔍 Analyze for Retinoblastoma", 
                  command=self.analyze_image, state='disabled')
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        # Status Gemma détaillé
        gemma_section = ttk.LabelFrame(control_frame, text="Gemma 3n Multimodal Status")
        gemma_section.pack(fill=tk.X, pady=10)
        
        self.gemma_status = ttk.Label(gemma_section, text="Initializing...", 
                                     foreground="blue", font=("Arial", 10, "bold"))
        self.gemma_status.pack(anchor=tk.W)
        
        self.memory_label = ttk.Label(gemma_section, text="Memory: --", 
                                     font=("Arial", 8))
        self.memory_label.pack(anchor=tk.W)
        
        # Boutons de gestion Gemma
        gemma_buttons = ttk.Frame(gemma_section)
        gemma_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(gemma_buttons, text="Reload Model", 
                  command=self.reload_gemma).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(gemma_buttons, text="Clear Memory", 
                  command=self.clear_gpu_memory).pack(side=tk.LEFT)
        
        # Paramètres d'analyse
        params_section = ttk.LabelFrame(control_frame, text="Detection Parameters")
        params_section.pack(fill=tk.X, pady=10)
        
        self.crop_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Cropped Eye Analysis", 
                       variable=self.crop_detection_var).pack(anchor=tk.W)
        
        self.multimodal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Multimodal Vision Analysis", 
                       variable=self.multimodal_var).pack(anchor=tk.W)
        
        self.enhanced_cv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_section, text="✓ Enhanced Computer Vision", 
                       variable=self.enhanced_cv_var).pack(anchor=tk.W)
        
        # Status et progression
        status_section = ttk.LabelFrame(control_frame, text="System Status")
        status_section.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_section, text="Starting...", 
                                     foreground="blue", font=("Arial", 9))
        self.status_label.pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(status_section, mode='determinate')
        self.progress.pack(fill=tk.X, pady=3)
        
        self.progress_detail = ttk.Label(status_section, text="", 
                                        font=("Arial", 8), foreground="gray")
        self.progress_detail.pack(anchor=tk.W)
        
        # Métriques de performance
        metrics_section = ttk.LabelFrame(control_frame, text="Performance Metrics")
        metrics_section.pack(fill=tk.X, pady=10)
        
        self.metrics_label = ttk.Label(metrics_section, text="No analysis yet", 
                                      font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Actions et exports
        actions_section = ttk.LabelFrame(control_frame, text="Actions")
        actions_section.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_section, text="📄 Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="🏥 Medical Report", 
                  command=self.generate_medical_report).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_section, text="📊 System Diagnostics", 
                  command=self.show_diagnostics).pack(fill=tk.X, pady=1)
        
        # Zone d'affichage principal
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook pour onglets
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Onglet analyse d'image
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="🖼️ Image Analysis")
        
        # Canvas avec barres de défilement
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
        
        # Onglet résultats détaillés
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
        
        # Onglet diagnostics système
        self.diagnostics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.diagnostics_frame, text="🔧 System Info")
        
        self.diagnostics_text = tk.Text(self.diagnostics_frame, wrap=tk.WORD, 
                                       font=("Consolas", 9), state='disabled')
        diag_scrollbar = ttk.Scrollbar(self.diagnostics_frame, orient="vertical", 
                                      command=self.diagnostics_text.yview)
        self.diagnostics_text.configure(yscrollcommand=diag_scrollbar.set)
        
        self.diagnostics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        diag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Barre de statut en bas
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.statusbar = ttk.Label(status_frame, text="RetinoblastoGemma Ready", 
                                  relief=tk.SUNKEN, font=("Arial", 9))
        self.statusbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Afficher les infos système initiales
        self.update_diagnostics()
    
    def initialize_system(self):
        """Initialise Gemma 3n multimodal avec optimisations mémoire"""
        def init_thread():
            try:
                self.update_status("🔄 Checking system requirements...", "blue")
                self.update_progress(5, "Validating dependencies...")
                
                # Vérifications critiques
                if not TORCH_AVAILABLE:
                    self.update_status("❌ PyTorch not available", "red")
                    return
                
                if not CORE_MODULES_AVAILABLE:
                    self.update_status("❌ Core modules not available", "red")
                    return
                
                if not GEMMA_MODEL_PATH.exists():
                    self.update_status("❌ Gemma 3n model not found", "red")
                    self.update_gemma_status("Model files missing", "red")
                    return
                
                self.update_progress(15, "Initializing eye detector...")
                
                # Initialiser le détecteur d'yeux
                try:
                    self.eye_detector = AdvancedEyeDetector()
                    logger.info("Eye detector initialized")
                except Exception as e:
                    logger.error(f"Eye detector initialization failed: {e}")
                    self.eye_detector = None
                
                self.update_progress(25, "Initializing Gemma 3n handler...")
                
                # Créer le gestionnaire Gemma
                self.gemma_handler = Gemma3nMultimodalHandler(GEMMA_MODEL_PATH)
                
                self.update_progress(30, "Loading Gemma 3n multimodal model...")
                self.update_gemma_status("Loading model (may take 3-5 minutes)...", "orange")
                
                # Charger le modèle avec callback de progression
                def progress_callback(percent, message):
                    progress_value = 30 + (percent * 0.65)  # 30% à 95%
                    self.update_progress(progress_value, message)
                    self.update_gemma_status(f"{message} ({percent:.0f}%)", "blue")
                
                success = self.gemma_handler.load_model_optimized(progress_callback)
                
                if success:
                    self.update_progress(98, "Finalizing initialization...")
                    
                    # Mise à jour des statuts
                    self.initialization_complete = True
                    self.update_progress(100, "System ready!")
                    self.update_status("✅ Gemma 3n Multimodal ready for analysis!", "green")
                    self.update_gemma_status("Ready - Multimodal Vision Active", "green")
                    
                    # Activer l'analyse
                    self.analyze_button.config(state='normal')
                    
                    # Mise à jour mémoire
                    self.update_memory_display()
                    
                    # Message de succès
                    messagebox.showinfo("System Ready", 
                        "🎉 Gemma 3n Multimodal loaded successfully!\n\n"
                        "✅ Vision capabilities active\n"
                        "✅ Ready for retinoblastoma detection\n"
                        "✅ Optimized for your hardware\n\n"
                        "Load a medical image to start analysis.")
                    
                else:
                    self.update_status("❌ Failed to load Gemma 3n model", "red")
                    self.update_gemma_status("Model loading failed", "red")
                    
                    messagebox.showerror("Model Loading Failed", 
                        "Failed to load Gemma 3n multimodal model.\n\n"
                        "Possible causes:\n"
                        "• Insufficient GPU memory\n"
                        "• Missing model files\n"
                        "• Incompatible model format\n\n"
                        "Check the diagnostics tab for details.")
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                self.update_status(f"❌ Initialization error: {e}", "red")
                self.update_gemma_status(f"Error: {e}", "red")
                
                messagebox.showerror("Initialization Error