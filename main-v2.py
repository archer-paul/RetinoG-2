    # [Le reste des méthodes essentielles...]
    
    def load_image(self):
        """Charge une image"""
        file_path = filedialog.askopenfilename(
            title="Select medical image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.update_status(f"Loaded: {Path(file_path).name}")
            logger.info(f"Image loaded: {file_path}")
    
    def display_image(self, image_path):
        """Affiche l'image dans le canvas"""
        try:
            image = Image.open(image_path)
            
            # Redimensionnement adaptatif
            canvas_width = max(800, self.canvas.winfo_width())
            canvas_height = max(600, self.canvas.winfo_height())
            
            # Conserver les proportions
            image.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            
            self.canvas.delete("all")
            self.canvas.create_image(20, 20, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("Display Error", f"Cannot display image: {e}")
    
    def analyze_image(self):
        """Analyse l'image avec Gemma 3n local"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.gemma_handler or not self.gemma_handler.initialized:
            messagebox.showwarning("Gemma Not Ready", "Gemma 3n model not loaded.")
            return
        
        def analysis_thread():
            try:
                start_time = time.time()
                self.update_status("Analyzing image for retinoblastoma...", "blue")
                self.update_progress(0)
                
                # Étape 1: Détection des régions d'yeux
                self.update_progress(20)
                eye_regions = self.detect_eye_regions()
                
                if not eye_regions:
                    self.update_status("No eye regions detected - check image quality", "orange")
                    messagebox.showwarning("No Eyes Detected", 
                        "No eye regions could be detected in this image.\n"
                        "Please ensure the image contains visible eyes.")
                    return
                
                logger.info(f"Detected {len(eye_regions)} eye regions")
                
                # Étape 2: Analyse avec Gemma 3n local
                self.update_progress(40)
                analysis_results = []
                
                for i, eye_region in enumerate(eye_regions):
                    progress = 40 + (i * 40) // len(eye_regions)
                    self.update_progress(progress)
                    self.update_status(f"Analyzing {eye_region['position']} eye with Gemma 3n...", "blue")
                    
                    result = self.gemma_handler.analyze_eye_image(
                        eye_region['image'], 
                        eye_region['position']
                    )
                    
                    result['eye_region'] = eye_region
                    analysis_results.append(result)
                    
                    logger.info(f"Analysis complete for {eye_region['position']} eye: "
                              f"Leukocoria={result.get('leukocoria_detected', False)}")
                
                # Étape 3: Visualisation des résultats
                self.update_progress(85)
                self.update_status("Generating visual results...", "blue")
                self.visualize_results(analysis_results)
                
                # Étape 4: Affichage des résultats
                self.update_progress(95)
                self.display_results(analysis_results)
                
                # Métriques finales
                processing_time = time.time() - start_time
                self.performance_metrics['total_analyses'] += 1
                self.performance_metrics['processing_times'].append(processing_time)
                
                positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
                if positive_detections > 0:
                    self.performance_metrics['detections_found'] += 1
                    self.update_status(f"ANALYSIS COMPLETE! WARNING: Possible retinoblastoma detected ({processing_time:.1f}s)", "red")
                    messagebox.showwarning("Positive Detection", 
                        f"⚠️ MEDICAL ALERT ⚠️\n\n"
                        f"Possible retinoblastoma detected in {positive_detections} eye(s).\n"
                        f"IMMEDIATE medical consultation recommended!")
                else:
                    self.update_status(f"Analysis complete! No concerning findings detected ({processing_time:.1f}s)", "green")
                
                self.update_progress(100)
                self.update_metrics()
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.update_status(f"Analysis failed: {e}", "red")
                messagebox.showerror("Analysis Error", f"Analysis failed: {e}")
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def detect_eye_regions(self):
        """Détecte les régions d'yeux - optimisé pour images croppées"""
        try:
            image = cv2.imread(self.current_image_path)
            if image is None:
                return []
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            eye_regions = []
            
            # Méthode 1: Détection MediaPipe (si disponible)
            try:
                import mediapipe as mp
                
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.2
                )
                
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        eye_regions.extend(self.extract_eyes_from_landmarks(face_landmarks, image))
                        
            except ImportError:
                logger.warning("MediaPipe not available - using fallback detection")
            
            # Méthode 2: Si pas de visage détecté ET mode cropped activé
            if not eye_regions and self.crop_detection_var.get():
                logger.info("No face detected, analyzing as cropped eye image")
                
                # Analyser l'image entière comme région d'oeil
                if w > h * 1.8:  # Image très horizontale = probablement deux yeux
                    mid_x = w // 2
                    eye_regions = [
                        {
                            'position': 'left',
                            'bbox': (0, 0, mid_x, h),
                            'image': Image.fromarray(rgb_image[:, :mid_x]),
                            'confidence': 0.7,
                            'is_cropped': True
                        },
                        {
                            'position': 'right',
                            'bbox': (mid_x, 0, w - mid_x, h),
                            'image': Image.fromarray(rgb_image[:, mid_x:]),
                            'confidence': 0.7,
                            'is_cropped': True
                        }
                    ]
                else:  # Image carrée ou verticale = un oeil
                    eye_regions.append({
                        'position': 'center',
                        'bbox': (0, 0, w, h),
                        'image': Image.fromarray(rgb_image),
                        'confidence': 0.8,
                        'is_cropped': True
                    })
            
            logger.info(f"Detected {len(eye_regions)} eye regions")
            return eye_regions
            
        except Exception as e:
            logger.error(f"Eye detection failed: {e}")
            return []
    
    def extract_eyes_from_landmarks(self, face_landmarks, image):
        """Extrait les yeux des landmarks MediaPipe"""
        h, w = image.shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        eye_regions = []
        
        # Indices MediaPipe pour les yeux
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for eye_name, indices in [('left', left_eye_indices), ('right', right_eye_indices)]:
            try:
                eye_points = [landmarks[i] for i in indices if i < len(landmarks)]
                if len(eye_points) >= 6:
                    xs = [p[0] for p in eye_points]
                    ys = [p[1] for p in eye_points]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Marge pour capturer toute la région
                    margin = 30
                    x1 = max(0, x_min - margin)
                    y1 = max(0, y_min - margin)
                    x2 = min(w, x_max + margin)
                    y2 = min(h, y_max + margin)
                    
                    if x2 > x1 and y2 > y1:
                        eye_image = image[y1:y2, x1:x2]
                        rgb_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
                        
                        eye_regions.append({
                            'position': eye_name,
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'image': Image.fromarray(rgb_eye),
                            'confidence': 0.9,
                            'is_cropped': False
                        })
            except Exception as e:
                logger.warning(f"Failed to extract {eye_name} eye: {e}")
        
        return eye_regions
    
    def visualize_results(self, analysis_results):
        """Visualise les résultats avec boîtes colorées"""
        try:
            original_image = Image.open(self.current_image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Dessiner les boîtes pour chaque région
            for result in analysis_results:
                eye_region = result.get('eye_region', {})
                bbox = eye_region.get('bbox')
                
                if not bbox:
                    continue
                
                x, y, w, h = bbox
                leukocoria_detected = result.get('leukocoria_detected', False)
                confidence = result.get('confidence', 0)
                risk_level = result.get('risk_level', 'low')
                
                # Couleur selon le résultat
                if leukocoria_detected and risk_level == 'high':
                    color = 'red'
                    width = 5
                elif leukocoria_detected and risk_level == 'medium':
                    color = 'orange'
                    width = 4
                elif leukocoria_detected:
                    color = 'yellow'
                    width = 3
                else:
                    color = 'green'
                    width = 3
                
                # Dessiner le rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
                
                # Label avec informations
                position = eye_region.get('position', 'unknown')
                urgency = result.get('urgency', 'routine')
                
                label_parts = [position.upper()]
                if leukocoria_detected:
                    label_parts.append(f"⚠️ {urgency.upper()}")
                label_parts.append(f"{confidence:.0f}%")
                
                label = " ".join(label_parts)
                
                # Position du texte
                text_y = y - 30 if y > 30 else y + h + 5
                
                # Fond pour le texte
                text_bbox = draw.textbbox((x, text_y), label)
                draw.rectangle(text_bbox, fill=color, outline=color)
                draw.text((x, text_y), label, fill='white' if color != 'yellow' else 'black')
            
            # Sauvegarder l'image annotée
            timestamp = int(time.time())
            annotated_path = RESULTS_DIR / f"retinoblastoma_analysis_{timestamp}.jpg"
            original_image.save(annotated_path, quality=95)
            
            # Afficher l'image annotée
            self.display_annotated_image(original_image)
            
            logger.info(f"Results visualized and saved to {annotated_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def display_annotated_image(self, annotated_image):
        """Affiche l'image annotée"""
        try:
            canvas_width = max(800, self.canvas.winfo_width())
            canvas_height = max(600, self.canvas.winfo_height())
            
            display_image = annotated_image.copy()
            display_image.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(display_image)
            
            self.canvas.delete("all")
            self.canvas.create_image(20, 20, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying annotated image: {e}")
    
    def display_results(self, analysis_results):
        """Affiche les résultats détaillés"""
        report = f"""RETINOBLASTOMA ANALYSIS REPORT - LOCAL GEMMA 3N
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {Path(self.current_image_path).name if self.current_image_path else 'Unknown'}
AI Engine: Gemma 3n (Local Model - 100% Offline)

DETECTION SUMMARY:
{'='*30}
"""
        
        total_eyes = len(analysis_results)
        positive_detections = sum(1 for r in analysis_results if r.get('leukocoria_detected', False))
        high_risk_count = sum(1 for r in analysis_results if r.get('risk_level') == 'high')
        immediate_cases = sum(1 for r in analysis_results if r.get('urgency') == 'immediate')
        
        report += f"Eyes analyzed: {total_eyes}\n"
        report += f"Positive detections: {positive_detections}\n"
        report += f"High-risk findings: {high_risk_count}\n"
        report += f"Immediate attention needed: {immediate_cases}\n\n"
        
        if positive_detections > 0:
            report += "🚨 MEDICAL ALERT: POSSIBLE RETINOBLASTOMA DETECTED\n"
            report += "IMMEDIATE PEDIATRIC OPHTHALMOLOGICAL CONSULTATION REQUIRED\n\n"
        else:
            report += "✅ No concerning findings detected\n"
            report += "Continue regular pediatric eye monitoring\n\n"
        
        report += "DETAILED ANALYSIS BY EYE:\n"
        report += "="*40 + "\n"
        
        for i, result in enumerate(analysis_results, 1):
            eye_region = result.get('eye_region', {})
            position = eye_region.get('position', 'unknown')
            is_cropped = eye_region.get('is_cropped', False)
            
            report += f"\n--- Eye {i}: {position.upper()} {'(Cropped Image)' if is_cropped else ''} ---\n"
            report += f"Leukocoria detected: {'⚠️ YES' if result.get('leukocoria_detected') else '✅ NO'}\n"
            report += f"Confidence level: {result.get('confidence', 0):.1f}%\n"
            report += f"Risk assessment: {result.get('risk_level', 'unknown').upper()}\n"
            report += f"Urgency level: {result.get('urgency', 'routine').upper()}\n"
            
            if result.get('affected_eye'):
                report += f"Affected eye: {result.get('affected_eye')}\n"
            
            pupil_desc = result.get('pupil_description', 'No description available')
            report += f"Pupil analysis: {pupil_desc}\n"
            
            reasoning = result.get('medical_reasoning', 'No reasoning provided')
            # Limiter la longueur pour la lisibilité
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            report += f"Medical reasoning: {reasoning}\n"
            
            recommendations = result.get('recommendations', 'No specific recommendations')
            report += f"Recommendations: {recommendations}\n"
            
            if result.get('error'):
                report += f"⚠️ Analysis notes: {result.get('error')}\n"
        
        report += f"\nTECHNICAL DETAILS:\n"
        report += "="*30 + "\n"
        report += f"AI Model: Gemma 3n (Local - Offline)\n"
        report += f"Model location: {GEMMA_MODEL_PATH}\n"
        report += f"Processing device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n"
        report += f"Cropped image analysis: {'Enabled' if self.crop_detection_var.get() else 'Disabled'}\n"
        report += f"Enhanced CV analysis: {'Enabled' if self.enhanced_analysis_var.get() else 'Disabled'}\n"
        
        if self.performance_metrics['processing_times']:
            avg_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            report += f"Average processing time: {avg_time:.1f}s\n"
        
        report += f"\nCRITICAL MEDICAL DISCLAIMER:\n"
        report += "="*40 + "\n"
        report += "⚠️ IMPORTANT: This analysis is provided by an AI screening system.\n"
        report += "This is NOT a medical diagnosis and should NOT replace professional\n"
        report += "medical evaluation by qualified pediatric ophthalmologists.\n\n"
        
        report += "IMMEDIATE ACTION REQUIRED if positive findings:\n"
        report += "1. ⏰ Contact pediatric ophthalmologist IMMEDIATELY\n"
        report += "2. 📋 Bring this report and original images to appointment\n"
        report += "3. 🚫 Do NOT delay seeking professional medical evaluation\n"
        report += "4. 📞 Emergency: Call your healthcare provider or emergency services\n\n"
        
        report += "Retinoblastoma facts:\n"
        report += "- Most common eye cancer in children (under 6 years)\n"
        report += "- 95% survival rate with EARLY detection and treatment\n"
        report += "- Can affect one or both eyes\n"
        report += "- Treatment options depend on stage and location\n"
        
        # Afficher dans l'onglet résultats
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        
        # Basculer vers l'onglet résultats
        self.notebook.select(self.results_frame)
        
        # Sauvegarder les résultats
        self.current_analysis_results = report
        
        logger.info("Analysis results displayed")
    
    def reload_gemma(self):
        """Recharge le modèle Gemma 3n"""
        def reload_thread():
            try:
                self.update_gemma_status("Reloading model...", "blue")
                self.gemma_handler = LocalGemmaHandler(GEMMA_MODEL_PATH)
                
                success = self.gemma_handler.load_model(
                    lambda p, m: self.update_gemma_status(f"{m} ({p:.0f}%)", "blue")
                )
                
                if success:
                    self.update_gemma_status("Model reloaded successfully", "green")
                    self.analyze_button.config(state='normal')
                else:
                    self.update_gemma_status("Model reload failed", "red")
                    self.analyze_button.config(state='disabled')
                    
            except Exception as e:
                self.update_gemma_status(f"Reload error: {e}", "red")
        
        threading.Thread(target=reload_thread, daemon=True).start()
    
    def export_results(self):
        """Exporte les résultats"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Medical Analysis",
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
                
                self.update_status(f"Results exported to {Path(file_path).name}", "green")
                messagebox.showinfo("Export Complete", f"Medical report exported to:\n{file_path}")
                
            except Exception as e:
                self.update_status(f"Export failed: {e}", "red")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def generate_report(self):
        """Génère un rapport médical HTML"""
        if not self.current_analysis_results:
            messagebox.showwarning("No Analysis", "Please perform an analysis first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"retinoblastoma_medical_report_{timestamp}.html"
        
        try:
            # Déterminer s'il y a des détections positives
            has_positive = "MEDICAL ALERT" in self.current_analysis_results
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Retinoblastoma Medical Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            line-height: 1.6;
            color: #333;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 12px; 
            text-align: center;
            margin-bottom: 30px;
        }}
        .alert-positive {{ 
            background-color: #fff5f5; 
            border: 3px solid #e53e3e;
            padding: 20px; 
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }}
        .alert-negative {{ 
            background-color: #f0fff4; 
            border: 3px solid #38a169;
            padding: 20px; 
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }}
        .results {{ 
            background-color: #f8f9fa;
            padding: 25px; 
            border-radius: 8px;
            margin: 20px 0;
        }}
        .disclaimer {{ 
            background-color: #fffbf0; 
            border: 2px solid #ed8936;
            padding: 20px; 
            border-radius: 8px;
            margin-top: 30px;
        }}
        pre {{ 
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 20px; 
            border-radius: 8px; 
            overflow-x: auto; 
            font-size: 14px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin: 2px;
        }}
        .badge-local {{ background-color: #4299e1; }}
        .badge-ai {{ background-color: #9f7aea; }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
        }}
        .emergency-contact {{
            background-color: #fed7d7;
            border: 2px solid #fc8181;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        h1, h2, h3 {{ color: #2d3748; }}
        .tech-badge {{ font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Retinoblastoma Medical Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div>
            <span class="badge badge-local tech-badge">LOCAL AI</span>
            <span class="badge badge-ai tech-badge">GEMMA 3N</span>
            <span class="badge tech-badge" style="background-color: #68d391;">OFFLINE</span>
        </div>
        <p><strong>Analysis System:</strong> Gemma 3n (100% Local Processing)</p>
        <p><strong>Image:</strong> {Path(self.current_image_path).name if self.current_image_path else 'Unknown'}</p>
    </div>"""
            
            if has_positive:
                html_content += """
    <div class="alert-positive">
        <h2>🚨 MEDICAL ALERT - IMMEDIATE ACTION REQUIRED</h2>
        <p style="font-size: 18px; font-weight: bold; color: #e53e3e;">
            Possible retinoblastoma detected. Contact pediatric ophthalmologist IMMEDIATELY.
        </p>
        <div class="emergency-contact">
            <strong>Emergency Actions:</strong><br>
            1. Call pediatric ophthalmologist TODAY<br>
            2. Bring this report and original images<br>
            3. Do NOT delay medical evaluation
        </div>
    </div>"""
            else:
                html_content += """
    <div class="alert-negative">
        <h2>✅ No Concerning Findings Detected</h2>
        <p style="font-size: 16px; color: #38a169;">
            The AI analysis did not detect signs of leukocoria in this image.
            Continue regular pediatric eye monitoring.
        </p>
    </div>"""
            
            html_content += f"""
    <div class="results">
        <h2>📊 Detailed Analysis Results</h2>
        <pre>{self.current_analysis_results}</pre>
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
    
    <div class="footer">
        <p><strong>Generated by RetinoblastoGemma v1.0</strong></p>
        <p>AI-Powered Retinoblastoma Screening with Local Gemma 3n</p>
        <p>🔒 100% Local Processing - No data transmitted</p>
        <p style="font-size: 12px; margin-top: 10px;">
            Report ID: RG_{timestamp} | 
            Model: Gemma 3n Local | 
            Processing: {'GPU' if torch.cuda.is_available() else 'CPU'}
        </p>
    </div>
</body>
</html>"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Ouvrir le rapport dans le navigateur
            import webbrowser
            webbrowser.open(f"file://{report_path.absolute()}")
            
            self.update_status(f"Medical report generated: {report_path.name}", "green")
            messagebox.showinfo("Report Generated", 
                f"Medical report saved and opened:\n{report_path}\n\n"
                f"This report can be shared with medical professionals.")
            
        except Exception as e:
            self.update_status(f"Report generation failed: {e}", "red")
            messagebox.showerror("Report Error", f"Failed to generate report:\n{e}")
    
    def update_status(self, message, color="blue"):
        """Met à jour le statut principal"""
        self.root.after(0, lambda: self.status_label.config(text=message, foreground=color))
        self.root.after(0, lambda: self.statusbar.config(text=message))
        logger.info(f"Status: {message}")
    
    def update_gemma_status(self, message, color="blue"):
        """Met à jour le statut Gemma"""
        self.root.after(0, lambda: self.gemma_status.config(text=message, foreground=color))
    
    def update_progress(self, value):
        """Met à jour la barre de progression"""
        self.root.after(0, lambda: self.progress.config(value=value))
    
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

def main():
    """Fonction principale optimisée pour Gemma 3n local"""
    try:
        # Vérifications préliminaires
        print("🏥 RETINOBLASTOGAMMA - LOCAL GEMMA 3N")
        print("="*50)
        
        # Créer les dossiers nécessaires
        for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
            directory.mkdir(exist_ok=True)
        
        # Vérifier Gemma 3n
        if not GEMMA_MODEL_PATH.exists():
            print(f"❌ Gemma 3n model not found at: {GEMMA_MODEL_PATH}")
            print(f"💡 Please ensure Gemma 3n is installed in models/gemma-3n/")
            print(f"🔧 Run: python check_gemma_local.py for diagnosis")
            input("Press Enter to continue anyway (will show error in app)...")
        else:
            print(f"✅ Gemma 3n model found at: {GEMMA_MODEL_PATH}")
        
        # Vérifier les dépendances critiques
        missing_deps = []
        
        if not TORCH_AVAILABLE:
            missing_deps.append("torch")
        if not TRANSFORMERS_AVAILABLE:
            missing_deps.append("transformers")
        
        try:
            import mediapipe
        except ImportError:
            missing_deps.append("mediapipe")
        
        if missing_deps:
            print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            print(f"📦 Install with: pip install {' '.join(missing_deps)}")
            choice = input("Continue anyway? (y/n): ")
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
            app = RetinoblastoGemmaLocal(root)
            logger.info("RetinoblastoGemma with Local Gemma 3n started")
            print("🚀 Application launched successfully!")
            print("💡 The app will initialize Gemma 3n after startup")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            messagebox.showerror("Initialization Error", 
                f"Failed to initialize application:\n{e}\n\n"
                "Check dependencies and Gemma 3n installation.")
            return
        
        # Gestionnaire de fermeture
        def on_closing():
            try:
                logger.info("Application closing...")
                # Libérer les ressources GPU si nécessaire
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                root.quit()
                root.destroy()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Démarrer l'interface
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        print("📋 Troubleshooting:")
        print("1. Run: python check_gemma_local.py")
        print("2. Check: pip install torch transformers mediapipe")
        print("3. Verify: models/gemma-3n/ exists and contains model files")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()"""
Interface principale pour RetinoblastoGemma - Version Locale Gemma 3n
Utilise le modèle Gemma 3n local pour hackathon (100% offline)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import logging
from pathlib import Path
import threading
import json
import time
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration logging sans emojis pour Windows
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
    logger.info("PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

# Configuration des chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
GEMMA_MODEL_PATH = MODELS_DIR / "gemma-3n"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"

# Créer les dossiers
for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

class LocalGemmaHandler:
    """Gestionnaire pour Gemma 3n local"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        logger.info(f"Initializing Local Gemma Handler with device: {self.device}")
    
    def load_model(self, progress_callback=None):
        """Charge le modèle Gemma 3n local"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Gemma model not found at {self.model_path}")
            
            logger.info(f"Loading Gemma 3n from {self.model_path}")
            
            if progress_callback:
                progress_callback(20, "Loading tokenizer...")
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Définir le pad token si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if progress_callback:
                progress_callback(50, "Loading model...")
            
            # Configuration du modèle optimisée
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Charger le modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_kwargs
            )
            
            if progress_callback:
                progress_callback(80, "Optimizing model...")
            
            # Optimisations
            if torch.cuda.is_available():
                self.model = self.model.half()
            
            self.model.eval()
            
            if progress_callback:
                progress_callback(100, "Model ready!")
            
            self.initialized = True
            logger.info("Gemma 3n local model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n: {e}")
            return False
    
    def analyze_eye_image(self, image_pil, eye_position="unknown"):
        """Analyse une image d'oeil avec Gemma 3n"""
        if not self.initialized:
            return self._create_fallback_result("Model not initialized")
        
        try:
            # Préparer le prompt médical spécialisé
            prompt = self._create_medical_prompt(eye_position)
            
            # Convertir l'image en description textuelle pour l'analyse
            # (Gemma 3n est text-only, donc on analyse les caractéristiques visuelles)
            image_features = self._extract_image_features(image_pil)
            
            # Créer le prompt complet
            full_prompt = f"{prompt}\n\nImage analysis features:\n{image_features}\n\nProvide medical analysis in JSON format:"
            
            # Tokenisation
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Génération avec paramètres médicaux optimisés
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # Très faible pour précision médicale
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Décoder la réponse
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parser la réponse JSON
            result = self._parse_medical_response(response, image_features)
            
            logger.info(f"Gemma 3n analysis complete for {eye_position} eye")
            return result
            
        except Exception as e:
            logger.error(f"Gemma 3n analysis failed: {e}")
            return self._create_fallback_result(f"Analysis error: {e}")
    
    def _create_medical_prompt(self, eye_position):
        """Crée un prompt médical spécialisé pour rétinoblastome"""
        return f"""You are a specialized medical AI for retinoblastoma detection.

MEDICAL CONTEXT:
Retinoblastoma is a serious eye cancer in children. The main early sign is leukocoria (white pupil reflex).

TASK:
Analyze the eye image features for signs of retinoblastoma/leukocoria.

EYE POSITION: {eye_position}

ANALYSIS CRITERIA:
1. Pupil appearance: Normal (dark) vs. Abnormal (white/gray/bright)
2. Symmetry compared to normal eye
3. Reflection patterns indicating possible tumor
4. Overall suspicion level

RESPONSE FORMAT (JSON):
{{
    "leukocoria_detected": boolean,
    "confidence": float (0-100),
    "risk_level": "low|medium|high",
    "affected_eye": "left|right|unknown",
    "pupil_description": "detailed description",
    "medical_reasoning": "clinical analysis",
    "recommendations": "immediate medical advice",
    "urgency": "routine|soon|urgent|immediate"
}}

IMPORTANT: Be conservative - err on side of caution for child safety."""
    
    def _extract_image_features(self, image_pil):
        """Extrait les caractéristiques visuelles de l'image pour l'analyse"""
        try:
            # Convertir en array numpy
            image_array = np.array(image_pil)
            
            if len(image_array.shape) == 3:
                # Image couleur
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Analyse des caractéristiques
            features = {
                "brightness_mean": float(np.mean(gray)),
                "brightness_std": float(np.std(gray)),
                "brightness_max": float(np.max(gray)),
                "brightness_min": float(np.min(gray)),
                "image_size": f"{image_pil.width}x{image_pil.height}",
                "contrast_score": float(np.std(gray)),
            }
            
            # Détection de cercles (pupilles potentielles) avec OpenCV
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=min(gray.shape)//4
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                features["circles_detected"] = len(circles)
                
                # Analyser le cercle le plus central (probable pupille)
                center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
                best_circle = None
                min_distance = float('inf')
                
                for (x, y, r) in circles:
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_circle = (x, y, r)
                
                if best_circle:
                    x, y, r = best_circle
                    # Analyser la région pupillaire
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    pupil_region = gray[mask > 0]
                    
                    if len(pupil_region) > 0:
                        features["pupil_brightness"] = float(np.mean(pupil_region))
                        features["pupil_contrast"] = float(np.std(pupil_region))
                        features["pupil_radius"] = int(r)
                        features["pupil_position"] = f"({x}, {y})"
                        
                        # Indicateur de leucocorie basé sur la luminosité
                        global_brightness = features["brightness_mean"]
                        pupil_brightness = features["pupil_brightness"]
                        
                        # Score de leucocorie (plus élevé = plus suspect)
                        leukocoria_score = max(0, (pupil_brightness - global_brightness) / (255 - global_brightness))
                        features["leukocoria_score"] = float(leukocoria_score)
                        
                        # Classification préliminaire
                        if leukocoria_score > 0.3:
                            features["preliminary_assessment"] = "SUSPICIOUS - Bright pupil detected"
                        elif leukocoria_score > 0.15:
                            features["preliminary_assessment"] = "MONITOR - Slightly bright pupil"
                        else:
                            features["preliminary_assessment"] = "NORMAL - Dark pupil"
            else:
                features["circles_detected"] = 0
                features["preliminary_assessment"] = "NO_PUPIL_DETECTED"
            
            # Créer une description textuelle
            description = f"""
Image Analysis Results:
- Size: {features['image_size']}
- Overall brightness: {features['brightness_mean']:.1f} (std: {features['brightness_std']:.1f})
- Contrast: {features['contrast_score']:.1f}
- Circles detected: {features['circles_detected']}
"""
            
            if "pupil_brightness" in features:
                description += f"""- Pupil brightness: {features['pupil_brightness']:.1f}
- Pupil contrast: {features['pupil_contrast']:.1f}
- Leukocoria score: {features.get('leukocoria_score', 0):.2f}
- Preliminary assessment: {features['preliminary_assessment']}"""
            
            return description
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return f"Feature extraction failed: {e}"
    
    def _parse_medical_response(self, response, image_features):
        """Parse la réponse médicale de Gemma"""
        try:
            # Chercher le JSON dans la réponse
            response = response.strip()
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validation et nettoyage
                result['leukocoria_detected'] = bool(result.get('leukocoria_detected', False))
                result['confidence'] = max(0, min(100, float(result.get('confidence', 50))))
                
                if result.get('risk_level') not in ['low', 'medium', 'high']:
                    result['risk_level'] = 'medium'
                
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse Gemma response: {e}")
            
            # Fallback: analyse basée sur les caractéristiques d'image
            return self._create_feature_based_result(image_features, response)
    
    def _create_feature_based_result(self, image_features, raw_response):
        """Crée un résultat basé sur l'analyse des caractéristiques d'image"""
        try:
            # Extraire le score de leucocorie des caractéristiques
            leukocoria_score = 0
            if "leukocoria_score" in image_features:
                import re
                match = re.search(r'leukocoria_score: ([\d.]+)', image_features)
                if match:
                    leukocoria_score = float(match.group(1))
            
            # Déterminer la détection basée sur le score
            detected = leukocoria_score > 0.2
            confidence = min(95, leukocoria_score * 100) if detected else max(20, (1 - leukocoria_score) * 100)
            
            if leukocoria_score > 0.4:
                risk_level = "high"
                urgency = "immediate"
            elif leukocoria_score > 0.2:
                risk_level = "medium"
                urgency = "urgent"
            else:
                risk_level = "low"
                urgency = "routine"
            
            return {
                "leukocoria_detected": detected,
                "confidence": confidence,
                "risk_level": risk_level,
                "affected_eye": "unknown",
                "pupil_description": f"Computer vision analysis - leukocoria score: {leukocoria_score:.2f}",
                "medical_reasoning": f"Analysis based on image brightness patterns. {raw_response[:100]}...",
                "recommendations": "Professional ophthalmological evaluation recommended" if detected else "Continue regular monitoring",
                "urgency": urgency,
                "analysis_method": "feature_based_fallback"
            }
            
        except Exception as e:
            logger.error(f"Feature-based analysis failed: {e}")
            return self._create_fallback_result(f"Analysis failed: {e}")
    
    def _create_fallback_result(self, error_msg):
        """Crée un résultat de fallback en cas d'erreur"""
        return {
            "leukocoria_detected": False,
            "confidence": 0,
            "risk_level": "unknown",
            "affected_eye": "unknown",
            "pupil_description": "Analysis failed",
            "medical_reasoning": error_msg,
            "recommendations": "Manual professional evaluation required",
            "urgency": "soon",
            "error": error_msg
        }

class RetinoblastoGemmaLocal:
    """Application principale avec Gemma 3n local"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RetinoblastoGemma - Local Gemma 3n")
        self.root.geometry("1400x900")
        
        # Configuration
        self.current_image_path = None
        self.current_analysis_results = None
        self.gemma_handler = None
        self.initialization_complete = False
        
        # Métriques
        self.performance_metrics = {
            'total_analyses': 0,
            'detections_found': 0,
            'processing_times': []
        }
        
        self.setup_ui()
        # Initialiser automatiquement
        self.root.after(1000, self.initialize_system)
    
    def setup_ui(self):
        """Interface utilisateur"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Panel de contrôles
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Boutons principaux
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(control_frame, text="Analyze for Retinoblastoma", 
                  command=self.analyze_image, state='disabled')
        self.analyze_button.pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Status Gemma
        gemma_frame = ttk.LabelFrame(control_frame, text="Gemma 3n Local Model")
        gemma_frame.pack(fill=tk.X, pady=5)
        
        self.gemma_status = ttk.Label(gemma_frame, text="Initializing...", foreground="blue")
        self.gemma_status.pack(anchor=tk.W)
        
        ttk.Button(gemma_frame, text="Reload Model", 
                  command=self.reload_gemma).pack(fill=tk.X, pady=2)
        
        # Paramètres
        params_frame = ttk.LabelFrame(control_frame, text="Analysis Parameters")
        params_frame.pack(fill=tk.X, pady=5)
        
        self.crop_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Enable Cropped Eye Analysis", 
                       variable=self.crop_detection_var).pack(anchor=tk.W)
        
        self.enhanced_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Enhanced Computer Vision", 
                       variable=self.enhanced_analysis_var).pack(anchor=tk.W)
        
        # Status général
        status_frame = ttk.LabelFrame(control_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Starting...", foreground="blue")
        self.status_label.pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)
        
        self.metrics_label = ttk.Label(status_frame, text="No analysis yet", font=("Arial", 8))
        self.metrics_label.pack(anchor=tk.W)
        
        # Actions
        actions_frame = ttk.LabelFrame(control_frame, text="Actions")
        actions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(actions_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=1)
        
        ttk.Button(actions_frame, text="Medical Report", 
                  command=self.generate_report).pack(fill=tk.X, pady=1)
        
        # Zone d'affichage principal
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Onglet image
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Analysis")
        
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
        self.notebook.add(self.results_frame, text="Analysis Results")
        
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)
        
        # Barre de statut
        self.statusbar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.statusbar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def initialize_system(self):
        """Initialise le système"""
        def init_thread():
            try:
                self.update_status("Checking system requirements...", "blue")
                self.update_progress(10)
                
                # Vérifier les dépendances
                if not TORCH_AVAILABLE:
                    self.update_status("PyTorch not available - limited functionality", "orange")
                    self.update_progress(100)
                    return
                
                if not TRANSFORMERS_AVAILABLE:
                    self.update_status("Transformers not available - cannot load Gemma", "red")
                    self.update_progress(100)
                    return
                
                self.update_progress(30)
                
                # Initialiser le gestionnaire Gemma
                self.update_status("Initializing Local Gemma 3n handler...", "blue")
                self.gemma_handler = LocalGemmaHandler(GEMMA_MODEL_PATH)
                
                self.update_progress(50)
                
                # Charger le modèle
                self.update_status("Loading Gemma 3n model (this may take several minutes)...", "blue")
                
                def progress_update(percent, message):
                    self.update_progress(50 + (percent * 0.4))  # 50-90%
                    self.update_gemma_status(message, "blue")
                
                success = self.gemma_handler.load_model(progress_update)
                
                if success:
                    self.update_progress(100)
                    self.update_status("System ready! Gemma 3n loaded successfully.", "green")
                    self.update_gemma_status("Gemma 3n Ready", "green")
                    self.analyze_button.config(state='normal')
                    self.initialization_complete = True
                else:
                    self.update_status("Failed to load Gemma 3n model", "red")
                    self.update_gemma_status("Model loading failed", "red")
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                self.update_status(f"Initialization failed: {e}", "red")
                self.update_gemma_status(f"Error: {e}", "red")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    # [Le reste des méthodes load_image, analyze_image, etc. restent similaires]
    # [Je continue avec les méthodes clés...]