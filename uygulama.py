import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
import json
from datetime import datetime
import os

class FacialParalysisDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("FACIPA - Facial Paralysis Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.current_image_path = None
        self.analysis_results = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="FACIAL PARALYSIS DETECTION SYSTEM", 
                              font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white', pady=10)
        title_label.pack(fill=tk.X)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Image upload and display
        left_frame = tk.LabelFrame(main_container, text="Image Input", font=('Arial', 12, 'bold'),
                                  bg='#f0f0f0', padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image display
        self.image_display = tk.Label(left_frame, text="Please upload a facial image", 
                                     bg='white', relief=tk.SUNKEN, width=60, height=25)
        self.image_display.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(left_frame, text="📁 Upload Facial Image", 
                              command=self.upload_image, bg='#3498db', fg='white',
                              font=('Arial', 12, 'bold'), padx=20, pady=10)
        upload_btn.pack(pady=5)
        
        # Analyze button
        self.analyze_btn = tk.Button(left_frame, text="🔍 Analyze Paralysis", 
                                    command=self.analyze_image, bg='#e74c3c', fg='white',
                                    font=('Arial', 12, 'bold'), padx=20, pady=10,
                                    state=tk.DISABLED)
        self.analyze_btn.pack(pady=5)
        
        # Right panel - Results display
        right_frame = tk.LabelFrame(main_container, text="Analysis Results", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Numerical Scores
        self.scores_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(self.scores_tab, text="📊 Numerical Scores")
        
        # Tab 2: Classification
        self.classification_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(self.classification_tab, text="🏥 Classification")
        
        # Tab 3: Visualization
        self.visualization_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(self.visualization_tab, text="📈 Visualization")
        
        self.setup_results_tabs()
    
    def setup_results_tabs(self):
        # Numerical Scores Tab
        scores_frame = tk.Frame(self.scores_tab, bg='#f0f0f0')
        scores_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.scores_text = tk.Text(scores_frame, height=20, width=50, font=('Arial', 11))
        self.scores_text.pack(fill=tk.BOTH, expand=True)
        
        # Classification Tab
        classification_frame = tk.Frame(self.classification_tab, bg='#f0f0f0')
        classification_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.classification_text = tk.Text(classification_frame, height=20, width=50, font=('Arial', 11))
        self.classification_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization Tab
        visualization_frame = tk.Frame(self.visualization_tab, bg='#f0f0f0')
        visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.visualization_label = tk.Label(visualization_frame, text="Visualization will appear here", 
                                           bg='white', relief=tk.SUNKEN, width=60, height=20)
        self.visualization_label.pack(pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Facial Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Image uploaded successfully!\nClick 'Analyze Paralysis' to proceed.")
    
    def display_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(image)
        self.image_display.config(image=photo, text="")
        self.image_display.image = photo
    
    def analyze_image(self):
        if not self.current_image_path:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        
        try:
            # Perform analysis
            results = self.perform_analysis()
            self.analysis_results = results
            
            # Display results in tabs
            self.display_numerical_scores(results)
            self.display_classification(results)
            self.display_visualization(results)
            
            messagebox.showinfo("Analysis Complete", "Facial paralysis analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
    
    def perform_analysis(self):
        # Load image
        image = cv2.imread(self.current_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.detector(gray)
        if len(faces) == 0:
            raise Exception("No face detected in the image")
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Extract landmarks
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        
        # Calculate symmetry scores
        symmetry_scores = self.calculate_symmetry_scores(landmarks_points)
        
        # Calculate House-Brackmann score
        hb_grade, hb_classification = self.calculate_house_brackmann(symmetry_scores)
        
        # Create visualization
        visualization_path = self.create_visualization(image.copy(), landmarks_points, symmetry_scores, hb_grade)
        
        return {
            'symmetry_scores': symmetry_scores,
            'house_brackmann': {
                'grade': hb_grade,
                'classification': hb_classification
            },
            'visualization_path': visualization_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_symmetry_scores(self, landmarks):
        # Calculate various symmetry metrics
        def calculate_feature_symmetry(left_points, right_points):
            left_center = np.mean([landmarks[i] for i in left_points], axis=0)
            right_center = np.mean([landmarks[i] for i in right_points], axis=0)
            
            # Calculate horizontal symmetry
            face_center_x = (landmarks[0][0] + landmarks[16][0]) / 2
            left_dist = abs(left_center[0] - face_center_x)
            right_dist = abs(right_center[0] - face_center_x)
            
            symmetry = 100 * (1 - abs(left_dist - right_dist) / max(left_dist, right_dist))
            return max(0, min(100, symmetry))
        
        # Define facial feature points
        left_eye = list(range(36, 42))
        right_eye = list(range(42, 48))
        left_brow = list(range(17, 22))
        right_brow = list(range(22, 27))
        left_mouth = [48, 49, 50, 58, 59, 60]
        right_mouth = [52, 53, 54, 55, 64, 65]
        
        eye_symmetry = calculate_feature_symmetry(left_eye, right_eye)
        brow_symmetry = calculate_feature_symmetry(left_brow, right_brow)
        mouth_symmetry = calculate_feature_symmetry(left_mouth, right_mouth)
        
        overall_symmetry = (eye_symmetry + brow_symmetry + mouth_symmetry) / 3
        
        return {
            'eye_symmetry': round(eye_symmetry, 2),
            'brow_symmetry': round(brow_symmetry, 2),
            'mouth_symmetry': round(mouth_symmetry, 2),
            'overall_symmetry': round(overall_symmetry, 2)
        }
    
    def calculate_house_brackmann(self, symmetry_scores):
        overall = symmetry_scores['overall_symmetry']
        
        if overall >= 95:
            return 1, "Normal - No dysfunction"
        elif overall >= 80:
            return 2, "Mild Dysfunction - Slight weakness"
        elif overall >= 60:
            return 3, "Moderate Dysfunction - Obvious but not disfiguring weakness"
        elif overall >= 40:
            return 4, "Moderately Severe Dysfunction - Obvious weakness and disfigurement"
        elif overall >= 20:
            return 5, "Severe Dysfunction - Only barely perceptible motion"
        else:
            return 6, "Total Paralysis - No movement"
    
    def create_visualization(self, image, landmarks, symmetry_scores, hb_grade):
        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Draw symmetry line
        height, width = image.shape[:2]
        cv2.line(image, (width//2, 0), (width//2, height), (255, 0, 0), 2)
        
        # Add results text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        line_height = 25
        
        texts = [
            f"House-Brackmann Grade: {hb_grade}/6",
            f"Overall Symmetry: {symmetry_scores['overall_symmetry']}%",
            f"Eye Symmetry: {symmetry_scores['eye_symmetry']}%",
            f"Brow Symmetry: {symmetry_scores['brow_symmetry']}%",
            f"Mouth Symmetry: {symmetry_scores['mouth_symmetry']}%"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(image, text, (10, y_offset + i * line_height), 
                       font, 0.5, (255, 255, 255), 2)
            cv2.putText(image, text, (10, y_offset + i * line_height), 
                       font, 0.5, (0, 0, 0), 1)
        
        # Save visualization
        output_path = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(output_path, image)
        return output_path
    
    def display_numerical_scores(self, results):
        self.scores_text.delete(1.0, tk.END)
        
        scores_text = """
=== NUMERICAL SYMMETRY SCORES ===

Overall Facial Symmetry: {overall_symmetry}%

Detailed Breakdown:
• Eye Region Symmetry: {eye_symmetry}%
• Brow Region Symmetry: {brow_symmetry}%
• Mouth Region Symmetry: {mouth_symmetry}%

Interpretation:
- 90-100%: Normal symmetry
- 80-89%: Mild asymmetry
- 60-79%: Moderate asymmetry
- 40-59%: Significant asymmetry
- 20-39%: Severe asymmetry
- 0-19%: Extreme asymmetry

These scores measure the bilateral symmetry 
between left and right facial features.
""".format(**results['symmetry_scores'])
        
        self.scores_text.insert(1.0, scores_text)
    
    def display_classification(self, results):
        self.classification_text.delete(1.0, tk.END)
        
        hb = results['house_brackmann']
        classification_text = f"""
=== HOUSE-BRACKMANN CLASSIFICATION ===

Grade: {hb['grade']}/6
Classification: {hb['classification']}

Grade Scale Explanation:

Grade 1: Normal facial function
Grade 2: Mild dysfunction (slight weakness)
Grade 3: Moderate dysfunction (obvious but not disfiguring)
Grade 4: Moderately severe dysfunction (obvious weakness)
Grade 5: Severe dysfunction (barely perceptible motion)
Grade 6: Total paralysis (no movement)

Clinical Significance:
- Grades 1-2: Usually requires minimal intervention
- Grades 3-4: May benefit from physical therapy
- Grades 5-6: Likely requires medical intervention

Recommendation: Consult with a healthcare professional 
for proper diagnosis and treatment planning.
"""
        
        self.classification_text.insert(1.0, classification_text)
    
    def display_visualization(self, results):
        # Display the analyzed image with landmarks
        image = Image.open(results['visualization_path'])
        image.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(image)
        self.visualization_label.config(image=photo, text="")
        self.visualization_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialParalysisDetector(root)
    root.mainloop()