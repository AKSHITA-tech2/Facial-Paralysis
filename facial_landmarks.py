import cv2
import dlib
import numpy as np
import json
from scipy.spatial import distance

class EnhancedFacialParalysisAnalyzer:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
    def calculate_symmetry_score(self, landmarks):
        """Calculate symmetry score between left and right facial features"""
        # Define left and right facial points
        left_eye_points = list(range(36, 42))
        right_eye_points = list(range(42, 48))
        left_brow_points = list(range(17, 22))
        right_brow_points = list(range(22, 27))
        left_mouth_points = [48, 49, 50, 58, 59, 60]
        right_mouth_points = [52, 53, 54, 55, 64, 65]
        
        symmetry_scores = {}
        
        # Eye symmetry
        left_eye_width = distance.euclidean(landmarks[36], landmarks[39])
        right_eye_width = distance.euclidean(landmarks[42], landmarks[45])
        eye_symmetry = 1 - abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width)
        
        # Mouth symmetry
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_center = landmarks[51]
        left_dist = distance.euclidean(mouth_left, mouth_center)
        right_dist = distance.euclidean(mouth_right, mouth_center)
        mouth_symmetry = 1 - abs(left_dist - right_dist) / max(left_dist, right_dist)
        
        # Brow symmetry
        left_brow_avg = np.mean([landmarks[i] for i in left_brow_points], axis=0)
        right_brow_avg = np.mean([landmarks[i] for i in right_brow_points], axis=0)
        brow_symmetry = 1 - distance.euclidean(left_brow_avg, right_brow_avg) / 100
        
        overall_symmetry = (eye_symmetry + mouth_symmetry + brow_symmetry) / 3
        
        symmetry_scores = {
            'eye_symmetry': round(eye_symmetry * 100, 2),
            'mouth_symmetry': round(mouth_symmetry * 100, 2),
            'brow_symmetry': round(brow_symmetry * 100, 2),
            'overall_symmetry': round(overall_symmetry * 100, 2)
        }
        
        return symmetry_scores
    
    def calculate_house_brackmann_score(self, symmetry_scores, landmarks):
        """Calculate House-Brackmann grading based on symmetry scores"""
        overall_symmetry = symmetry_scores['overall_symmetry']
        
        if overall_symmetry >= 90:
            return 1, "Normal"
        elif overall_symmetry >= 80:
            return 2, "Mild Dysfunction"
        elif overall_symmetry >= 60:
            return 3, "Moderate Dysfunction"
        elif overall_symmetry >= 40:
            return 4, "Moderately Severe Dysfunction"
        elif overall_symmetry >= 20:
            return 5, "Severe Dysfunction"
        else:
            return 6, "Total Paralysis"
    
    def analyze_facial_movement(self, image, landmarks):
        """Analyze facial movement and paralysis indicators"""
        # Calculate facial angles and distances
        results = {}
        
        # Mouth angle calculation
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_center_top = landmarks[51]
        mouth_center_bottom = landmarks[57]
        
        # Calculate mouth asymmetry
        horizontal_asymmetry = abs(mouth_left[0] - mouth_center_top[0]) - abs(mouth_right[0] - mouth_center_top[0])
        vertical_asymmetry = abs(mouth_center_top[1] - mouth_center_bottom[1])
        
        # Eye closure analysis
        left_eye_height = distance.euclidean(landmarks[37], landmarks[41])
        right_eye_height = distance.euclidean(landmarks[43], landmarks[47])
        eye_closure_asymmetry = abs(left_eye_height - right_eye_height)
        
        results.update({
            'mouth_horizontal_asymmetry': round(abs(horizontal_asymmetry), 2),
            'mouth_vertical_asymmetry': round(vertical_asymmetry, 2),
            'eye_closure_asymmetry': round(eye_closure_asymmetry, 2)
        })
        
        return results
    
    def process_image(self, image_path):
        """Main processing function"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return {"error": "No face detected"}
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert landmarks to list of points
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        
        # Calculate scores
        symmetry_scores = self.calculate_symmetry_score(landmarks_points)
        hb_grade, hb_classification = self.calculate_house_brackmann_score(symmetry_scores, landmarks_points)
        movement_analysis = self.analyze_facial_movement(image, landmarks_points)
        
        # Create visualization
        visualization = self.create_visualization(image.copy(), landmarks_points, symmetry_scores, hb_grade)
        
        return {
            'symmetry_scores': symmetry_scores,
            'house_brackmann_grade': hb_grade,
            'house_brackmann_classification': hb_classification,
            'movement_analysis': movement_analysis,
            'visualization_path': visualization
        }
    
    def create_visualization(self, image, landmarks, symmetry_scores, hb_grade):
        """Create comprehensive visualization with scores and analysis"""
        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Draw symmetry lines
        height, width = image.shape[:2]
        cv2.line(image, (width//2, 0), (width//2, height), (255, 0, 0), 1)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        cv2.putText(image, f"House-Brackmann Grade: {hb_grade}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Overall Symmetry: {symmetry_scores['overall_symmetry']}%", 
                   (10, y_offset + 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Eye Symmetry: {symmetry_scores['eye_symmetry']}%", 
                   (10, y_offset + 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Mouth Symmetry: {symmetry_scores['mouth_symmetry']}%", 
                   (10, y_offset + 90), font, 0.5, (255, 255, 255), 1)
        
        # Save visualization
        output_path = "paralysis_analysis_visualization.png"
        cv2.imwrite(output_path, image)
        
        return output_path