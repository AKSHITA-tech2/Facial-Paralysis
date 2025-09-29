import csv
import os
from datetime import datetime

class MedicalDataCollector:
    def __init__(self, data_file='medical_facial_data.csv'):
        self.data_file = data_file
        self.initialize_dataset()
    
    def initialize_dataset(self):
        """Initialize CSV with medical data structure"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'image_path', 'house_brackmann_grade', 
                    'expert_rating', 'eye_closure_ratio', 'mouth_deviation_score',
                    'brow_elevation_asymmetry', 'patient_age', 'paralysis_duration',
                    'etiology', 'treatment_status', 'notes'
                ])
    
    def add_medical_sample(self, image_path, landmarks, expert_grade, patient_info=None):
        """Add a new sample to the medical dataset"""
        features = self.extract_features(landmarks)
        
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                image_path,
                expert_grade,
                expert_grade,  # expert_rating same as grade for now
                features['eye_closure_ratio'],
                features['mouth_deviation_score'],
                features['brow_elevation_asymmetry'],
                patient_info.get('age', '') if patient_info else '',
                patient_info.get('duration_days', '') if patient_info else '',
                patient_info.get('etiology', '') if patient_info else '',
                patient_info.get('treatment', '') if patient_info else '',
                patient_info.get('notes', '') if patient_info else ''
            ])