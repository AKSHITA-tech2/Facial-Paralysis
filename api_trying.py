from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from facial_landmarks import EnhancedFacialParalysisAnalyzer

app = Flask(__name__)
analyzer = EnhancedFacialParalysisAnalyzer("shape_predictor_68_face_landmarks.dat")

@app.route('/analyze', methods=['POST'])
def analyze_facial_paralysis():
    try:
        if 'image' not in request.files and 'image_base64' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = f"temp_{image_file.filename}"
            image_file.save(image_path)
        else:
            # Handle base64 image
            image_data = request.json['image_base64']
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image_path = "temp_image.jpg"
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
        
        # Analyze the image
        results = analyzer.process_image(image_path)
        
        # Include visualization as base64
        if 'visualization_path' in results:
            with open(results['visualization_path'], 'rb') as f:
                visualization_base64 = base64.b64encode(f.read()).decode('utf-8')
            results['visualization_base64'] = visualization_base64
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Facial Paralysis Analysis'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)