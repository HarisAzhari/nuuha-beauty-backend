# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class SkinConditionClassifier:
    def __init__(self, model_path):
        self.input_shape = [1, 128, 128, 3]
        self.rescale_factor = 0.00392156862745098  # 1/255
        self.class_labels = [
            "Acne",
            "Actinic Keratosis",
            "Basal Cell Carcinoma",
            "Eczema",
            "Rosacea"
        ]
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            logging.info(f"Model loaded successfully from {model_path}")
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
    def predict(self, image):
        try:
            # Preprocess image
            image = image.resize((128, 128))
            img_array = np.array(image, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array * self.rescale_factor
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process results
            probabilities = output_data[0]
            predictions = {
                label: float(prob) 
                for label, prob in zip(self.class_labels, probabilities)
            }
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

# Initialize classifier
try:
    MODEL_PATH = 'skin_disease_model.tflite'  # Make sure this is in the same directory
    classifier = SkinConditionClassifier(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to initialize classifier: {str(e)}")
    classifier = None  # Allow the server to start even if model fails to load

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None
    })

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    try:
        if not classifier:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500

        logging.info('Received analyze-face request')
        
        # Get the image data from request
        data = request.json
        if not data or 'image' not in data:
            logging.error('No image data provided')
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
            
        try:
            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            logging.info(f'Received image data length: {len(image_data)}')
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            logging.info(f'Decoded image bytes length: {len(image_bytes)}')
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            logging.info(f'Image opened successfully. Size: {image.size}, Mode: {image.mode}')
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logging.info(f'Converting image from {image.mode} to RGB')
                image = image.convert('RGB')
            
            # Get predictions
            predictions = classifier.predict(image)
            logging.info(f'Generated predictions: {predictions}')
            
            # Get top prediction
            top_condition = max(predictions.items(), key=lambda x: x[1])
            logging.info(f'Top condition: {top_condition}')
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'topCondition': {
                    'condition': top_condition[0],
                    'probability': float(top_condition[1])
                }
            })
            
        except Exception as e:
            logging.error(f'Error processing image: {str(e)}')
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
            
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Add host parameter to make the server accessible from other devices
    app.run(debug=True, host='0.0.0.0', port=5002)
            
            