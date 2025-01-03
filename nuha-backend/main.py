# First file: face_analyzer.py
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import math
import logging
import sys
from PIL import Image
import io
import time
import threading
from datetime import datetime
import google.generativeai as genai
from rembg import remove


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://nuhabeauty-web.vercel.app*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# [Previous ConditionTimer class remains the same]
class ConditionTimer:
    def __init__(self):
        self.start_time = None
        self.required_duration = 3.0  # 3 seconds
        self.lock = threading.Lock()
        
    def start(self):
        with self.lock:
            if self.start_time is None:
                self.start_time = time.time()
                logging.info("Timer started")
    
    def reset(self):
        with self.lock:
            if self.start_time is not None:
                logging.info("Timer reset")
            self.start_time = None
    
    def check_completion(self):
        with self.lock:
            if self.start_time is None:
                return False, 0
            
            elapsed = time.time() - self.start_time
            if elapsed >= self.required_duration:
                return True, self.required_duration
            return False, elapsed

class GeminiManager:
    def __init__(self):
        self.api_keys = [
            "AIzaSyBpLpIg_5apQEoXgP2Eg3kuGVTUyiwy0vE",
            "AIzaSyAXGk_zpIGP_6VSzxVLFSBsk8ePN7uc1-E",
            "AIzaSyDiEuFsPIya8em34GDtytDYXsOC1aJ48h8"
        ]
        self.models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        self.current_key_index = 0
        self.current_model_index = 0
        self.exhausted_combinations = set()
        
    def _next_model(self):
        """Switch to next available model"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        return self.models[self.current_model_index]
    
    def _next_key(self):
        """Switch to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return self.api_keys[self.current_key_index]
    
    def _get_combination_key(self):
        """Get current key-model combination identifier"""
        return f"{self.api_keys[self.current_key_index]}_{self.models[self.current_model_index]}"
    
    def get_next_available_combination(self):
        """Get next available API key and model combination"""
        initial_combination = self._get_combination_key()
        
        while True:
            current_combination = self._get_combination_key()
            
            # If we've tried all combinations
            if len(self.exhausted_combinations) == len(self.api_keys) * len(self.models):
                # Reset exhausted combinations and start over
                self.exhausted_combinations.clear()
                logging.warning("All combinations were exhausted. Resetting tracked combinations.")
            
            # If current combination is not exhausted, use it
            if current_combination not in self.exhausted_combinations:
                return self.api_keys[self.current_key_index], self.models[self.current_model_index]
            
            # Try next model first
            if self.current_model_index < len(self.models) - 1:
                self._next_model()
            else:
                # If we've tried all models, move to next key and reset model index
                self._next_key()
                self.current_model_index = 0
            
            # If we're back to where we started, all combinations are exhausted
            if self._get_combination_key() == initial_combination:
                raise Exception("All API key and model combinations are exhausted")
    
    def mark_current_exhausted(self):
        """Mark current combination as exhausted"""
        self.exhausted_combinations.add(self._get_combination_key())
        logging.warning(f"Marked combination as exhausted: {self._get_combination_key()}")
    
    def generate_content(self, prompt, image):
        """Generate content with automatic failover"""
        attempts = 0
        max_attempts = len(self.api_keys) * len(self.models)
        last_error = None
        
        while attempts < max_attempts:
            try:
                api_key, model = self.get_next_available_combination()
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model_name=model)
                
                logging.info(f"Attempting with API key ending in ...{api_key[-4:]} and model {model}")
                response = model_instance.generate_content([prompt, image])
                return response
                
            except Exception as e:
                error_message = str(e).lower()
                attempts += 1
                last_error = e
                
                # Check for quota-related errors
                if any(err in error_message for err in ["quota", "rate limit", "429"]):
                    logging.warning(f"Quota exceeded for combination. Marking as exhausted.")
                    self.mark_current_exhausted()
                    continue
                
                # If it's not a quota error, try next combination anyway
                logging.warning(f"Error with current combination: {str(e)}")
                self.mark_current_exhausted()
                continue
                
        # If we get here, all attempts failed
        raise Exception(f"All API keys and models exhausted. Last error: {str(last_error)}")

class FaceAnalyzer:
    def __init__(self):
        # Load the required cascades
        try:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            logging.info("Cascades loaded successfully")
        except Exception as e:
            logging.error(f"Error loading cascades: {str(e)}")
            raise

        # Constants - calibrated for webcam
        self.actual_width = 15.0
        self.focal_length = 500.0
        self.distance_scaling = 0.5
        
        # Initialize condition timer
        self.condition_timer = ConditionTimer()

    # [All previous FaceAnalyzer methods remain exactly the same]
    def analyze_lighting(self, frame, face_roi=None):
        """Analyze lighting conditions in the frame or face ROI"""
        if face_roi is not None:
            analysis_region = face_roi
        else:
            analysis_region = frame

        try:
            yuv = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:,:,0]

            mean_brightness = float(y_channel.mean())
            std_brightness = float(y_channel.std())
            overexposed = float((y_channel > 240).mean() * 100)
            underexposed = float((y_channel < 30).mean() * 100)
            
            lighting_score = 100 - (
                abs(mean_brightness - 127) * 0.4 +
                max(0, overexposed * 2) +
                max(0, underexposed * 2) +
                max(0, (30 - std_brightness)) * 2 +
                max(0, (std_brightness - 80)) * 2
            )
            lighting_score = max(0, min(100, lighting_score))
            
            is_good_lighting = (
                40 < mean_brightness < 200 and
                20 < std_brightness < 80 and
                overexposed < 10 and
                underexposed < 15
            )
            
            return {
                'score': float(lighting_score),
                'is_good': bool(is_good_lighting),
                'details': {
                    'mean_brightness': float(mean_brightness),
                    'contrast': float(std_brightness),
                    'overexposed_percent': float(overexposed),
                    'underexposed_percent': float(underexposed)
                }
            }
        except Exception as e:
            logging.error(f"Error in lighting analysis: {str(e)}")
            return None

    def calculate_angle(self, p1, p2):
        """Calculate angle between two points"""
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        return math.atan2(delta_y, delta_x) * (180.0 / math.pi)

    def calculate_distance(self, face_width):
        """Calculate distance based on face width with scaling adjustment"""
        raw_distance = (self.actual_width * self.focal_length) / face_width
        adjusted_distance = raw_distance * self.distance_scaling
        return float(adjusted_distance)

    def analyze_face(self, image_data):
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                self.condition_timer.reset()
                return {
                    'success': False,
                    'error': 'No face detected',
                    'timer_status': {
                        'is_complete': False,
                        'elapsed_time': 0
                    }
                }

            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            lighting_analysis = self.analyze_lighting(face_roi)
            
            roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(w//3, h//3)
            )

            valid_eyes = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                rel_x = (ex + ew/2) / w
                rel_y = (ey + eh/2) / h
                if 0.1 < rel_y < 0.5:
                    valid_eyes.append((eye_center, rel_x))

            valid_eyes.sort(key=lambda x: x[1])

            position_analysis = {
                'is_straight': False,
                'is_perfect_distance': False,
                'angle': 0.0,
                'distance': 0.0,
                'eye_height_difference': 0.0
            }

            if len(valid_eyes) >= 2:
                eye_centers = [valid_eyes[0][0], valid_eyes[-1][0]]
                angle = self.calculate_angle(eye_centers[0], eye_centers[1])
                eye_height_diff = abs(eye_centers[1][1] - eye_centers[0][1]) / h
                eye_distance = math.sqrt(
                    (eye_centers[1][0] - eye_centers[0][0])**2 + 
                    (eye_centers[1][1] - eye_centers[0][1])**2
                )

                is_straight = (abs(angle) < 10 and eye_height_diff < 0.05 and 0.3 < eye_distance/w < 0.7)
                distance = self.calculate_distance(w)
                is_perfect_distance = 10 <= distance <= 15

                position_analysis = {
                    'is_straight': bool(is_straight),
                    'is_perfect_distance': bool(is_perfect_distance),
                    'angle': float(angle),
                    'distance': float(distance),
                    'eye_height_difference': float(eye_height_diff)
                }

                all_conditions_met = (
                    is_straight and 
                    is_perfect_distance and 
                    lighting_analysis['is_good']
                )

                if all_conditions_met:
                    self.condition_timer.start()
                else:
                    self.condition_timer.reset()

                timer_complete, elapsed_time = self.condition_timer.check_completion()

            else:
                self.condition_timer.reset()
                timer_complete, elapsed_time = False, 0

            return {
                'success': True,
                'face_detected': True,
                'face_position': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'lighting': lighting_analysis,
                'position': position_analysis,
                'timer_status': {
                    'is_complete': timer_complete,
                    'elapsed_time': elapsed_time
                },
                'conditions_met': {
                    'straight': position_analysis['is_straight'],
                    'distance': position_analysis['is_perfect_distance'],
                    'lighting': lighting_analysis['is_good']
                }
            }

        except Exception as e:
            logging.error(f"Error in face analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timer_status': {
                    'is_complete': False,
                    'elapsed_time': 0
                }
            }

# Initialize analyzer
try:
    analyzer = FaceAnalyzer()
except Exception as e:
    logging.error(f"Failed to initialize analyzer: {str(e)}")
    analyzer = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'analyzer_loaded': analyzer is not None
    })

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Analyzer not initialized'
            }), 500

        logging.info('Received analyze-face request')
        
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
            image_bytes = base64.b64decode(image_data)
            
            # Validate image is not empty before analysis
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logging.error('Invalid or empty image data')
                return jsonify({
                    'success': False,
                    'error': 'Invalid or empty image data'
                }), 400
            
            result = analyzer.analyze_face(image_bytes)
            return jsonify(result)
            
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

def get_analysis_prompt():
    return """You are a professional Korean skincare specialist for Nuuha Beauty products. Analyze this facial image in detail and provide specific product recommendations.

    Important Instructions:
    1. First determine if the face needs treatment (has skin concerns) or just enhancement (healthy skin).
    2. If you detect any skin concerns, mark status_face as "treatment" and fill the disease object.
    3. If the skin is healthy, mark status_face as "enhancement" and fill the enhancement_remark object.
    4. Never fill both disease and enhancement_remark - use only one based on status_face.
    5. Assess overall skin condition from: "excellent", "good", "moderate", "concerning"

    Analyze for these potential conditions:
    1. Acne-Related:
    - Active acne (whiteheads, blackheads, cystic, pustules)
    - Blemishes
    - Acne scarring
    - Post-inflammatory erythema (PIE)

    2. Surface Issues:
    - Enlarged pores
    - Texture problems
    - Tiny bumps
    - Dullness
    - Uneven skin tone

    3. Moisture & Oil:
    - Oiliness
    - Dryness
    - Dehydration

    4. Aging & Damage:
    - Wrinkles
    - Fine lines
    - Sun damage
    - Dark spots
    - Hyperpigmentation

    5. Sensitivity:
    - Redness
    - Inflammation
    - Damaged skin barrier
    - Sensitive skin

    For healthy skin, consider these enhancement aspects:
    - Brightness and radiance potential
    - Hydration optimization
    - Texture refinement
    - Preventive care
    - Pore minimization
    - Protection needs

    THE RESPONSE MUST EXACTLY MATCH THIS JSON STRUCTURE:
    {
        "status": "success",
        "analysis": {
            "products": [
                {
                    "enhancement_remark": {
                        "confidence_percent": [0.0-1.0],
                        "feature": "[skin feature or area to enhance]",
                        "recommendation": "[specific enhancement suggestion with period at end.]"
                    } OR "disease": {
                        "confidence_percent": [0.0-1.0],
                        "name": "[skin condition name]"
                    },
                    "frequency": "[usage frequency with period at end.]",
                    "how_to_use": "[detailed application instructions with period at end.]",
                    "name": "[EXACT product name from: NUUHA BEAUTY MUGWORT HYDRA BRIGHT GENTLE DAILY FOAM CLEANSER / NUUHA BEAUTY 4 IN 1 HYDRA BRIGHT ULTIMATE KOREAN WATER MIST / NUUHA BEAUTY 4X BRIGHTENING COMPLEX ADVANCED GLOW SERUM / NUUHA BEAUTY 10X SOOTHING COMPLEX HYPER RELIEF SERUM / NUUHA BEAUTY 7X PEPTIDE ULTIMATE GLASS SKIN MOISTURISER / NUUHA BEAUTY ULTRA GLOW BRIGHTENING SERUM SUNSCREEN SPF50+ PA++++]",
                    "step": [1-5]
                }
            ],
            "skin_condition": "[excellent/good/moderate/concerning]",
            "status_face": "[treatment/enhancement]"
        }
    }

    IMPORTANT FORMAT RULES:
    - All text fields must end with a period
    - Keep exact field ordering as shown in the example
    - Products array must be the first field in analysis object
    - skin_condition and status_face must come after products array
    - Each product must maintain exact field ordering: enhancement_remark/disease, frequency, how_to_use, name, step
    - Response must be pure JSON with no additional text"""

def clean_gemini_response(response_text):
    try:
        # Strip any non-JSON content
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        json_str = response_text[start:end]
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"Response text: {response_text}")
        raise

# Initialize the Gemini manager globally
gemini_manager = GeminiManager()

@app.route('/analyze-skin', methods=['POST'])
def analyze_skin():
    try:
        data = request.json
 
        if 'image' not in data:
            return jsonify({"error": "Image data is required"}), 400
            
        # Remove data URL prefix if present
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get response using Gemini manager
        try:
            response = gemini_manager.generate_content(get_analysis_prompt(), image)
            # Get the clean response and extract just the analysis part
            cleaned_response = clean_gemini_response(response.text)
            if 'analysis' in cleaned_response:
                analysis = cleaned_response['analysis']
            else:
                analysis = cleaned_response
                
            return jsonify({
                "status": "success",
                "analysis": analysis
            })
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            return jsonify({
                "status": "error",
                "error": "Failed to analyze image after trying all available API keys and models."
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Add this new endpoint to your Flask app
@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    try:
        logging.info('Received remove-background request')
        
        data = request.json
        if not data or 'image' not in data:
            logging.error('No image data provided')
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
            
        try:
            # Get image data and handle data URL format
            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Create PIL Image from bytes
            input_image = Image.open(io.BytesIO(image_bytes))
            
            # Remove background
            output_image = remove(input_image)
            
            # Convert back to base64
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{output_base64}'
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
    app.run(debug=True, host='0.0.0.0', port=5003)