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

class FaceAnalyzer:
    def __init__(self):
        # Load the required cascades
        
        try:
            self.face_cascade = cv2.CascadeClassifier('nuuhabeauty-faceanalyzer-backend/haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier('nuuhabeauty-faceanalyzer-backend/haarcascade_eye.xml')
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

            # Initialize position analysis
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

                # Check all conditions
                all_conditions_met = (
                    is_straight and 
                    is_perfect_distance and 
                    lighting_analysis['is_good']
                )

                # Handle timer
                if all_conditions_met:
                    self.condition_timer.start()
                else:
                    self.condition_timer.reset()

                # Check if timer is complete
                timer_complete, elapsed_time = self.condition_timer.check_completion()

            else:
                # Reset timer if eyes are not detected
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)