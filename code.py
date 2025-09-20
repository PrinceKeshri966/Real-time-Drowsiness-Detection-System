import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import mediapipe as mp
import threading
import time
import pygame
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns

class DrowsinessDetector:
    def __init__(self):
        # ML Components
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmarks indices for MediaPipe
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouth landmarks
        self.MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
        # State tracking
        self.is_trained = False
        self.drowsiness_history = deque(maxlen=30)  # 1 second history at 30fps
        self.blink_counter = 0
        self.yawn_counter = 0
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds
        
        # Initialize pygame for alerts
        pygame.mixer.init()
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y] for lm in eye_landmarks])
        
        # Calculate distances
        A = np.linalg.norm(points[1] - points[5])  # Vertical distance 1
        B = np.linalg.norm(points[2] - points[4])  # Vertical distance 2
        C = np.linalg.norm(points[0] - points[3])  # Horizontal distance
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio"""
        points = np.array([[lm.x, lm.y] for lm in mouth_landmarks])
        
        # Calculate mouth opening
        A = np.linalg.norm(points[2] - points[6])  # Vertical distance 1
        B = np.linalg.norm(points[3] - points[5])  # Vertical distance 2
        C = np.linalg.norm(points[0] - points[4])  # Horizontal distance
        
        mar = (A + B) / (2.0 * C)
        return mar
    
    def calculate_head_pose(self, face_landmarks):
        """Calculate head pose angles"""
        # Get specific landmarks for head pose estimation
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[18]
        left_eye_corner = face_landmarks.landmark[33]
        right_eye_corner = face_landmarks.landmark[263]
        
        # Convert to numpy arrays
        nose = np.array([nose_tip.x, nose_tip.y])
        chin_point = np.array([chin.x, chin.y])
        left_eye = np.array([left_eye_corner.x, left_eye_corner.y])
        right_eye = np.array([right_eye_corner.x, right_eye_corner.y])
        
        # Calculate angles
        eye_center = (left_eye + right_eye) / 2
        nose_to_eye = eye_center - nose
        nose_to_chin = chin_point - nose
        
        # Head tilt (roll)
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        # Head nod (pitch) - approximation
        pitch_angle = np.arctan2(nose_to_chin[1], nose_to_chin[0])
        
        return eye_angle, pitch_angle
    
    def extract_features(self, face_landmarks):
        """Extract comprehensive drowsiness features"""
        if face_landmarks is None:
            return None
        
        # Get eye landmarks
        left_eye_landmarks = [face_landmarks.landmark[i] for i in self.LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[i] for i in self.RIGHT_EYE]
        mouth_landmarks = [face_landmarks.landmark[i] for i in self.MOUTH]
        
        # Calculate ratios
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mar(mouth_landmarks)
        
        # Head pose
        roll_angle, pitch_angle = self.calculate_head_pose(face_landmarks)
        
        # Additional features
        eye_difference = abs(left_ear - right_ear)  # Asymmetry
        
        # Temporal features (if history available)
        ear_variance = 0
        mar_variance = 0
        if len(self.drowsiness_history) > 5:
            recent_ears = [h[0] for h in list(self.drowsiness_history)[-5:]]
            recent_mars = [h[1] for h in list(self.drowsiness_history)[-5:]]
            ear_variance = np.var(recent_ears)
            mar_variance = np.var(recent_mars)
        
        features = np.array([
            avg_ear, mar, roll_angle, pitch_angle, 
            eye_difference, ear_variance, mar_variance,
            left_ear, right_ear
        ])
        
        return features
    
    def create_synthetic_data(self, num_samples=1000):
        """Create synthetic training data for demonstration"""
        print("Generating synthetic training data...")
        
        # Normal state (awake)
        normal_ear = np.random.normal(0.25, 0.05, num_samples//2)
        normal_mar = np.random.normal(0.05, 0.02, num_samples//2)
        normal_roll = np.random.normal(0, 0.1, num_samples//2)
        normal_pitch = np.random.normal(1.5, 0.2, num_samples//2)
        normal_asymmetry = np.random.normal(0.02, 0.01, num_samples//2)
        normal_ear_var = np.random.normal(0.001, 0.0005, num_samples//2)
        normal_mar_var = np.random.normal(0.0005, 0.0002, num_samples//2)
        normal_left_ear = normal_ear + np.random.normal(0, 0.01, num_samples//2)
        normal_right_ear = normal_ear + np.random.normal(0, 0.01, num_samples//2)
        
        # Drowsy state
        drowsy_ear = np.random.normal(0.15, 0.03, num_samples//2)  # Lower EAR
        drowsy_mar = np.random.normal(0.08, 0.03, num_samples//2)  # Higher MAR (yawning)
        drowsy_roll = np.random.normal(0.2, 0.15, num_samples//2)  # Head tilting
        drowsy_pitch = np.random.normal(1.2, 0.3, num_samples//2)  # Head nodding
        drowsy_asymmetry = np.random.normal(0.05, 0.02, num_samples//2)  # More asymmetry
        drowsy_ear_var = np.random.normal(0.003, 0.001, num_samples//2)  # More variation
        drowsy_mar_var = np.random.normal(0.002, 0.0008, num_samples//2)
        drowsy_left_ear = drowsy_ear + np.random.normal(0, 0.02, num_samples//2)
        drowsy_right_ear = drowsy_ear + np.random.normal(0, 0.02, num_samples//2)
        
        # Combine features
        X = np.vstack([
            np.column_stack([normal_ear, normal_mar, normal_roll, normal_pitch, 
                           normal_asymmetry, normal_ear_var, normal_mar_var,
                           normal_left_ear, normal_right_ear]),
            np.column_stack([drowsy_ear, drowsy_mar, drowsy_roll, drowsy_pitch,
                           drowsy_asymmetry, drowsy_ear_var, drowsy_mar_var,
                           drowsy_left_ear, drowsy_right_ear])
        ])
        
        y = np.array([0] * (num_samples//2) + [1] * (num_samples//2))  # 0: awake, 1: drowsy
        
        return X, y
    
    def train_model(self, X=None, y=None):
        """Train the drowsiness detection model"""
        if X is None or y is None:
            X, y = self.create_synthetic_data()
        
        print("Training drowsiness detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Awake', 'Drowsy']))
        
        # Feature importance
        feature_names = ['EAR', 'MAR', 'Head_Roll', 'Head_Pitch', 'Eye_Asymmetry', 
                        'EAR_Variance', 'MAR_Variance', 'Left_EAR', 'Right_EAR']
        importances = self.model.feature_importances_
        
        print("\nFeature Importances:")
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.3f}")
        
        self.is_trained = True
        return accuracy
    
    def predict_drowsiness(self, features):
        """Predict drowsiness state"""
        if not self.is_trained or features is None:
            return 0, 0.0
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0])
        
        return prediction, confidence
    
    def play_alert(self):
        """Play alert sound (create a simple beep)"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            # Create a simple beep sound
            duration = 1000  # milliseconds
            freq = 440  # Hz
            
            # Generate beep sound
            sample_rate = 22050
            frames = int(duration * sample_rate / 1000)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
            self.last_alert_time = current_time
            print("⚠️ DROWSINESS ALERT! ⚠️")
    
    def real_time_detection(self):
        """Run real-time drowsiness detection"""
        if not self.is_trained:
            print("Model not trained! Training with synthetic data...")
            self.train_model()
        
        print("Starting real-time drowsiness detection...")
        print("Press 'q' to quit")
        print("Press 's' to show statistics")
        
        cap = cv2.VideoCapture(0)
        drowsy_frames = 0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            total_frames += 1
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract features
                    features = self.extract_features(face_landmarks)
                    
                    if features is not None:
                        # Predict drowsiness
                        prediction, confidence = self.predict_drowsiness(features)
                        
                        # Update history
                        self.drowsiness_history.append((features[0], features[1], prediction))
                        
                        # Determine alert status
                        if prediction == 1 and confidence > 0.7:
                            drowsy_frames += 1
                            status = "DROWSY"
                            color = (0, 0, 255)  # Red
                            
                            # Check if we should trigger alert
                            recent_predictions = [h[2] for h in list(self.drowsiness_history)[-10:]]
                            if sum(recent_predictions) >= 7:  # 7 out of last 10 frames
                                threading.Thread(target=self.play_alert).start()
                        else:
                            status = "AWAKE"
                            color = (0, 255, 0)  # Green
                        
                        # Draw face landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                            None, self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                        )
                        
                        # Display information
                        cv2.putText(frame, f"Status: {status}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"EAR: {features[0]:.3f}", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"MAR: {features[1]:.3f}", (10, 140),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw drowsiness percentage
                        drowsy_percent = (drowsy_frames / max(total_frames, 1)) * 100
                        cv2.putText(frame, f"Drowsy: {drowsy_percent:.1f}%", (10, 170),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Drowsiness Detection System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_statistics(drowsy_frames, total_frames)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_statistics(self, drowsy_frames, total_frames):
        """Show detection statistics"""
        print("\n=== Detection Statistics ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Drowsy frames detected: {drowsy_frames}")
        print(f"Drowsiness percentage: {(drowsy_frames/max(total_frames,1))*100:.2f}%")
        print(f"Average processing rate: ~30 FPS")
    
    def save_model(self, filename='drowsiness_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='drowsiness_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = True
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found!")
            return False

def main():
    detector = DrowsinessDetector()
    
    while True:
        print("\n=== Real-time Drowsiness Detection System ===")
        print("🚗 Driver Safety AI Application")
        print("\n1. Train model (with synthetic data)")
        print("2. Start real-time detection")
        print("3. Save model")
        print("4. Load model")
        print("5. Run demo with statistics")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            detector.train_model()
        
        elif choice == '2':
            detector.real_time_detection()
        
        elif choice == '3':
            detector.save_model()
        
        elif choice == '4':
            detector.load_model()
        
        elif choice == '5':
            print("\n=== Demo Mode ===")
            print("This will show real-time detection with detailed statistics")
            print("Look normal, then try:")
            print("- Close your eyes for extended periods")
            print("- Yawn frequently")
            print("- Tilt your head")
            detector.real_time_detection()
        
        elif choice == '6':
            print("Stay safe on the road! 🚗")
            break
        
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()