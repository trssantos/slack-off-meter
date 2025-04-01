import cv2
import time
import numpy as np
import os
import pygame
import datetime
import csv
from pathlib import Path
import mediapipe as mp
from ultralytics import YOLO

class SlackOffMeter:
    def __init__(self, alert_timeout=60):
        # Initialize YOLO model
        self.model = YOLO("yolov8n.pt")  # Use YOLOv8 nano model
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Parameters
        self.alert_timeout = alert_timeout  # Use the provided timeout value
        self.last_focused_time = time.time()
        self.is_focused = True
        self.slack_count = 0
        self.session_start = time.time()
        self.total_focused_time = 0
        self.total_slack_time = 0
        self.last_state_time = time.time()
        
        # Face detection sensitivity (higher = more lenient)
        self.face_sensitivity = 0.7  # Default setting
        
        # Create directory for logs
        self.log_dir = Path("slack_meter_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'state', 'duration'])
        
        # Initialize sound
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.wav") if os.path.exists("alert.wav") else None
        
        # Funny messages
        self.slack_messages = [
            "Back to work, human!",
            "Those TPS reports won't write themselves!",
            "Your productivity is leaving the chat...",
            "Focus detected: ERROR 404",
            "Boss mode activated!",
            "Did someone say coffee break? Not yet!",
            "Your to-do list is getting lonely",
            "Procrastination level: Expert",
            "Warning: Productivity leak detected!"
        ]
        
        # State variables
        self.debug_mode = True
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.line_thickness = 2
        
        print(f"SlackOffMeter initialized with alert timeout of {self.alert_timeout} seconds!")
        print("Press 'q' to quit, 'd' to toggle debug mode, '+'/'-' to adjust sensitivity.")

    def check_gaze_direction(self, face_landmarks, image_shape):
        """Determine if the person is looking at the screen based on face landmarks"""
        if not face_landmarks:
            return False
            
        # Get landmarks for eyes
        # Eye corners
        left_eye_left = face_landmarks.landmark[33]
        left_eye_right = face_landmarks.landmark[133]
        right_eye_left = face_landmarks.landmark[362]
        right_eye_right = face_landmarks.landmark[263]
        
        # Nose and chin landmarks for vertical position
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[199]
        
        # 1. Calculate horizontal eye alignment (for left/right head rotation)
        left_eye_width = abs(left_eye_right.x - left_eye_left.x)
        right_eye_width = abs(right_eye_right.x - right_eye_left.x)
        eye_width_ratio = min(left_eye_width, right_eye_width) / max(left_eye_width, right_eye_width)
        
        # If one eye appears much narrower than the other, the head is likely turned
        horizontal_aligned = eye_width_ratio > (1.0 - self.face_sensitivity)
        
        # 2. Check if head is tilted down too much
        nose_chin_ratio = (chin.y - nose_tip.y) / 0.2  # Normalized by expected ratio
        vertical_aligned = nose_chin_ratio > self.face_sensitivity
        
        # 3. Check if eyes are approximately level (for head tilt)
        left_eye_y = (left_eye_left.y + left_eye_right.y) / 2
        right_eye_y = (right_eye_left.y + right_eye_right.y) / 2
        eye_level_diff = abs(left_eye_y - right_eye_y)
        eyes_level = eye_level_diff < (0.03 * (2.0 - self.face_sensitivity))
        
        # For debugging
        self.face_metrics = {
            "eye_width_ratio": eye_width_ratio,
            "nose_chin_ratio": nose_chin_ratio,
            "eye_level_diff": eye_level_diff,
            "horizontal_aligned": horizontal_aligned,
            "vertical_aligned": vertical_aligned,
            "eyes_level": eyes_level
        }
        
        # Combine checks - lenient version that prioritizes horizontal alignment
        return horizontal_aligned and (vertical_aligned or eyes_level)

    def log_state_change(self, current_state):
        """Log state changes to CSV file"""
        now = time.time()
        duration = now - self.last_state_time
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'focused' if current_state else 'slacking',
                f"{duration:.2f}"
            ])
        
        # Update times
        if self.is_focused:
            self.total_focused_time += duration
        else:
            self.total_slack_time += duration
            
        self.last_state_time = now

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip the frame horizontally for a more natural view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get frame height and width
                h, w, _ = frame.shape
                
                # Process with MediaPipe
                face_results = self.face_mesh.process(rgb_frame)
                
                # Variables to track current state
                is_person_detected = False
                is_phone_detected = False
                is_facing_computer = False
                
                # Create a visualization frame
                viz_frame = frame.copy()
                
                # Run YOLO detection for person and phone
                results = self.model(frame)
                
                # Parse YOLO results
                for detection in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = detection
                    class_name = results[0].names[int(class_id)]
                    
                    if conf > 0.5:  # Confidence threshold
                        # Draw bounding box
                        cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                      (0, 255, 0) if class_name == "person" else (0, 0, 255), 2)
                        
                        # Add label
                        cv2.putText(viz_frame, f"{class_name} {conf:.2f}", 
                                    (int(x1), int(y1 - 10)), self.font, self.font_scale, 
                                    (0, 255, 0) if class_name == "person" else (0, 0, 255), 
                                    self.line_thickness)
                        
                        if class_name == "person":
                            is_person_detected = True
                        elif class_name == "cell phone":
                            is_phone_detected = True
                
                # Check face orientation with MediaPipe
                self.face_metrics = {} # Reset metrics
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Draw face mesh in debug mode
                        if self.debug_mode:
                            self.mp_drawing.draw_landmarks(
                                viz_frame,
                                face_landmarks,
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                        # Check gaze direction - face detected = person detected
                        is_person_detected = True
                        is_facing_computer = self.check_gaze_direction(face_landmarks, (h, w))
                
                # Determine focus state
                currently_focused = is_person_detected and is_facing_computer and not is_phone_detected
                
                # Update timer and state
                current_time = time.time()
                if currently_focused != self.is_focused:
                    # Log state change
                    self.log_state_change(currently_focused)
                    
                    if currently_focused:
                        cv2.putText(viz_frame, "Back to work! Good job.", (10, h - 70), 
                                    self.font, self.font_scale, (0, 255, 0), self.line_thickness)
                    else:
                        self.last_focused_time = current_time
                        
                    self.is_focused = currently_focused
                
                # Check if alert should be triggered
                if not currently_focused and (current_time - self.last_focused_time > self.alert_timeout):
                    message = np.random.choice(self.slack_messages)
                    cv2.putText(viz_frame, f"ALERT: {message}", (10, h - 40), 
                                self.font, self.font_scale, (0, 0, 255), self.line_thickness)
                    
                    # Play sound if available
                    if self.alert_sound:
                        self.alert_sound.play()
                        
                    # Reset timer and increase counter
                    self.last_focused_time = current_time
                    self.slack_count += 1
                
                # Display status on frame
                status = "FOCUSED" if currently_focused else "SLACKING OFF"
                color = (0, 255, 0) if currently_focused else (0, 0, 255)
                cv2.putText(viz_frame, status, (10, 30), 
                            self.font, 1.2, color, self.line_thickness)
                
                # Display timer if not focused
                if not currently_focused:
                    time_unfocused = int(current_time - self.last_focused_time)
                    cv2.putText(viz_frame, f"Slack time: {time_unfocused}s / {self.alert_timeout}s", 
                                (10, 70), self.font, self.font_scale, (0, 0, 255), self.line_thickness)
                
                # Display sensitivity
                cv2.putText(viz_frame, f"Sensitivity: {self.face_sensitivity:.1f}", 
                            (10, h - 100), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                
                # Calculate session stats
                session_duration = current_time - self.session_start
                focus_percentage = (self.total_focused_time / session_duration) * 100 if session_duration > 0 else 0
                
                # Display session stats
                cv2.putText(viz_frame, f"Session: {int(session_duration // 60)}m {int(session_duration % 60)}s", 
                            (w - 300, 30), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                cv2.putText(viz_frame, f"Focus rate: {focus_percentage:.1f}%", 
                            (w - 300, 60), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                cv2.putText(viz_frame, f"Slack alerts: {self.slack_count}", 
                            (w - 300, 90), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                
                # Display detection info in debug mode
                if self.debug_mode:
                    debug_info = [
                        f"Person detected: {is_person_detected}",
                        f"Phone detected: {is_phone_detected}",
                        f"Facing computer: {is_facing_computer}"
                    ]
                    
                    # Add face metrics if available
                    if self.face_metrics:
                        for key, value in self.face_metrics.items():
                            if isinstance(value, bool):
                                debug_info.append(f"{key}: {value}")
                            else:
                                debug_info.append(f"{key}: {value:.3f}")
                    
                    y_pos = 120
                    for info in debug_info:
                        cv2.putText(viz_frame, info, (10, y_pos), 
                                    self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                        y_pos += 30
                
                # Calculate FPS
                self.frame_count += 1
                if current_time - self.fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = current_time
                
                # Display FPS
                cv2.putText(viz_frame, f"FPS: {self.fps}", (10, h - 10), 
                            self.font, self.font_scale, (255, 255, 255), self.line_thickness)
                
                # Show frame
                cv2.imshow("Slack Off Meter", viz_frame)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    self.face_sensitivity = min(1.0, self.face_sensitivity + 0.1)
                    print(f"Face sensitivity increased to {self.face_sensitivity:.1f}")
                elif key == ord('-') or key == ord('_'):
                    self.face_sensitivity = max(0.1, self.face_sensitivity - 0.1)
                    print(f"Face sensitivity decreased to {self.face_sensitivity:.1f}")
                
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            
            # Log final state
            if self.is_focused or not self.is_focused:
                self.log_state_change(self.is_focused)
            
            # Show final stats
            session_duration = time.time() - self.session_start
            focus_percentage = (self.total_focused_time / session_duration) * 100 if session_duration > 0 else 0
            
            print("\n=== Session Summary ===")
            print(f"Duration: {int(session_duration // 60)}m {int(session_duration % 60)}s")
            print(f"Focus time: {int(self.total_focused_time // 60)}m {int(self.total_focused_time % 60)}s")
            print(f"Slack time: {int(self.total_slack_time // 60)}m {int(self.total_slack_time % 60)}s")
            print(f"Focus rate: {focus_percentage:.1f}%")
            print(f"Slack alerts: {self.slack_count}")
            print(f"Session log saved to: {self.log_file}")

if __name__ == "__main__":
    # Create an alert sound if it doesn't exist
    if not os.path.exists("alert.wav"):
        print("Creating default alert sound...")
        import wave
        import struct
        
        # Create a simple beep sound
        framerate = 44100
        duration = 0.5  # seconds
        frequency = 440  # A4 note
        
        with wave.open("alert.wav", "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(framerate)
            
            for i in range(int(framerate * duration)):
                value = int(32767 * 0.9 * np.sin(2 * np.pi * frequency * i / framerate))
                data = struct.pack('<h', value)
                f.writeframesraw(data)
    
    # Ask user for slack time threshold
    default_timeout = 60
    try:
        print("\n=== Slack Off Meter Configuration ===")
        user_input = input(f"Enter slack time threshold in seconds (default {default_timeout}s): ")
        
        # Use default if empty input
        if user_input.strip():
            alert_timeout = int(user_input)
            print(f"Slack time threshold set to {alert_timeout} seconds")
        else:
            alert_timeout = default_timeout
            print(f"Using default threshold of {alert_timeout} seconds")
    except ValueError:
        alert_timeout = default_timeout
        print(f"Invalid input. Using default threshold of {alert_timeout} seconds")
    
    # Fun productivity quote
    quotes = [
        "The key is not to prioritize what's on your schedule, but to schedule your priorities.",
        "Productivity is never an accident. It is always the result of a commitment to excellence.",
        "Until we can manage time, we can manage nothing else.",
        "The way to get started is to quit talking and begin doing.",
        "Focus on being productive instead of busy.",
        "Your focus determines your reality. May the productivity be with you."
    ]
    print(f"\n\"{np.random.choice(quotes)}\"\n")
    
    print("Starting SlackOffMeter...")
    meter = SlackOffMeter(alert_timeout=alert_timeout)
    meter.run()
