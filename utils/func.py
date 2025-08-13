import cv2
import numpy as np
import pygame
from dataclasses import dataclass
from typing import Optional, Tuple
from ultralytics import YOLO
import time

# Configuration (should be in separate config.py)
class ProjectConfig:
    YOLO_MODEL_PATH = "path/to/yolo_model.pt"
    MAR_THRESHOLD = 0.5  # Example value, needs calibration
    YAWN_CONSEC_FRAMES = 15
    TIME_THRESHOLD = 3.0  # Seconds
    ALARM_SOUND = "alarm.wav"

pj = ProjectConfig()

@dataclass
class Detection:
    box: np.ndarray
    class_id: int
    confidence: float

def get_results(results):
    boxes = []
    lst_cls = []
    
    # Get the first result object (single frame)
    result = results[0]
    
    # Extract boxes and classes
    for box in result.boxes:
        # Convert box coordinates to numpy array
        b = box.xyxy[0].cpu().numpy().tolist()  # convert to list for easier handling
        c = int(box.cls[0].cpu().numpy())  # get first class value as integer
        boxes.append(b)
        lst_cls.append(c)
    
    return boxes, lst_cls  # return exactly 2 values

def crop_image(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Safe image cropping with boundary checks"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
    except (ValueError, TypeError, IndexError) as e:
        print(f"Cropping error: {e}")
        return None

def is_inside(box1, box2, threshold=0.2):  # More lenient threshold
    """Check if box1 is inside box2"""
    if box1 is None or box2 is None:
        return False
    
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    
    # Calculate center points
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check if center point is inside head box with margin
    return (X1 < center_x < X2 and Y1 < center_y < Y2)

def calculate_area(box):
    """Calculate area of a bounding box"""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_mar(mouth_box, head_box):
    """
    Calculate mouth aspect ratio optimized for yawn detection
    """
    try:
        mouth_height = mouth_box[3] - mouth_box[1]
        mouth_width = mouth_box[2] - mouth_box[0]
        head_height = head_box[3] - head_box[1]
        
        # Calculate vertical opening ratio
        vertical_ratio = mouth_height / head_height
        
        # Calculate aspect ratio
        aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Combined MAR that emphasizes vertical opening
        mar = vertical_ratio * 4.0  # Increase weight of vertical opening
        
        print(f"Mouth metrics - Height: {mouth_height:.1f}, Width: {mouth_width:.1f}, MAR: {mar:.3f}")
        return mar
    except Exception as e:
        print(f"Error calculating MAR: {e}")
        return 0.0


class DrowsinessDetector:
    def __init__(self, detect_model: str, clf_model):
        self.detect_model = detect_model
        self.clf_model = clf_model
        self.yolo_model = None
        self.alarm_sound = None
        self.reset_state()
        
        pygame.mixer.init()
        self.load_models()
        
    def reset_state(self):
        """Initialize all tracking variables"""
        self.eye_status1 = None
        self.eye_status2 = None
        self.start_time = 0.0
        self.time_close_eyes = 0.0
        self.count_start = False
        self.alarm_on = False
        self.yawn_counter = 0
        self.yawn_start_time = 0.0
        self.is_yawn_active = False
        self.both_eyes_closed = False
        self.mouth_box = None  # Store current mouth box

    def load_models(self):
        """Load required models with error handling"""
        try:
            if self.detect_model == "yolo":
                self.yolo_model = YOLO(pj.YOLO_MODEL_PATH)
                
            self.alarm_sound = pygame.mixer.Sound(pj.ALARM_SOUND)
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def process_yolo_detections(self, frame: np.ndarray) -> np.ndarray:
        """Process YOLO detections and update state"""
        try:
            results = self.yolo_model.predict(frame, conf=0.45, verbose=False)
            boxes, classes = get_results(results)
            
            heads = []
            mouths = []
            eyes = []

            for box, cls in zip(boxes, classes):
                if cls == 1:
                    heads.append(box)
                elif cls == 2:
                    mouths.append(box)
                else:  # Assuming other classes are eyes
                    eyes.append(box)

            # Find largest head
            main_head = max(heads, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)
            
            # Process mouths
            max_mar = 0
            self.mouth_box = None  # Reset mouth box
            
            for mouth in mouths:
                if main_head is not None and is_inside(mouth, main_head):
                    mar = calculate_mar(mouth, main_head)
                    if mar > max_mar:
                        max_mar = mar
                        self.mouth_box = mouth  # Store the mouth box with highest MAR

            # Update yawn state
            current_time = time.time()
            if max_mar > pj.MAR_THRESHOLD:
                if not self.is_yawn_active:
                    self.yawn_start_time = current_time
                    self.is_yawn_active = True
                
                # Check if yawn has lasted more than 2 seconds
                if current_time - self.yawn_start_time >= 2.0:
                    self.yawn_counter = pj.YAWN_CONSEC_FRAMES
                    print("Yawn detected for more than 2 seconds!")
                else:
                    self.yawn_counter = 0
            else:
                self.is_yawn_active = False
                self.yawn_counter = 0
                self.mouth_box = None

            # Draw mouth box if detected
            if self.mouth_box is not None:
                x1, y1, x2, y2 = map(int, self.mouth_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return results[0].plot()
        except Exception as e:
            print(f"Detection processing error: {e}")
            return frame

    def update_alarm_state(self, frame: np.ndarray):
        """Update alarm state and display warnings"""
        if self.yawn_counter >= pj.YAWN_CONSEC_FRAMES:
            print("Yawn Alert Triggered!") 
            cv2.putText(frame, "ALERT: Yawning Detected!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.trigger_alarm()
        elif self.is_yawn_active:
            # Show countdown for yawn duration
            elapsed = time.time() - self.yawn_start_time
            cv2.putText(frame, f"Yawning: {elapsed:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Drowsiness detection logic
        if self.both_eyes_closed or self.yawn_counter >= pj.YAWN_CONSEC_FRAMES:
            current_time = time.time()
            if not self.count_start:
                self.start_time = current_time
                self.count_start = True
            
            self.time_close_eyes = current_time - self.start_time
            self.update_drowsiness_display(frame)

    def trigger_alarm(self):
        """Handle alarm triggering with thread safety"""
        if not self.alarm_on and self.alarm_sound:
            try:
                self.alarm_on = True
                self.alarm_sound.play()
            except Exception as e:
                print(f"Alarm error: {e}")

    def update_drowsiness_display(self, frame: np.ndarray):
        """Update display based on eye closure duration"""
        if self.time_close_eyes >= pj.TIME_THRESHOLD:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 255), -1)
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 255, 255), -1)
            cv2.putText(frame, f"Eyes closed: {self.time_close_eyes:.1f}s", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def detect_drowsiness(self, frame: np.ndarray) -> np.ndarray:
        """Main processing pipeline"""
        try:
            processed_frame = self.process_yolo_detections(frame)
            self.update_alarm_state(processed_frame)
            return processed_frame
        except Exception as e:
            print(f"Detection error: {e}")
            return frame