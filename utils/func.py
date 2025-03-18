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

def is_inside(child_box: np.ndarray, parent_box: np.ndarray, threshold: float = 0.8) -> bool:
    """Check if child box is mostly contained in parent box"""
    if child_box is None or parent_box is None:
        return False
    
    # Calculate intersection area
    inter_x1 = max(child_box[0], parent_box[0])
    inter_y1 = max(child_box[1], parent_box[1])
    inter_x2 = min(child_box[2], parent_box[2])
    inter_y2 = min(child_box[3], parent_box[3])
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return False
    
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    child_area = (child_box[2] - child_box[0]) * (child_box[3] - child_box[1])
    return (intersection / child_area) > threshold

def calculate_area(box):
    """Calculate area of a bounding box"""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_mar(mouth_box: np.ndarray, head_box: np.ndarray) -> float:
    if mouth_box is None or head_box is None:
        return 0.0
    
    mouth_height = max(1, mouth_box[3] - mouth_box[1])  # Prevent division by zero
    head_height = max(1, head_box[3] - head_box[1])  

    return mouth_height / head_height


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
        self.both_eyes_closed = False

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
                print(f"Detected class {cls} with box: {box}") 
                if cls == 1:
                    heads.append(box)
                elif cls == 2:
                    mouths.append(box)
                else:  # Assuming other classes are eyes
                    eyes.append(box)
            print(f"Total Heads: {len(heads)}, Total Mouths: {len(mouths)}, Total Eyes: {len(eyes)}")
            # Find largest head
            main_head = max(heads, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=None)
            
            # Process mouths
            for mouth in mouths:
                if main_head is not None and is_inside(mouth, main_head):
                    mar = calculate_mar(mouth, main_head)
                    max_mar = max(max_mar, mar)
                    print(f"MAR: {mar:.2f}, Threshold: {pj.MAR_THRESHOLD}")

            # Update yawn counter
            if max_mar > pj.MAR_THRESHOLD:
                self.yawn_counter = min(self.yawn_counter + 1, pj.YAWN_CONSEC_FRAMES)
            else:
                self.yawn_counter = max(self.yawn_counter - 0.5, 0)
            print(f"Yawn Counter: {self.yawn_counter}")

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

        # Drowsiness detection logic
        if self.both_eyes_closed:
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