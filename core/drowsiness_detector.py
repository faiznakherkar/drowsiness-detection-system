import cv2
from utils.config import project_config as pj
from keras.utils import img_to_array
import time
import numpy as np
from playsound import playsound
from threading import Thread
from ultralytics import YOLO
from utils.func import calculate_mar, get_results, calculate_area, is_inside, crop_image
import os
import pygame
from typing import Optional

# Try to import Twilio, but don't fail if it's not available
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("Warning: Twilio package not installed. SMS alerts will be disabled.")


class DrowsinessDetector:
    def __init__(self, detect_model, clf_model) -> None:
        self.detect_model = detect_model
        self.clf_model = clf_model
        self.eye_status1 = None
        self.eye_status2 = None
        self.start_time = 0
        self.end_time = 0
        self.count_start = 0
        self.time_close_eyes = 0
        self.load_model = 0
        self.alarm_on = 0
        self.yawn_counter = 0
        self.yawning = False
        self.frame_counter = 0
        self.sms_sent = False  # Flag to track if SMS has been sent
        self.twilio_client = None  # Initialize as None

        # Initialize Twilio client if available and configured
        if TWILIO_AVAILABLE and pj.validate_twilio_config():
            try:
                self.twilio_client = Client(pj.TWILIO_ACCOUNT_SID, pj.TWILIO_AUTH_TOKEN)
                print("Twilio client initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Twilio client: {e}")
                self.twilio_client = None

        self.face_cascade = None
        self.left_eye_cascade = None
        self.right_eye_cascade = None
        self.both_eye_close = None
        self.yolo_model = None
        self.load_detect_model(detect_model)
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound(pj.ALARM_SOUND)

    def load_detect_model(self, detect_model):
        if detect_model == "cascade":
            self.face_cascade = cv2.CascadeClassifier(pj.FACE_CASCADE_PATH)
            self.left_eye_cascade = cv2.CascadeClassifier(pj.LEFT_EYE_CASCADE_PATH)
            self.right_eye_cascade = cv2.CascadeClassifier(pj.RIGHT_EYE_CASCADE_PATH)
        if detect_model == "yolo":
            import torch
            # Store the original torch.load function
            original_load = torch.load
            # Create a new function that wraps the original with weights_only=False
            torch.load = lambda f, *args, **kwargs: original_load(f, *args, weights_only=False, **kwargs)
            self.yolo_model = YOLO(pj.YOLO_MODEL_PATH)
        self.load_model = True

    def process_eye_frame(self, eye_frame):
        processed_eye_frame = cv2.resize(eye_frame, (145, 145))
        processed_eye_frame = processed_eye_frame.astype("float") / 255.0
        processed_eye_frame = img_to_array(processed_eye_frame)
        processed_eye_frame = np.expand_dims(processed_eye_frame, axis=0)
        return processed_eye_frame

    def get_eye_status(self, frame):
        if self.load_model and self.detect_model == "cascade":
            framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(framegray, 1.3, 5)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = framegray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]
                left_eye = self.left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = self.right_eye_cascade.detectMultiScale(roi_gray)
                self.eye_status1 = None
                self.eye_status2 = None
                for x1, y1, w1, h1 in left_eye:
                    cv2.rectangle(
                        roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2
                    )
                    eye1 = roi_color[y1 : y1 + h1, x1 : x1 + w1]
                    pred1 = self.clf_model.predict(self.process_eye_frame(eye1))
                    self.eye_status1 = np.argmax(pred1)
                    break

                for x2, y2, w2, h2 in right_eye:
                    cv2.rectangle(
                        roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2
                    )
                    eye2 = roi_color[y2 : y2 + h2, x2 : x2 + w2]
                    pred2 = self.clf_model.predict(self.process_eye_frame(eye2))
                    self.eye_status2 = np.argmax(pred2)
                    break
        if self.load_model and self.detect_model == "yolo":
            results = self.yolo_model.predict(frame, conf=0.2, verbose=False)
            frame = results[0].plot()
            boxes, lst_cls = get_results(results)

            boxes_heads = []
            boxes_eyes = []
            boxes_mouths = []

            # Debug print for all detections
            print("\nDetected objects:")
            for i, (box, cls) in enumerate(zip(boxes, lst_cls)):
                print(f"Class {cls} at {box}")
                if cls == 1:  # head
                    boxes_heads.append(box)
                elif cls == 2:  # mouth/yawn
                    boxes_mouths.append(box)
                else:  # eyes
                    boxes_eyes.append(box)

            print("Head boxes:",boxes_heads)
            print("Mouth boxes:",boxes_mouths)
            print("Eye boxes:",boxes_eyes)

            # Find largest head box
            largest_box_head = None
            if boxes_heads:
                largest_box_head = max(boxes_heads, key=calculate_area)
                print(f"Head detected at: {largest_box_head}")

            # Enhanced yawn detection
            if largest_box_head is not None:
                max_mar = 0
                mouth_detected = False
                current_time = time.time()

                for box in boxes_mouths:
                    if is_inside(box, largest_box_head):
                        mar = calculate_mar(box, largest_box_head)
                        mouth_detected = True
                        max_mar = max(max_mar, mar)
                        print(f"Mouth detected - MAR: {mar:.3f}, Threshold: {pj.MAR_THRESHOLD}")

                        if mar > pj.MAR_THRESHOLD:
                            if not self.yawning:
                                self.yawn_start_time = current_time
                                self.yawning = True
                            # Check if yawn has lasted more than 2 seconds
                            if current_time - self.yawn_start_time >= 2.0:
                                self.yawn_counter = pj.YAWN_CONSEC_FRAMES
                                print("Yawn detected for more than 2 seconds!")
                            else:
                                self.yawn_counter = 0
                            break

                # Update yawn state
                if not mouth_detected or max_mar <= pj.MAR_THRESHOLD:
                    self.yawning = False
                    self.yawn_counter = 0
                
                print(f"Final yawn state - Counter: {self.yawn_counter}, Yawning: {self.yawning}")

            # Check eyes
            self.both_eye_close = False
            status = []
            for box in boxes_eyes:
                if is_inside(box, largest_box_head):
                    eye_frame = crop_image(frame, box)
                    processed_eye_frame = self.process_eye_frame(eye_frame)
                    pred = self.clf_model.predict(processed_eye_frame)
                    status.append(np.argmax(pred))
            
            if len(status) == 0:
                pass
            elif all(s == 2 for s in status):
                self.both_eye_close = True

        return frame

    def start_alarm(self, sound):
        """Play the alarm sound"""
        try:
            if not pygame.mixer.get_busy():  # Only play if not already playing
                self.alarm_sound.play(maxtime=3000)  # Play for 3 seconds max
        except Exception as e:
            print(f"Error playing alarm sound: {e}")

    def send_sms_alert(self):
        """Send SMS alert using Twilio"""
        if not self.twilio_client:
            print("Warning: Twilio client not available. SMS alert not sent.")
            return

        try:
            if not self.sms_sent:
                message = self.twilio_client.messages.create(
                    messaging_service_sid=pj.TWILIO_MESSAGING_SERVICE_SID,
                    body='Your Dear Faiz Nakherkar has fallen asleep in his car and is not responding, please contact him immediately - Alert!',
                    to=pj.TWILIO_TO_NUMBER
                )
                print(f"SMS sent successfully! SID: {message.sid}")
                self.sms_sent = True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            # Don't set sms_sent to True if there was an error
            # This allows retrying on the next frame

    def reset_sms_state(self):
        """Reset SMS state when eyes are open"""
        self.sms_sent = False

    def detect_drowsiness(self, frame):
        frame = self.get_eye_status(frame)
        
        # More sensitive drowsiness detection
        drowsy = (self.yawning and self.yawn_counter >= pj.YAWN_CONSEC_FRAMES) or \
                 (self.eye_status1 == 2 and self.eye_status2 == 2) or \
                 self.both_eye_close
        
        # Reset alarm and SMS state only if definitely not drowsy
        if not drowsy and self.yawn_counter < (pj.YAWN_CONSEC_FRAMES / 3):
            if self.alarm_on:
                self.alarm_sound.stop()
                self.alarm_on = False
            self.time_close_eyes = 0
            self.count_start = False
            self.reset_sms_state()  # Reset SMS state when eyes are open
        
        # Yawning detection display
        if self.yawning:  # Show warning for any yawn detection
            # Calculate warning intensity
            warning_intensity = min(1.0, self.yawn_counter / pj.YAWN_CONSEC_FRAMES)
            
            if warning_intensity > 0.7:
                # Red warning for significant yawning
                cv2.rectangle(frame, (0, 70), (frame.shape[1], 140), (0, 0, 255), -1)
                cv2.putText(frame, "DROWSINESS ALERT: Yawning Detected!", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add yawn duration display
                elapsed = time.time() - self.yawn_start_time
                cv2.putText(frame, f"Yawn Duration: {elapsed:.1f}s", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if not self.alarm_on:
                    self.alarm_on = True
                    try:
                        t = Thread(target=self.start_alarm, args=(pj.ALARM_SOUND,))
                        t.daemon = True
                        t.start()
                    except Exception as e:
                        print(f"Error starting alarm thread: {e}")
            else:
                # Yellow warning for initial yawn detection
                cv2.rectangle(frame, (0, 70), (frame.shape[1], 140), (0, 255, 255), -1)
                elapsed = time.time() - self.yawn_start_time
                cv2.putText(frame, f"Warning: Yawning ({elapsed:.1f}s)", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Drowsiness detection
        if (self.eye_status1 == 2 and self.eye_status2 == 2) or self.both_eye_close:
            if not self.count_start:
                self.start_time = time.time()
                self.count_start = True
            if self.count_start:
                self.end_time = time.time()
                self.time_close_eyes = self.end_time - self.start_time

                # Check if we should send SMS
                if self.time_close_eyes >= pj.SMS_THRESHOLD:
                    self.send_sms_alert()

            if self.time_close_eyes >= pj.TIME_THRESHOLD:
                # Draw red filled rectangle at top of frame
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "DANGER! DROWSINESS DETECTED!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Eyes Closed for: {round(self.time_close_eyes, 1)} seconds",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                if not self.alarm_on:
                    self.alarm_on = True
                    try:
                        t = Thread(target=self.start_alarm, args=(pj.ALARM_SOUND,))
                        t.daemon = True
                        t.start()
                    except Exception as e:
                        print(f"Error starting alarm thread: {e}")
            else:
                # Yellow warning for eyes closed but not yet drowsy
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"Warning: Eyes Closed for {round(self.time_close_eyes, 1)} seconds",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
        else:
            # Green status for eyes open
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 255, 0), -1)
            cv2.putText(
                frame,
                "Status: Alert - Eyes Open",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            self.alarm_on = False
            self.time_close_eyes = 0
            self.count_start = False
            self.reset_sms_state()  # Reset SMS state when eyes are open

        return frame