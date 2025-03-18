# %%
import keras
import cv2
import time
from utils.config import project_config as pj
from core import drowsiness_detector
import sys
import pygame


cap = cv2.VideoCapture(0)
model = keras.models.load_model(pj.MODEL_PATH1)
alarm_on = False
show_fps = 1
detector = drowsiness_detector.DrowsinessDetector("yolo", model)

if show_fps:
    num_frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add debug prints
        print("Processing frame...")
        frame = detector.detect_drowsiness(frame)
        print(f"Yawn counter: {detector.yawn_counter}")
        print(f"Alarm state: {detector.alarm_on}")
        
        cv2.imshow('Drowsiness Detection', frame)
        
        # Check for 'q' key or window close button
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27 or cv2.getWindowProperty('Drowsiness Detection', cv2.WND_PROP_VISIBLE) < 1:
            print("Closing application...")
            break

finally:
    # Cleanup
    print("Cleaning up...")
    detector.alarm_sound.stop()  # Stop any playing alarm
    pygame.mixer.quit()  # Cleanup pygame
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)  # Force exit the program