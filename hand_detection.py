import cv2
import mediapipe as mp
import math
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # For drawing if needed for debugging within HandDetector

class HandDetector(QObject):
    # Signal to emit hand data
    # (right_hand_data, left_hand_data)
    # Each hand data will be a tuple: (index_tip_coords, is_pinching, raw_landmarks)
    hand_data_signal = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.pinch_threshold = 40 # From your original code

    def detect_pinch(self, landmarks, img_shape):
        if landmarks:
            thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = img_shape
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)
            distance = math.hypot(x2 - x1, y2 - y1)
            return distance < self.pinch_threshold, (x2, y2)
        return False, None

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        right_hand_data = (None, False, None) # (index_tip_coords, is_pinching, raw_landmarks)
        left_hand_data = (None, False, None)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = img.shape
                index_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                # Check for pinch
                is_pinching, pinch_point = self.detect_pinch(hand_landmarks.landmark, img.shape)

                if label == "Right":
                    right_hand_data = (index_tip_coords, is_pinching, hand_landmarks.landmark)
                elif label == "Left":
                    left_hand_data = (index_tip_coords, is_pinching, hand_landmarks.landmark)
        
        # Emit the combined hand data
        self.hand_data_signal.emit((right_hand_data, left_hand_data))

class CameraThread(QThread):
    # Signal to emit raw frames (optional, for debugging or if you wanted to show the camera feed)
    # frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, hand_detector):
        super().__init__()
        self.hand_detector = hand_detector
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.running = False
            return

        while self.running:
            success, img = self.cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1) # Mirror the image
            self.hand_detector.process_frame(img)
            # self.frame_signal.emit(img) # Emit frame if needed for display elsewhere

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait() # Wait for the thread to finish cleanly