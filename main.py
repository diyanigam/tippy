import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class VirtualKeyboard:
    def __init__(self, keys=[["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                             ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                             ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
                             ["Space", "Backspace"]]):
        self.keys = keys
        self.button_positions = []
        self.pinch_threshold = 40  # Increased for better detection
        self.last_pressed_time = 0
        self.debounce_time = 0.5  # Half second debounce
        self.pressed_key_color = (0, 255, 0)
        self.default_key_color = (255, 0, 0)
        self.current_hovered_button = None
        self.last_pressed_key = None

    def draw_keyboard(self, img):
        h, w, _ = img.shape
        rows = len(self.keys)
        cols = max(len(row) for row in self.keys)
        key_width = w // cols - 10
        key_height = h // (rows + 2) - 10

        self.button_positions = []
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = j * (key_width + 10) + 5
                y = (i + 1) * (key_height + 10) + 5
                color = self.pressed_key_color if self.current_hovered_button == key else self.default_key_color
                cv2.rectangle(img, (x, y), (x + key_width, y + key_height), color, cv2.FILLED)
                cv2.putText(img, key, (x + 10, y + key_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.button_positions.append(((x, y), key_width, key_height, key))
        return img

    def detect_pinch(self, hand_landmarks, img_shape):
        if hand_landmarks and len(hand_landmarks) >= 21:
            thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP.value]
            index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            h, w, _ = img_shape

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)
            return distance < self.pinch_threshold, (index_x, index_y)
        return False, None

    def check_key_press(self, finger_x, finger_y):
        current_time = time.time()
        if current_time - self.last_pressed_time < self.debounce_time:
            return None
        for (x, y), w, h, key in self.button_positions:
            if x < finger_x < x + w and y < finger_y < y + h:
                self.last_pressed_time = current_time
                self.last_pressed_key = key
                print(f"Pressed key: {key}")  # Debug print of key press

                # Map special keys to pyautogui keys
                key_to_press = key.lower()
                if key.lower() == "space":
                    key_to_press = "space"
                elif key.lower() == "backspace":
                    key_to_press = "backspace"

                try:
                    pyautogui.press(key_to_press)
                except Exception as e:
                    print(f"Error pressing key {key_to_press}: {e}")

                return key
        return None

def main():
    cap = cv2.VideoCapture(0)
    keyboard = VirtualKeyboard()
    model_path = 'models/hand_landmarker.task'

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            img = cv2.flip(img, 1)
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = landmarker.detect(mp_image)
            hand_landmarks_list = detection_result.hand_landmarks

            keyboard.current_hovered_button = None  # reset hovered key
            img = keyboard.draw_keyboard(img)

            if hand_landmarks_list:
                for hand_landmarks in hand_landmarks_list:
                    # Manually convert to NormalizedLandmarkList
                    landmark_proto = landmark_pb2.NormalizedLandmarkList()
                    for lm in hand_landmarks:
                        landmark_proto.landmark.append(
                            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        )

                    mp_drawing.draw_landmarks(
                        img,
                        landmark_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )


                hand_landmarks = hand_landmarks_list[0]
                is_pinching, pinch_point = keyboard.detect_pinch(hand_landmarks, img.shape)

                index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                h, w, _ = img.shape
                finger_x, finger_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw fingertip circle
                cv2.circle(img, (finger_x, finger_y), 10, (0, 255, 0), cv2.FILLED)

                # Highlight hovered key based on fingertip
                for (x, y), key_w, key_h, key in keyboard.button_positions:
                    if x < finger_x < x + key_w and y < finger_y < y + key_h:
                        keyboard.current_hovered_button = key
                        break

                # If pinching, check key press at pinch point
                if is_pinching and pinch_point:
                    pressed_key = keyboard.check_key_press(pinch_point[0], pinch_point[1])
                    # You can do something with pressed_key here (like print or update state)

                # Visualize pinch point
                if pinch_point:
                    cv2.circle(img, pinch_point, 15, (0, 255, 255), cv2.FILLED)

            cv2.imshow('Virtual Keyboard', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
