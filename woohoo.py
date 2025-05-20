import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class VirtualKeyboard:
    def __init__(self):
        self.keys = [
            ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
            ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
            ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
            ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
            ["Space"]
        ]
        self.button_positions = []
        self.pinch_threshold = 40
        self.last_pressed_time = 0
        self.debounce_time = 0.5
        self.pressed_key_color = (0, 255, 0)
        self.hover_key_color = (255, 0, 255)
        self.default_key_color = (255, 0, 0)
        self.current_hovered_key = None
        self.last_pressed_key = None
        self.last_pressed_key_time = 0

    def draw_keyboard(self, img, hover_point = None):
        h, w, _ = img.shape
        kb_width = int(0.8 * w)
        kb_height = int(0.5 * h)
        start_x = (w - kb_width) // 2
        start_y = h - kb_height - 20

        rows = len(self.keys)
        max_cols = max(len(row) for row in self.keys)
        key_width = kb_width // max_cols - 5
        key_height = kb_height // rows - 5

        self.button_positions = []
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = start_x + j * (key_width + 5)
                y = start_y + i * (key_height + 5)

                # Handle wider keys
                w_multiplier = 1
                if key in ["Tab", "Caps", "Enter", "Shift"]:
                    w_multiplier = 1.5
                elif key == "Backspace":
                    w_multiplier = 2
                elif key == "Space":
                    w_multiplier = 6

                width = int(key_width * w_multiplier)

                if self.last_pressed_key == key and time.time() - self.last_pressed_key_time < self.debounce_time:
                    color = self.pressed_key_color
                elif hover_point and x < hover_point[0] < x + width and y < hover_point[1] < y + key_height:
                    color = self.hover_key_color
                    self.current_hovered_key = key
                else:
                    color = self.default_key_color

                cv2.rectangle(img, (x, y), (x + width, y + key_height), color, cv2.FILLED)
                cv2.putText(img, key, (x + 10, y + key_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.button_positions.append(((x, y), width, key_height, key))
        return img

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

    def check_key_press(self, finger_x, finger_y):
        current_time = time.time()
        if current_time - self.last_pressed_time < self.debounce_time:
            return None
        for (x, y), w, h, key in self.button_positions:
            if x < finger_x < x + w and y < finger_y < y + h:
                self.last_pressed_time = current_time
                self.last_pressed_key_time = current_time
                self.last_pressed_key = key
                print(f"Pressed key: {key}")

                key_to_press = key.lower()

                special_keys = {
                    "space": "space",
                    "enter": "enter",
                    "backspace": "backspace",
                    "tab": "tab",
                    "caps": "capslock",
                    "shift": "shift"
                }

                key_to_press = special_keys.get(key_to_press, key_to_press)

                try:
                    pyautogui.press(key_to_press)
                except Exception as e:
                    print(f"Error pressing key {key_to_press}: {e}")
                return key
        return None

def main():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {width}x{height}")

    keyboard = VirtualKeyboard()
    prev_x, prev_y = 0, 0
    smooth_factor = 10

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7
    ) as hands:

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            right_hand_landmarks = None
            left_hand_landmarks = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Right":
                        right_hand_landmarks = hand_landmarks.landmark
                    elif label == "Left":
                        left_hand_landmarks = hand_landmarks.landmark


            hover_point = None
            if right_hand_landmarks:
                index = right_hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = img.shape
                hover_point = int(index.x * w), int(index.y * h)

            img = keyboard.draw_keyboard(img, hover_point)

            if right_hand_landmarks:
                #img = keyboard.draw_keyboard(img)
                is_pinching, pinch_point = keyboard.detect_pinch(right_hand_landmarks, img.shape)

                if pinch_point:
                    cv2.circle(img, pinch_point, 15, (0, 255, 255), cv2.FILLED)

                if is_pinching and pinch_point:
                    pressed_key = keyboard.check_key_press(pinch_point[0], pinch_point[1])
                    if pressed_key:
                        print(f"Right hand pinched: {pressed_key}")

                mp_drawing.draw_landmarks(
                    img, results.multi_hand_landmarks[results.multi_hand_landmarks.index(hand_landmarks)],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            # else:
            #     keyboard.current_hovered_button = None

            if left_hand_landmarks:
                index_finger = left_hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = img.shape
                finger_x, finger_y = int(index_finger.x * w ), int(index_finger.y * h )

                screen_x = np.interp(finger_x, [0, w], [0, SCREEN_WIDTH])
                screen_y = np.interp(finger_y, [0, h], [0, SCREEN_HEIGHT])
                smooth_x = prev_x + (screen_x - prev_x) / smooth_factor
                smooth_y = prev_y + (screen_y - prev_y) / smooth_factor

                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y

                is_pinch, _ = keyboard.detect_pinch(left_hand_landmarks, img.shape)
                if is_pinch:
                    pyautogui.click()
                    time.sleep(0.1)

                mp_drawing.draw_landmarks(
                    img, results.multi_hand_landmarks[results.multi_hand_landmarks.index(hand_landmarks)],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

            cv2.imshow("Virtual Keyboard & Mouse", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
