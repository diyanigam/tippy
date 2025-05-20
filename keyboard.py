import sys
import cv2 # Only for constants like FONT_HERSHEY_SIMPLEX if needed, but QPainter has its own fonts
import numpy as np
import time
import pyautogui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSlot

# Import HandDetector and CameraThread from our hand_detector.py file
from hand_detection import HandDetector, CameraThread

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class TransparentKeyboardWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Keyboard")
        self.setGeometry(0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2) # Position at bottom half of screen
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.keys = [
            ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
            ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
            ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
            ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
            ["Space"]
        ]
        self.button_positions = [] # Stores ((x, y), width, height, key_text) for each button

        # --- Hand and Interaction State ---
        self.right_hand_index_tip = None # (x, y) coordinates
        self.right_hand_pinching = False
        self.left_hand_index_tip = None # (x, y) coordinates
        self.left_hand_pinching = False

        self.last_pressed_key = None
        self.last_pressed_key_time = 0
        self.debounce_time = 0.5
        self.pressed_key_color = QColor(0, 255, 0, 200) # Green, semi-transparent
        self.hover_key_color = QColor(255, 0, 255, 150) # Magenta, semi-transparent
        self.default_key_color = QColor(255, 0, 0, 100) # Red, semi-transparent

        self.current_hovered_key_info = None # Stores ((x, y), width, height, key_text) of the hovered key

        # Mouse smoothing
        self.prev_mouse_x, self.prev_mouse_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self.smooth_factor = 10

        self.init_ui()

    def init_ui(self):
        self.calculate_key_positions()

        # Set up a timer to continuously update the UI based on hand data
        # This timer will trigger paintEvent periodically
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update) # Calls update() which triggers paintEvent
        self.update_timer.start(30) # ~33 FPS for UI refresh

        # Setup hand detection and camera thread
        self.hand_detector = HandDetector()
        self.camera_thread = CameraThread(self.hand_detector)
        self.hand_detector.hand_data_signal.connect(self.handle_hand_data)
        self.camera_thread.start()

    def calculate_key_positions(self):
        # Calculate keyboard dimensions relative to window size
        kb_width = int(0.95 * self.width())
        kb_height = int(0.95 * self.height())
        start_x = (self.width() - kb_width) // 2
        start_y = (self.height() - kb_height) // 2

        rows = len(self.keys)
        max_cols = max(len(row) for row in self.keys)
        key_width = kb_width // max_cols - 5
        key_height = kb_height // rows - 5

        self.button_positions = []
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = start_x + j * (key_width + 5)
                y = start_y + i * (key_height + 5)

                w_multiplier = 1
                if key in ["Tab", "Caps", "Enter", "Shift"]:
                    w_multiplier = 1.5
                elif key == "Backspace":
                    w_multiplier = 2
                elif key == "Space":
                    w_multiplier = 6

                width = int(key_width * w_multiplier)

                self.button_positions.append(((x, y), width, key_height, key))

    @pyqtSlot(tuple)
    def handle_hand_data(self, data):
        right_hand_data, left_hand_data = data

        self.right_hand_index_tip = right_hand_data[0]
        self.right_hand_pinching = right_hand_data[1]

        self.left_hand_index_tip = left_hand_data[0]
        self.left_hand_pinching = left_hand_data[1]

        # Process mouse movement for left hand
        if self.left_hand_index_tip:
            # Map camera coordinates to screen coordinates
            camera_w, camera_h = 640, 480 # Assuming a typical camera resolution, adjust if known
            finger_x_cam, finger_y_cam = self.left_hand_index_tip

            screen_x = np.interp(finger_x_cam, [0, camera_w], [0, SCREEN_WIDTH])
            screen_y = np.interp(finger_y_cam, [0, camera_h], [0, SCREEN_HEIGHT])

            # Apply smoothing
            smooth_x = self.prev_mouse_x + (screen_x - self.prev_mouse_x) / self.smooth_factor
            smooth_y = self.prev_mouse_y + (screen_y - self.prev_mouse_y) / self.smooth_factor

            pyautogui.moveTo(smooth_x, smooth_y)
            self.prev_mouse_x, self.prev_mouse_y = smooth_x, smooth_y

        # Handle left hand click
        if self.left_hand_pinching:
            pyautogui.click()
            time.sleep(0.1) # Debounce clicks

        # Check for key presses with the right hand
        if self.right_hand_pinching and self.right_hand_index_tip:
            self.check_key_press(self.right_hand_index_tip[0], self.right_hand_index_tip[1])


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the keyboard
        current_time = time.time()
        self.current_hovered_key_info = None # Reset hovered key for this frame

        for (x, y), w, h, key_text in self.button_positions:
            key_rect = QRect(x, y, w, h)
            color = self.default_key_color

            # Check for hover
            if self.right_hand_index_tip:
                hover_point_qt = QPoint(self.right_hand_index_tip[0], self.right_hand_index_tip[1])
                # Map camera coordinates to local window coordinates for hover detection
                # This mapping needs to be carefully considered.
                # For simplicity, let's assume camera coordinates are roughly proportional to screen.
                # A more precise mapping might be needed if camera FOV significantly differs.
                # Here, we map a central region of the camera feed to the keyboard area.
                # Assuming 640x480 camera resolution and a keyboard width of 0.95 * self.width()
                # keyboard_x_start = (self.width() - 0.95 * self.width()) // 2
                # keyboard_y_start = (self.height() - 0.95 * self.height()) // 2
                # This needs to be calibrated. For now, a direct mapping from camera to window:
                mapped_x = np.interp(hover_point_qt.x(), [0, 640], [0, self.width()])
                mapped_y = np.interp(hover_point_qt.y(), [0, 480], [0, self.height()])
                mapped_hover_point = QPoint(int(mapped_x), int(mapped_y))

                if key_rect.contains(mapped_hover_point):
                    color = self.hover_key_color
                    self.current_hovered_key_info = ((x, y), w, h, key_text)


            # Check for pressed key visual feedback
            if self.last_pressed_key == key_text and current_time - self.last_pressed_key_time < self.debounce_time:
                color = self.pressed_key_color

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRoundedRect(key_rect, 5, 5) # Draw rounded rectangle for keys

            painter.setPen(QPen(Qt.white))
            font = QFont("Arial", 14)
            painter.setFont(font)
            painter.drawText(key_rect, Qt.AlignCenter, key_text)

        # Draw right index finger circle (hover indicator)
        if self.right_hand_index_tip:
            # Map camera coords to window coords for drawing
            mapped_x = np.interp(self.right_hand_index_tip[0], [0, 640], [0, self.width()])
            mapped_y = np.interp(self.right_hand_index_tip[1], [0, 480], [0, self.height()])
            
            finger_center = QPoint(int(mapped_x), int(mapped_y))
            
            if self.right_hand_pinching:
                painter.setBrush(QBrush(QColor(0, 255, 255, 200))) # Cyan when pinching
            else:
                painter.setBrush(QBrush(QColor(255, 255, 0, 150))) # Yellow normally
            
            painter.setPen(QPen(Qt.black, 2))
            painter.drawEllipse(finger_center, 15, 15) # Draw a circle

    def check_key_press(self, finger_x_cam, finger_y_cam):
        current_time = time.time()
        if current_time - self.last_pressed_key_time < self.debounce_time:
            return None

        # Map camera coordinates to local window coordinates for hit-testing
        mapped_finger_x = np.interp(finger_x_cam, [0, 640], [0, self.width()])
        mapped_finger_y = np.interp(finger_y_cam, [0, 480], [0, self.height()])


        for (x, y), w, h, key_text in self.button_positions:
            if x < mapped_finger_x < x + w and y < mapped_finger_y < y + h:
                self.last_pressed_key_time = current_time
                self.last_pressed_key = key_text
                print(f"Pressed key: {key_text}")

                key_to_press = key_text.lower()

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
                return key_text
        return None

    def closeEvent(self, event):
        # Ensure the camera thread stops when the window closes
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    keyboard_window = TransparentKeyboardWindow()
    keyboard_window.show()
    sys.exit(app.exec_())