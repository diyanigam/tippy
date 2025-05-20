import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# No resolution change, keep original frame size

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Create black background image same size as frame
        black_bg = np.zeros_like(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on black background
                mp_drawing.draw_landmarks(
                    black_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

        # Show the black background with skeleton
        cv2.imshow("Hand Skeleton on Black Background", black_bg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
