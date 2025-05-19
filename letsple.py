import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Use lowest resolution for faster processing
FRAME_WIDTH, FRAME_HEIGHT = 160, 120
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    frame_counter = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Process every 5 frames only to save compute
        if frame_counter % 5 == 0:
            results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = small_frame.shape
                # Get thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))

                # Draw small circles on tips
                cv2.circle(small_frame, thumb_pos, 5, (0, 255, 0), -1)
                cv2.circle(small_frame, index_pos, 5, (0, 255, 0), -1)

                # Simple pinch detection
                dist = math.hypot(thumb_pos[0] - index_pos[0], thumb_pos[1] - index_pos[1])
                if dist < 30:
                    cv2.putText(small_frame, "Pinch!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

        # Show small frame scaled up to window
        cv2.imshow("Fast Hand Tracking", cv2.resize(small_frame, (FRAME_WIDTH*3, FRAME_HEIGHT*3)))

        frame_counter += 1

        # FPS calculation & display every second
        if frame_counter % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - last_time)
            last_time = current_time
            print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
