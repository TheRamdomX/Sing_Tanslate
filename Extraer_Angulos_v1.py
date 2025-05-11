import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.95) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                height, width, _ = frame.shape

                def get_point(id):
                    return np.array([landmarks[id].x * width, landmarks[id].y * height])

                angles = {}

                # Función para calcular ángulo respecto al centro (wrist)
                def calculate_center_angle(center, point):
                    vector = point - center
                    angle = degrees(atan2(vector[1], vector[0]))
                    return round(angle, 2)

                wrist = get_point(0)

                # Lista de conexiones para sacar inclinaciones de cada línea
                connections = [
                    # Pulgar
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    # Índice
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    # Medio
                    (0, 9), (9,10), (10,11), (11,12),
                    # Anular
                    (0,13), (13,14), (14,15), (15,16),
                    # Meñique
                    (0,17), (17,18), (18,19), (19,20)
                ]

                for idx, (start_id, end_id) in enumerate(connections):
                    start_point = get_point(start_id)
                    end_point = get_point(end_id)
                    angles[f"Line_{start_id}_{end_id}"] = calculate_center_angle(wrist, end_point)

                y_offset = 30
                for name, angle in angles.items():
                    cv2.putText(frame, f"{name}: {angle}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 18

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('Hand Angles', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
