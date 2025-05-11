import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan2
import tensorflow as tf 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.95)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

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
            
            def calculate_center_angle(center, point):
                vector = point - center
                angle = degrees(atan2(vector[0], vector[1]))
                angle = abs(angle)
                if angle > 180:
                    angle = 360 - angle
                return round(angle, 2)
            
            wrist = get_point(0)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9,10), (10,11), (11,12),
                (0,13), (13,14), (14,15), (15,16),
                (0,17), (17,18), (18,19), (19,20)
            ]
            
            angles = []
            for start_id, end_id in connections:
                end_point = get_point(end_id)
                angles.append(calculate_center_angle(wrist, end_point))
            
            # Predecir la letra
            input_data = np.array([angles])
            prediction = model.predict(input_data, verbose=0)
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction) * 100
            
            # Mostrar resultado
            cv2.putText(frame, f"Letra: {predicted_class} ({confidence:.1f}%)", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow('Reconocimiento de Letras', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

