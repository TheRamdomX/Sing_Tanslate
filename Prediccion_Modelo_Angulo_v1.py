# Predict.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from math import degrees, atan2
from sklearn.preprocessing import StandardScaler  # Importar StandardScaler

# Cargar el modelo y el scaler
model = tf.keras.models.load_model('/home/matias/Escritorio/Sing_Tanslate/Modelos/V1/V1.3/Model.keras')  # Corregir nombre del archivo
scaler_params = np.load('/home/matias/Escritorio/Sing_Tanslate/Modelos/V1/V1.3/scaler_params.npy', allow_pickle=True).item()
scaler = StandardScaler()
scaler.mean_ = scaler_params['mean']
scaler.scale_ = scaler_params['scale']

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def get_point_3d(landmarks, id):
    return np.array([landmarks[id].x, landmarks[id].y, landmarks[id].z])

def calculate_reference_axis(landmarks):
    wrist = get_point_3d(landmarks, 0)
    middle_base = get_point_3d(landmarks, 9)
    ring_base = get_point_3d(landmarks, 13)
    middle_point = (middle_base + ring_base) / 2
    reference_axis = middle_point - wrist
    reference_axis = reference_axis / np.linalg.norm(reference_axis)
    return reference_axis, wrist

def calculate_angle_with_axis(point, reference_axis, wrist):
    vector = point - wrist
    projection = vector - np.dot(vector, reference_axis) * reference_axis
    angle = degrees(atan2(projection[0], projection[1]))
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Calcular eje de referencia
        reference_axis, wrist_3d = calculate_reference_axis(landmarks)
        
        # Extraer características (ángulos)
        angles = []
        for (start_id, end_id) in connections:
            end_point_3d = get_point_3d(landmarks, end_id)
            angle = calculate_angle_with_axis(end_point_3d, reference_axis, wrist_3d)
            angles.append(angle)
        
        # Normalizar características
        angles = np.array(angles).reshape(1, -1)
        angles_normalized = scaler.transform(angles)
        
        # Realizar predicción
        prediction = model.predict(angles_normalized)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Dibujar landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Mostrar predicción
        cv2.putText(frame, f"Prediccion: {chr(65 + predicted_class)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confianza: {confidence:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cv2.namedWindow('Hand Sign Prediction', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar frame
    processed_frame = process_frame(frame)
    
    # Mostrar resultado
    cv2.imshow('Hand Sign Prediction', processed_frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()