import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el scaler
model = tf.keras.models.load_model('/home/matias/Escritorio/Sing_Tanslate/Modelos/V3/V3.0/Model.keras')
scaler_params = np.load('/home/matias/Escritorio/Sing_Tanslate/Modelos/V3/V3.0/scaler_params.npy', allow_pickle=True).item()
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

def calculate_relative_position(point, reference_point):
    return point - reference_point

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Obtener punto de referencia (muñeca - punto 0)
        wrist_3d = get_point_3d(landmarks, 0)
        
        # Calcular posiciones relativas para cada punto
        features = []
        for i in range(1, 21):  # Empezamos desde 1 porque 0 es la muñeca
            point_3d = get_point_3d(landmarks, i)
            relative_pos = calculate_relative_position(point_3d, wrist_3d)
            
            # Guardar coordenadas x, y, z relativas
            features.extend([relative_pos[0], relative_pos[1], relative_pos[2]])
        
        # Convertir a array y normalizar
        features = np.array(features).reshape(1, -1)
        features_normalized = scaler.transform(features)
        
        # Realizar predicción
        prediction = model.predict(features_normalized)
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