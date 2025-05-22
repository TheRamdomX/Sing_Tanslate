import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from math import degrees, atan2

# Configuración de paths
input_folder = "/home/matias/Escritorio/Sing_Tanslate/Data"
output_csv = "DataSet_v2.csv"

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_point_3d(landmarks, id):
    #Obtiene coordenadas 3D normalizadas
    return np.array([landmarks[id].x, landmarks[id].y, landmarks[id].z])

def calculate_reference_axis(landmarks):
    #Calcula el eje de referencia usando muñeca y dedos medio-anular
    wrist = get_point_3d(landmarks, 0)
    middle_base = get_point_3d(landmarks, 9)
    ring_base = get_point_3d(landmarks, 13)
    
    # Punto medio entre base del dedo medio y anular
    middle_point = (middle_base + ring_base) / 2
    reference_axis = middle_point - wrist
    
    # Normalización
    norm = np.linalg.norm(reference_axis)
    if norm > 0:
        reference_axis = reference_axis / norm
    
    return reference_axis, wrist

def calculate_angle_with_axis(point, reference_axis, wrist):
    #Calcula el ángulo entre un punto y el eje de referencia
    vector = point - wrist
    
    try:
        # Proyección en el plano perpendicular al eje
        dot_product = np.dot(vector, reference_axis)
        projection = vector - dot_product * reference_axis
        
        # Manejo de casos límite
        norm_proj = np.linalg.norm(projection)
        if norm_proj < 1e-6:
            return 0.0
            
        angle = degrees(atan2(projection[0], projection[1]))
        angle = abs(angle)
        return round(angle % 180, 2)
    except:
        return 0.0

def calculate_distance(point1, point2):
    #Calcula la distancia euclidiana entre dos puntos
    return np.linalg.norm(point1 - point2)

def calculate_angle_between_points(point1, point2, reference_axis, wrist):
    #Calcula el ángulo entre dos puntos adyacentes con respecto al eje de referencia
    vector = point2 - point1
    return calculate_angle_with_axis(vector, reference_axis, wrist)

def main():
    data = []
    skipped_images = 0
    processed_images = 0

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,  
        min_detection_confidence=0.8,  
        min_tracking_confidence=0.8) as hands:

        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Error leyendo imagen: {img_path}")
                skipped_images += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = hand_landmarks.landmark
                    
                    # Calcular eje de referencia
                    reference_axis, wrist_3d = calculate_reference_axis(landmarks)
                    
                    # Obtener etiqueta (primera letra del nombre de archivo)
                    label = img_name[0]
                    features = {"Label": label}
                    
                    # 1. Ángulos de cada punto con respecto al eje
                    for i in range(21):
                        point_3d = get_point_3d(landmarks, i)
                        angle = calculate_angle_with_axis(point_3d, reference_axis, wrist_3d)
                        features[f"Angle_Point_{i}"] = angle
                    
                    # 2. Distancias desde la muñeca (punto 0) a cada punto
                    for i in range(1, 21):
                        point_3d = get_point_3d(landmarks, i)
                        distance = calculate_distance(wrist_3d, point_3d)
                        features[f"Distance_0_to_{i}"] = round(distance, 4)
                    
                    # 3. Ángulos entre puntos adyacentes
                    connections = [
                        (1, 2), (2, 3), (3, 4),  # Pulgar
                        (5, 6), (6, 7), (7, 8),  # Índice
                        (9, 10), (10, 11), (11, 12),  # Medio
                        (13, 14), (14, 15), (15, 16),  # Anular
                        (17, 18), (18, 19), (19, 20)  # Meñique
                    ]
                    
                    for start_id, end_id in connections:
                        start_point = get_point_3d(landmarks, start_id)
                        end_point = get_point_3d(landmarks, end_id)
                        angle = calculate_angle_between_points(start_point, end_point, reference_axis, wrist_3d)
                        features[f"Angle_Between_{start_id}_{end_id}"] = angle
                    
                    data.append(features)
                
                processed_images += 1
            else:
                print(f"No se detectaron manos en la imagen {img_name}")
                skipped_images += 1

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"\nResultados:")
        print(f"- CSV guardado como {output_csv}")
        print(f"- Imágenes procesadas: {processed_images}")
        print(f"- Manos detectadas: {len(data)}")
        print(f"- Imágenes omitidas: {skipped_images}")
        print(f"- Características por muestra: {len(data[0]) if data else 0}")
    else:
        print("No se detectaron manos en ninguna imagen.")

if __name__ == "__main__":
    main()