import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from math import degrees, atan2

input_folder = "/home/matias/Escritorio/Sing_Tanslate/Data"
output_csv = "DataSet.csv"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_point(landmarks, id, width, height):
    return np.array([landmarks[id].x * width, landmarks[id].y * height])

def get_point_3d(landmarks, id):
    return np.array([landmarks[id].x, landmarks[id].y, landmarks[id].z])

def calculate_reference_axis(landmarks):
    # Obtener puntos 3D
    wrist = get_point_3d(landmarks, 0)
    middle_base = get_point_3d(landmarks, 9)
    ring_base = get_point_3d(landmarks, 13)
    
    # Calcular punto medio entre base del dedo medio y anular
    middle_point = (middle_base + ring_base) / 2
    
    # Calcular vector del eje de referencia
    reference_axis = middle_point - wrist
    reference_axis = reference_axis / np.linalg.norm(reference_axis)
    
    return reference_axis, wrist

def calculate_angle_with_axis(point, reference_axis, wrist):
    # Vector desde la muñeca al punto
    vector = point - wrist
    
    # Proyectar el vector en el plano perpendicular al eje de referencia
    projection = vector - np.dot(vector, reference_axis) * reference_axis
    
    # Calcular ángulo en el plano
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

data = []

Count = 0

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.60) as hands:

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error leyendo imagen: {img_path}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            height, width, _ = frame.shape

            # Calcular eje de referencia
            reference_axis, wrist_3d = calculate_reference_axis(landmarks)
            wrist = get_point(landmarks, 0, width, height)

            label = img_name[0]
            angles = {"Label": label}

            for (start_id, end_id) in connections:
                end_point_3d = get_point_3d(landmarks, end_id)
                angle = calculate_angle_with_axis(end_point_3d, reference_axis, wrist_3d)
                angles[f"Line_{start_id}_{end_id}"] = angle

            data.append(angles)
        else:
            print(f"No se detectaron manos en la imagen {img_name}")
            Count += 1

if data:
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV guardado como {output_csv}")
    print(f"Se procesaron {len(data)} imágenes.")
    print(f"Se omitieron {Count} imágenes sin detección de manos.")
else:
    print("No se detectaron manos en ninguna imagen.")
