import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from math import degrees, atan2

# Configuración de paths
input_folder = "/home/matias/Escritorio/Sing_Tanslate/Data"
output_csv = "DataSet_Mejorado.csv"

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def enhance_contrast(image):
    """Mejora el contraste de la imagen usando CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Aplica un filtro de enfoque a la imagen"""
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def get_point(landmarks, id, width, height):
    """Obtiene coordenadas 2D normalizadas"""
    return np.array([landmarks[id].x * width, landmarks[id].y * height])

def get_point_3d(landmarks, id):
    """Obtiene coordenadas 3D normalizadas"""
    return np.array([landmarks[id].x, landmarks[id].y, landmarks[id].z])

def calculate_reference_axis(landmarks):
    """Calcula el eje de referencia mejorado"""
    wrist = get_point_3d(landmarks, 0)
    middle_base = get_point_3d(landmarks, 9)
    ring_base = get_point_3d(landmarks, 13)
    pinky_base = get_point_3d(landmarks, 17)
    
    # Usar más puntos para un eje más estable
    middle_point = (middle_base + ring_base + pinky_base) / 3
    reference_axis = middle_point - wrist
    
    # Normalización robusta
    norm = np.linalg.norm(reference_axis)
    if norm > 0:
        reference_axis = reference_axis / norm
    
    return reference_axis, wrist

def calculate_angle_with_axis(point, reference_axis, wrist):
    """Cálculo de ángulo mejorado con manejo de errores"""
    vector = point - wrist
    
    try:
        # Proyección más estable numéricamente
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

def compute_joint_angles(landmarks):
    """Calcula ángulos directos entre segmentos adyacentes"""
    angles = {}
    connections = [
        (1, 2, 3), (2, 3, 4),       # Pulgar
        (5, 6, 7), (6, 7, 8),       # Índice
        (9, 10, 11), (10, 11, 12),  # Medio
        (13, 14, 15), (14, 15, 16), # Anular
        (17, 18, 19), (18, 19, 20)  # Meñique
    ]
    
    for a, b, c in connections:
        vec1 = get_point_3d(landmarks, a) - get_point_3d(landmarks, b)
        vec2 = get_point_3d(landmarks, c) - get_point_3d(landmarks, b)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            angle = degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles[f"Joint_{a}_{b}_{c}"] = round(angle, 2)
        else:
            angles[f"Joint_{a}_{b}_{c}"] = 0.0
    
    return angles

def calculate_normalized_distances(landmarks):
    """Calcula distancias normalizadas usando la longitud del dedo índice como referencia"""
    index_length = np.linalg.norm(get_point_3d(landmarks, 5) - get_point_3d(landmarks, 8))
    distances = {}
    
    if index_length > 0:
        for i in range(21):
            for j in range(i+1, 21):
                dist = np.linalg.norm(get_point_3d(landmarks, i) - get_point_3d(landmarks, j))
                distances[f"NormDist_{i}_{j}"] = round(dist / index_length, 4)
    
    return distances

def process_image(image, hands):
    """Procesa una imagen individual con manejo de errores"""
    # Preprocesamiento
    enhanced_img = enhance_contrast(image.copy())
    frame_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    # Primer intento de detección
    results = hands.process(frame_rgb)
    
    # Segundo intento con imagen mejorada si falla
    if not results.multi_hand_landmarks:
        sharp_img = sharpen_image(enhanced_img)
        frame_rgb = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
    
    return results

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

            results = process_image(frame, hands)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = hand_landmarks.landmark
                    height, width, _ = frame.shape
                    
                    # Calcular eje de referencia
                    reference_axis, wrist_3d = calculate_reference_axis(landmarks)
                    
                    # Obtener etiqueta (primera letra del nombre de archivo)
                    label = img_name[0]
                    features = {
                        "Label": label,
                        "Handedness": results.multi_handedness[hand_idx].classification[0].label
                    }
                    
                    # Calcular características
                    line_angles = {}
                    for (start_id, end_id) in mp_hands.HAND_CONNECTIONS:
                        end_point_3d = get_point_3d(landmarks, end_id)
                        angle = calculate_angle_with_axis(end_point_3d, reference_axis, wrist_3d)
                        line_angles[f"Line_{start_id}_{end_id}"] = angle
                    
                    # Añadir características adicionales
                    joint_angles = compute_joint_angles(landmarks)
                    norm_distances = calculate_normalized_distances(landmarks)
                    
                    # Combinar todas las características
                    features.update(line_angles)
                    features.update(joint_angles)
                    features.update(norm_distances)
                    
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