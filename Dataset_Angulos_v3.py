import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Configuración de paths
input_folder = "/home/matias/Escritorio/Sing_Tanslate/Data"
output_csv = "DataSet_v3.csv"

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_point_3d(landmarks, id):
    #Obtiene coordenadas 3D normalizadas
    return np.array([landmarks[id].x, landmarks[id].y, landmarks[id].z])

def calculate_relative_position(point, reference_point):
    #Calcula la posición relativa de un punto con respecto a un punto de referencia
    return point - reference_point

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
                    
                    # Obtener etiqueta (primera letra del nombre de archivo)
                    label = img_name[0]
                    features = {"Label": label}
                    
                    # Obtener punto de referencia (muñeca - punto 0)
                    wrist_3d = get_point_3d(landmarks, 0)
                    
                    # Calcular posiciones relativas para cada punto
                    for i in range(1, 21):  # Empezamos desde 1 porque 0 es la muñeca
                        point_3d = get_point_3d(landmarks, i)
                        relative_pos = calculate_relative_position(point_3d, wrist_3d)
                        
                        # Guardar coordenadas x, y, z relativas
                        features[f"Point_{i}_x"] = round(relative_pos[0], 6)
                        features[f"Point_{i}_y"] = round(relative_pos[1], 6)
                        features[f"Point_{i}_z"] = round(relative_pos[2], 6)
                    
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
        print(f"- Total de puntos procesados: 20 (19 puntos relativos a la muñeca)")
        print(f"- Dimensiones por punto: 3 (x, y, z)")
    else:
        print("No se detectaron manos en ninguna imagen.")

if __name__ == "__main__":
    main() 