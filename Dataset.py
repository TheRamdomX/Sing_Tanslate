import cv2
import mediapipe as mp
import numpy as np
import os

# Directorios de entrada y salida
input_dir = '/home/matias/Escritorio/Señas/Img/Train/Z/'
output_dir = '/home/matias/Escritorio/Señas/data/Train/Z/'

# Configuración de MediaPipe
mediahands = mp.solutions.hands
hands = mediahands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01, min_tracking_confidence=0.01)
mpdraw = mp.solutions.drawing_utils

# Colores para los dedos (BGR)
colors = [(255, 0, 0),  # Pulgar - Azul
          (0, 255, 0),  # Índice - Verde
          (0, 0, 255),  # Medio - Rojo
          (255, 255, 0), # Anular - Cian
          (255, 0, 255)] # Meñique - Magenta

# Procesar imágenes
for i in range(2997, 5997):
    # Generar rutas de entrada y salida
    input_image_path = os.path.join(input_dir, f'Z{i - 2992 }.jpg')
    output_image_path = os.path.join(output_dir, f'{i}.jpg')

    # Leer la imagen de entrada
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"No se pudo leer la imagen de entrada: {input_image_path}")
        continue

    # Crear una imagen negra con las mismas dimensiones que la imagen original
    black_image = np.zeros_like(image)

    # Convertir la imagen a RGB
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    resultado = hands.process(imgRGB)

    # Si se detectan manos
    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            # Dibujar landmarks de la mano con colores distintos en cada dedo
            for j, lm in enumerate(handLms.landmark):
                # Obtener la posición de los landmarks en píxeles
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Definir un color por defecto
                color = (0, 255, 255)  # Amarillo claro por defecto

                # Asignar un color según el dedo
                if j in range(1, 5):        # Pulgar
                    color = colors[0]
                elif j in range(5, 9):      # Índice
                    color = colors[1]
                elif j in range(9, 13):     # Medio
                    color = colors[2]
                elif j in range(13, 17):    # Anular
                    color = colors[3]
                elif j in range(17, 21):    # Meñique
                    color = colors[4]

                # Dibujar cada landmark con el color correspondiente
                cv2.circle(black_image, (cx, cy), 8, color, cv2.FILLED)
            
            # Dibujar las conexiones de la mano en la imagen negra
            mpdraw.draw_landmarks(black_image, handLms, mediahands.HAND_CONNECTIONS)

    # Guardar la imagen de salida con el esqueleto de la mano dibujado
    cv2.imwrite(output_image_path, black_image)

    print(f"Procesada la imagen {input_image_path} y guardada en {output_image_path}")

print("Procesamiento completado.")
