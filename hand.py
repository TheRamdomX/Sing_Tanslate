import cv2
import mediapipe as mp
import numpy as np

# Abre la cámara
cap = cv2.VideoCapture(0)

mediahands = mp.solutions.hands 
hands = mediahands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9, min_tracking_confidence=0.8)

mpdraw = mp.solutions.drawing_utils

# Colores para los dedos (BGR)
colors = [(255, 0, 0),  # Pulgar - Azul
          (0, 255, 0),  # Índice - Verde
          (0, 0, 255),  # Medio - Rojo
          (255, 255, 0),# Anular - Cian
          (255, 0, 255)]# Meñique - Magenta

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se puede recibir el frame (se ha terminado la transmisión o ha ocurrido un error)")
        break

    h, w, c = frame.shape

    # Convierte el frame a RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa la imagen para detectar manos
    resultado = hands.process(imgRGB)

    # Si se detectan manos
    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            # Dibujar landmarks de la mano con colores distintos en cada dedo
            for i, lm in enumerate(handLms.landmark):
                # Obtener la posición de los landmarks en píxeles
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Definir un color por defecto
                color = (0, 255, 255) # Amarillo claro por defecto

                # Asignar un color según el dedo
                if i in range(1, 5):        # Pulgar
                    color = colors[0]
                elif i in range(5, 9):      # Índice
                    color = colors[1]
                elif i in range(9, 13):     # Medio
                    color = colors[2]
                elif i in range(13, 17):    # Anular
                    color = colors[3]
                elif i in range(17, 21):    # Meñique
                    color = colors[4]

                # Dibujar cada landmark con el color correspondiente
                cv2.circle(frame, (cx, cy), 8, color, cv2.FILLED)
            
            # Dibujar las conexiones de la mano
            mpdraw.draw_landmarks(frame, handLms, mediahands.HAND_CONNECTIONS)

    # Muestra el frame en una ventana
    cv2.imshow('Iriun Webcam', frame)

    # Espera por la tecla 'q' para salir del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Libera el dispositivo de la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()