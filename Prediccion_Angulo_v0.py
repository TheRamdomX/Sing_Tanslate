import cv2
import mediapipe as mp
import numpy as np
from math import *

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de la cámara
cap = cv2.VideoCapture(0)
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

# Diccionario de letras y sus patrones de dedos
LETRAS = {
    'A': [1,1,0,0,0,0],
    'B': [0,0,1,1,1,1],
    'C': [1,0,1,0,0,0],
    'D': [0,0,0,0,0,1],
    'E': [0,0,0,0,0,0],
    'F': [1,1,1,1,1,0],
    'G': [1,0,0,1,0,0],
    'H': [1,0,0,1,1,0],
    'I': [0,0,1,0,0,0],
    'J': [1,0,1,1,0,0],
    'K': [1,1,0,0,1,1],
    'L': [1,1,0,0,0,1],
    'M': [1,0,0,1,0,1],
    'N': [0,1,0,0,1,1],
    'O': [1,0,1,0,0,0],
    'P': [0,1,1,1,1,1],
    'Q': [0,1,0,1,0,0],
    'R': [1,1,0,1,0,0],
    'S': [1,0,1,0,1,0],
    'T': [0,1,1,0,1,0],
    'U': [0,0,1,0,0,1],
    'V': [0,1,0,0,1,1],
    'W': [0,1,0,1,1,1],
    'X': [1,0,0,0,1,1],
    'Y': [1,1,1,0,0,0],
    'Z': [0,1,1,0,0,0]
}

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo entre tres puntos"""
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    
    if l1 and l3 != 0:
        num_den = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
        if -1 < num_den < 1:
            return round(degrees(abs(acos(num_den))))
    return 0

def detectar_letra(dedos):
    """Detecta la letra basada en el patrón de dedos"""
    for letra, patron in LETRAS.items():
        if dedos == patron:
            return letra
    return None

def main():
    # Crear ventana antes del bucle principal
    cv2.namedWindow('Predicción de Letras', cv2.WINDOW_NORMAL)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Obtener puntos de referencia
                    landmarks = hand_landmarks.landmark
                    
                    # Calcular ángulos para cada dedo
                    angulos = []
                    for dedo in ['PINKY', 'RING_FINGER', 'MIDDLE_FINGER', 'INDEX_FINGER', 'THUMB']:
                        if dedo == 'THUMB':
                            # Ángulos del pulgar (interno y externo)
                            p1 = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x * width,
                                         landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height])
                            p2 = np.array([landmarks[mp_hands.HandLandmark.THUMB_IP].x * width,
                                         landmarks[mp_hands.HandLandmark.THUMB_IP].y * height])
                            p3 = np.array([landmarks[mp_hands.HandLandmark.THUMB_MCP].x * width,
                                         landmarks[mp_hands.HandLandmark.THUMB_MCP].y * height])
                            p4 = np.array([landmarks[mp_hands.HandLandmark.WRIST].x * width,
                                         landmarks[mp_hands.HandLandmark.WRIST].y * height])
                            
                            angulos.extend([
                                calcular_angulo(p1, p2, p3),  # Ángulo interno
                                calcular_angulo(p1, p3, p4)   # Ángulo externo
                            ])
                        else:
                            # Ángulos para los otros dedos
                            p1 = np.array([landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_TIP')].x * width,
                                         landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_TIP')].y * height])
                            p2 = np.array([landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_PIP')].x * width,
                                         landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_PIP')].y * height])
                            p3 = np.array([landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_MCP')].x * width,
                                         landmarks[getattr(mp_hands.HandLandmark, f'{dedo}_MCP')].y * height])
                            
                            angulos.append(calcular_angulo(p1, p2, p3))

                    # Convertir ángulos a dedos extendidos (1) o doblados (0)
                    dedos = []
                    # Pulgar externo
                    dedos.append(1 if angulos[5] > 125 else 0)
                    # Pulgar interno
                    dedos.append(1 if angulos[4] > 150 else 0)
                    # Otros dedos
                    for i in range(4):
                        dedos.append(1 if angulos[i] > 90 else 0)

                    # Detectar letra
                    letra = detectar_letra(dedos)
                    
                    # Dibujar landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Mostrar letra detectada
                    if letra:
                        cv2.rectangle(frame, (0,0), (100,100), (255,255,255), -1)
                        cv2.putText(frame, letra, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 2, cv2.LINE_AA)

            # Mostrar frame en la ventana existente
            cv2.imshow('Predicción de Letras', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()