import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

def main():
    st.title("Streamlit + OpenCV + MediaPipe Example")

    # Configuración de MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: No se pudo abrir la cámara.")
        return

    # Inicializar Streamlit
    stframe = st.empty()

    # Bucle principal
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Error: No se pudo recibir fotograma de la cámara.")
            break

        # Procesar el fotograma con MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Dibujar landmarks en el fotograma
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Mostrar el fotograma en Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
