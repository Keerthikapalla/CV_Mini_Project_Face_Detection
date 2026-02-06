
import streamlit as st
import cv2
import numpy as np

st.title("Face Detection Mini Project")

uploaded = st.file_uploader("Upload Face Image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(image, caption="Detected Faces", channels="BGR")
    st.write(f"Total Faces Detected: {len(faces)}")
