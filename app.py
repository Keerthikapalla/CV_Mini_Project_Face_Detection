from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces (improved parameters)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Draw GREEN rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),   # Bright green
            5              # Thick border
        )

    # Save output image
    cv2.imwrite("static/output.jpg", img)

    return render_template(
        "index.html",
        faces=len(faces),
        image="output.jpg"
    )

if __name__ == '__main__':
    app.run(debug=True)
