import cv2
import numpy as np
import tensorflow as tf
from utils import Utils

img_size = (224, 224)
class_indices = {0: "messi", 1: "salah",
                 2: "seif"}

model = Utils.load_model('face_recognition_vgg16.h5')


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Crop face region
        label, confidence = Utils.predict_face(
            face, model, class_indices, img_size)

        color = (0, 255, 0) if confidence > 0.7 else (
            0, 0, 255)  # Green if confident, Red if unsure
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text = f"{label}: {confidence*100:.2f}%"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
