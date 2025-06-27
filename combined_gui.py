import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import messagebox

# Load your models
drowsiness_model = load_model("drowsiness_model.h5")
age_model = load_model("final_age_model_4bins.h5")

# Age bin mapping
age_bins = ['0-20', '21-40', '41-60', '61+']

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)
sleeping_count = 0
ages_detected = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    sleeping_count = 0
    ages_detected.clear()

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64,64))
        face_array = img_to_array(face_resized) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Predict age
        age_pred = age_model.predict(face_array)
        age_bin = age_bins[np.argmax(age_pred)]
        ages_detected.append(age_bin)

        # Detect eyes within the face
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        closed_eyes = 0

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
                continue
            
            eye_resized = cv2.resize(eye_roi, (64,64))
            eye_array = img_to_array(eye_resized) / 255.0
            eye_array = np.expand_dims(eye_array, axis=0)
            
            eye_pred = drowsiness_model.predict(eye_array)
            pred_label = np.argmax(eye_pred)

            if pred_label == 0:  # Closed
                closed_eyes += 1
        
        if closed_eyes >= 1:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, f"Sleeping", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            sleeping_count += 1
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Awake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Show predicted age bin
        cv2.putText(frame, f"Age: {age_bin}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    cv2.imshow("Drowsiness & Age Detection", frame)

    if sleeping_count > 0:
        # Show popup once for sleeping detection
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("Alert", f"Sleeping people: {sleeping_count} | Ages: {ages_detected}")
        root.destroy()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
