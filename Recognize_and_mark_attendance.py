import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import pickle

# ---------- Function to mark attendance ----------
def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # Create attendance folder if not exists
    if not os.path.exists("attendance"):
        os.makedirs("attendance")

    filename = f"attendance/attendance_{date_string}.csv"

    # Load existing or create new DataFrame
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # If not already marked today
    if not ((df["Name"] == name) & (df["Date"] == date_string)).any():
        new_data = pd.DataFrame([[name, date_string, time_string]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(filename, index=False)
        return f"{name}'s attendance marked at {time_string}."
    else:
        return f"{name} already marked today."


# ---------- UI Setup ----------
st.title('📸 Face Recognition Attendance System')

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('trained_model.yml')

# Load label mapping
with open('labels.pkl', 'rb') as f:
    label_dict = pickle.load(f)
id_to_name = {v: k for k, v in label_dict.items()}

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
setframe = st.empty()
stop_requested = st.button("Stop Recognition")

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop for live recognition
while cap.isOpened() and not stop_requested:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = model.predict(face_img)

        if confidence < 80:
            name = id_to_name[label]
            msg = mark_attendance(name)  # ✅ Call attendance function
            color = (0, 255, 0)
        else:
            name = "Unknown"
            msg = "Face not recognized"
            color = (0, 0, 255)

        # Draw box and text
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    setframe.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()

st.success("🛑 Recognition stopped. Attendance marked in CSV inside 'attendance/' folder.")
