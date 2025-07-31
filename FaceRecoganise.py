import streamlit as st
import cv2
import os
from datetime import datetime

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Face Data Collection for Attendance System")

# Streamlit inputs
name = st.text_input("Enter Your Name")
User_id = st.text_input("Enter Your ID")
start = st.button("Start Your Capture")

if start:
    if not name or not User_id:
        st.warning("Please enter both Name and ID")
    else:

        save_path = f'dataset/{User_id}_{name}'
        os.makedirs(save_path, exist_ok=True)

        # Start webcam
        cap = cv2.VideoCapture(0)
        img_count = 0
        stframe = st.empty()
        
        while img_count < 100:
            ret , frame = cap.read()
            if not ret:
                st.error("Webcam not working")
            else:
                # convert to greyscale
                gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect face in the frame
                faces= face_cascade.detectMultiScale(gray, 1.3, 5)

                # loop through all detected faces

                for(x,y,w,h) in faces:
                    img_count+=1

                    # crop and resize the face regin

                    faces_img=gray[y:y+h, x:x+w]
                    faces_img=cv2.resize(faces_img,(200, 200))

                    # saving the face image
                    img_path = f'{save_path}/{img_count}.jpg'
                    cv2.imwrite(img_path, faces_img)

                    # Drawing a rectange
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                
                # show the frame in streamlit
                stframe.image(frame, channels="BGR")
        cap.release()
        st.success(f'captured 100 face images for {name} (Id: {User_id})')