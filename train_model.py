import cv2
import numpy as np
import os
import pickle

# use OpenCV’s LBPHFaceRecognizer
# LBPH (Local Binary Pattern Histogram) model.


# path to the dataset
dataset_path='dataset'

# Data Holders

faces=[]
labels=[]

# label encode

label_dict = {}
current_label = 0

# Traverse the datset folder

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    label_dict[folder_name] = current_label

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name) 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label+=1

# Convert to numpy arrays
faces_np=np.array(faces)
labels_np=np.array(labels)

# create and train the LBPH recoganiser

model=cv2.face.LBPHFaceRecognizer_create()
model.train(faces_np, labels_np)

# save the model

model.save('trained_model.yml')
np.save('label_dict.npy', label_dict)
with open('labels.pkl','wb') as f:
    pickle.dump(label_dict, f)

print('✅ Model training complete and saved!')