import cv2
import numpy as np
import os

dataset_dir = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face_EigenFaceRecognizer.create()
# recognizer = cv2.face_FisherFaceRecognizer.create()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(dataset_dir):
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(gray_img)
        
        for (x, y, w, h) in faces:
            face_samples.append(gray_img[y:y+h, x:x+w])
            ids.append(face_id)
    
    return face_samples, ids

print("Training faces. It will take a few seconds. Wait ...")
faces, ids = get_images_and_labels(dataset_dir)
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write('trainer.yml')
print("Model trained and saved as 'trainer.yml'.")
