import cv2
import os

dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
user_id = input("Enter user ID (enter 'q' to quit): ")

while user_id.lower() != 'q':
    num_samples = 0
    max_samples = 100    # Number of images to capture

    cap = cv2.VideoCapture(0)

    while num_samples < max_samples:
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x, y, w, h) in faces:
            num_samples += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_dir}/user.{user_id}.{num_samples}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Capture', frame)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {num_samples} samples for user ID {user_id}.")

    user_id = input("Enter user ID (enter 'q' to quit): ")
