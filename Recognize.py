import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_count = len(faces)  # Get the number of detected faces
    # cv2.putText(frame, f"Faces detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    if face_count > 1:
        print(f"Multiple faces detected. Count: {face_count}")

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face_img)
        match_percentage = 100 - conf  # Convert confidence to match percentage

        if conf < 80:
            # cv2.putText(frame, f"User {id_}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255,0), 2)
            cv2.putText(frame, f"User {id_} ({match_percentage:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2)

        else:
            cv2.putText(frame, "User Not Found", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
