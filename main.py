import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

mp_face = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A+B)/(2.0*C)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    attention_score = 0

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            landmarks = [(int(p.x*w), int(p.y*h)) for p in face.landmark]

            x,y,w1,h1 = cv2.boundingRect(np.array(landmarks))
            face_img = frame[y:y+h1, x:x+w1]

            if face_img.size != 0:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48,48))
                gray = gray.reshape(1,48,48,1)/255

                emotion = emotion_labels[np.argmax(model.predict(gray))]
                cv2.putText(frame, emotion, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)

            attention_score = 85  # simplified scoring

    cv2.putText(frame, f"Attention: {attention_score}%", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Emotion & Attention Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
