import numpy as np
import cv2
import time
from tensorflow import keras
from keras.models import model_from_json, Sequential
from keras.utils import img_to_array

# Load model
emotion_name = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# Load json and create model
with open('./models/emotion_model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

classifier = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})
classifier.load_weights("./models/emotion_model1.h5")

# Load face cascade
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

def detect_emotion(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    emotion_output = "No face detected"

    for (x, y, w, h) in faces:
        cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            emotion_output = emotion_name[maxindex]
            label_position = (x, y)
            cv2.putText(image, emotion_output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, emotion_output

def main():
    print("Real Time Face Emotion Detection Application")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, emotion_output = detect_emotion(frame)

        cv2.imshow('Webcam Preview', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'c' to capture the image and recognize emotion
        if key == ord('c'):
            print("Capturing image...")
            result_image_path = "result_image.png"
            cv2.imwrite(result_image_path, frame)

            # Save the emotion recognition to a text file
            with open("emotion_recognition.txt", "w") as f:
                f.write(f"Detected Emotion: {emotion_output}\n")

            print(f"Image saved as {result_image_path} and emotion saved to emotion_recognition.txt")

        # Press 'q' to quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
