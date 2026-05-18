import os
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from tensorflow.keras.models import load_model

input_folder = "Images/"
output_folder = "my_Faces_Dataset/"
classifier_path = "face_classifier.pkl"
model_path = "facenet_keras.h5"
face_size = (160, 160)
confidence_threshold = 60

os.makedirs(output_folder, exist_ok=True)

face_detector = MTCNN()
facenet_model = load_model(model_path)


def extract_faces():
    for person_name in os.listdir(input_folder):
        person_input_path = os.path.join(input_folder, person_name)
        person_output_path = os.path.join(output_folder, person_name)
        os.makedirs(person_output_path, exist_ok=True)

        if not os.path.isdir(person_input_path):
            continue

        for img_file in os.listdir(person_input_path):
            img_path = os.path.join(person_input_path, img_file)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected_faces = face_detector.detect_faces(image_rgb)

            for i, face in enumerate(detected_faces):
                x, y, w, h = face["box"]
                x, y = max(0, x), max(0, y)

                face_crop = image_rgb[y:y+h, x:x+w]

                if face_crop.size == 0:
                    continue

                face_crop = Image.fromarray(face_crop).resize(face_size)
                face_crop.save(f"{person_output_path}/{img_file[:-4]}_{i}.jpg")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(50)

    print("?????????????????????? Face extraction complete!")
    cv2.destroyAllWindows()


def get_embedding(face_pixels):
    face_pixels = face_pixels.astype("float32")
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.predict(face_pixels)[0]
    return embedding


def train_classifier():
    embeddings, labels = [], []

    for person_name in os.listdir(output_folder):
        person_folder = os.path.join(output_folder, person_name)

        if not os.path.isdir(person_folder):
            continue

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            face_image = Image.open(img_path).resize(face_size)
            face_image = np.array(face_image)
            embedding = get_embedding(face_image)
            embeddings.append(embedding)
            labels.append(person_name)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    normalizer = Normalizer(norm="l2")
    embeddings = normalizer.transform(embeddings)

    svm_classifier = SVC(kernel="linear", probability=True)
    svm_classifier.fit(embeddings, labels)

    with open(classifier_path, "wb") as f:
        pickle.dump(svm_classifier, f)

    print("??????????????????????Face recognition model trained and saved as 'face_classifier.pkl'.")
    return svm_classifier


def recognize_faces_in_image(image_path, svm_classifier):
    image = cv2.imread(image_path)

    if image is None:
        print("?????????????????????Error: Image not found!")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector.detect_faces(image_rgb)

    for face in detected_faces:
        x, y, w, h = face["box"]
        x, y = max(0, x), max(0, y)

        face_crop = image_rgb[y:y+h, x:x+w]
        face_crop = Image.fromarray(face_crop).resize(face_size)
        face_crop = np.array(face_crop)

        embedding = get_embedding(face_crop)
        embedding = np.expand_dims(embedding, axis=0)

        person_name = svm_classifier.predict(embedding)[0]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(image, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_faces_in_video(video_path, svm_classifier):
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("??????????????????????????? Error: Could not open video!")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_detector.detect_faces(frame_rgb)

        for face in detected_faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_crop = frame_rgb[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            face_crop = Image.fromarray(face_crop).resize(face_size)
            face_crop = np.array(face_crop)

            embedding = get_embedding(face_crop)
            embedding = np.expand_dims(embedding, axis=0)

            person_name = svm_classifier.predict(embedding)[0]
            confidence = svm_classifier.predict_proba(embedding).max() * 100
            label = f"{person_name} ({confidence:.1f}%)" if confidence > confidence_threshold else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

extract_faces()


svm_classifier = train_classifier()

# recognize_faces_in_video("Test/test.mp4", svm_classifier)
recognize_faces_in_image("Test/test (1).jpg", svm_classifier)