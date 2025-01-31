import os
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from os.path import isfile


class Utils:
    @staticmethod
    def replace_top(bottom_model, num_classes, train_datagen):
        top_model = Flatten()(bottom_model.output)
        # top_model = GlobalAveragePooling2D()(top_model)
        # top_model = Dense(1024,activation='relu')(top_model)
        # top_model = Dense(1024,activation='relu')(top_model)
        top_model = Dense(512, activation='relu')(top_model)
        top_model = Dense(256, activation='relu')(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(len(train_datagen.class_indices),
                          activation='softmax')(top_model)
        return top_model

    @staticmethod
    def convert_images_to_jpeg(directory):
        for filename in os.listdir(directory):
            if not filename.endswith('.jpeg'):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                new_filename = os.path.splitext(filename)[0] + '.jpeg'
                new_img_path = os.path.join(directory, new_filename)
                img.convert('RGB').save(new_img_path, 'JPEG')
                os.remove(img_path)

    @staticmethod
    def rename_images_in_directory(directory):
        files = [f for f in os.listdir(directory) if f.endswith('.jpeg')]
        files.sort()
        for i, filename in enumerate(files):
            new_filename = f"{i+1}.jpeg"
            new_filepath = os.path.join(directory, new_filename)
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{i+1}_{counter}.jpeg"
                new_filepath = os.path.join(directory, new_filename)
                counter += 1
            os.rename(os.path.join(directory, filename), new_filepath)

    @staticmethod
    def compile_fit_save(model, train_generator, val_generator, NUM_EPOCHS, model_name):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #     'model.h5', save_best_only=True)
        history = model.fit(train_generator, epochs=NUM_EPOCHS,
                            validation_data=val_generator)
        model.save(f'{model_name}.h5')
        return history

    @staticmethod
    def preprocess_image(face, img_size=(224, 224)):
        """Preprocess the face image for the VGG16 model."""
        face = cv2.resize(face, img_size)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face

    @staticmethod
    def predict_face(face, model, class_indices, img_size=(224, 224)):
        """Predict the face class using the trained model."""
        processed_face = Utils.preprocess_image(face, img_size)
        prediction = model.predict(processed_face)
        class_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return class_indices[class_label], confidence

    @staticmethod
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def get_embedding(image_path, embedder):
        """Extracts embedding using FaceNet."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))  # Required size for FaceNet
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        embedding = embedder.embeddings(img)
        return embedding[0]  # Extract first (only) embedding

    @staticmethod
    def predict_person(image_path, class_indices, model, embedder, le):
        frame = cv2.imread(image_path)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            print("No face detected")
            return

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_embedding = Utils.get_embedding(image_path, embedder)

            name = model.predict([face_embedding])[0]
            if isinstance(model, SVC):
                name = le.inverse_transform([name])[0]

            print(name)

            # Display the result
            cv2.putText(frame, f"Prediction: {
                        name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    @staticmethod
    def prepare_embeddings(dataset_path, embedder):
        persons = {}
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if isfile(os.path.join(folder_path, file)):
                        name = folder
                        embedding = Utils.get_embedding(
                            os.path.join(folder_path, file), embedder)
                        persons[name] = embedding
                        break
        return persons
