# Face Recognition with VGG16, FaceNet, and Machine Learning Classifiers

This repository contains a comprehensive implementation of face recognition using **VGG16** for fine-tuning and transfer learning, combined with **FaceNet** for one-shot learning. The project also integrates traditional machine learning classifiers like **SVM (Support Vector Machine)** and **K-NN (K-Nearest Neighbors)** for face recognition tasks. The repository is designed to provide a robust pipeline for face recognition, from data preprocessing to model training and prediction.

---

## Key Features

1. **Fine-Tuning and Transfer Learning with VGG16**:
   - Fine-tune the VGG16 model on a custom dataset for face recognition.
   - Perform transfer learning by freezing specific layers of VGG16 and training the top layers for classification.

2. **One-Shot Learning with FaceNet**:
   - Use FaceNet to generate embeddings for face images.
   - Train **K-NN** and **SVM** classifiers on the embeddings for one-shot face recognition.

3. **Machine Learning Classifiers**:
   - **K-NN**: A k-nearest neighbors classifier with cosine similarity for face recognition.
   - **SVM**: A support vector machine classifier with a linear kernel for face recognition.

4. **Real-Time Face Recognition**:
   - A `predict.py` script for real-time face recognition using a webcam or video feed.
   - Detects faces, preprocesses them, and predicts the person's identity using the trained VGG16 model.

5. **Utilities**:
   - A `utils.py` module with helper functions for:
     - Preprocessing images.
     - Generating embeddings using FaceNet.
     - Training and saving models.
     - Predicting faces using trained models.

---

## Directory Structure

```
└── face-recognition-vgg16-facenet/
    ├── README.md                   # Project overview and instructions
    ├── LICENSE                     # MIT License for the project
    ├── main.ipynb                  # Jupyter notebook for training and evaluation
    ├── predict.py                  # Script for real-time face recognition
    ├── utils.py                    # Utility functions for preprocessing, training, and prediction
    └── data/                       # Dataset directory
        └── README.md               # Information about the dataset
```

---

## How It Works

1. **Dataset Preparation**:
   - The dataset is organized into subdirectories, where each subdirectory represents a person and contains their face images.
   - Data augmentation techniques (e.g., rotation, scaling, flipping) are applied to improve model generalization.

2. **Fine-Tuning VGG16**:
   - The VGG16 model is loaded with pre-trained weights from ImageNet.
   - The last few layers are fine-tuned on the custom face dataset.

3. **Transfer Learning with VGG16**:
   - The VGG16 model is used as a feature extractor by freezing all layers.
   - A custom classification head is added and trained on the face dataset.

4. **One-Shot Learning with FaceNet**:
   - FaceNet generates 128-dimensional embeddings for face images.
   - K-NN and SVM classifiers are trained on these embeddings for one-shot recognition.

5. **Real-Time Prediction**:
   - The `predict.py` script uses OpenCV to detect faces in real-time.
   - Detected faces are preprocessed and passed to the trained model for prediction.

---

## Usage

1. **Training**:
   - Run the `main.ipynb` notebook to fine-tune VGG16, perform transfer learning, and train FaceNet with K-NN/SVM.

2. **Real-Time Prediction**:
   - Use the `predict.py` script for real-time face recognition:
     ```bash
     python predict.py
     ```

3. **Dataset**:
   - Place your face dataset in the `data/` directory, with each subdirectory named after the person and containing their images.

---

## Dependencies

- Python 3.x
- TensorFlow/Keras
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- `keras-facenet` (for FaceNet embeddings)

Install the required dependencies using:

```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib keras-facenet
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Future Work

- Implement **Siamese Networks** for one-shot learning.
- Add support for more advanced face detection models (e.g., MTCNN, YOLO).
- Extend the project to handle video-based face recognition.

---
