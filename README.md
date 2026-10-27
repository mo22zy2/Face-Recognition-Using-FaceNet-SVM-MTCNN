# Face Recognition Using FaceNet + SVM + MTCNN

A face recognition system that detects faces with **MTCNN**, extracts embeddings with **FaceNet**, and classifies them with an **SVM** classifier. Includes a Tkinter GUI and a no-webcam CLI pipeline.

## Features

- Face detection using MTCNN (robust to varied poses and lighting)
- Face embedding extraction via pre-trained FaceNet (Keras model)
- SVM classifier with L2-normalized embeddings for recognition
- GUI application (`Final_Project.py`) with buttons for:
  - Upload image / video for recognition
  - Open webcam for real-time recognition
  - Extract faces from source images
  - Train/retrain the classifier
- No-webcam pipeline (`No_webcam.py`) for headless environments

## Project Structure

```
FaceRecog Project/
├── Final_Project.py         # GUI application (Tkinter)
├── No_webcam.py             # CLI pipeline (no GUI, no webcam)
├── face_classifier.pkl      # Trained SVM + normalizer (generated)
├── facenet_keras.h5         # Pre-trained FaceNet model
├── Images/                  # Source images organized by person
│   ├── Your images
│ 
├── my_Faces_Dataset/        # Extracted face crops (generated)
│   └── {person}/
└── Test/                    # Test images / videos for inference
```

## Setup

1. **Clone and enter the project**
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place the FaceNet model**  
   Download `facenet_keras.h5` and place it in the `FaceRecog Project/` directory.  
   (If you already have it, no action needed.)

4. **Add source images** (optional)  
   Place face images in `FaceRecog Project/Images/{person_name}/`.  
   Each subdirectory name will become a class label.

## Usage

### GUI Application

```bash
python "FaceRecog Project/Final_Project.py"
```

1. Click **Extract Faces** to detect/crop faces from `Images/` into `my_Faces_Dataset/`
2. Click **Train Model** to train the SVM classifier
3. Use **Upload Image**, **Upload Video**, or **Open Webcam** to recognize faces

### No-Webcam Pipeline

```bash
python "FaceRecog Project/No_webcam.py"
```

This runs the full pipeline automatically: extract → train → recognize a test image.

## How It Works

1. **Face Detection** — MTCNN locates faces in each image and returns bounding boxes
2. **Face Extraction** — Detected faces are cropped, resized to 160×160, and saved
3. **Embedding Generation** — Each face is passed through FaceNet to produce a 128-D embedding
4. **Normalization** — Embeddings are L2-normalized
5. **Classification** — A linear SVM is trained on the normalized embeddings
6. **Recognition** — New faces go through steps 2-4, then the SVM predicts the label

## Requirements

- Python 3.8+
- OpenCV, NumPy, Pillow, MTCNN, scikit-learn, TensorFlow/Keras
- See `requirements.txt` for exact versions
