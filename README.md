# Real-Time Emotion Detection from Facial Expressions

## Overview
This project implements an end-to-end real-time facial emotion recognition system using Python and Deep Learning. It captures live webcam video, uses OpenCV's Haar Cascade for face detection, and a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify facial expressions into one of 7 emotions:
1. Angry (0)
2. Disgust (1)
3. Fear (2)
4. Happy (3)
5. Sad (4)
6. Surprise (5)
7. Neutral (6)

## Project Structure
```
emotion_detection/
│
├── data/
│   ├── train/                   # Contains 80% of data (approx 28,709 images)
│   ├── val/                     # Contains 10% of data (created by dataset_loader.py)
│   └── test/                    # Contains 10% of data
│
├── models/                      # Folder to store the trained model (.h5)
│
├── src/
│   ├── dataset_loader.py        # Script to create the validation split
│   ├── preprocessing.py         # Image augmentation and generators
│   ├── model.py                 # CNN architecture definition
│   ├── train.py                 # Script to train the model
│   ├── evaluate.py              # Script to evaluate model on test data
│   └── real_time_detection.py   # Webcam deployment script
│
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset**
   Download the FER-2013 dataset (e.g., from Kaggle). Extract it such that your `data` folder contains a `train` folder and a `test` folder, with subdirectories for each emotion.

3. **Prepare Validation Split**
   FER-2013 typically comes with just `train` and `test` splits. Run the loader to create a validation split (10% validation, 10% test from the original 20% test split):
   ```bash
   python src/dataset_loader.py
   ```

4. **Train the Model**
   ```bash
   python src/train.py
   ```

5. **Evaluate the Model**
   ```bash
   python src/evaluate.py
   ```

6. **Run Real-Time Detection**
   ```bash
   python src/real_time_detection.py
   ```
