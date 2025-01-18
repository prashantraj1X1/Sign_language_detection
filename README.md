# Gesture Recognition System
<img src="https://github.com/TLxGHOST/Sign_language_detection/blob/main/giphy.gif">
## Overview
This project involves creating a gesture recognition system that captures hand gestures using a webcam, stores labeled images for training, and utilizes a deep learning model to predict gestures in real time. The labels can also be passed to a Large Language Model (LLM) for contextual insights and suggestions.

## Features
- **Real-Time Gesture Capture**: Uses a webcam to capture images of hand gestures.
- **Dynamic Labeling**: Allows users to assign labels to gestures during capture.
- **Data Storage**: Saves images in a structured folder format based on labels.
- **Model Training**: Trains a Convolutional Neural Network (CNN) on the captured data.
- **Real-Time Prediction**: Predicts gestures in real-time using a webcam feed.
- **LLM Integration**: Need to be worked on

## Requirements
- Python 3.8+
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `tensorflow`
  - `sklearn`
  - `openai` (for LLM integration)

Install the required libraries using:
```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn openai
```

## Project Structure
```
.
├── CollectedData/                # Folder containing labeled gesture images
│   ├── Label1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   ├── Label2/
│       ├── 1.jpg
│       ├── 2.jpg
├── saved_labels.txt             # File storing gesture labels
├── Models/                      # Folder containing labeled gesture images
│   ├── model1/
|   ├── model2/
|   ├── model3/                  # Trained gesture recognition model
├── label_class.npy              # Label encoder for mapping labels
└── Final(Revised).ipynb         # Main script for capturing, training, and predicting
```

## Usage

### 1. Capturing Gestures
Run the script to start capturing gestures:
```bash
python main.py
```
- Press `'s'` to save a labeled gesture.
- Press `'q'` to quit.

Captured images will be stored in `CollectedData/` under subfolders named after labels.

### 2. Training the Model
Use the captured data to train a CNN model:
```python
# In your script, call the train_model function
train_model('CollectedData/', 'model.h5', 'label_encoder_classes.npy')
```
This saves the trained model as `model.h5`(change the name as per the model to be used eg:- model1.h5, model2.h5, model3.h5) and label encoder as `label_class.npy`.

### 3. Real-Time Gesture Prediction
Run the script for real-time prediction:
```python
real_time_prediction('model.h5', 'label_class.npy')
```
- The predicted label will be displayed on the webcam feed.
- Press `'s'` to save the predicted label to `saved_labels.txt`.

## Key Functions
- **Gesture Capture**:
  - Captures and labels hand gesture images.
  - Displays landmarks using Mediapipe.
  - Saves cropped hand images for training.

- **Model Training**:
  - Builds and trains a CNN model.
  - Saves the trained model and label encoder.

- **Real-Time Prediction**:
  - Uses the trained model to predict gestures in real time.
  - Displays predictions on the webcam feed.

- **LLM Integration**:
  - Still under considertation

## Future Enhancements
- Extend the dataset with more gestures and labels.
- Implement multi-hand gesture recognition.
- Use of video data for training and prediction.
- Implement a more advanced model for gesture recognition.
- Enhance the UI for gesture capture and prediction.
- Integrate additional AI models for more advanced recognition.


## Acknowledgments
- Mediapipe for hand tracking and landmarks.
- TensorFlow/Keras for deep learning.

