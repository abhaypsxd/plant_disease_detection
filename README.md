# Plant Disease Detection

A deep learning-based system for detecting diseases in plant leaves using Convolutional Neural Networks (CNN).

## Overview

This project implements a machine learning model that can identify 38 different classes of plant diseases across various crops including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. The model is trained on the "New Plant Diseases Dataset" from Kaggle and achieves over 91% validation accuracy.

## Features

- Detects 38 different plant diseases across 14+ plant species
- Uses a CNN architecture with TensorFlow/Keras
- Achieves 96% training accuracy and 91% validation accuracy
- Includes comprehensive evaluation metrics and visualizations
- Provides easy-to-use inference for new plant leaf images

## Dataset

The project uses the "New Plant Diseases Dataset" from Kaggle, which contains augmented images of healthy and diseased plant leaves. The dataset is organized into training and validation sets.

Dataset source: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Model Architecture

The model uses a Convolutional Neural Network (CNN) with the following architecture:

- Multiple convolutional layers with increasing filters (32 → 64 → 128 → 256)
- Max pooling layers for dimensionality reduction
- Flatten layer to convert 2D feature maps to 1D feature vectors
- Dense layer with 1024 units and ReLU activation
- Output layer with 38 units (one for each disease class) and softmax activation

## Requirements

```
tensorflow==2.10.0
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
streamlit
librosa==0.10.1
opencv-python
```

## Project Structure

- `setup.ipynb`: Downloads and extracts the dataset from Kaggle
- `train_plant_dataset.ipynb`: Preprocesses data, defines the model architecture, and trains the model
- `performance_eval.ipynb`: Evaluates model performance with metrics and visualizations
- `test_model.ipynb`: Demonstrates how to use the model for inference on new images
- `trained_model.keras`: The trained CNN model (pre-trained and ready to use)
- `training_hist.json`: Training history with loss and accuracy metrics
- `requirements.txt`: List of required Python packages

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run `setup.ipynb` to download and extract the dataset (requires Kaggle API credentials)
4. Alternatively, download the dataset manually from Kaggle and extract it to the appropriate location

## Usage

### Training the Model

If you want to train the model yourself:

1. Run `setup.ipynb` to download and prepare the dataset
2. Run `train_plant_dataset.ipynb` to train the model
3. The trained model will be saved as `trained_model.keras`

### Using the Pre-trained Model

To use the pre-trained model for inference:

1. Load the model:
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('trained_model.keras')
   ```

2. Preprocess your image:
   ```python
   import cv2
   import numpy as np
   
   # Load and preprocess the image
   image = tf.keras.preprocessing.image.load_img('your_image.jpg', target_size=(128, 128))
   input_array = tf.keras.preprocessing.image.img_to_array(image)
   input_array = np.array([input_array])  # Create batch dimension
   ```

3. Make a prediction:
   ```python
   prediction = model.predict(input_array)
   result_index = np.argmax(prediction)
   ```

4. Map the result to a disease class:
   ```python
   class_names = ['Apple___Apple_scab', 'Apple___Black_rot', ...]  # Full list in test_model.ipynb
   predicted_disease = class_names[result_index]
   ```

## Performance

The model achieves:
- Training accuracy: 96.2%
- Validation accuracy: 91.4%

Detailed performance metrics and visualizations can be found in `performance_eval.ipynb`.

## Future Improvements

- Implement a web or mobile application for easy access
- Add more plant species and disease classes
- Improve model accuracy through transfer learning with pre-trained models
- Implement explainable AI techniques to highlight affected areas of the leaf


## Acknowledgements

- Dataset provided by Kaggle user vipoooool
- TensorFlow and Keras for the deep learning framework
- Scikit-learn for evaluation metrics
