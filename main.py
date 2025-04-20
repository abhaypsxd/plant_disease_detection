import streamlit as st
import tensorflow as tf
import numpy as np

#prediction function
def model_prediction(test_image,):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home","About","Disease Detection"])

if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""# Plant Disease Detection

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

""")

elif(app_mode=="About"):
    st.header("About")
    st.markdown(""" #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train: 70295 Images
    2. Valid: 17572 Images
    3. Test: 33 Images
""")

elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an image")
    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    if(st.button("Predict")):
        with st.spinner("The model is predicting"):
            result_index = model_prediction(test_image)
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            st.success("Model predicts it's a {}".format(class_name[result_index]))
            