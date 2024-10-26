# Alzheimer-Disease-Diagnosis

### Alzheimer MRI Classification Using CNN
## Project Overview
Unlock the potential of deep learning in healthcare! This project leverages a Convolutional Neural Network (CNN) to classify MRI images of the brain for early detection of Alzheimer's disease. By utilizing advanced techniques, this model aims to provide accurate predictions, helping to enhance patient outcomes and support medical research.

## Introduction
As Alzheimer's disease continues to affect millions worldwide, early diagnosis is crucial for effective treatment and care. This project explores the classification of MRI scans using a CNN architecture, demonstrating the power of machine learning in addressing real-world health challenges.

## Dataset
The dataset used in this project is the Alzheimer MRI dataset, containing MRI scans classified into four categories:
-Mild Demented
-Moderate Demented
-Non-Demented
-Very Mild Demented
The dataset is sourced from Kaggle and can be easily accessed and downloaded using the Kaggle API.

## Installation
To get started, ensure you have Python installed along with the required libraries. You can install the necessary packages using:
---> pip install tensorflow pandas seaborn matplotlib imbalanced-learn split-folders

## Model Training
The CNN architecture includes multiple convolutional layers and dropout for regularization, optimizing the model for improved accuracy. The model is trained using the Adam optimizer and evaluates accuracy metrics.
## model code
model = keras.models.Sequential()
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer="he_normal"))

# Additional layers omitted for brevity
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
