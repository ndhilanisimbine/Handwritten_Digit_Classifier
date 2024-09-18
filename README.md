# Handwritten_Digit_Classifier
A Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset


# MNIST Digit Classifier
# Overview:

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset. The model is built using TensorFlow and Keras and is trained to recognize digits based on grayscale images.

# Features:

Uses a CNN model for digit classification.
Trained and tested on the MNIST dataset (60,000 training images, 10,000 test images).
Achieves high accuracy in recognizing handwritten digits.
Uses TensorFlow's GradientTape for manual gradient calculation and backpropagation.
Technologies Used
TensorFlow: A powerful machine learning framework.
Keras: High-level neural networks API for easy model building.
Python: Main programming language used.
NumPy: For numerical operations.
Matplotlib: For data visualization.
# Dataset:

The MNIST dataset contains 70,000 images of handwritten digits. Each image is 28x28 pixels and is labeled as one of the digits from 0 to 9.

# Model Architecture
Two convolutional layers with ReLU activation.
MaxPooling layers to reduce spatial dimensions.
A fully connected (Dense) layer with 128 neurons.
Output layer with 10 neurons and Softmax activation for classification.
