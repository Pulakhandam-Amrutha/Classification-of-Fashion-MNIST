# Fashion MNIST Image Classification using CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 70,000 grayscale images, each 28x28 pixels, depicting 10 categories of clothing and accessories. This project uses deep learning techniques to accurately classify these images into their respective categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)

## Project Overview
The goal of this project is to build a robust image classification model using a CNN to predict the category of a fashion item based on the image provided. The model is trained and tested on the Fashion MNIST dataset, which contains images of items such as t-shirts, trousers, dresses, and shoes.

### Categories in the Dataset
The dataset contains 10 classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Dataset
The Fashion MNIST dataset contains images of various fashion items.

- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image size**: 28x28 pixels
- **Channels**: 1 (grayscale)

You can download the Fashion MNIST dataset directly via TensorFlow or Keras using:

```python
from tensorflow.keras.datasets import fashion_mnist
```

### Model Architecture
The model is a Convolutional Neural Network (CNN) designed to classify the fashion images into one of 10 classes. The architecture includes:
- **Input layer**: 28x28 grayscale images (1 channel)
- **Conv2D layer**: Extracts features from the image (several convolutional layers with ReLU activation)
- **MaxPooling layer**: Reduces the spatial dimensions of the feature maps
- **Fully connected (Dense) layers**: Maps the extracted features to class probabilities
- **Softmax layer**: Outputs the probability distribution over the 10 classes

## Installation

Follow these steps to set up the project environment:

1. **Check Python Version**  
   Ensure you have Python 3.x installed by running the following command:

   ```bash
   python --version
   ```
2. **Install Dependencies**
After creating and activating the virtual environment (or in your existing environment), install the required packages using requirements.txt:
```bash
pip install -r requirements.txt
```
### Results
After training the model, you should expect to achieve an accuracy of around 90% on the test dataset. The following metrics can be used to evaluate the model performance:

**Accuracy**: ~90%
**Loss**: < 0.22
