# MNIST Handwritten Digit Classification

## Overview
This project trains a neural network to classify handwritten digits (0-9) using the **MNIST dataset**. The dataset consists of grayscale images of size **28x28 pixels**. The model is built using **TensorFlow** and **Keras**.

## Dataset
- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Training Data**: 60,000 images
- **Testing Data**: 10,000 images
- **Classes**: Digits from 0 to 9

## Dependencies
To run this project, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## Setup & Execution
1. Clone or download the repository.
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Mnist_Classification.ipynb
   ```
3. Follow the notebook cells to preprocess data, train the model, and evaluate performance.

## Model Architecture
- **Flatten Layer**: Converts 28x28 images into a 1D array.
- **Dense Layers**:
  - Fully connected layers with activation functions.
  - Uses ReLU activation for hidden layers.
  - Uses Softmax activation for the output layer.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam.

## Results
- The model achieves high accuracy on the MNIST dataset.
- Performance is evaluated using accuracy and loss metrics.

## Future Improvements
- Experiment with **CNNs** for better accuracy.
- Implement **data augmentation** to improve generalization.
- Deploy the model as a **web application** using Flask or FastAPI.
