## Multiclass Classification with Pins Face Recognition Dataset

# Objective
This project aims to classify images of faces into one of 105 categories using a neural network built with Keras and TensorFlow. The network is trained to recognize different celebrities from the Pins Face Recognition dataset, achieving high accuracy on unseen validation data.

# Dataset Description
The Pins Face Recognition dataset contains images of various celebrities organized into 105 folders, with each folder representing a different individual. The dataset allows for the training and evaluation of the model on face recognition tasks. Each image will be preprocessed to detect and crop faces before being fed into the classification model.

# Instructions for Running the Code
1. Clone the repository from GitHub.
2. Download and unzip the dataset folder into the specified directory.
3. Download the dataset from that link "https://www.kaggle.com/datasets/hereisburak/pins-face-recognition"
4. Open colab and load the project file.
5. Install the dependencies.
6. Run all cells in the colab from the top of the page.
7. In the second cell choose the directory where the zip file of the dataset exist 

# Dependencies and Installation
- Python 3.8+
- TensorFlow 2.4+
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

To install the dependencies, use the following command in the terminal:
'pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn'

Note: No need to install the dependencies if you run on colab(build-in)

# Expected Results and Model Performance
The model is expected to achieve an accuracy of at least 85% on the validation set. During training, you can monitor the accuracy and loss for both the training and validation sets. The final model evaluation includes:

-Validation accuracy
-Classification report
-Confusion matrix
-Training and validation accuracy/loss plots



