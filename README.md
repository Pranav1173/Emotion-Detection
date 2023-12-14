***Project Title: Emotion Detection Model***
**Overview:**
This project focuses on building an emotion detection model using Convolutional Neural Networks (CNNs) to classify facial expressions into seven different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model is trained on a dataset of facial expression images and implemented using the TensorFlow and Keras frameworks.


**Prerequisites:**
Python
TensorFlow
Keras
OpenCV
Matplotlib
PIL
scikit-learn


**Project Structure:**
*train/:* Directory containing the training dataset images categorized into emotion classes.
*test/:* Directory containing the test dataset images for model validation.
*model_emo.h5:* Trained model weights file.
*model_1.json:* Trained model architecture file in JSON format.
*haarcascade_frontalface_default.xml:* Haarcascade file for face detection.


**Model Architecture:**
The emotion detection model is designed with a convolutional neural network (CNN) architecture, utilizing various layers for effective feature extraction and emotion classification. The architecture can be described as follows:

*Convolutional Layers:*

Convolutional Layer 1:
Filters: 32
Kernel Size: (3,3)
Activation: ReLU
Batch Normalization
Max Pooling: (2,2)
Dropout: 25%

Convolutional Layer 2:
Filters: 64
Kernel Size: (5,5)
Activation: ReLU
Batch Normalization
Max Pooling: (2,2)
Dropout: 25%

Convolutional Layer 3:
Filters: 128
Kernel Size: (3,3)
Activation: ReLU
Batch Normalization
Max Pooling: (2,2)
Dropout: 25%

*Flatten Layer:*
Flattens the output from the last convolutional layer to prepare for dense layers.

*Dense Layers:*
Dense Layer 1:
Nodes: 256
Activation: ReLU
Batch Normalization
Dropout: 25%

*Output Layer:*
Dense Layer with 7 nodes (equal to the number of emotions) and softmax activation for multi-class classification.


**Model Compilation:**
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy


**Training:**
The model is trained using an image data generator to augment the training dataset. Training details, such as the number of epochs, steps per epoch, and validation steps, are specified. Model checkpoints are saved during training to monitor and save the best weights based on validation accuracy.


**Graphical Interface:**
The graphical interface is implemented using Tkinter and allows users to upload an image for emotion detection. The interface displays the detected emotion and includes a "Detect Emotion" button for initiating the prediction.


*Note:*
Ensure that the required libraries and dependencies are installed before running the script.
The haarcascade_frontalface_default.xml file is crucial for face detection before emotion prediction.
