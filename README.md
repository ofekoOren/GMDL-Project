# Object Set Recognition (OSR) for MNIST
This repository contains Python code presenting a solution for Object Set Recognition (OSR) applied to the MNIST dataset.
The solution utilizes a combination of Convolutional Neural Networks (CNN) and an autoencoder.
The primary goal is to classify MNIST digits into their respective classes while labeling any unrelated data as "unknown."

### Introduction
Object Set Recognition (OSR) involves the task of categorizing objects within a dataset.
In this project, our main goal is to recognize and classify handwritten digits in the MNIST dataset.
The project utilizes the powerful combination of Convolutional Neural Networks (CNN) and an autoencoder for this purpose.

### Dependencies
- Python 3
- NumPy
- PyTorch
- Matplotlib
- Torchvision
- scikit-learn

### Models
##### Convolutional Neural Network (CNN)
The CNN model is designed for image classification tasks.
It consists of two convolutional layers followed by max-pooling layers, and two fully connected layers for final predictions.

##### Autoencoder
The Autoencoder model is used for feature extraction.
It compresses the input image into a lower-dimensional representation and then reconstructs it.

### Training
The training functions for both the CNN and Autoencoder models are provided in the notebook.
Training parameters, such as learning rate and number of epochs, can be adjusted based on specific requirements.

### Evaluation
After training the models, the notebook includes an evaluation section that tests the final models on the test dataset and generates confusion matrices.
The models' weights are saved for future use.

### Conclusion
This project showcases the capabilities of combining CNNs and autoencoders for Object Set Recognition on the MNIST dataset. The intelligent handling of unrelated data as "unknown" enhances the practicality and adaptability of the solution.
