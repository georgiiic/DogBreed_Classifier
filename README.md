# Dog Breed Classification

This project uses machine learning to classify dog breeds. The following steps are involved in this project:

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Loading and Exploration](#data-loading-and-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building and Training](#model-building-and-training)
6. [Evaluation](#evaluation)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
Convolutional Neural Networks (CNNs) have shown great success in image classification tasks. In this notebook, we will use CNNs to classify images of dogs into their respective breeds. We will go through the following steps:

1. Import necessary libraries
2. Load and preprocess the data
3. Build and train the CNN model
4. Evaluate the model and visualize the results

Let's get started!

## Setup

The project is implemented in Python and relies on a variety of libraries for data analysis, machine learning model development, and visualization. 

Here is a brief introduction to the key libraries used:

1. **TensorFlow and Keras**: Used for creating and training the deep learning models.
2. **Pandas**: Used for data manipulation and analysis.
3. **NumPy**: Used for numerical computations and working with arrays.
4. **Plotly, Seaborn, and Matplotlib**: Used for data visualization and creating informative plots.
5. **WordCloud**: Used for creating word clouds.
6. **PIL (Python Imaging Library)**: Used for opening, manipulating, and saving many different image file formats.
7. **os, glob, shutil, tarfile**: Used for file operations.
8. **TensorBoard**: Used for visualizing learning metrics and model architecture.
9. **VisualKeras**: Used for creating a visual representation of a Keras model.

These libraries are listed in the `requirements.txt` file and can be installed using pip:

```bash
pip install -r requirements.txt
```
Please note that the original project was developed in a Google Colab environment. Some code may be specific to that environment and may require adjustments if running the code in a different environment.

## Data Loading and Exploration

The project utilizes the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) for training the dog breed classification model. The dataset is explored through various data analysis and visualization methods to gain insights into the distribution of classes, sample images, and the overall structure of the data.

Here is the distribution of our dataset by dog breed :

![newplot (20)](https://github.com/georgiiic/DogBreed_Classifier/assets/96066482/a9baa38c-0dfa-484c-9119-60a9ec140212)

For more specific details about the data exploration process and the insights derived from it, please refer to the 'Data Loading and Exploration' section in the project notebook.

## Data Preprocessing

In the data preprocessing stage, generators are instantiated to efficiently load the dataset. This approach ensures that the entire dataset does not need to be loaded into memory all at once, which can be particularly beneficial when working with large datasets. The generators handle the data in batches, providing the model with a continuous stream of training samples. This method not only enhances the efficiency of data handling but also facilitates data augmentation, preprocessing, and real-time data feeding to the model. For more specific details on how the generators are implemented and used, please refer to the 'Data Preprocessing' section in the project notebook.

Here is a WordCloud of our targets :
![image](https://github.com/georgiiic/DogBreed_Classifier/assets/96066482/ed56c5d6-7107-49bc-8de1-7d0c475b66d7)

## Model Building and Training

I started the project by trying to build a CNN from scratch :

![téléchargé](https://github.com/georgiiic/DogBreed_Classifier/assets/96066482/0d8d0cf5-6f16-44e1-b6c4-a1003d8ae8e8)


Then I tried adding data augmentation and regularizer :

![téléchargé (1)](https://github.com/georgiiic/DogBreed_Classifier/assets/96066482/b11153ef-f52d-4d6f-9888-b0ada06d93da)

Here you can see a graph of both their performance :
![newplot (21)](https://github.com/georgiiic/DogBreed_Classifier/assets/96066482/fd7336eb-bc15-4fd1-abcc-1a0f59390839)


Buth in order two obtain better results, I could either find more pictures or use Transfer Learning and employed multiple deep learning architectures for the task of dog breed classification:

1. **InceptionV3**: An architecture that leverages factorized convolutions and aggressive dimension reductions to create a more efficient deep learning model.
2. **InceptionResNetV2**: A model that combines the advantages of Inception networks and residual connections.
3. **EfficientNetB7**: A model that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient.

During the training process, the models are trained for different numbers of epochs, ranging from 15 to 50. The models are trained using the `fit` function in Keras, with early stopping implemented to prevent overfitting and unnecessary computation. For more specific details on each model's architecture and training process, please refer to the 'Model Building and Training' section in the project notebook.


## Evaluation
After the model is trained, its performance is evaluated in this section. It includes steps like testing the model on unseen data, checking accuracy, precision, recall, and other metrics.

## Conclusion
The conclusion section summarizes the findings and results of the project.

## References
The references section cites the resources referred to in the project.

This project includes a total of 63 code cells.
