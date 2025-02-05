# Heart-Disease-Detection-using-Combined-Neural-Networks
This repository contains a hybrid deep learning model that combines GraphSAGE (Graph Sample and Aggregation) with a ResNet50 image feature extractor for multi-label classification of medical images. The model is designed to classify chest X-ray images from the MIMIC-CXR dataset, leveraging both image data and tabular features.

# Table of Contents
Overview

Dataset

Model Architecture

Installation

Usage

Results

Contributing

License

# Overview
This project implements a hybrid deep learning model that combines:

ResNet50: A pre-trained convolutional neural network (CNN) for extracting features from chest X-ray images.

GraphSAGE: A graph neural network (GNN) for processing tabular features and their relationships using a graph structure.

The model is trained to classify chest X-ray images into multiple labels, such as "Atelectasis," "Cardiomegaly," and "Pneumonia," using both image and tabular data.

# Dataset
The model is trained on the MIMIC-CXR dataset, which contains chest X-ray images and associated tabular data. The dataset includes the following features:

Image data: Chest X-ray images in JPEG format.

Tabular data: Binary labels for 10 medical conditions (e.g., "Atelectasis," "Cardiomegaly").

Graph structure: An adjacency matrix constructed using cosine similarity between tabular features.

Dataset Structure
The dataset is split into train, validation, and test sets.

Each image is associated with a set of tabular features and a multi-label target vector.

# Model Architecture
The hybrid model consists of two main branches:

Image Branch:

Uses a pre-trained ResNet50 model to extract features from chest X-ray images.

Features are pooled using Global Average Pooling and passed to the output layer.

GraphSAGE Branch:

Processes tabular features using a GraphSAGE layer.

The graph structure is defined by an adjacency matrix constructed using cosine similarity.

Features are aggregated and passed to the output layer.

The outputs of both branches are concatenated and fed into a dense layer with a sigmoid activation function for multi-label classification.

# Installation
To set up the environment and run the code, follow these steps:

Clone the repository:
git clone https://github.com/your-username/hybrid-graphsage-mimic-cxr.git
cd hybrid-graphsage-mimic-cxr
Install dependencies:
Ensure you have Python 3.8 or higher installed. Then, install the required libraries:

pip install -r requirements.txt
Download the dataset:

Download the MIMIC-CXR dataset and place it in the /content/mimic-cxr.zip directory.

Extract the dataset using the provided script.

# Usage
Preprocess the data:

The script will automatically preprocess the images and tabular data, and construct the adjacency matrix.

Train the model:
Run the script to train the hybrid GraphSAGE model:
python train.py
Evaluate the model:

After training, the model will be evaluated on the test dataset, and metrics (precision, recall, F1-score) will be displayed.

Modify the model:

You can modify the model architecture, hyperparameters, or dataset paths in the script.

# Results
The model's performance is evaluated using the following metrics:

Precision: The ratio of true positives to the total predicted positives.

Recall: The ratio of true positives to the total actual positives.

F1-Score: The harmonic mean of precision and recall.

Example results:

Validation F1-Score: 0.98

Test F1-Score: 0.97

# Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
 
Create a new branch for your feature or bugfix.

Commit your changes and push to the branch.

Submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
The MIMIC-CXR dataset is provided by the MIT Laboratory for Computational Physiology.

The Spektral library is used for implementing GraphSAGE and other graph neural network layers.

