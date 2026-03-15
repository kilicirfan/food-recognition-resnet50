Food Recognition Project

This project is a simple food classification system using deep learning.

We used the UNIMIB2016 food dataset, which contains tray images from a university cafeteria. Each image includes different food items placed on a tray.

Before training, the food regions are cropped using the provided annotations, resized, and then sent to the neural network model.

Project Pipeline

Tray Image → Food Crop → CNN Model → Food Prediction

Dataset Information

Dataset: UNIMIB2016 Food Dataset

Total images: 1027

Training images: 650

Test images: 360

Number of classes: 65 food types

Example classes include:

pizza

pasta

pane

carote

mandarini

Experiments

We tested different deep learning models and training methods.

Experiment 1 — ResNet50 (Baseline)

We trained a ResNet50 model using the original dataset with 65 classes.

Experiment 2 — ResNet50 (Filtered Dataset)

Classes with very few samples were removed.
The dataset was reduced from 65 classes to 42 classes to create a more balanced dataset.

Experiment 3 — ResNet50 (Data Augmentation)

We applied data augmentation techniques such as rotation and color changes during training to improve generalization.

Experiment 4 — VGG16

We trained a VGG16 model using the same dataset to compare another CNN architecture.

Results
Experiment	Model	Method	Test Accuracy
Experiment 1	ResNet50	Baseline	31.2%
Experiment 2	ResNet50	Filtered dataset	30.8%
Experiment 3	ResNet50	Data augmentation	25.8%
Experiment 4	VGG16	CNN comparison	28.4%

The best result was obtained with the ResNet50 baseline model.

Technologies Used

Python

PyTorch

Torchvision

NumPy

OpenCV
