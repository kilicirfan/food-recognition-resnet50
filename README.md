This project is a simple food classification system using deep learning.

We used the **UNIMIB2016 food dataset**, which contains tray images with different food items. Each food item is annotated and cropped before training.

The model used in this project is **ResNet50**, a convolutional neural network commonly used for image classification.

Project pipeline:

Tray Image → Food Crop → ResNet50 Model → Food Prediction

Dataset information:
- Total images: 1027
- Training images: 650
- Test images: 360
- Number of classes: 65


Experiment Results

Baseline Model:
ResNet50
Classes: 65
Validation Accuracy: 36.5%
Test Accuracy: 31.2%

Improved Model:
ResNet50 with class filtering and weighted loss
Classes: 42
Test Accuracy: 30.8%


The model was trained using Python and PyTorch.
