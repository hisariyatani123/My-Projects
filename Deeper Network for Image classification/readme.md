# Deep Neural Networks for Image Classification
A comparative study of VGG-19, ResNet-34, and DenseNet-121 architectures on MNIST and CIFAR-10 datasets with model modifications.

## Overview
This project implements and compares three deep CNN architectures:
- **VGG-19** with batch normalization modifications
- **ResNet-34** with dropout layers
- **DenseNet-121** (original implementation)

Key features:
- Comparative analysis of model performance
- Implementation of regularization techniques
- Training optimization with GPU acceleration
- Visualization of accuracy/loss metrics

## Architecture Modifications
| Model        | Modifications                          |
|--------------|----------------------------------------|
| VGG-19       | Added batch normalization              |
| ResNet-34    | Added dropout (0.2) in FC layer        |
| DenseNet-121 | Original implementation                |

##  Dataset Preparation
### MNIST
- 70,000 grayscale images (28×28)
- Preprocessing: Resize to 32×32, horizontal flip, normalization

### CIFAR-10
- 60,000 RGB images (32×32)
- Preprocessing: Padding (4px), random crop, normalization

