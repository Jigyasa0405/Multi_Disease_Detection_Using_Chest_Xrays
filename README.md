# Multi-Disease Detection in Chest X-Rays Using an Explainable Deep Learning Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the code and methodology for a deep learning project aimed at developing a robust, transparent, and clinically relevant system for detecting multiple diseases from chest X-ray images.

## Overview

The primary goal of this project is to build a multi-disease classification model that not only achieves high diagnostic accuracy but also provides interpretable insights into its decision-making process. This is crucial for building trust with clinicians and facilitating the adoption of AI in medical diagnostics.

The framework is built upon:
- **Multi-Architecture Comparison:** Evaluation of three powerful CNN architectures: **ResNet-50**, **DenseNet-121**, and **EfficientNet-B4**.
- **Balanced, Multi-Source Dataset:** A balanced dataset of 60,000 images (10,000 per class) created by merging and strategically resampling data from the NIH ChestX-ray14, COVIDx CXR-2, and Tuberculosis (Montgomery & China) datasets.
- **Explainable AI (XAI):** Integration of **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps that visualize the anatomical regions the model focuses on for its predictions.

## Key Features

- **Multi-Class Classification:** Simultaneously identifies 6 distinct thoracic conditions:
    - COVID-19
    - Cardiomegaly
    - Normal
    - Pleural Effusion
    - Pneumonia
    - Tuberculosis
- **High Performance:** Achieves test accuracies of up to **89.13%** (EfficientNet-B4) and robust AUC scores.
- **Model Interpretability:** Generates visual explanations (Grad-CAM heatmaps) to show *why* a model made a specific diagnosis, aligning attention with clinically relevant anatomy.
- **Comprehensive Evaluation:** Models are evaluated using a range of metrics, including accuracy, precision, recall, F1-score, and macro AUC-ROC.
- **Reproducible Pipeline:** A structured workflow from data loading and preprocessing to model training, evaluation, and XAI visualization.

## Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** PyTorch, torchvision
- **Data Manipulation:** Pandas, NumPy
- **Image Processing:** OpenCV, Pillow
- **Visualization:** Matplotlib
- **Explainable AI:** PyTorch Grad-CAM
- **Environment:** Jupyter Notebook, Kaggle

## Dataset

The project uses a combination of three public chest X-ray datasets to ensure diversity and coverage of target diseases:

1.  **NIH ChestX-ray14:** Provides a large-scale baseline with various thoracic pathologies.
2.  **COVIDx CXR-2:** Contains a significant number of COVID-19 positive cases.
3.  **TB Datasets (Montgomery & China):** Provides cases for tuberculosis detection.

To address severe class imbalance in the combined dataset, a strategic resampling strategy was implemented, creating a balanced dataset of **60,000 images** with exactly 10,000 images per class. The data was then split into **70% training, 15% validation, and 15% test** sets with patient-level stratification.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

1. The authors of the NIH ChestX-ray14, COVIDx CXR-2, and TB datasets for making their data publicly available.
2. The PyTorch and torchvision teams for their excellent deep learning frameworks.
3. The creators of the PyTorch Grad-CAM library.
