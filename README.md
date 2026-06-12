# Diabetic Retinopathy Detection using CNN

A deep learning pipeline for automated detection and classification of **Diabetic Retinopathy (DR)** from retinal fundus images, built with PyTorch.

## Motivation
Diabetic Retinopathy affects a significant proportion of individuals with long-standing diabetes and is a leading cause of preventable blindness. Manual interpretation of retinal images is time-consuming and subject to human error. This project develops an automated CNN-based classification system to assist healthcare professionals in accurate, efficient DR screening.

## Problem Statement
This project classifies retinal fundus images into:

| Label | Class |
|-------|-------|
| 0 | Diabetic Retinopathy |
| 1 | No Diabetic Retinopathy |

## Key Technical Contributions
- **Custom Dataset Class:** Handles CSV-based label mapping instead of standard folder-structured datasets
- **Data Augmentation:** Random flips, rotation, and ImageNet normalization
- **Model Architectures:** Custom CNN + Transfer Learning (ResNet18)
- **Training Strategy:** Adam optimizer with ReduceLROnPlateau scheduler and best model checkpointing

## Technologies
- Python 3.10+, PyTorch 2.0+, TorchVision
- scikit-learn, pandas, matplotlib, seaborn

## Author
**Mariam Khamis Madata**
MSc Data Science (First Class) - Chandigarh University
Co-author: *Emerging of Machine Learning and Deep Learning Technology: Addressing in Intelligent Wireless Network Optimization* (peer-reviewed)

## Status
Active Development - data pipeline complete, model training in progress.
