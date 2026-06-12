# Diabetic Retinopathy Detection using CNN

A deep learning pipeline for automated 5-class classification of Diabetic Retinopathy from retinal fundus images, built with PyTorch and transfer learning (ResNet18).

## Results
- **Test Accuracy: 83%**
- **Validation Accuracy: 80.69%**

| Class | F1 Score |
|-------|----------|
| No_DR | 0.98 |
| Moderate | 0.77 |
| Mild | 0.67 |
| Proliferate_DR | 0.54 |
| Severe | 0.34 |

## Dataset
Diabetic Retinopathy 224x224 (2019 Data) from Kaggle
3,662 retinal fundus images across 5 severity classes

## Model
Pretrained ResNet18 with fine-tuned classification head for 5-class DR grading

## Training
- Train/Val/Test split: 70/15/15
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Epochs: 10
- Hardware: GPU (CUDA)

## Technologies
- Python, PyTorch, TorchVision, scikit-learn

## Author
**Mariam Khamis Madata**
MSc Data Science (First Class) - Chandigarh University

## Status
Training complete. Evaluation done.
