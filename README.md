# Fruit Image Classification with AlexNet

## Overview

In today's digital era, image classification using deep convolutional neural networks (CNNs) has become a powerful tool for various applications, including object recognition and classification. This study utilizes AlexNet, a deep CNN architecture consisting of multiple layers of convolutional and pooling operations, coupled with its ability to capture intricate features within images, making it well-suited for the task of fruit classification.

Through extensive training on a dataset comprising diverse images of fruits, the aim is to leverage AlexNet's capabilities to accurately classify fruits into distinct categories.

## Table of Contents

- [Data Gathering](#data-gathering)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Model Selection](#model-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Image Classification (Class Prediction)](#image-classification-class-prediction)
- [Conclusion](#conclusion)

## Data Gathering

Gathering a diverse and comprehensive dataset of fruit images is crucial for training a robust classification model and it also tends to reduce the possibility of model overfitting. 

### Dataset Specifications
- **Number of classes**: 8 fruit types
- **Training images per class**: 2,100 images
- **Testing images per class**: 103 images
- **Total training images**: 16,800
- **Total testing images**: 824

### Fruit Classes
The fruit types/classes include:
- Strawberries
- Oranges
- Bananas
- Mangoes
- Pineapples
- Grapes
- Apples
- Watermelon

> **Note**: Images were scraped from several online repositories to ensure diversity and reduce model bias.

## Data Preprocessing and Augmentation

The preprocessing steps undertaken were necessary to ensure the consistency and suitability of the dataset for training.

### Image Resizing
- **Input size**: 227 x 227 pixels (as required by AlexNet architecture)
- **Purpose**: Prevent errors during training and evaluation

### Cross-Validation
- **Method**: 5-fold cross-validation
- **Purpose**: Increase dataset robustness and thorough model evaluation
- **Advantage**: Each iteration includes a distinct combination of training and validation sets

### Data Augmentation

#### Training Dataset Augmentations
The following augmentations were applied to the training dataset:

| Augmentation Type | Parameters | Purpose |
|------------------|------------|---------|
| Random Horizontal Flip | 50% probability | Reduce orientation bias |
| Random Rotation | ±20 degrees | Simulate different camera angles |
| Color Jitter | Max delta: 0.1 | Imitate lighting variations |
| Random Grayscale | 20% probability | Remove color information variability |
| Gaussian Blur | Kernel size: 5 pixels | Simulate image quality changes |
| Tensor Conversion & Normalization | Standard range | Prepare for model training |

#### Test and Validation Sets
- **Augmentation**: None applied
- **Rationale**: Simulate real-life scenarios without artificial modifications

### Dataset Preview
```
📁 Dataset Structure
├── 🍓 Strawberries (2,100 train + 103 test)
├── 🍊 Oranges (2,100 train + 103 test)
├── 🍌 Bananas (2,100 train + 103 test)
├── 🥭 Mangoes (2,100 train + 103 test)
├── 🍍 Pineapples (2,100 train + 103 test)
├── 🍇 Grapes (2,100 train + 103 test)
├── 🍎 Apples (2,100 train + 103 test)
└── 🍉 Watermelon (2,100 train + 103 test)
```

## Model Selection

**AlexNet** was chosen as the deep CNN architecture for this classification task due to:

- ✅ Proven effectiveness in image classification
- ✅ Multiple convolutional layers followed by pooling layers
- ✅ Fully connected layers with softmax classification
- ✅ Ability to capture intricate features within images
- ✅ Well-suited architecture depth and complexity for fruit classification

## Model Training

### Implementation Details
- **Framework**: Python with PyTorch library
- **Pre-trained Model**: AlexNet with transfer learning
- **Fine-tuning Strategy**: Frozen upper layers, fine-tuned last layers

### Hyperparameter Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 10 | Balance learning and overfitting prevention |
| **Loss Function** | Cross-entropy | Optimal for multi-class classification |
| **Optimizer** | SGD | Best performance after extensive testing |
| **Batch Size** | 32 | Balance between convergence and resource usage |
| **Number of Classes** | 8 | Customized for fruit classification task |

### Training Process
1. Load pre-trained AlexNet model
2. Freeze upper layer weights
3. Fine-tune final layers for 8-class classification
4. Apply cross-validation training across 5 folds

## Model Evaluation

### Evaluation Metrics
The following metrics were utilized to assess model performance:

- **Training Accuracy**: Model performance on training dataset
- **Validation Accuracy**: Generalization ability on unseen data
- **Confusion Matrix**: Detailed classification performance breakdown

### Evaluation Process
1. Compute training and validation accuracies
2. Generate confusion matrix on test dataset
3. Analyze misclassification patterns
4. Identify challenging class distinctions

## Results

### Cross-Validation Performance

| Model | Accuracy |
|-------|----------|
| Fold 1 | 0.8536 (85.36%) |
| Fold 2 | 0.8940 (89.40%) |
| Fold 3 | 0.9036 (90.36%) |
| Fold 4 | 0.9182 (91.82%) |
| **Fold 5** | **0.9220 (92.20%)** ⭐ |

### Key Findings
- 🏆 **Best Performance**: Fold 5 achieved the highest accuracy of 92.20%
- 📈 **Consistent Improvement**: Progressive accuracy increase across folds
- ✅ **Model Robustness**: Demonstrated effective training process
- 🚀 **Deployment Ready**: Top-performing model suitable for practical applications

## Image Classification (Class Prediction)

### Implementation
1. **Model Selection**: Fold 5 identified as top-performing model
2. **Model Persistence**: Saved best model for future use
3. **Custom Function**: Defined AlexNet instantiation function
4. **Prediction Pipeline**: Seamless loading and classification of sample images

### Usage Example
```python
# Load saved model
model = load_alexnet_model('fold5_model.pth')

# Classify fruit image
prediction = classify_fruit_image(model, 'sample_fruit.jpg')
print(f"Predicted class: {prediction}")
```

## Conclusion

This project demonstrates a **simplistic yet effective approach** to image classification leveraging deep CNN models. The AlexNet-based fruit classifier achieved impressive results with a peak accuracy of **92.20%**.

### Key Achievements
- ✅ Successful implementation of transfer learning with AlexNet
- ✅ Robust data preprocessing and augmentation pipeline
- ✅ Effective cross-validation strategy
- ✅ High classification accuracy across multiple fruit types

### Future Improvements
While the results are promising, several enhancements could further improve model performance:

#### Additional Metrics
- 📊 **Precision**: Class-specific accuracy measurements
- 📊 **Recall**: Sensitivity analysis for each fruit type
- 📊 **F1-Score**: Balanced precision-recall evaluation
- 📊 **Specificity**: True negative rate analysis

#### Model Enhancements
- 🔧 **Architecture Modifications**: Addition of normalization and flattening layers
- 🔧 **Extended Fine-tuning**: More comprehensive layer customization
- 🔧 **Larger Datasets**: Increased training data for better generalization
- 🔧 **Advanced Augmentation**: More sophisticated data transformation techniques

### Repository Structure
```
📁 fruit-classification-alexnet/
├── 📄 README.md
├── 📁 data/
│   ├── 📁 train/
│   ├── 📁 test/
│   └── 📁 validation/
├── 📁 notebooks/
│   └── 📓 fruit_classification_analysis.ipynb
└── 📄 requirements.txt
```

---

> **💡 Pro Tip**: This implementation serves as a solid foundation for fruit classification tasks and can be extended to other image classification problems with minimal modifications.

**Contributors**: Opeyemi Aniwura  
