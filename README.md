# Lab Report: Computer Vision with PyTorch

## Introduction
Computer vision, a subfield of artificial intelligence, focuses on enabling machines to interpret and understand visual information from the world. In this lab, we explore various computer vision techniques using the PyTorch library, a powerful open-source machine learning framework. The lab is divided into two parts, each addressing different aspects of computer vision tasks.

## Part 1: CNN Classifier
### Objective
The objective of Part 1 is to build, train, and evaluate different neural network architectures, specifically Convolutional Neural Networks (CNNs) and Faster R-CNN, for image classification tasks using the MNIST dataset. We aim to understand the principles behind these architectures and compare their performance on the classification task.

### Implementation
1. **CNN Architecture**: We implemented a CNN architecture using PyTorch, leveraging its built-in modules such as convolutional layers, pooling layers, and fully connected layers. The architecture was designed to take grayscale images of handwritten digits from the MNIST dataset as input and output the predicted class label for each image.
   
2. **Faster R-CNN**: We adapted the Faster R-CNN architecture, a state-of-the-art object detection model, for the image classification task on MNIST. This involved treating each digit as an object and using bounding boxes for classification. While this approach may seem unconventional for MNIST, it provided an interesting comparison with the CNN model.
   
3. **Training and Evaluation**: Both CNN and Faster R-CNN models were trained using the training set of the MNIST dataset and evaluated on the test set. We used standard metrics such as accuracy, F1 score, loss, and training time to assess the performance of each model.

### Results and Analysis
- **CNN Performance**: The CNN model achieved high accuracy and F1 score on the MNIST dataset, demonstrating its effectiveness for image classification tasks. It showed robust performance and relatively fast convergence during training.
  
- **Faster R-CNN Performance**: The adaptation of Faster R-CNN for MNIST classification was unconventional but provided valuable insights into the flexibility of the model architecture. However, the model's performance may have been affected by the mismatch between the model's design and the characteristics of the dataset.
  
- **Comparison**: By comparing the results of both models, we observed differences in performance metrics, highlighting the strengths and limitations of each approach. While CNN performed well on the classification task, Faster R-CNN showcased the versatility of object detection models in different scenarios.

## Part 2: Vision Transformer (ViT)
### Objective
The objective of Part 2 is to explore the emerging paradigm of Vision Transformers (ViTs) and apply them to the MNIST classification task. We aim to understand the underlying principles of ViTs and compare their performance with traditional CNN models.

### Implementation
1. **ViT Model**: Following a tutorial, we implemented a Vision Transformer model from scratch in PyTorch. ViTs represent a novel approach to image classification, relying on self-attention mechanisms to capture global dependencies in the input image. The model architecture included embedding layers, transformer blocks, and a classification head.
   
2. **MNIST Classification**: The ViT model was adapted to perform image classification on the MNIST dataset. We adjusted the input size and other parameters to fit the characteristics of the dataset, ensuring compatibility between the model and the task.

### Results and Analysis
- **ViT Performance**: The ViT model demonstrated competitive performance on the MNIST dataset, showcasing the potential of transformer-based architectures for image classification tasks. Despite being a relatively new approach, ViTs showed promising results and offered a fresh perspective on traditional computer vision tasks.
  
- **Comparison with CNN**: Comparing the results obtained from ViT with those from CNN models provided valuable insights into the relative strengths and weaknesses of each approach. While CNN models have been the cornerstone of computer vision for many years, ViTs represent a promising alternative that warrants further exploration.
  
- **Interpretation**: Analyzing the results allowed us to understand the behavior of ViTs in the context of image classification. We observed similarities and differences in performance metrics compared to CNN models, shedding light on the unique characteristics of ViTs and their potential applications in various domains.

## Conclusion
Through this lab, we gained hands-on experience in building and comparing different neural network architectures for computer vision tasks. We explored traditional CNNs, object detection with Faster R-CNN, and the emerging paradigm of Vision Transformers. By analyzing the results and understanding the nuances of each approach, we are better equipped to tackle real-world computer vision problems and contribute to advancements in the field.

---
By implementing these models and analyzing their performance, we have deepened our understanding of computer vision techniques and their applications in various domains. This lab serves as a foundation for further exploration and experimentation in the field of computer vision with PyTorch.
