# Computer Vision From Scratch 🚀

## Overview
This repository provides **end-to-end implementations of modern computer vision architectures built from scratch using PyTorch**. The project is designed to build a deep, practical understanding of how vision models work internally, starting from basic Convolutional Neural Networks (CNNs) and progressing through VGG-style networks, ResNet, and finally Vision Transformers (ViT).

The focus is **learning by implementation**, not using pretrained models or high-level abstractions.

---

## Motivation
Most tutorials rely heavily on pretrained models and hidden abstractions. This repository aims to:
- Understand *why* architectures are designed the way they are
- Implement core ideas manually
- Build strong intuition for modern vision backbones

---

## Architectures Covered

### 1. CNN from Scratch
- Basic convolutional neural network
- Convolution, ReLU, MaxPooling, Fully Connected layers
- Manual training and evaluation loop
- Dataset: CIFAR-10

### 2. VGG-like Network
- Deep CNN with stacked convolutional blocks
- Fixed 3×3 kernels
- Increasing channel depth
- Demonstrates the effect of depth on feature learning

### 3. ResNet (Residual Networks)
- Residual blocks with skip connections
- Solves vanishing gradient problem
- Modular architecture design
- ResNet-18 style implementation

### 4. Advanced CNN Concepts
- Batch Normalization
- Dropout
- Data Augmentation
- Training stability and generalization

### 5. Vision Transformer (ViT)
- Image patch embedding
- Positional encoding
- Multi-head self-attention
- Transformer encoder for vision tasks
- Classification head

---

## Repository Structure
```
Computer-Vision-From-Scratch/
│
├── 01_CNN_Basics/
│   └── cnn_from_scratch.ipynb
│
├── 02_VGG_Like/
│   └── vgg_like_from_scratch.ipynb
│
├── 03_ResNet/
│   └── resnet_from_scratch.ipynb
│
├── 04_Advanced_CNNs/
│   └── regularization_and_augmentation.ipynb
│
├── 05_Vision_Transformer/
│   └── vit_from_scratch.ipynb
│
├── requirements.txt
└── README.md
```

---

## Tech Stack
- Python 3
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Dataset
- **CIFAR-10**
  - 60,000 color images
  - 10 classes
  - Image size: 32×32

Used consistently across all CNN-based experiments for fair comparison.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/MMS-1017/CNN-to-ViT-From-Scratch
cd CNN-to-ViT-From-Scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open any notebook and start experimenting.

---

## Notes
- All models are trained from scratch
- Code is intentionally explicit for educational clarity
- Performance is secondary to understanding

---

## Future Work
- Vision Transformers with larger datasets
- Hybrid CNN-Transformer models
- Self-supervised learning (e.g., MAE, SimCLR)
- Model optimization and deployment

---
