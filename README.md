# Breast Cancer Classification using BUSI Dataset

**Student:** Vadde Sowmya 

**Roll No:** CS23B1077

**Institution:** Indian Institute of Information Technology Raichur  

**Course:** Deep Learning for Medical Imaging  

## 📌 Project Overview
This project implements a deep learning classification system for breast ultrasound images. The model categorizes images into three diagnostic classes: **Benign, Malignant, and Normal** tissue. The core objective of this study is to evaluate the effectiveness of dynamic data augmentation in preventing neural network overfitting on limited medical imaging data.

## 📊 Dataset
* **Source:** Breast Ultrasound Images Dataset (BUSI)
* **Total Images Used:** 780 (Raw ultrasound images only; ground-truth segmentation masks were excluded)
* **Class Distribution:** Benign (~56%), Malignant (~27%), Normal (~17%)
* **Data Split:** Stratified 70% Training (546), 15% Validation (117), 15% Testing (117)

## 🛠️ Methodology

### 1. Preprocessing & Augmentation
* Images resized to `128x128` pixels and normalized to a `0-1` scale.
* Class weighting was applied to penalize the model for missing minority classes.
* **Training Augmentation:** Random horizontal flipping, rotation (up to 18°), and zooming (up to 10%) were applied dynamically to the training set to prevent memorization.

### 2. Model Architecture
A custom Sequential Convolutional Neural Network (CNN) built with TensorFlow/Keras:
* **Feature Extraction:** 3 Convolutional Blocks (Conv2D + MaxPooling2D) using 32, 64, and 128 filters.
* **Classification Head:** Flatten layer $\rightarrow$ Dense layer (128 neurons) $\rightarrow$ Dropout (0.5) to prevent overfitting $\rightarrow$ Dense Output (3 neurons, Softmax).

## 📈 Comparative Analysis: The Impact of Augmentation
To prove the necessity of augmentation, two iterative experiments were conducted:

1. **Model A (Baseline - No Augmentation):** Trained for 20 epochs. Exhibited severe overfitting. Training accuracy reached **95.4%**, while validation accuracy stalled at **75.2%**. The model was memorizing exact pixel locations.
2. **Model B (With Augmentation):** By introducing dynamic augmentation and training for an extended **50 epochs**, the model was forced to learn underlying tumor structures. The generalization gap closed completely, preventing overfitting.

## 🏆 Final Results (Model B - 50 Epochs)
Evaluated on the strictly held-out test set (117 unseen images):
* **Overall Accuracy:** 74%
* **Benign F1-Score:** 0.79
* **Malignant Recall:** 0.68
* **Normal F1-Score:** 0.65

**Clinical Insight:** The model demonstrated strong precision, particularly in identifying benign tumors with minimal false alarms. Crucially, it achieved a 68% recall rate for malignant cases, correctly identifying the majority of true cancers without confusing them for normal tissue.

## 💻 Tech Stack
* Python
* TensorFlow / Keras
* OpenCV (`cv2`)
* Scikit-Learn (stratified splitting, class weights, evaluation metrics)
* Matplotlib & Seaborn (visualizations)
