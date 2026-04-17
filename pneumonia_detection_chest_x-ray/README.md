# 🩺 Pneumonia Detection using Deep Learning, Machine Learning & Explainable AI (XAI)

## 📌 Project Overview

This project presents an intelligent system for **automatic pneumonia detection** from chest X-ray images by combining:

* 🧠 Deep Learning (CNN)
* 🤖 Machine Learning (RF, SVM, KNN)
* 🔍 Explainable AI (Grad-CAM, LIME, SHAP)

The goal is to **assist radiologists** by providing accurate predictions along with **visual explanations**, improving trust and interpretability in medical diagnosis.

---

## 🎯 Objectives

* Detect pneumonia from chest X-ray images
* Improve classification accuracy using hybrid models
* Provide explainable predictions using XAI techniques
* Build a user-friendly inference system

---

## 🧠 Methodology

### 🔹 1. Deep Learning (CNN)

* A custom **22-layer Convolutional Neural Network**
* Extracts high-level image features
* Trained using chest X-ray dataset

### 🔹 2. Feature Extraction

* Intermediate CNN layer (`feature_dense`) used as feature vector
* Generates meaningful representations for ML models

### 🔹 3. Machine Learning Models

* 🌲 Random Forest (RF)
* 📈 Support Vector Machine (SVM)
* 📍 K-Nearest Neighbors (KNN)

These models classify extracted features for improved performance.

### 🔹 4. Hybrid Model

* CNN + ML combination
* Final decision using **majority voting**

---

## 🔍 Explainable AI (XAI)

### 🔥 Grad-CAM

* Highlights infected regions in the lungs
* Shows where the CNN is focusing

### 🧩 LIME

* Explains prediction using superpixel segmentation
* Identifies important regions influencing the decision

### 📊 SHAP

* Explains feature contributions
* Provides model interpretability at feature level

---

## 📂 Dataset

* Chest X-ray dataset (Pneumonia vs Normal)
* Structured as:

```
train/
 ├── NORMAL/
 └── PNEUMONIA/

val/
 ├── NORMAL/
 └── PNEUMONIA/

test/
 ├── NORMAL/
 └── PNEUMONIA/
```

---

## ⚙️ Technologies Used

* Python 🐍
* TensorFlow / Keras
* Scikit-learn
* OpenCV / PIL
* LIME
* SHAP
* Google Colab

---

## 📁 Project Structure

```
Research_base_Model/
│
├── training_notebook.ipynb
├── pneumonia_xai_prediction.ipynb
│
├── final_cnn_model.h5
├── feature_extractor.h5
├── feature_scaler.pkl
├── cnn_rf_model.pkl
├── cnn_svm_model.pkl
├── cnn_knn_model.pkl
│
└── chest_xray_dataset/
```

---

## 🚀 How to Run

### 🔹 Step 1: Train Model

* Run `training_notebook.ipynb`
* Save trained models

### 🔹 Step 2: Prediction + XAI

* Open `pneumonia_xai_prediction.ipynb`
* Load saved models
* Run:

```python
user_predict_and_explain("image_path")
```

---

## 📊 Output

The system provides:

* ✅ Pneumonia / Normal prediction
* 📊 CNN + RF + SVM + KNN results
* 🔥 Grad-CAM heatmap
* 🧩 LIME explanation
* 📊 SHAP feature importance

---

## 🧠 Key Contributions

* Hybrid CNN + ML architecture
* High accuracy classification
* Integrated Explainable AI
* User-level prediction interface

---

## 📌 Applications

* Clinical decision support
* Medical image analysis
* AI-assisted diagnosis systems

---

## 🎓 Future Work

* Deploy as web application (Streamlit)
* Use larger datasets
* Improve real-time inference
* Integrate more advanced XAI methods

---

## 👨‍💻 Author

Developed as part of research-based machine learning project.

---

## 📜 License

This project is for academic and research purposes only.

