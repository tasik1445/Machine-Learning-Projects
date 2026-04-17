## 🖼️ Results & Explainability

### 📌 Model Prediction Output

The system successfully classifies chest X-ray images into **PNEUMONIA** or **NORMAL** using a hybrid approach.

* CNN provides probability-based prediction
* Random Forest, SVM, and KNN perform classification on extracted features
* Final decision is obtained using **majority voting**

✔ In the shown example, all models predicted **PNEUMONIA**, leading to a confident final diagnosis.

---

### 🔥 Grad-CAM Visualization

Grad-CAM highlights the **regions of the lungs** that influenced the CNN’s decision.

* Warmer colors (red/yellow) indicate **high importance**
* The model focuses on **infected lung areas**
* Helps verify that prediction is based on **medically relevant regions**

---

### 🧩 LIME Explanation

LIME explains the prediction by identifying **important image segments (superpixels)**.

* Highlighted regions represent areas contributing most to the prediction
* Provides **local interpretability**
* Confirms that abnormal lung regions influence the model output

---

### 📊 SHAP Explanation

SHAP explains how different features contribute to the final prediction.

* Shows **positive/negative impact** of features
* Helps understand model behavior at a deeper level
* Adds transparency to the hybrid ML model

---

## 🧠 Overall Interpretation

All three XAI techniques collectively validate the model:

* Grad-CAM → where the model is looking
* LIME → which regions matter locally
* SHAP → how features contribute

👉 This ensures the system is not just accurate, but also **interpretable and trustworthy**, which is essential for real-world medical applications.

