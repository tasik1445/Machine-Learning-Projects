# 🩺 Diabetes Prediction Project

This project predicts whether a patient has diabetes using **Logistic Regression** and **Decision Tree** models.

---

## 📂 Project Structure
```
diabetes-prediction-project/
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── log_model.pkl
│   └── tree_model.pkl
│
├── src/
│   ├── train.py      # Train models
│   └── predict.py    # Predict with trained models
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. Clone repo:
```bash
git clone https://github.com/yourusername/diabetes-prediction-project.git
cd diabetes-prediction-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train models:
```bash
python src/train.py
```

4. Run prediction:
```bash
python src/predict.py
```

---

## 📊 Models Used
- Logistic Regression (with StandardScaler)
- Decision Tree Classifier

---

✅ Models will be saved in the `models/` folder.  
✅ Input is an 8-feature patient record.  
✅ Outputs comparison between Logistic Regression and Decision Tree.
