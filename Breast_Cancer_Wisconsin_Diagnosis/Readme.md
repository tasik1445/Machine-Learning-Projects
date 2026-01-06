🩺 Breast Cancer Wisconsin Diagnosis using Logistic Regression
📌 Project Overview

This project implements Logistic Regression from scratch using NumPy to classify breast tumors as Malignant (Cancerous) or Benign (Non-cancerous) using the Breast Cancer Wisconsin Diagnostic Dataset.

The goal is to demonstrate:

Data preprocessing and normalization

Binary classification using Logistic Regression

Forward & backward propagation

Gradient descent optimization

Model evaluation using accuracy

This project highlights how machine learning can assist in early detection and diagnosis of breast cancer.

📂 Dataset

Name: Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository / Kaggle

Target Variable: diagnosis

M → Malignant (1)

B → Benign (0)

Dropped Columns

id (identifier only)

Unnamed: 32 (empty column)

🛠️ Technologies & Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn (only for train_test_split)

⚠️ Logistic Regression is implemented from scratch, not using sklearn.linear_model.

⚙️ Project Workflow

Load Dataset

Data Cleaning & Encoding

Feature Normalization (Min-Max Scaling)

Train-Test Split (85% Train / 15% Test)

Model Architecture

Weight & bias initialization

Sigmoid activation function

Forward & Backward Propagation

Gradient Descent Optimization

Prediction & Evaluation

🧠 Model Details

Algorithm: Logistic Regression

Loss Function: Binary Cross-Entropy

Optimizer: Gradient Descent

Learning Rate: 0.01

Iterations: 1000

Decision Threshold: 0.5

📊 Model Performance

The model prints accuracy for both training and testing data:

Train accuracy: XX%
Test accuracy: XX%


(Exact values may vary due to random weight initialization)

▶️ How to Run the Project

1️⃣ Clone the Repository
git clone https://github.com/your-username/breast-cancer-logistic-regression.git
cd breast-cancer-logistic-regression

2️⃣ Install Dependencies
pip install numpy pandas matplotlib scikit-learn

3️⃣ Run the Script
python breast_cancer_wisconsin_diagnosis_using_logistic_regression.py


📌 Make sure the dataset path is correctly set if you are not using Google Colab.

📁 Project Structure
├── data.csv
├── breast_cancer_wisconsin_diagnosis_using_logistic_regression.py
├── README.md

🚀 Future Improvements

Use sklearn Logistic Regression for comparison

Add confusion matrix & ROC curve

Hyperparameter tuning

Cross-validation

Deploy as a web app (Flask / Streamlit)

👤 Author

Tasikul Islam
Department: Information and Communication Engineering
Daffodil International University

📜 License

This project is for educational purposes only.
Free to use, modify, and share.
