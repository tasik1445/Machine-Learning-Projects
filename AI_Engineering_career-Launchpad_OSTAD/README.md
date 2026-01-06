## 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview:

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques.
A Random Forest Classifier is trained on passenger data such as age, gender, ticket class, fare, and family information.

The project demonstrates:

Data preprocessing

Feature selection

Supervised classification

Model training and evaluation

## 📂 Dataset:

Name: Titanic Dataset

Source: Public GitHub Dataset (DataScienceDojo)

Target Variable: Survived

1 → Survived

0 → Did Not Survive

Selected Features

Pclass – Passenger class

Sex – Gender

Age – Age of passenger

SibSp – Number of siblings/spouses aboard

Parch – Number of parents/children aboard

Fare – Ticket fare

Embarked – Port of embarkation

## 🛠️ Technologies & Libraries Used:

Python

Pandas

Scikit-learn

RandomForestClassifier

train_test_split

accuracy_score

confusion_matrix

## ⚙️ Project Workflow:

Load the Titanic dataset

Select relevant features

Encode categorical variables

Male → 0, Female → 1

Embarked: S → 0, C → 1, Q → 2

Handle missing values

Split data into training and testing sets

Train a Random Forest Classifier

Evaluate model performance

## 🧠 Model Details:

Algorithm: Random Forest Classifier

Test Size: 20%

Random State: 42

Evaluation Metrics:

Accuracy Score

Confusion Matrix

## 📊 Model Evaluation:

The model evaluates performance using:

Accuracy Score

Confusion Matrix

Example output:

Accuracy: XX%
Confusion Matrix:
[[TN FP]
 [FN TP]]


(Results may vary due to randomness in model training.)

▶️ How to Run the Project:

1️⃣ Clone the Repository
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction

2️⃣ Install Required Libraries
pip install pandas scikit-learn

3️⃣ Run the Script
python survival_prediction.py

📁 Project Structure
├── survival_prediction.py
├── README.md

🚀 Future Improvements

Handle missing values more effectively

Add feature scaling

Compare multiple ML models

Add visualization (EDA & confusion matrix heatmap)

Deploy using Streamlit or Flask

👤 Author

Tasikul Islam
Department: Information and Communication Engineering
Daffodil International University


