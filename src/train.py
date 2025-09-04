import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import os

# Paths
DATA_PATH = "../data/diabetes.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load Dataset
data = pd.read_csv(DATA_PATH)
print("✅ Dataset Loaded Successfully!")

# Replace 0 with NaN in specific columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(data.drop('Outcome', axis=1))
y = data['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Evaluate
print("✅ Logistic Regression Accuracy:", accuracy_score(y_test, log_model.predict(X_test_scaled)))
print("✅ Decision Tree Accuracy:", accuracy_score(y_test, tree_model.predict(X_test)))

# Save Models
joblib.dump(imputer, f"{MODEL_DIR}/imputer.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(log_model, f"{MODEL_DIR}/log_model.pkl")
joblib.dump(tree_model, f"{MODEL_DIR}/tree_model.pkl")

print("✅ Models Saved Successfully in 'models/' directory!")
