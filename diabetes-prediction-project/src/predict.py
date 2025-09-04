import numpy as np
import pandas as pd
import joblib

def predict_diabetes_compare(patient_data):
    """
    patient_data format:
    [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    """
    # Load models
    imputer = joblib.load("../models/imputer.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    log_model = joblib.load("../models/log_model.pkl")
    tree_model = joblib.load("../models/tree_model.pkl")

    # Convert to array
    patient_data = np.array([patient_data])

    # Handle missing values same way
    patient_data = imputer.transform(patient_data)

    # Logistic Regression Prediction
    patient_scaled = scaler.transform(patient_data)
    pred_log = log_model.predict(patient_scaled)[0]

    # Decision Tree Prediction
    pred_tree = tree_model.predict(patient_data)[0]

    # Results
    result_dict = {
        'Model': ['Logistic Regression', 'Decision Tree'],
        'Prediction': [pred_log, pred_tree],
        'Outcome': ['Diabetes (1)' if pred_log==1 else 'No Diabetes (0)',
                    'Diabetes (1)' if pred_tree==1 else 'No Diabetes (0)']
    }
    return pd.DataFrame(result_dict)


if __name__ == "__main__":
    # Example patient
    test_patient = [2, 120, 70, 20, 79, 25.5, 0.5, 30]
    print("\nPrediction Results Comparison:")
    print(predict_diabetes_compare(test_patient))
