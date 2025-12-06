from dotenv import load_dotenv
import os
import numpy as np
import joblib

load_dotenv()

base_path = os.getenv("MODEL_BASE_PATH")
model = joblib.load(f"{base_path}_model.pkl")
scaler = joblib.load(f"{base_path}_scaler.pkl")
encoder = joblib.load(f"{base_path}_encoder.pkl")

features = input("Enter features to predict")

def predict_mine_or_rock(input_str, model, scaler, encoder):
    # Validate input
    features = np.array(input_str.split(','), dtype=float).reshape(1, -1)
    if features.shape[1] != 60:
        raise ValueError(f"Expected 60 features, got {features.shape[1]}")
    
    # Apply same preprocessing as training
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction_encoded = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Convert back to original label
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    confidence = np.max(prediction_proba)
    
    return prediction, confidence

prediction, confidence = predict_mine_or_rock(features, model, scaler, encoder)
print("Prediction : ", prediction)
print("Confidence : ", confidence)