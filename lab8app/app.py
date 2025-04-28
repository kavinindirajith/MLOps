# lab8app/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

# Create the FastAPI app
app = FastAPI()

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load your model from MLflow
model = mlflow.pyfunc.load_model("models:/heart_attack_rf/Production") 

# Define the input data format
class InputData(BaseModel):
    age: float
    gender: float
    region: float
    income_level: float
    hypertension: float
    diabetes: float
    cholesterol_level: float
    obesity: float
    waist_circumference: float
    family_history: float
    smoking_status: float
    alcohol_consumption: float
    physical_activity: float
    dietary_habits: float
    air_pollution_exposure: float
    stress_level: float
    sleep_hours: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    fasting_blood_sugar: float
    cholesterol_hdl: float
    cholesterol_ldl: float
    triglycerides: float
    EKG_results: float
    previous_heart_disease: float
    medication_usage: float
    participated_in_free_screening: float

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Convert the input data to a DataFrame
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])

    # (Optional) Reorder columns if your model expects a specific order
    column_order = [
        'age', 'gender', 'region', 'income_level', 'hypertension', 'diabetes',
        'cholesterol_level', 'obesity', 'waist_circumference', 'family_history',
        'smoking_status', 'alcohol_consumption', 'physical_activity', 'dietary_habits',
        'air_pollution_exposure', 'stress_level', 'sleep_hours', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'fasting_blood_sugar', 'cholesterol_hdl', 'cholesterol_ldl',
        'triglycerides', 'EKG_results', 'previous_heart_disease', 'medication_usage',
        'participated_in_free_screening'
    ]
    df = df[column_order]

    # Make the prediction
    prediction = model.predict(df)

    # Return the prediction
    return {"prediction": prediction.tolist()}
