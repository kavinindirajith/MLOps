import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "age": 60,
    "gender": 1,
    "region": 1,
    "income_level": 2,
    "hypertension": 0,
    "diabetes": 1,
    "cholesterol_level": 210,
    "obesity": 1,
    "waist_circumference": 95,
    "family_history": 1,
    "smoking_status": 0,
    "alcohol_consumption": 1,
    "physical_activity": 2,
    "dietary_habits": 1,
    "air_pollution_exposure": 1,
    "stress_level": 2,
    "sleep_hours": 6,
    "blood_pressure_systolic": 130,
    "blood_pressure_diastolic": 85,
    "fasting_blood_sugar": 105,
    "cholesterol_hdl": 50,
    "cholesterol_ldl": 120,
    "triglycerides": 150,
    "EKG_results": 1,
    "previous_heart_disease": 0,
    "medication_usage": 1,
    "participated_in_free_screening": 0
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
