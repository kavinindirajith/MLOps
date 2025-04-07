import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def main():
    input_path = "data/heart_attack_prediction_indonesia.csv"
    output_path = "data/processed_data.csv"

    df = pd.read_csv(input_path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop missing values
    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Save processed data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
