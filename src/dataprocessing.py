import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "heart_attack"

def load_data(path="data/processed_data.csv"):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET] if TARGET in df else None

    return X, y

def load_train_test():
    df = load_data()
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
