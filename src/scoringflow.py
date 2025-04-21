from metaflow import FlowSpec, step
from dataprocessing import preprocess_data
from model_utils import load_registered_model
import pandas as pd

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        print("Loading holdout data...")
        self.df = pd.read_csv("data/holdout_data.csv")
        self.true_labels = self.df['heart_attack']
        self.df = self.df.drop(columns=['heart_attack'])
        self.X = self.df.copy()
        self.next(self.predict)

    # @step
    # def preprocess(self):
    #     print("Preprocessing holdout data...")
    #     self.X, _ = preprocess_data(self.df)
    #     self.next(self.predict)

    @step
    def predict(self):
        print("Loading registered model and making predictions...")
        model = load_registered_model()
        self.df['predicted_heart_attack'] = model.predict(self.X)
        print(self.df[['predicted_heart_attack']].head())
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")

if __name__ == '__main__':
    ScoringFlow()
