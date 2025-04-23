from metaflow import FlowSpec, step, resources, timeout, retry, catch, conda_base
from dataprocessing import preprocess_data
from model_utils import load_registered_model
import pandas as pd

@conda_base(libraries={
    "scikit-learn": "1.3.0",
    "pandas": "2.2.2",
    "joblib": "1.3.2",
    "mlflow": "2.11.3"
})

class ScoringFlow(FlowSpec):

    @catch(var="load_error")
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="load_error")
    @step
    def start(self):
        print("Loading holdout data...")
        self.df = pd.read_csv("data/holdout_data.csv")
        self.true_labels = self.df['heart_attack']
        self.df = self.df.drop(columns=['heart_attack'])
        self.X = self.df.copy()
        self.next(self.predict)

    @step
    @resources(cpu=2, memory=4096)
    def predict(self):
        print("Loading registered model and making predictions...")
        model = load_registered_model()
        self.df['predicted_heart_attack'] = model.predict(self.X)
        print(self.df[['predicted_heart_attack']].head())
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")
