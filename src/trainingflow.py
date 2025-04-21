from metaflow import FlowSpec, step, Parameter
from dataprocessing import load_train_test, preprocess_data
from model_utils import train_model, evaluate_model, register_model

class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)

    @step
    def start(self):
        print("Loading and splitting data...")
        self.train_df, self.test_df = load_train_test()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Preprocessing data...")
        self.X_train, self.y_train = preprocess_data(self.train_df)
        self.X_test, self.y_test = preprocess_data(self.test_df)
        self.next(self.train)

    @step
    def train(self):
        print("Training model...")
        self.model = train_model(self.X_train, self.y_train, self.seed)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Evaluating model...")
        self.accuracy, self.report = evaluate_model(self.model, self.X_test, self.y_test)
        print(f"Accuracy: {self.accuracy}\nReport:\n{self.report}")
        self.next(self.register)

    @step
    def register(self):
        print("Registering model...")
        register_model(self.model)
        self.next(self.end)

    @step
    def end(self):
        print("Training pipeline completed.")

if __name__ == '__main__':
    TrainingFlow()
