from metaflow import FlowSpec, step, card
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class IrisFlow(FlowSpec):

    @card
    @step
    def start(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.next(self.split_data)
    @card
    @step
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.models = ['random_forest', 'mlp', 'svm']
        self.next(self.train_models, foreach='models')  # dùng foreach nên KHÔNG cần @parallel

    @card
    @step
    def train_models(self):
        model_name = self.input

        if model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'mlp':
            model = MLPClassifier(random_state=42, max_iter=500)
        elif model_name == 'svm':
            model = SVC()

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        self.model_name = model_name
        self.accuracy = accuracy
        self.next(self.collect_results)

    @step
    def collect_results(self, inputs):
        self.results = {input.model_name: input.accuracy for input in inputs}
        self.next(self.end)

    @step
    def end(self):
        print("Kết quả đánh giá độ chính xác:")
        for model, acc in self.results.items():
            print(f"{model}: {acc:.4f}")

if __name__ == '__main__':
    IrisFlow()
