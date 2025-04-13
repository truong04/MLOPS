import mlflow
import mlflow.sklearn
from metaflow import FlowSpec, step, card
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class IrisFlow(FlowSpec):

    @card
    @step
    def start(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.next(self.train_model)
    
    @card
    @step
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, average='weighted')  # Chỉ số Precision (Weighted)
            self.recall = recall_score(y_test, y_pred, average='weighted')        # Chỉ số Recall (Weighted)
            self.f1 = f1_score(y_test, y_pred, average='weighted')  

            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.log_metric("precision", self.precision)
            mlflow.log_metric("recall", self.recall)
            mlflow.log_metric("f1_score", self.f1)
            mlflow.sklearn.log_model(self.model, "model")
            print(f"Model saved in run {mlflow.active_run().info.run_id}")
            
        self.next(self.end)
    
    @card
    @step
    def end(self):
        print("Accuracy:", self.accuracy)


if __name__ == '__main__':
    IrisFlow()