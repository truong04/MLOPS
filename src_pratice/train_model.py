from metaflow import FlowSpec, step, card
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class TrainModelFlow(FlowSpec):
    
    @card
    @step
    def start(self):
        from sklearn.datasets import load_iris
        data = load_iris()
        self.X = data.data
        self.y = data.target
        self.next(self.train)

    @step
    def train(self):
        print("Huấn luyện mô hình")
        model = RandomForestClassifier()
        model.fit(self.X, self.y)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Đánh giá mô hình")
        preds = self.model.predict(self.X)
        self.accuracy = accuracy_score(self.y, preds)
        print(f"Độ chính xác: {self.accuracy}")
        self.next(self.save)

    @step
    def save(self):
        joblib.dump(self.model, "model.pkl")
        print("Đã lưu model tại model.pkl")
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline ML hoàn tất!")

if __name__ == '__main__':
    TrainModelFlow()