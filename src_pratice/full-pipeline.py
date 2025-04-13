from metaflow import FlowSpec, step, card
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class FullMLFlow(FlowSpec):
    
    @card
    @step
    def start(self):
        print("🔹 Bắt đầu pipeline: tải dữ liệu Iris")
        data = load_iris()
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.next(self.split_data)
    
    @card
    @step
    def split_data(self):
        print("🔹 Chia dữ liệu thành train/test")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.next(self.train_model)
    
    @card
    @step
    def train_model(self):
        print("🔹 Huấn luyện mô hình RandomForest")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate_model)
    
    @card
    @step
    def evaluate_model(self):
        print("🔹 Đánh giá mô hình")
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        self.accuracy = acc
        print(f"✅ Độ chính xác: {acc:.4f}")

        from metaflow.cards import Markdown
        self.card = Markdown(f"""
        ## Đánh giá mô hình
        
        **Độ chính xác:** {acc:.4f}  
        **Số lượng mẫu test:** {len(self.y_test)}  
        **Số cây trong rừng:** 100  
        """)

        self.next(self.save_model)
    
    @card
    @step
    def save_model(self):
        print("🔹 Lưu mô hình vào file `rf_model.pkl`")
        joblib.dump(self.model, "rf_model.pkl")
        self.next(self.end)
    
    @card
    @step
    def end(self):
        print("🎉 Pipeline hoàn tất!")

if __name__ == '__main__':
    FullMLFlow()
