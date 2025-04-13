import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_log_model():
    # Tải bộ dữ liệu Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo mô hình RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Bắt đầu một MLflow run
    with mlflow.start_run():
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán và tính toán các độ đo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  # Chỉ số Precision (Weighted)
        recall = recall_score(y_test, y_pred, average='weighted')        # Chỉ số Recall (Weighted)
        f1 = f1_score(y_test, y_pred, average='weighted')                # Chỉ số F1 (Weighted)
        
        # Log các tham số và kết quả
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log mô hình
        mlflow.sklearn.log_model(model, "model")

        # In ra thông tin về run đã lưu
        print(f"Model saved in run {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_and_log_model()