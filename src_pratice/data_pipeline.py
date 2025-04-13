from metaflow import FlowSpec, step
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPipeline(FlowSpec):

    @step
    def start(self):
        print("Đọc data")
        self.df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        print("Data preprocessing")
        self.df['target'] = self.df['species'].astype('category').cat.codes
        self.features = self.df.drop(['species', 'target'], axis = 1)
        self.labels = self.df['target']
        self.next(self.split)

    @step
    def split(self):
        print("Chia train/test")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        self.train = (X_train, y_train)
        self.test = (X_test, y_test)
        self.next(self.end)

    @step
    def end(self):
        print("Hoàn thành xử lý dữ liệu")

if __name__ == '__main__':
    DataPipeline()