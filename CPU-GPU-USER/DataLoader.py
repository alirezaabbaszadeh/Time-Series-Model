import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self, file_path, time_steps=168, split_ratio=0.8):
        """
        سازنده کلاس DataLoader.

        Parameters:
        - file_path (str): مسیر فایل CSV.
        - time_steps (int): تعداد گام‌های زمانی برای ایجاد دنباله.
        - split_ratio (float): نسبت تقسیم داده به آموزش و تست.
        """
        self.file_path = file_path
        self.time_steps = time_steps
        self.split_ratio = split_ratio
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def load_data(self):
        """
        بارگذاری داده‌ها از فایل CSV.

        Returns:
        - data (DataFrame): دیتافریم بارگذاری شده.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file at {self.file_path} was not found.")
        data = pd.read_csv(self.file_path)
        return data
    
    def preprocess_data(self, data):
        """
        انتخاب ویژگی‌ها و هدف، و استانداردسازی آن‌ها.

        Parameters:
        - data (DataFrame): دیتافریم اصلی.

        Returns:
        - X_scaled (ndarray): ویژگی‌های استانداردسازی شده.
        - y_scaled (ndarray): هدف استانداردسازی شده.
        """
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['Close'].values
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled
    
    def create_sequences(self, X, y):
        """
        ایجاد دنباله‌های زمانی برای مدل.

        Parameters:
        - X (ndarray): ویژگی‌های استانداردسازی شده.
        - y (ndarray): هدف استانداردسازی شده.

        Returns:
        - Xs (ndarray): دنباله‌های ورودی.
        - ys (ndarray): مقادیر هدف مربوط به هر دنباله.
        """
        Xs, ys = [], []
        for i in range(len(X) - self.time_steps):
            Xs.append(X[i:i + self.time_steps])
            ys.append(y[i + self.time_steps])
        return np.array(Xs), np.array(ys)
    
    def split_data(self, X, y):
        """
        تقسیم داده‌ها به مجموعه‌های آموزش و تست.

        Parameters:
        - X (ndarray): دنباله‌های ورودی.
        - y (ndarray): مقادیر هدف.

        Returns:
        - X_train (ndarray): داده‌های آموزش.
        - X_test (ndarray): داده‌های تست.
        - y_train (ndarray): هدف‌های آموزش.
        - y_test (ndarray): هدف‌های تست.
        """
        split = int(self.split_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test
    
    def get_data(self):
        """
        اجرای کامل فرآیند بارگذاری و پیش‌پردازش داده‌ها.

        Returns:
        - X_train (ndarray): داده‌های آموزش.
        - X_test (ndarray): داده‌های تست.
        - y_train (ndarray): هدف‌های آموزش.
        - y_test (ndarray): هدف‌های تست.
        - scaler_y (StandardScaler): شی استانداردسازی هدف.
        """
        data = self.load_data()
        X_scaled, y_scaled = self.preprocess_data(data)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        X_train, X_test, y_train, y_test = self.split_data(X_seq, y_seq)
        return X_train, X_test, y_train, y_test, self.scaler_y