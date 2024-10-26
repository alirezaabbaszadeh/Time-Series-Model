import tensorflow as tf
import json
import os

class Trainer:
    """
    کلاس Trainer برای مدیریت فرآیند آموزش مدل‌های TensorFlow.

    Attributes:
        model (tf.keras.Model): مدل TensorFlow که قرار است آموزش داده شود.
        X_train (np.ndarray): داده‌های ویژگی برای آموزش.
        y_train (np.ndarray): برچسب‌های داده‌های آموزشی.
        X_val (np.ndarray): داده‌های ویژگی برای اعتبارسنجی.
        y_val (np.ndarray): برچسب‌های داده‌های اعتبارسنجی.
        epochs (int): تعداد دورهای آموزش.
        batch_size (int): اندازه بچ برای آموزش.
        history (tf.keras.callbacks.History): تاریخچه آموزش مدل.
        history_path (str, optional): مسیر ذخیره تاریخچه آموزش به صورت فایل JSON.
    """

    def __init__(self, model, X_train, y_train, X_val, y_val, epochs=1, batch_size=16, history_path=None):
        """
        سازنده کلاس Trainer.

        Args:
            model (tf.keras.Model): مدل TensorFlow برای آموزش.
            X_train (np.ndarray): داده‌های ویژگی آموزشی.
            y_train (np.ndarray): برچسب‌های آموزشی.
            X_val (np.ndarray): داده‌های ویژگی اعتبارسنجی.
            y_val (np.ndarray): برچسب‌های اعتبارسنجی.
            epochs (int, optional): تعداد دوره‌های آموزش. پیش‌فرض 100.
            batch_size (int, optional): اندازه بچ برای آموزش. پیش‌فرض 64.
            history_path (str, optional): مسیر ذخیره تاریخچه آموزش. پیش‌فرض None.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None
        self.history_path = history_path  # مسیر ذخیره تاریخچه

    def train(self):
        """
        متد برای شروع فرآیند آموزش مدل.

        این متد شامل تنظیم callbackهای مورد نیاز، اجرای آموزش مدل و ذخیره تاریخچه آموزش (در صورت تعیین مسیر) می‌باشد.

        Returns:
            tf.keras.callbacks.History: تاریخچه آموزش مدل.
        """
        callbacks = [
            # توقف زودهنگام آموزش اگر بهبود در اعتبارسنجی دیده نشود
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # اگر مسیر ذخیره‌سازی تاریخچه مشخص شده باشد، از Callback سفارشی برای ذخیره تاریخچه استفاده می‌کنیم
        if self.history_path:
            callbacks.append(self.SaveHistoryCallback(self.history_path))
        
        # اجرای فرآیند آموزش مدل
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_val, self.y_val),
            verbose=1,
            callbacks=callbacks
        )
        return self.history

    class SaveHistoryCallback(tf.keras.callbacks.Callback):
        """
        Callback سفارشی برای ذخیره تاریخچه آموزش به صورت فایل JSON.

        Attributes:
            history_path (str): مسیر ذخیره تاریخچه آموزش.
        """

        def __init__(self, history_path):
            """
            سازنده Callback سفارشی.

            Args:
                history_path (str): مسیر ذخیره تاریخچه آموزش.
            """
            super().__init__()
            self.history_path = history_path

        def on_train_end(self, logs=None):
            """
            متد فراخوانی شده در انتهای فرآیند آموزش.

            این متد تاریخچه آموزش را به فرمت JSON ذخیره می‌کند.

            Args:
                logs (dict, optional): اطلاعات لاگ‌های آموزش. پیش‌فرض None.
            """
            # تبدیل تاریخچه به دیکشنری
            history_dict = self.model.history.history
            # ذخیره به فرمت JSON
            with open(self.history_path, 'w') as f:
                json.dump(history_dict, f)
            print(f"Training history saved to {self.history_path}")
