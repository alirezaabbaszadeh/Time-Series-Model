from ModelBuilder import ModelBuilder
from Trainer import Trainer
from DataLoader import DataLoader
from Evaluator import Evaluator
from HistoryManager import HistoryManager

class TimeSeriesModel:
    """
    A class to encapsulate the entire pipeline for building, training, evaluating,
    and saving a time series prediction model.

    Attributes:
        file_path (str): Path to the input data file.
        data_loader (DataLoader): Instance responsible for loading and preprocessing data.
        model (tf.keras.Model): The machine learning model.
        trainer (Trainer): Instance responsible for training the model.
        evaluator (Evaluator): Instance responsible for evaluating the model.
        history_path (str): Path to store the training history.
    """

    def __init__(self, file_path, history_path="C:/AAA/training_history.json"):
        """
        Initializes the TimeSeriesModel with the specified data file path and history path.

        Args:
            file_path (str): Path to the input data file.
            history_path (str, optional): Path to store the training history. Defaults to "C:/AAA/training_history.json".
        """
        self.file_path = file_path
        self.data_loader = DataLoader(file_path)  # Initialize DataLoader with the given file path
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.history_path = history_path  # مسیر ذخیره تاریخچه

    def run(self):
        """
        Executes the full pipeline: data loading, model building, training, evaluation,
        and saving the trained model.
        """
        # بارگذاری و پیش‌پردازش داده‌ها (Load and preprocess data)
        X_train, X_test, y_train, y_test, scaler_y = self.data_loader.get_data()
        print("Data loaded and preprocessed successfully.")

        # ساخت مدل (Build the model)
        model_builder = ModelBuilder(time_steps=X_train.shape[1], num_features=X_train.shape[2])
        self.model = model_builder.build_model()
        self.model.summary()  # نمایش خلاصه مدل

        # آموزش مدل با مشخص کردن مسیر ذخیره تاریخچه (Train the model and specify history path)
        self.trainer = Trainer(
            model=self.model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=1,  # می‌توانید تعداد اپک‌ها را افزایش دهید
            batch_size=256,
            history_path=self.history_path  # ارسال مسیر به Trainer
        )
        history = self.trainer.train()
        print("Model training completed.")

        # مدیریت تاریخچه (Manage training history)
        history_manager = HistoryManager(self.history_path)
        print("Training history managed.")

        # ارزیابی مدل (Evaluate the model)
        self.evaluator = Evaluator(
            model=self.model,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y,
            history_manager=history_manager  # ارسال HistoryManager به Evaluator
        )
        print("Starting model evaluation...")

        self.evaluator.plot_loss(history)  # نمایش نمودار loss
        print("Loss plot generated.")

        # پیش‌بینی و مقیاس‌بندی مجدد نتایج (Predict and rescale results)
        y_pred_rescaled, y_test_rescaled = self.evaluator.predict()
        print("Predictions made and rescaled.")

        # محاسبه معیارهای ارزیابی (Calculate evaluation metrics)
        mae, mse, rmse, r2 = self.evaluator.calculate_metrics()
        self.evaluator.print_metrics(mae, mse, rmse, r2)
        print("Evaluation metrics calculated and printed.")

        # نمایش نمودارهای پیش‌بینی و توزیع خطا (Plot prediction and error distribution)
        self.evaluator.plot_predictions()
        self.evaluator.plot_error_distribution()
        print("Prediction and error distribution plots generated.")

        # چاپ پیش‌بینی‌ها (Print predictions)
        self.evaluator.print_predictions()
        print("Predictions printed.")

        # ذخیره مدل (Save the trained model)
        model_save_path = 'C:/AAA/CNN_BiLSTM_Attention_Model.h5'
        self.model.save(model_save_path)
        print(f"Model saved successfully to {model_save_path}.")
