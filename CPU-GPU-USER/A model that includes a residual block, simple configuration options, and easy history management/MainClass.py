import os
import datetime
import json
from ModelBuilder import ModelBuilder
from Trainer import Trainer
from DataLoader import DataLoader
from Evaluator import Evaluator
from HistoryManager import HistoryManager
import matplotlib.pyplot as plt

class TimeSeriesModel:
    """
    A class to encapsulate the entire pipeline for building, training, evaluating,
    and saving a time series prediction model.
    """

    def __init__(self, file_path, base_dir="C:/AAA/", epochs=1, batch_size=16, block_configs=None):
        """
        Initializes the TimeSeriesModel with the specified parameters.

        Args:
            file_path (str): Path to the input data file.
            base_dir (str, optional): Base directory to store all outputs. Defaults to "C:/AAA/".
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 16.
            block_configs (list, optional): Configuration for the residual blocks in the model.
        """
        self.file_path = file_path
        self.base_dir = base_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.block_configs = block_configs

        # ایجاد دایرکتوری‌ها اگر وجود نداشته باشند
        os.makedirs(self.base_dir, exist_ok=True)

        # ایجاد نام پوشه جدید برای هر اجرا
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        block_summary = "_".join([f"{cfg['filters']}f_{cfg.get('kernel_size', 3)}k" for cfg in block_configs])
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}_epochs{self.epochs}_batch{self.batch_size}_{block_summary}")
        os.makedirs(self.run_dir, exist_ok=True)

        # مسیرهای ذخیره‌سازی
        self.history_path = os.path.join(self.run_dir, 'training_history.json')
        self.model_save_path = os.path.join(self.run_dir, 'model.h5')
        self.loss_plot_path = os.path.join(self.run_dir, 'loss_plot.png')
        self.prediction_plot_path = os.path.join(self.run_dir, 'prediction_plot.png')
        self.error_distribution_plot_path = os.path.join(self.run_dir, 'error_distribution_plot.png')

        # Initialize components
        self.data_loader = DataLoader(file_path)
        self.model = None
        self.trainer = None
        self.evaluator = None

    def save_plot(self, fig, path):
        """
        ذخیره نمودار در مسیر مشخص شده.
        """
        if fig is not None:
            fig.savefig(path)
            plt.close(fig)
            print(f"Plot saved successfully to {path}.")
        else:
            print("Error: Figure is None and cannot be saved.")


    def save_hyperparameters(self):
        """
        ذخیره تنظیمات هایپرپارامترهای مدل و اطلاعات مربوط به معماری در فایل.
        """
        hyperparams = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'block_configs': self.block_configs
        }
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Hyperparameters saved to {os.path.join(self.run_dir, 'hyperparameters.json')}")

    def run(self):
        """
        Executes the full pipeline: data loading, model building, training, evaluation,
        and saving the trained model.
        """
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test, scaler_y = self.data_loader.get_data()
            print("Data loaded and preprocessed successfully.")

            # Save hyperparameters
            self.save_hyperparameters()

            # Build the model
            model_builder = ModelBuilder(time_steps=X_train.shape[1], num_features=X_train.shape[2], block_configs=self.block_configs)
            self.model = model_builder.build_model()
            self.model.summary()

            # Train the model
            self.trainer = Trainer(
                model=self.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=self.epochs,
                batch_size=self.batch_size,
                history_path=self.history_path
            )
            history = self.trainer.train()
            print("Model training completed.")

            # Manage training history
            history_manager = HistoryManager(self.history_path)
            history_manager.save_history(history)  # Save history to JSON file
            print(f"Training history saved to {self.history_path}.")

            # Evaluate the model
            self.evaluator = Evaluator(
                model=self.model,
                X_test=X_test,
                y_test=y_test,
                scaler_y=scaler_y,
                history_manager=history_manager
            )
            print("Starting model evaluation...")

            # Plot loss and save
            fig_loss = self.evaluator.plot_loss(history)
            self.save_plot(fig_loss, self.loss_plot_path)
            print(f"Loss plot saved to {self.loss_plot_path}.")

            # Make predictions and rescale results
            y_pred_rescaled, y_test_rescaled = self.evaluator.predict()
            print("Predictions made and rescaled.")

            # Calculate evaluation metrics
            mae, mse, rmse, r2 = self.evaluator.calculate_metrics()
            self.evaluator.print_metrics(mae, mse, rmse, r2)
            print("Evaluation metrics calculated and printed.")

            # Plot predictions and save
            fig_pred = self.evaluator.plot_predictions()
            self.save_plot(fig_pred, self.prediction_plot_path)
            print(f"Prediction plot saved to {self.prediction_plot_path}.")

            # Plot error distribution and save
            fig_error = self.evaluator.plot_error_distribution()
            self.save_plot(fig_error, self.error_distribution_plot_path)
            print(f"Error distribution plot saved to {self.error_distribution_plot_path}.")

            # Save the trained model
            self.model.save(self.model_save_path)
            print(f"Model saved successfully to {self.model_save_path}.")

        except Exception as e:
            print(f"An error occurred during the model pipeline execution: {e}")











