import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from typing import Optional, Tuple

class Evaluator:
    """
    Evaluator class for assessing the performance of a machine learning model.
    
    Attributes:
        model: The trained machine learning model to evaluate.
        X_test (array-like): Test dataset features.
        y_test (array-like): True labels for the test dataset.
        scaler_y: Scaler object used to inverse transform the predictions and true labels.
        history_manager: Optional object to manage training history.
        y_pred (array-like): Predicted values by the model.
        y_pred_rescaled (array-like): Rescaled predicted values.
        y_test_rescaled (array-like): Rescaled true labels.
    """
    
    def __init__(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        scaler_y, 
        history_manager: Optional[object] = None
    ):
        """
        Initializes the Evaluator with the necessary components.
        
        Args:
            model: Trained machine learning model.
            X_test (np.ndarray): Features of the test dataset.
            y_test (np.ndarray): True labels of the test dataset.
            scaler_y: Scaler used to inverse transform the target variable.
            history_manager: Optional manager for training history.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.y_pred = None
        self.y_pred_rescaled = None
        self.y_test_rescaled = None
        self.history_manager = history_manager  # Manages training history

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates predictions using the model and rescales them to the original scale.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Rescaled predicted values and rescaled true labels.
        """
        # Generate predictions using the model
        self.y_pred = self.model.predict(self.X_test)
        
        # Inverse transform the scaled predictions to original scale
        self.y_pred_rescaled = self.scaler_y.inverse_transform(self.y_pred)
        
        # Inverse transform the scaled true labels to original scale
        self.y_test_rescaled = self.scaler_y.inverse_transform(self.y_test)
        
        return self.y_pred_rescaled, self.y_test_rescaled

    def calculate_metrics(self) -> Tuple[float, float, float, float]:
        """
        Calculates evaluation metrics for the model's predictions.
        
        Returns:
            Tuple[float, float, float, float]: MAE, MSE, RMSE, and R² score.
        """
        mae = mean_absolute_error(self.y_test_rescaled, self.y_pred_rescaled)
        mse = mean_squared_error(self.y_test_rescaled, self.y_pred_rescaled)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test_rescaled, self.y_pred_rescaled)
        return mae, mse, rmse, r2

    def plot_loss(self, history: Optional[object] = None) -> None:
        """
        Plots the training and validation loss over epochs.
        
        Args:
            history: Optional history object containing loss information.
                     If not provided, it uses the history_manager if available.
        """
        if history:
            # Plot loss from the provided history
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Train Loss', color='blue')
            plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
            plt.title('Model Loss During Training')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
        elif self.history_manager:
            # Plot loss using the history_manager's stored history
            self.history_manager.plot_history()
        else:
            print("No history available to plot.")

    def plot_predictions(self) -> None:
        """
        Plots the actual vs predicted close prices.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.y_test_rescaled, label='Actual Close Price', color='blue')
        plt.plot(self.y_pred_rescaled, label='Predicted Close Price', color='red', alpha=0.7)
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Sample')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_error_distribution(self) -> None:
        """
        Plots the distribution of prediction errors.
        """
        # Calculate prediction errors
        errors = self.y_test_rescaled - self.y_pred_rescaled
        
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50, color='purple', edgecolor='black')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def print_metrics(self, mae: float, mse: float, rmse: float, r2: float) -> None:
        """
        Prints the calculated evaluation metrics.
        
        Args:
            mae (float): Mean Absolute Error.
            mse (float): Mean Squared Error.
            rmse (float): Root Mean Squared Error.
            r2 (float): R² Score.
        """
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R² Score: {r2}")

    def print_predictions(self, num_samples: int = 10) -> None:
        """
        Prints a specified number of actual vs predicted values.
        
        Args:
            num_samples (int): Number of samples to print. Defaults to 10.
        """
        for i in range(min(num_samples, len(self.y_test_rescaled))):
            actual = self.y_test_rescaled[i][0] if isinstance(self.y_test_rescaled[i], (list, np.ndarray)) else self.y_test_rescaled[i]
            predicted = self.y_pred_rescaled[i][0] if isinstance(self.y_pred_rescaled[i], (list, np.ndarray)) else self.y_pred_rescaled[i]
            print(f"Sample {i+1}: Actual = {actual}, Predicted = {predicted}")
