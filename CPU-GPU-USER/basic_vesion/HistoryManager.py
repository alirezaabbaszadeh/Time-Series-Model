import json
import os
import matplotlib.pyplot as plt

class HistoryManager:
    """
    A class to manage and visualize the training history of a machine learning model.

    Attributes:
        history_path (str): The file path to the JSON file containing training history.
        history (dict): A dictionary to store the loaded training history.
    """

    def __init__(self, history_path):
        """
        Initializes the HistoryManager with the path to the history file.

        Args:
            history_path (str): The file path to the JSON file containing training history.
        """
        self.history_path = history_path
        self.history = None


    def load_history(self):
        """
        Loads the training history from a JSON file.

        Raises:
            FileNotFoundError: If the history file does not exist at the specified path.

        Returns:
            dict: The loaded training history.
        """
        if not os.path.exists(self.history_path):
            raise FileNotFoundError(f"The history file at {self.history_path} was not found.")
        
        with open(self.history_path, 'r') as f:
            self.history = json.load(f)
        
        return self.history

    def plot_history(self):
        """
        Plots the training and validation loss over epochs.
        """
        if not self.history:
            self.load_history()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['loss'], label='Train Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Model Loss During Training')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
