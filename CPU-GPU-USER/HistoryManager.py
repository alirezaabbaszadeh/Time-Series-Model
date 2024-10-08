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
        # Check if the history file exists
        if not os.path.exists(self.history_path):
            raise FileNotFoundError(f"The history file at {self.history_path} was not found.")
        
        # Open and load the JSON history file
        with open(self.history_path, 'r') as f:
            self.history = json.load(f)
        
        return self.history

    def plot_history(self):
        """
        Plots the training and validation loss over epochs.

        This method assumes that the history contains 'loss' and 'val_loss' keys.
        It visualizes the training process by plotting these metrics.
        """
        # Load history if it hasn't been loaded yet
        if not self.history:
            self.load_history()
        
        # Create a new figure with specified size
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.plot(self.history['loss'], label='Train Loss', color='blue')
        
        # Plot validation loss
        plt.plot(self.history['val_loss'], label='Validation Loss', color='orange')
        
        # Set the title of the plot
        plt.title('Model Loss During Training')
        
        # Label for the y-axis
        plt.ylabel('Mean Absolute Error (MAE)')
        
        # Label for the x-axis
        plt.xlabel('Epoch')
        
        # Display legend in the upper right corner
        plt.legend(loc='upper right')
        
        # Enable grid for better readability
        plt.grid(True)
        
        # Display the plot
        plt.show()
