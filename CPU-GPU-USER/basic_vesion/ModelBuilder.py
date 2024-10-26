from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    Attention,
    LeakyReLU
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2
from BahdanauAttention import BahdanauAttention 
from tensorflow.keras.layers import BatchNormalization

class ModelBuilder:
    def __init__(self, time_steps, num_features):
        """
        Constructor for the ModelBuilder class.

        Parameters:
        - time_steps (int): Number of time steps in the input data.
        - num_features (int): Number of features in the input data.
        """
        self.time_steps = time_steps
        self.num_features = num_features

    def build_model(self):
        """
        Builds a CNN-BiLSTM-Attention model.

        Returns:
        - model (Model): The constructed and compiled Keras model.
        """
        # Define the input layer with the specified shape
        input_layer = Input(shape=(self.time_steps, self.num_features))
        
        # Convolutional layers to extract local patterns
        x = input_layer
        # Layer 1: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  # Normalize after activation for stable training
        x = MaxPooling1D(pool_size=25, padding='same')(x)

        # Layer 2: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=5, padding='same')(x)

        # Layer 3: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=5, padding='same')(x)

        # Layer 4: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=512, kernel_size=3, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=5, padding='same')(x)

        # Layer 5: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=1024, kernel_size=3, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=10, padding='same')(x)

        # Layer 6: Conv1D + LeakyReLU + BatchNormalization
        x = Conv1D(filters=2048, kernel_size=3, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        # Bidirectional LSTM layers to capture dependencies in both directions
        for _ in range(4):  # Number of LSTM layers between 2 to 3
            x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(x)
        
        # Attention mechanism to focus on important parts of the sequence
        # پیاده‌سازی Bahdanau Attention
        attention = BahdanauAttention(units=32)  # تعداد واحدها باید متناسب با خروجی BiLSTM باشد
        context_vector, attention_weights = attention(x, x)
        
        # Flatten and Dropout layers
        x = Flatten()(context_vector)
        x = Dropout(0.5)(x)  # Dropout to prevent overfitting
        
        # Output layer with a single neuron for regression with L2 Regularization
        output = Dense(1, kernel_regularizer=l2(0.01))(x)
        
        # Create the model by specifying inputs and outputs
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile the model using Adam optimizer and Mean Squared Error loss
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        return model









