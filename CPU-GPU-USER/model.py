import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Attention,
    Dropout
)
# from google.colab import drive
from tensorflow.keras.optimizers import Adam # type: ignore
import os




# Step 1: Mount Google Drive
# drive.mount('/content/drive')

# Step 2: Define File Path
# file_path = '/content/drive/MyDrive/Colab Notebooks/پول/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv'
file_path = "C:\AAA\csv\EURUSD_Candlestick_1_Hour_ASK_01.01.2015-28.09.2024.csv"


# Step 3: Verify File Exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path and filename.")





# Step 4: Load Data
data = pd.read_csv(file_path)


# Step 5: Select Features and Target
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Close'].values


# Step 6: Standardize Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))


# Step 7: Create Sequences
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)


# Step 8: Split Data into Training and Testing Sets
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Step 9: Define the Model Using Functional API
input_layer = Input(shape=(time_steps, X_train.shape[2]))

# CNN Layer
conv = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(input_layer)
pool = MaxPooling1D(pool_size=1, padding='same')(conv)

# BiLSTM Layer
bilstm = Bidirectional(LSTM(64, return_sequences=True))(pool)

# Attention Layer
attention = Attention()([bilstm, bilstm])

# Optional Dropout Layer
dropout = Dropout(0.2)(attention)

# Flatten Layer
flatten = Flatten()(dropout)

# Output Layer
output = Dense(1)(flatten)

# Create the Model
model = Model(inputs=input_layer, outputs=output)

# Summary of the Model
model.summary()

# Step 10: Compile the Model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Step 11: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 12: Make Predictions
y_pred = model.predict(X_test)

# Step 13: Inverse Transform the Predictions and True Values
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Step 14: Print the Results
for i in range(len(y_test_rescaled)):
    print(f"Actual: {y_test_rescaled[i][0]}, Predicted: {y_pred_rescaled[i][0]}")
