import tensorflow as tf
from tensorflow.keras.layers import Layer

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: output of the LSTM (hidden state)
        # values: all hidden states from the BiLSTM
        
        # Expand query to match values' dimensions
        query_with_time_axis = tf.expand_dims(query, 1)

        # Score function (using learned weights W1, W2, and V)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Compute the attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute the context vector as a weighted sum of the values
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
