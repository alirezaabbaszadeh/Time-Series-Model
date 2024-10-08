from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    Add,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from BahdanauAttention import BahdanauAttention  # فرض بر این است که این کلاس پیاده‌سازی شده است

class ModelBuilder:
    def __init__(self, time_steps, num_features, block_configs):
        """
        سازنده کلاس ModelBuilder.

        پارامترها:
        - time_steps (int): تعداد تایم‌استپ‌ها در داده ورودی.
        - num_features (int): تعداد ویژگی‌ها در داده ورودی.
        - block_configs (list of dict): تنظیمات بلوک‌ها شامل تعداد فیلتر، کرنل سایز و پولینگ سایز.
        """
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs

    def residual_block(self, x, filters, kernel_size, pool_size, block_num):
        """
        یک بلوک باقیمانده با قابلیت تنظیم تعداد فیلترها، کرنل سایز و پولینگ سایز.

        پارامترها:
        - x: ورودی به بلوک.
        - filters: تعداد فیلترها.
        - kernel_size: اندازه کرنل.
        - pool_size: اندازه پولینگ.
        - block_num: شماره بلاک برای نام‌گذاری.

        بازگشت:
        - x: خروجی پس از اعمال لایه‌های باقیمانده.
        """
        shortcut = x
        # اولین لایه Conv1D
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv1_{block_num}')(x)
        x = BatchNormalization(name=f'bn1_{block_num}')(x)
        x = LeakyReLU(name=f'leaky_relu1_{block_num}')(x)
        
        # دومین لایه Conv1D
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv2_{block_num}')(x)
        x = BatchNormalization(name=f'bn2_{block_num}')(x)
        
        # بررسی نیاز به تطبیق ابعاد shortcut
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', name=f'shortcut_conv_{block_num}')(shortcut)
            shortcut = BatchNormalization(name=f'shortcut_bn_{block_num}')(shortcut)
        
        # اتصال باقیمانده
        x = Add(name=f'add_{block_num}')([shortcut, x])
        x = LeakyReLU(name=f'leaky_relu2_{block_num}')(x)
        
        # اگر pool_size تعریف شده باشد، لایه MaxPooling1D را اعمال می‌کنیم
        if pool_size:
            x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'max_pool_{block_num}')(x)
        
        return x

    def build_model(self):
        """
        ساخت مدل CNN-BiLSTM-Attention.

        بازگشت:
        - model (Model): مدل Keras ساخته شده و کامپایل شده.
        """
        # لایه ورودی
        input_layer = Input(shape=(self.time_steps, self.num_features))
        x = input_layer

        # ایجاد بلوک‌های باقیمانده با تنظیمات انعطاف‌پذیر
        for i, config in enumerate(self.block_configs, 1):
            filters = config['filters']
            kernel_size = config.get('kernel_size', 3)  # به صورت پیش‌فرض 3
            pool_size = config.get('pool_size', 2)      # به صورت پیش‌فرض 2
            x = self.residual_block(x, filters, kernel_size, pool_size, i)

        # ادامه مدل همانند قبل
        # لایه‌های BiLSTM
        for _ in range(4):
            x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)

        # مکانیزم توجه
        attention = BahdanauAttention(units=512)
        context_vector, attention_weights = attention(x, x)

        # Flatten و Dropout
        x = Flatten()(context_vector)
        x = Dropout(0.2)(x)

        # لایه خروجی با یک نورون (برای رگرسیون) با منظم‌سازی L2
        output = Dense(1, kernel_regularizer=l2(0.0001))(x)

        # ساخت مدل با مشخص کردن ورودی‌ها و خروجی‌ها
        model = Model(inputs=input_layer, outputs=output)

        # کامپایل مدل با استفاده از Adam optimizer و Mean Squared Error loss
        optimizer = AdamW(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        return model








# from tensorflow.keras.models import Model  # type: ignore
# from tensorflow.keras.layers import (  # type: ignore
#     Input,
#     Conv1D,
#     MaxPooling1D,
#     Bidirectional,
#     LSTM,
#     Dense,
#     Flatten,
#     Dropout,
#     Attention,
#     LeakyReLU
# )
# from tensorflow.keras.optimizers import Adam  # type: ignore
# from tensorflow.keras.regularizers import l2
# from BahdanauAttention import BahdanauAttention 
# class ModelBuilder:
#     def __init__(self, time_steps, num_features):
#         """
#         Constructor for the ModelBuilder class.

#         Parameters:
#         - time_steps (int): Number of time steps in the input data.
#         - num_features (int): Number of features in the input data.
#         """
#         self.time_steps = time_steps
#         self.num_features = num_features

#     def build_model(self):
#         """
#         Builds a CNN-BiLSTM-Attention model.

#         Returns:
#         - model (Model): The constructed and compiled Keras model.
#         """
#         # Define the input layer with the specified shape
#         input_layer = Input(shape=(self.time_steps, self.num_features))
        
#         # Convolutional layers to extract local patterns
#         x = input_layer
#         # Layer 1: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
#         x = Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)  # Normalize after activation for stable training
#         x = MaxPooling1D(pool_size=1, padding='same')(x)

#         # Layer 2: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
#         x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)
#         x = MaxPooling1D(pool_size=1, padding='same')(x)

#         # Layer 3: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
#         x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)
#         x = MaxPooling1D(pool_size=1, padding='same')(x)

#         # Layer 4: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
#         x = Conv1D(filters=512, kernel_size=3, padding='same')(x)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)
#         x = MaxPooling1D(pool_size=1, padding='same')(x)

#         # Layer 5: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
#         x = Conv1D(filters=1024, kernel_size=3, padding='same')(x)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)
#         x = MaxPooling1D(pool_size=2, padding='same')(x)

#         # Layer 6: Conv1D + LeakyReLU + BatchNormalization
#         x = Conv1D(filters=2048, kernel_size=3, padding='same')(x)
#         x = LeakyReLU()(x)
#         x = BatchNormalization()(x)
        
#         # Bidirectional LSTM layers to capture dependencies in both directions
#         for _ in range(4):  # Number of LSTM layers between 2 to 3
#             x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.4, recurrent_dropout=0.4))(x)
        
#         # Attention mechanism to focus on important parts of the sequence
#         # پیاده‌سازی Bahdanau Attention
#         attention = BahdanauAttention(units=512)  # تعداد واحدها باید متناسب با خروجی BiLSTM باشد
#         context_vector, attention_weights = attention(x, x)
        
#         # Flatten and Dropout layers
#         x = Flatten()(context_vector)
#         x = Dropout(0.5)(x)  # Dropout to prevent overfitting
        
#         # Output layer with a single neuron for regression with L2 Regularization
#         output = Dense(1, kernel_regularizer=l2(0.0001))(x)
        
#         # Create the model by specifying inputs and outputs
#         model = Model(inputs=input_layer, outputs=output)
        
#         # Compile the model using Adam optimizer and Mean Squared Error loss
#         optimizer = Adam(learning_rate=0.0001)
#         model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
#         return model









