# run.py
from MainClass import TimeSeriesModel

# تعریف تنظیمات برای هر بلوک
# block_configs = [
#     {'filters': 64, 'kernel_size': 5, 'pool_size': 3},  # به طور صریح kernel_size و pool_size مقداردهی شده‌اند
#     {'filters': 128, 'kernel_size': 4, 'pool_size': 2},  # kernel_size و pool_size خاصی تعیین شده‌اند
#     {'filters': 256, 'pool_size': 2},  # فقط pool_size تعیین شده است، مقدار پیش‌فرض kernel_size اعمال می‌شود (3)
#     {'filters': 512},  # نه kernel_size و نه pool_size تعیین نشده‌اند، مقدار پیش‌فرض هر دو اعمال می‌شود
# ]

block_configs = [
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 128, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 256, 'kernel_size': 5, 'pool_size': 2},
    {'filters': 512, 'kernel_size': 5, 'pool_size': 2},
    {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 1024, 'kernel_size': 3, 'pool_size': None}  # بدون MaxPooling در آخرین بلوک
]

# تنظیم پارامترهای مدل
time_series_model = TimeSeriesModel(
    file_path="C:/AAA/csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv", 
    base_dir="C:/AAA/",
    epochs=1, 
    batch_size=1024,
    block_configs=block_configs  # ارسال تنظیمات بلوک‌ها
)

# اجرای کل فرآیند آموزش و ارزیابی
time_series_model.run()














# # Run.py

# from MainClass import TimeSeriesModel

# # مسیر فایل CSV
# file_path = 'C:/AAA/csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv'

# # مسیر ذخیره تاریخچه‌ی آموزش
# history_path = 'C:/AAA/training_history.json'

# # مسیر ذخیره مدل
# model_save_path = 'C:/AAA/CNN_BiLSTM_Attention_Model.h5'

# # ایجاد نمونه از کلاس اصلی
# ts_model = TimeSeriesModel(file_path, history_path=history_path)

# # اجرای مدل
# ts_model.run()
