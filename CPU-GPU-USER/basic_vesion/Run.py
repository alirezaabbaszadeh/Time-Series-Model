# Run.py

from MainClass import TimeSeriesModel

# مسیر فایل CSV
file_path = 'C:/AAA/basic_vesion/csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv'

# مسیر ذخیره تاریخچه‌ی آموزش
history_path = 'C:/AAA/basic_vesion/training_history.json'

# مسیر ذخیره مدل
model_save_path = 'C:/AAA/basic_vesion/CNN_BiLSTM_Attention_Model.h5'

# ایجاد نمونه از کلاس اصلی
ts_model = TimeSeriesModel(file_path, history_path=history_path)

# اجرای مدل
ts_model.run()
