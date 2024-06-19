# scripts/arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('../data/AAPL_data.csv', index_col='Date', parse_dates=True)
close_prices = data['Close']

# Обучение модели ARIMA
model = ARIMA(close_prices, order=(5, 1, 0))  # Пример параметров ARIMA
model_fit = model.fit()

# Прогнозирование
forecast = model_fit.forecast(steps=30)  # Пример: прогноз на 30 дней вперёд

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(close_prices.index, close_prices, label='Actual')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecasting')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
