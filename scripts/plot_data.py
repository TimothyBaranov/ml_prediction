import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('../data/AAPL_data.csv')

# Преобразование столбца 'Date' в формат datetime
df['Date'] = pd.to_datetime(df['Date'])

# Установка столбца 'Date' в качестве индекса
df.set_index('Date', inplace=True)

# Построение графика цен закрытия
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
# Сохранение графика в файл
plt.savefig('../plots/apple_stock_close_price.png')
plt.show()
