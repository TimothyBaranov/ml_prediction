import yfinance as yf
import pandas as pd

# Загрузка данных
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-12-31')

# Сохранение данных в CSV файл
data.to_csv('../data/AAPL_data.csv')
print(f'Data for {ticker} downloaded and saved to data/AAPL_data.csv')
