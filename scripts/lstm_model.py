# scripts/lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('../data/AAPL_data.csv', index_col='Date', parse_dates=True)
close_prices = data['Close'].values.reshape(-1, 1)

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Разделение на обучающий и тестовый наборы
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size,
                                    :], scaled_data[train_size:len(scaled_data), :]

# Функция для создания датасета


def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# Создание датасета для обучения и тестирования
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Изменение формы для LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Модель LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

# Прогнозирование
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Инвертирование нормализации для получения исходных значений
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Расчет MSE
mse_train = np.mean(
    np.square(train_data[time_step:len(train_predict)+time_step] - train_predict))
mse_test = np.mean(
    np.square(test_data[time_step:len(test_predict)+time_step] - test_predict))

print(f'Train Mean Squared Error: {mse_train}')
print(f'Test Mean Squared Error: {mse_test}')

# Визуализация результатов
train = data[:train_size]
valid = data[train_size:]
valid['Predictions'] = np.nan
valid['Predictions'].iloc[time_step:len(
    test_predict)+time_step] = test_predict[:, 0]

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()
