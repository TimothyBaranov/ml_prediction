import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Загрузка данных


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Подготовка данных для модели LSTM


def prepare_data(df, window_size):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Построение и обучение модели LSTM


def build_lstm_model(X_train, y_train, X_test, y_test, scaler):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[es])

    return model, history

# Прогнозирование будущих значений


def predict_future(model, df, scaler, window_size):
    # Последние данные для начального окна
    last_window = df['Close'].values[-window_size:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)
    X_input = last_window_scaled.reshape(1, window_size, 1)

    future_predictions = []
    months_to_predict = 120  # Прогноз на 120 месяцев (10 лет)

    for _ in range(months_to_predict):
        next_pred_scaled = model.predict(X_input)
        future_predictions.append(next_pred_scaled[0, 0])
        X_input = np.roll(X_input, -1)
        X_input[-1, -1, 0] = next_pred_scaled[0, 0]

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1))

    return future_predictions.flatten()


# Основная функция


def main():
    # Загрузка данных
    file_path = './data/aapl_data.csv'
    df = load_data(file_path)

    # Подготовка данных
    window_size = 60  # Размер окна для LSTM
    X, y, scaler = prepare_data(df, window_size)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Построение и обучение модели LSTM
    model, history = build_lstm_model(X_train, y_train, X_test, y_test, scaler)

    # Визуализация обучения
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Прогнозирование будущих значений
    future_predictions = predict_future(model, df, scaler, window_size)

    # Вывод результатов
    print("Прогноз на следующие 12 месяцев:")
    print(future_predictions)


if __name__ == "__main__":
    main()
