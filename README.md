# Прогнозирование цен на акции с использованием ARIMA и LSTM

Этот проект реализует модели ARIMA и LSTM для прогнозирования цен на акции компании Apple (AAPL).

## Структура проекта

ml_stock_prediction/
│
├── data/
│ └── AAPL_data.csv # Файл с данными о ценах акций
│
├── scripts/
│ ├── download_data.py # Скрипт для загрузки данных
│ ├── arima_model.py # Скрипт для модели ARIMA
│ └── lstm_model.py # Скрипт для модели LSTM
│
├── .gitignore # Файл для исключений Git
├── requirements.txt # Файл с зависимостями
└── README.md # Описание проекта

## Установка

1. Клонируйте репозиторий:
#репризоторий + url  
cd ml_stock_prediction

2. Установите зависимости:

pip install -r requirements.txt


## Запуск

### Загрузка данных

Запустите скрипт для загрузки данных:

python scripts/download_data.py

### ARIMA модель

Запустите скрипт для модели ARIMA:

python scripts/arima_model.py

## Дополнительные комментарии

- Пожалуйста, убедитесь, что у вас установлены все необходимые зависимости перед запуском скриптов.
# ml_prediction
