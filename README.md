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

```bash
git clone <repository_url>
cd ml_stock_prediction
```
Установите зависимости:
```
 install -r requirements.txt
```
Запуск
Загрузка данных
Для загрузки данных выполните следующую команду:

```bash
python3 scripts/download_data.py
```
ARIMA модель
Для запуска модели ARIMA выполните:
```bash
python3 scripts/arima_model.py
```
LSTM модель
Для запуска модели LSTM выполните:
```
python3 scripts/lstm_model.py
```
Дополнительные комментарии
Убедитесь, что все необходимые зависимости установлены перед запуском скриптов.

```markdown
### Пояснения
- В Windows пути к файлам и папкам указываются с использованием обратных слэшей (`\`), 
- Вместо `pip install -r requirements.txt` используется `pip3 install -r requirements.txt`.
- Все команды для запуска скриптов указаны с использованием `python3`.
```
