import pandas as pd


df = pd.read_csv('../data/aapl_data.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
