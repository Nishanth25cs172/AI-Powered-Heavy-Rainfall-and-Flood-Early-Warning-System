import pandas as pd

df = pd.read_csv('data/Daily Rainfall Data - India (2009-2024).csv')
print('=== RAINFALL DATASET INSPECTION ===')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('\nData types:\n', df.dtypes)
print('\nMissing values:\n', df.isnull().sum())
print('\nFirst 5 rows:\n', df.head(5))
print('\nUnique states ({} total):\n{}'.format(df['state_name'].nunique(), df['state_name'].unique().tolist()))
print('\nDate min/max:', df['date'].min(), 'to', df['date'].max())
print('\nActual rainfall summary:\n', df['actual'].describe())
print('\nNormal rainfall summary:\n', df['normal'].describe())
print('\nDeparture summary:\n', df['deviation'].describe())
