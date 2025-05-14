import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('C:\Bahan\Komputasi\\dataAi.csv')

# Split data
df_train, df_test = train_test_split(df, shuffle=False)

# Create and fit model
m = Prophet()
m.fit(df_train)

# Make future dataframe
future = m.make_future_dataframe(periods=730)

# Predict
forecast = m.predict(future)

# Evaluate
forecast_df = forecast[['ds', 'yhat']]
actual = df_test['y']

# Plot hasil
m.plot(forecast)
m.plot_components(forecast)