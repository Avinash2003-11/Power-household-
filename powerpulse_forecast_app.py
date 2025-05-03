
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("AIML_PowerPulse: Household Energy Usage Forecast")
st.write("Predict future household energy consumption using machine learning.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("household_energy_usage.csv", parse_dates=['Date'])
    return data

data = load_data()
st.line_chart(data.set_index('Date'))

# Feature Engineering
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
X = data[['DayOfWeek', 'Month']]
y = data['Energy_kWh']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")

# Forecast for the next 7 days
future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=7)
future_df = pd.DataFrame({
    'Date': future_dates,
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month
})

future_X = future_df[['DayOfWeek', 'Month']]
future_df['Predicted_Energy_kWh'] = model.predict(future_X)

st.subheader("Forecast for Next 7 Days")
st.dataframe(future_df[['Date', 'Predicted_Energy_kWh']])

# Plot forecast
fig, ax = plt.subplots()
ax.plot(future_df['Date'], future_df['Predicted_Energy_kWh'], marker='o')
ax.set_title('Forecasted Energy Usage')
ax.set_xlabel('Date')
ax.set_ylabel('Energy (kWh)')
st.pyplot(fig)
