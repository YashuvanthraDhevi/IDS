import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objs as go

def prepare_data(file_path):
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load and preprocess the data
    data = pd.read_excel(file_path)
    time_series = data[['year', 'debt']].set_index('year')
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series)
    
    return time_series, scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_hybrid_model(scaled_data, seq_length=10, alpha=0.8):
    # Prepare sequences
    X, y = create_sequences(scaled_data, seq_length)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # LSTM Model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=16, epochs=500, validation_data=(X_test, y_test), verbose=0)
    
    return model, X_test, y_test

def evaluate_hybrid_model(model, X_test, y_test, time_series, scaler, alpha=0.8):
    # Make predictions
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    # ARIMA Model
    arima_model = ARIMA(time_series, order=(1,2,2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(y_test))
    arima_forecast = arima_forecast.values.reshape(-1, 1)

    # Combine LSTM and ARIMA predictions
    hybrid_predictions = alpha * test_predictions + (1 - alpha) * arima_forecast

    # Evaluation metrics
    metrics = {
        'MAE': mean_absolute_error(y_test_actual, hybrid_predictions),
        'MSE': mean_squared_error(y_test_actual, hybrid_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test_actual, hybrid_predictions)),
        'MAPE': np.mean(np.abs((y_test_actual - hybrid_predictions) / y_test_actual)) * 100
    }
    
    return metrics, arima_fit, hybrid_predictions

def forecast_future(model, X_test, time_series, scaler, arima_fit, future_steps=10, alpha=0.8):
    # LSTM Forecast
    lstm_input = X_test[-1].reshape(1, X_test.shape[1], 1)
    lstm_forecast = []
    for _ in range(future_steps):
        pred = model.predict(lstm_input, verbose=0)
        lstm_forecast.append(pred[0, 0])
        # Update input sequence with the new prediction
        lstm_input = np.append(lstm_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    lstm_forecast = np.array(lstm_forecast).reshape(-1, 1)
    lstm_forecast = scaler.inverse_transform(lstm_forecast)

    # ARIMA Forecast
    arima_future_forecast = arima_fit.forecast(steps=future_steps)
    arima_future_forecast = arima_future_forecast.values.reshape(-1, 1)

    # Hybrid Forecast (Weighted Combination)
    hybrid_forecast = alpha * lstm_forecast + (1 - alpha) * arima_future_forecast

    # Prepare forecast dataframe
    future_years = np.arange(time_series.index[-1] + 1, time_series.index[-1] + future_steps + 1)
    forecast_df = pd.DataFrame({'Year': future_years, 'Hybrid_Forecast': hybrid_forecast.flatten()})
    
    return forecast_df

def load_and_process_data(file_path):
    # Comprehensive function to load, process, train, and forecast
    time_series, scaled_data, scaler = prepare_data(file_path)
    model, X_test, y_test = train_hybrid_model(scaled_data)
    metrics, arima_fit, hybrid_predictions = evaluate_hybrid_model(model, X_test, y_test, time_series, scaler)
    forecast_df = forecast_future(model, X_test, time_series, scaler, arima_fit)
    
    return time_series, metrics, forecast_df

def main():
    st.title('Debt Forecasting Dashboard')
    
    # Sidebar for file upload
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open('temp_uploaded_file.xlsx', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Process data
        try:
            time_series, metrics, forecast_df = load_and_process_data('temp_uploaded_file.xlsx')
            
            # Display Metrics
            st.header('Model Performance Metrics')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('MAE', f'{metrics["MAE"]:.4f}')
            with col2:
                st.metric('MSE', f'{metrics["MSE"]:.4f}')
            with col3:
                st.metric('RMSE', f'{metrics["RMSE"]:.4f}')
            with col4:
                st.metric('MAPE', f'{metrics["MAPE"]:.4f}%')
            
            # Historical and Forecast Visualization
            st.header('Debt Forecast Visualization')
            
            # Plotly Interactive Line Chart
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=time_series.index, 
                y=time_series['debt'], 
                mode='lines', 
                name='Historical Debt',
                line=dict(color='blue')
            ))
            
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'], 
                y=forecast_df['Hybrid_Forecast'], 
                mode='lines', 
                name='Forecasted Debt',
                line=dict(color='red', dash='dot')
            ))
            
            fig.update_layout(
                title='Debt Forecast: Historical and Projected',
                xaxis_title='Year',
                yaxis_title='Debt',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Table
            st.header('Future Debt Forecast')
            st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing the file: {e}")

if __name__ == '__main__':
    main()