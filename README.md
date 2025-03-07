# IDS
Analyzing and Forecasting International Debt Statistics

### Project Overview
This project specifically examines external debt owed to the World Bank by various debtor countries, which accounts for a significant portion of global sovereign debt. It focuses on analyzing and forecasting international debt patterns using machine learning techniques, specifically examining the relationship between time (years) and debt amounts. The study employs exploratory data analysis and time series analysis to understand historical debt trajectories and develop predictive models for future debt levels based on historical year-to-year debt variations. We aim to identify patterns in debt accumulation and create accurate forecasting models based on historical year-to-year debt variations. The analysis concentrates on evaluating the accuracy of different machine learning approaches in debt forecasting. The insights from this study can aid in the development of more effective debt restructuring frameworks, contributing to financial stability and equitable development.

### Methodology
1. Data Collection

   Source - World Bank

   Attributes - year(1970-2023), creditor(300), debtor(134), indicator(572), debt
   
2. Data Preprocessing

   Handling missing values

   Outlier detection

   Stationarity check
   
3. Model Selection

   a. Statistical model - ARIMA

   b. ML model - Random Forest

   c. DL model - LSTM

   d. Hybrid model - ARIMA + LSTM

4. Model Training

   Data Splitting into training and testing datasets

   Model fitting using training dataset and evaluation using testing dataset 

5. Model Evalution

   MAE

   MSE

   RMSE

   MAPE

   R^2 Score

6. Forecasting

    Prediction of debt using appropriate model
