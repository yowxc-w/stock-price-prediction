# Stock Price Prediction

Prediction of the S&P500 by stationarize it with moving average.

(Important Note: it is not to forecast the actual closing price)

## Data Charateristics
Continuous manner  
One Feature Only  
Non-Stationary  
No-Well Defined Range

## Model Selection
1. Naive Forecast Model (Random Walk)
2. Linear Regression Model
3. Auto Regression Model
4. LSTM Model
5. SVR Model

## Time-Series Forecasting Approaches
1. Direct Multi-step Forecast  
Build one model for each steps 
```python
prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model2(obs(t-2), obs(t-3), ..., obs(t-n))
``` 
2. Recursive Multi-step Forecast  
Use the remaining value plus the predicted value to predict the next one step.  
```python
prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model(prediction(t+1), obs(t-1), ..., obs(t-n))
```
3. Direct-Recursive Hybrid Forecast  
The output from the first model is used as an input for the second model. 
```python
prediction(t+1) = model1(obs(t-1), obs(t-2), ..., obs(t-n))
prediction(t+2) = model2(prediction(t+1), obs(t-1), ..., obs(t-n))
``` 
4. Multiple Output Forecast  
Only if the forecasting module is able to do so.
```python
prediction(t+1), prediction(t+2) = model(obs(t-1), obs(t-2), ..., obs(t-n))
```
