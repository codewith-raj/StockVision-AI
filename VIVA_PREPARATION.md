# Viva Preparation

## Questions and answers

1. **What is deep learning?** A machine learning approach using multilayer neural networks to learn representations from data.
2. **What is time-series forecasting?** Predicting future values from time-ordered observations.
3. **What is a GRU?** A recurrent neural network with update and reset gates that control information flow through its hidden state.
4. **Why GRU instead of a basic RNN?** Gates provide a mechanism to retain or discard information, helping with longer dependencies and vanishing gradients.
5. **GRU versus LSTM?** GRU combines memory control into update/reset gates and has no separate cell state; LSTM uses input, forget and output gates plus a cell state.
6. **What is the update gate?** It controls how much previous hidden state is retained in the new state.
7. **What is the reset gate?** It controls how much previous state contributes to the candidate state.
8. **What is a lookback window?** The fixed number of preceding observations supplied for one prediction; here it is 60 trading observations.
9. **Why split chronologically?** Future observations must remain unseen when training on past observations; random splits can leak temporal information.
10. **What is data leakage?** Information unavailable at prediction time influences training or evaluation.
11. **What is MinMaxScaler?** A transformation mapping feature values to a chosen range based on observed minima and maxima.
12. **Why fit scaling only on training data?** Validation/test statistics must not influence model preparation.
13. **What is MSE?** The mean of squared prediction errors; it penalizes large errors more heavily.
14. **What is MAE?** The mean absolute prediction error, expressed here in dollars.
15. **What is RMSE?** The square root of MSE, in the target's units.
16. **What is MAPE?** Mean absolute error divided by actual magnitude, expressed as a percentage; it needs care around zero targets.
17. **What is dropout?** A regularization method that randomly disables some activations during training.
18. **What is Adam?** An adaptive gradient-based optimizer using running estimates of gradient moments.
19. **What is early stopping?** Ending training when validation performance stops improving and optionally restoring the best weights.
20. **What is overfitting?** Learning training-specific patterns that fail to generalize to unseen observations.
21. **What is a naive baseline?** A simple reference predictor; here tomorrow's close is predicted as today's close.
22. **What are limitations of stock prediction?** Prices are noisy and affected by unmodeled events, structural changes, and information absent from OHLCV.
23. **How does this model predict?** It receives the previous 60 days of five OHLCV features and outputs one scaled next-day Close estimate, which is inverse-scaled to dollars.
24. **What is the input shape?** `(batch, 60, 5)`: batches of sequences, 60 time steps, five features per step.
25. **How is it evaluated?** On later held-out dates with MAE, RMSE, MSE and MAPE, compared to persistence on the same dates.
26. **What are the findings?** After 19 epochs the GRU's test MAE/RMSE/MAPE were $45.56/$52.16/41.73%, while persistence reached $1.04/$1.48/1.08%. The baseline was substantially better.
27. **What is the test interval?** 2012-11-19 through 2017-11-10, after chronological training and validation intervals.
28. **Why did the GRU perform poorly?** The model forecasts normalized absolute prices and later AAPL prices exceed the training MinMax range. Neural networks do not automatically extrapolate price-scale changes.
29. **What is the GRU input shape?** A single sample has shape `(60, 1)`: 60 preceding closes and one feature. A batch adds the leading sample dimension.
30. **What does training-only scaling mean here?** The scaler learns the training Close minimum and maximum, then applies that same transformation to validation/test values; it does not refit on future data.

## Two-minute explanation

My project is StockVision AI, a time-series forecasting case study using a Gated Recurrent Unit network. The aim is to estimate Apple's next trading-day closing price from the preceding 60 Close values. The selected historical AAPL archive contains 8,364 dated rows from September 1984 to November 2017. I parse and sort dates, check missing values and duplicates, and use a chronological 70/15/15 train/validation/test split. The MinMax scaler is fitted only on training data, and each sample's window stops before its target date.

I sort and validate the data chronologically, use a 60-trading-day lookback, and divide target dates into 70 percent training, 15 percent validation, and 15 percent testing. The scaler is fit only on training data. Each input window ends before its target date, so the target day's values are not inputs. Validation supports early stopping, and training sequences are not shuffled.

The model has GRU layers with 64 and 32 units, 0.2 dropout after each, Dense(16, ReLU), then a linear output. Adam minimizes MSE; early stopping monitors validation loss and training sequences are not shuffled. On 2012–2017 test dates the GRU MAE/RMSE/MAPE were $45.56/$52.16/41.73%; persistence was $1.04/$1.48/1.08%. The GRU did not beat the baseline. The later price scale lies outside the training scaler's range, highlighting why absolute-price extrapolation is difficult. Historical price data also omits news, fundamentals, and sudden market changes; results do not imply trading profitability.
