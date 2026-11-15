# Final Compliance Report

| Requirement | Evidence | Status | Note |
|---|---|---|---|
| PDF has professor's six exact section headings | `StockVision_AI_Project_Report.pdf` | PASS | All six numbered headings appear exactly as specified |
| Exactly three primary submission files | `23FE10ITE00256/` | PASS | PDF, executed notebook, and trained `.keras` model only |
| Student/course/case information | Final notebook and `PROJECT_REPORT.md` | PASS | Ritik Raj, 23FE10ITE00256, MUJ, Case Study 7 |
| Historical AAPL dataset | `dataset/AAPL.csv`, `outputs/metrics.json` | PASS | Actual provided archive copy; 8,364 rows, 1984-09-07–2017-11-10 |
| Data inspection and cleaning | `load_data` in pipeline/notebook | PASS | CSV read, Date parsed, Close numeric, sorted, invalid/duplicate rows handled; no missing/duplicate required data found |
| 60-step next trading-day Close prediction | `prepare` | PASS | Single Close feature; asserts `(60,1)` sequences and target-date separation |
| Chronological 70/15/15 split | `prepare` | PASS | 5,854 / 1,255 / 1,255 target rows before lookback context; 5,794 train sequences |
| Training-only MinMaxScaler | `prepare` | PASS | Fitted to first 70% rows only |
| GRU architecture and parameters | Pipeline model summary | PASS | GRU64, dropout, GRU32, dropout, Dense16 ReLU, Dense1; Adam/MSE/MAE, seed 42, shuffle=False |
| Training/validation loss and MAE | `outputs/training_validation_loss.png`, `training_validation_mae.png`, `metrics.json` | PASS | Trained for 19 epochs with patience 8 |
| Held-out evaluation and baseline | `outputs/model_comparison.csv`, `test_predictions.csv` | PASS | Same target dates; inverse-transformed price metrics |
| Prediction/historical/error plots | `outputs/*.png` | PASS | Actual vs predicted, historical close, residual distribution, loss/MAE, architecture diagram |
| Model saving/loading/prediction | `23FE10ITE00256/stockvision_gru.keras` | PASS | Submission copy loaded with TensorFlow 2.21.0; input `(None,60,1)`, output `(None,1)`, 22,817 params; predicted finite output for a real training-scaled test window |
| Notebook execution | `StockVision_AI_GRU_Final.ipynb` | PASS | Executed sequentially in a clean Jupyter kernel; 5 code cells executed and zero cell errors |
| Academic report and viva | `PROJECT_REPORT.md`, `VIVA_PREPARATION.md` | PASS | Actual measured results and limitations documented |
| Original project credit/license | `README.md`, report | PARTIAL | Repository credited; no license file or Git metadata was supplied, so license terms could not be verified |

## Execution record

The pipeline ran successfully with Python 3.12, TensorFlow and required libraries on CPU. Source syntax compilation and notebook JSON validation passed. Keras `plot_model` was unavailable because pydot/Graphviz is absent; the pipeline generated a Matplotlib architecture diagram instead. Native Windows TensorFlow GPU support was unavailable, so execution used CPU. The trained model and all required results were generated.

## Overall status: PASS WITH LIMITATIONS

The GRU did **not** outperform naive persistence. On the test dates 2012-11-19–2017-11-10, GRU MAE/RMSE/MAPE were $45.56/$52.16/41.73%, versus $1.04/$1.48/1.08% for persistence. This poor extrapolation is reported honestly. The model uses univariate price levels, and the validation/test price regime is substantially beyond the training-only MinMax range. Do not claim predictive improvement or trading usefulness.

The local folder `23FE10ITE00256/` contains exactly the three required submission files, and their SHA-256 hashes match the verified development artifacts. It is ready to upload to the separately required Google Drive folder.
