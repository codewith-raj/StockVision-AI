# Repository Audit Report

Inspected before changes: `README.md`, `StockMarketPredictionGRU.ipynb`, `checktfgpu.ipynb`, and `.gitignore`. No CSV, Python source, requirements, result images, trained model, report, license file, or Git metadata was present. Notebook cells were inspected directly.

| Requirement | Existing implementation | Evidence | Status | Required change | Priority |
|---|---|---|---|---|---|
| Official data and AAPL | Downloads AAPL with yfinance, not assigned Kaggle CSV | Original notebook cells 1–3 | FAIL | Use supplied Kaggle AAPL.csv | P0 |
| Data cleaning | No explicit sorting, duplicate, missing or finite checks | Cells 2–3 | FAIL | Parse, sort, clean and validate | P0 |
| Features/target | Uses only High; predicts next High | Cells 4–10 | PARTIAL | OHLCV inputs, next-day Close | P0 |
| Scaling | Fits scaler on training High only | Cells 6, 9 | PASS | Retain train-only fitting | P0 |
| Partitions | Calendar 2016–2020 train and 2021+ test; no validation | Cells 4–5 | PARTIAL | Chronological 70/15/15 | P0 |
| Sequences | 60-step windows; no explicit date/assertion checks | Cells 7–10 | PARTIAL | Check target alignment and shapes | P0 |
| GRU architecture | Single 125-unit GRU, Dense(1) | Cell 12 | PARTIAL | Stacked 64/32 GRU, dropout, dense head | P0 |
| Validation/training | 10 epochs; no validation, callback; default shuffle | Cell 13 | FAIL | Validation early stopping, shuffle=False | P0 |
| Metrics | Test RMSE only | Cells 17–18 | PARTIAL | MAE/RMSE/MSE/MAPE | P1 |
| Baseline | None | All code cells | FAIL | Same-date persistence baseline | P1 |
| Graphs | Actual/predicted without dates; no loss curves | Cells 15–16 | FAIL | Loss, historical, dated prediction, residual plots | P1 |
| Architecture diagram | Summary only | Cell 12 | PARTIAL | Add model diagram | P1 |
| Saving/reproducibility | Seeds set; no serialization | Cells 0, 12–14 | PARTIAL | Save model/metrics and use portable paths | P1 |
| Dependencies/entry point | README install suggestion; no dependency file/source; yfinance network requirement | README, cells 0–2 | FAIL | Add requirements and local-data pipeline | P0 |
| Academic material | Minimal README and markdown | README and notebook | FAIL | Add explanations, report, viva material | P2 |
| Attribution/license | README lacked attribution; license/git absent | Workspace listing | NOT VERIFIED | Attribute original; license cannot be established | P2 |

## Follow-up dataset discovery and final execution

The user subsequently supplied `D:/Stocks/`, `D:/ETFs/`, and `D:/Data/` as local data locations. `D:/Stocks/aapl.us.txt` was inspected and found readable with Date, Open, High, Low, Close, Volume, and OpenInt columns; 8,364 observations from 1984-09-07 through 2017-11-10; no missing required values or duplicate dates. A byte-for-byte copy was placed at `dataset/AAPL.csv` for the project. The original copy was not modified.

With Python 3.12 and dependencies provisioned through uv, `stockvision_pipeline.py` completed successfully on CPU, trained for 19 epochs, evaluated test dates 2012-11-19–2017-11-10, saved the model, metric files and figures. `StockVision_AI_GRU_Final.ipynb` was then executed sequentially with nbconvert in a clean Jupyter kernel: five code cells ran and zero errors were recorded. TensorFlow native Windows GPU support was unavailable; Keras diagram plotting lacked pydot/Graphviz, so a Matplotlib diagram fallback was produced.

## Changes

Added `StockVision_AI_GRU_Final.ipynb`, `stockvision_pipeline.py`, `requirements.txt`, `dataset/AAPL.csv`, output artifacts, trained model, `PROJECT_REPORT.md`, `StockVision_AI_Project_Report.pdf`, `FINAL_COMPLIANCE_REPORT.md`, and `VIVA_PREPARATION.md`. Updated README. Original notebooks are preserved.
