# StockVision AI — AAPL forecasting with GRU

**Student:** Ritik Raj · **Registration:** 23FE10ITE00256 · **Course:** Deep Learning · **Manipal University Jaipur** · **Case Study 7**

This submission adapts [ax5hay/Stock-Market-Prediction-GRU](https://github.com/ax5hay/Stock-Market-Prediction-GRU). The original notebooks remain in the project. No license file or `.git` metadata was supplied in the workspace, so the original license terms could not be verified.

## Dataset

The input `dataset/AAPL.csv` is an unmodified byte-for-byte copy of `D:/Stocks/aapl.us.txt` from the historical US stock files supplied in the workspace. It has Date, Open, High, Low, Close, Volume and OpenInt columns, 8,364 rows, 1984-09-07 to 2017-11-10, and no missing required values or duplicate dates. This project uses only Close, as specified by the final model input `(60, 1)`. The other user-provided stock and ETF folders were not needed for this AAPL case study.

## Run

```powershell
uv run --python 3.12 --with-requirements requirements.txt python stockvision_pipeline.py
```

Or install `requirements.txt`, then run `python stockvision_pipeline.py`. To run the self-contained notebook, open `StockVision_AI_GRU_Final.ipynb` and execute cells in order. For Colab, upload the notebook and execute it; upload AAPL data if prompted. Training outputs are saved to `outputs/`, and the Keras model to `models/`.

## Measured result

Training completed for 19 epochs with early stopping. On the chronological test partition (2012-11-19 through 2017-11-10), persistence achieved MAE **$1.04**, RMSE **$1.48**, MAPE **1.08%**. The GRU achieved MAE **$45.56**, RMSE **$52.16**, MAPE **41.73%**, so it did not beat the baseline. The rising AAPL price scale lies well outside the training range; this is a major limitation of this close-level model and split. See `PROJECT_REPORT.md` for metrics and plots.

The prediction task is an academic experiment and does not establish a profitable strategy or investment advice.

## Professor submission package

`23FE10ITE00256/` is prepared locally with exactly the three requested files: the final PDF report, executed final notebook, and trained `stockvision_gru.keras` model. Upload this folder to the required Google Drive location for submission; no Drive account connection was available in this workspace.
