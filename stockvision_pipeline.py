"""StockVision AI: chronological AAPL next-day close forecasting with GRU."""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "AAPL.csv"
RESULTS = ROOT / "outputs"
LOOKBACK, SEED = 60, 42
FEATURES = ["Close"]


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        try:
            from google.colab import files
            print("Upload the AAPL historical dataset CSV/TXT file.")
            uploaded = files.upload()
            if not uploaded:
                raise FileNotFoundError("No file uploaded. Put AAPL.csv in dataset/ and retry.")
            name, content = next(iter(uploaded.items()))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except ImportError as exc:
            raise FileNotFoundError(f"AAPL dataset not found at {path}. Copy AAPL.csv there or, in Colab, upload the dataset when prompted.") from exc
    df = pd.read_csv(path)
    df.columns = [str(c).strip().title() for c in df.columns]
    required = ["Date", *FEATURES]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=required).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Need at least 100 valid daily observations for chronological splits and a 60-day lookback.")
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Non-finite feature values remain after cleaning.")
    assert df.Date.is_monotonic_increasing and not df.Date.duplicated().any()
    return df


def prepare(df):
    n = len(df)
    train_end, val_end = int(n * .70), int(n * .85)
    assert 0 < LOOKBACK < train_end < val_end < n
    # Fit all feature scaling parameters on training rows only.
    scaler = MinMaxScaler().fit(df.loc[:train_end-1, FEATURES])
    scaled = scaler.transform(df[FEATURES]).astype(np.float32)
    close_idx = FEATURES.index("Close")
    # A target at row t uses rows t-60...t-1, never row t or later.
    X, y, dates, previous_close = [], [], [], []
    for t in range(LOOKBACK, n):
        X.append(scaled[t-LOOKBACK:t])
        y.append(scaled[t, close_idx])
        dates.append(df.Date.iloc[t])
        previous_close.append(df.Close.iloc[t-1])
    X, y, dates = np.asarray(X), np.asarray(y, dtype=float), pd.DatetimeIndex(dates)
    target_rows = np.arange(LOOKBACK, n)
    train = target_rows < train_end
    valid = (target_rows >= train_end) & (target_rows < val_end)
    test = target_rows >= val_end
    assert train.any() and valid.any() and test.any()
    assert dates[train].max() < dates[valid].min() < dates[test].min()
    assert not (set(dates[train]) & set(dates[valid]) or set(dates[valid]) & set(dates[test]))
    return (X[train], y[train], X[valid], y[valid], X[test], y[test], dates[test],
            np.asarray(previous_close)[test], scaler, close_idx, (train_end, val_end))


def main(df=None, prepared=None):
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    if df is None:
        df = load_data()
    if prepared is None:
        prepared = prepare(df)
    (ROOT/"models").mkdir(exist_ok=True)
    (Xtr, ytr, Xv, yv, Xte, yte, test_dates, persistence, scaler,
     close_idx, boundaries) = prepared
    print(f"Rows: {len(df)} | train/validation/test target rows: {boundaries[0]}, {boundaries[1]-boundaries[0]}, {len(df)-boundaries[1]}")
    print(f"Sequences: {Xtr.shape}, {Xv.shape}, {Xte.shape}; features={FEATURES}; target=next trading day's Close")
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(df.Date, df.Close); ax.set(title="AAPL historical daily closing price", xlabel="Date", ylabel="USD"); fig.tight_layout(); fig.savefig(RESULTS/"historical_prices.png", dpi=160); plt.close(fig)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURES))),
        tf.keras.layers.GRU(64, return_sequences=True), tf.keras.layers.Dropout(.2),
        tf.keras.layers.GRU(32), tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(16, activation="relu"), tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse", metrics=["mae"])
    model.summary()
    try:
        tf.keras.utils.plot_model(model, to_file=str(RESULTS/"architecture.png"), show_shapes=True, show_layer_names=True)
    except Exception as exc:
        print(f"Keras plot_model unavailable ({exc}); writing Matplotlib diagram.")
        fig, ax = plt.subplots(figsize=(9, 2.5)); ax.axis("off")
        labels=["Input\n(60, 1)","GRU 64\nsequences", "Dropout\n0.2", "GRU 32", "Dropout\n0.2", "Dense 16\nReLU", "Dense 1\nlinear"]
        for i, label in enumerate(labels):
            x=.04+i*.16; ax.text(x,.5,label,ha="center",va="center",bbox=dict(boxstyle="round",fc="#e8f1fb"))
            if i<len(labels)-1: ax.annotate("",(x+.12,.5),(x+.05,.5),arrowprops=dict(arrowstyle="->"))
        fig.savefig(RESULTS/"architecture.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    history = model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=50,batch_size=32,shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True)],verbose=2)
    train_eval = model.evaluate(Xtr, ytr, verbose=0, return_dict=True)
    val_eval = model.evaluate(Xv, yv, verbose=0, return_dict=True)
    pred = model.predict(Xte,verbose=0).ravel()
    # Invert the training-fitted Close scaling before price metrics.
    pred = (pred - scaler.min_[close_idx]) / scaler.scale_[close_idx]
    actual = (yte - scaler.min_[close_idx]) / scaler.scale_[close_idx]
    def metrics(a,p):
        return {"MAE":float(mean_absolute_error(a,p)),"RMSE":float(np.sqrt(mean_squared_error(a,p))),
                "MSE":float(mean_squared_error(a,p)),"MAPE_percent":float(mean_absolute_percentage_error(a,p)*100)}
    gru, naive = metrics(actual,pred), metrics(actual,persistence)
    comparison=pd.DataFrame([{"Model":"Naive persistence",**naive},{"Model":"GRU",**gru}])
    comparison.to_csv(RESULTS/"model_comparison.csv",index=False)
    (RESULTS/"metrics.json").write_text(json.dumps({"ticker":"AAPL","source_file":"dataset/AAPL.csv (copied byte-for-byte from D:/Stocks/aapl.us.txt)","observations":len(df),"data_start":str(df.Date.min().date()),"data_end":str(df.Date.max().date()),"test_start":str(test_dates.min().date()),"test_end":str(test_dates.max().date()),"epochs_run":len(history.history["loss"]),"training_mae_scaled":float(train_eval["mae"]),"validation_mae_scaled":float(val_eval["mae"]),"training_mae_usd":float(train_eval["mae"]/scaler.scale_[close_idx]),"validation_mae_usd":float(val_eval["mae"]/scaler.scale_[close_idx]),"gru":gru,"naive_persistence":naive},indent=2))
    pd.DataFrame({"Date":test_dates,"Actual":actual,"GRU_Predicted":pred,"Naive_Predicted":persistence,"Residual":actual-pred}).to_csv(RESULTS/"test_predictions.csv",index=False)
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(history.history["loss"],label="Training loss"); ax.plot(history.history["val_loss"],label="Validation loss"); ax.set(xlabel="Epoch",ylabel="MSE (scaled close²)",title="Training and validation loss"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/"training_validation_loss.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(history.history["mae"],label="Training MAE"); ax.plot(history.history["val_mae"],label="Validation MAE"); ax.set(xlabel="Epoch",ylabel="MAE (scaled close)",title="Training and validation MAE"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/"training_validation_mae.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(test_dates,actual,label="Actual"); ax.plot(test_dates,pred,label="GRU"); ax.plot(test_dates,persistence,label="Naive",alpha=.7); ax.set(xlabel="Date",ylabel="Close (USD)",title="Held-out test predictions"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/"actual_vs_predicted.png",dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4)); axs[0].plot(test_dates,actual-pred); axs[0].axhline(0,color="black",lw=1); axs[0].set(title="Test residuals",xlabel="Date",ylabel="Actual - predicted (USD)"); axs[1].hist(actual-pred,bins=30); axs[1].set(title="Residual distribution",xlabel="Residual (USD)",ylabel="Count"); fig.tight_layout(); fig.savefig(RESULTS/"residuals.png",dpi=160); plt.close(fig)
    model.save(ROOT/"models"/"stockvision_gru.keras")
    print(comparison.to_string(index=False)); print("Test dates align to target rows:",test_dates.min().date(),"through",test_dates.max().date())

if __name__ == "__main__":
    main()



