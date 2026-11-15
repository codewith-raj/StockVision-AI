# Dataset details

`AAPL.csv` is a byte-for-byte copy of `D:/Stocks/aapl.us.txt`, a historical US stock OHLCV archive provided in the user's workspace. The source file was not edited. The file has 8,364 observations dated 1984-09-07 through 2017-11-10 and columns Date, Open, High, Low, Close, Volume, OpenInt. Required values had no missing entries and Date had no duplicates. This experiment selects Close only. The original source remains in `D:/Stocks/`.

For Google Colab without the local workspace copy, run the notebook; its loader prompts to upload the AAPL historical CSV/TXT file and places it at `dataset/AAPL.csv` in the Colab runtime.
