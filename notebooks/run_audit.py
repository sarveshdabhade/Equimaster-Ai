"""Run data audit (same logic as 01_data_audit.ipynb)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ticker = "RELIANCE.NS"
file_path = f"data/{ticker}.csv"

if not os.path.exists(file_path):
    print("[X] File not found! Check your data_loader.py")
    sys.exit(1)

print(f"[OK] Found data for {ticker}")
# yfinance CSV has 2 metadata rows; columns after that: Date, Close, High, Low, Open, Volume
df = pd.read_csv(
    file_path,
    skiprows=2,
    names=["Date", "Close", "High", "Low", "Open", "Volume"],
    parse_dates=["Date"],
    index_col=0,
)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

print("\nData Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head().to_string())

print("\nMissing Values Check:")
print(df.isnull().sum())
print("\nZeros in Close:", (df["Close"] == 0).sum())

# Save plot
plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Close Price", color="blue", linewidth=1)
plt.title(f"{ticker} - 10 Year Price History")
plt.xlabel("Days")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
out = os.path.join(os.path.dirname(__file__), "audit_plot.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
plt.close()
print(f"\n[OK] Plot saved to {out}")
