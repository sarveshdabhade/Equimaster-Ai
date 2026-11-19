"""Plot a versioned model's saved chronological backtest."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def predict_and_plot(ticker="RELIANCE.NS", horizon=1):
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.artifacts import resolve_artifact
    artifact = resolve_artifact(ticker, horizon)
    if artifact["legacy"]:
        raise ValueError("Retrain to generate a verified backtest")
    frame = pd.read_csv(artifact["model"].parent / f"backtest_h{horizon}.csv", parse_dates=["target_date"])
    figure, axes = plt.subplots(figsize=(14, 6))
    axes.plot(frame["target_date"], frame["actual_close"], label="Actual")
    axes.plot(frame["target_date"], frame["predicted_close"], label="Forecast")
    axes.legend()
    output = ROOT / "images" / f"prediction_{ticker}.png"
    figure.savefig(output)
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(predict_and_plot())
