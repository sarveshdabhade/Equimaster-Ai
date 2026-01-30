# Equimaster-Ai

**Financial Intelligence Platform** — AI-powered analysis using deep learning and clustering.

---

## Project Vision

Equimaster-Ai is a **Financial Intelligence Platform** that combines multiple AI and ML techniques to deliver actionable insights from market and textual data. The system uses:

- **LSTM** — time-series and sequence modeling for price/volume patterns
- **BERT** — natural language understanding for news, reports, and sentiment
- **CNN** — pattern and feature extraction from financial data
- **K-Means** — clustering for segmentation and regime detection

The goal is to provide a single, professional toolkit for research, backtesting, and decision support in financial markets.

---

## Tech Stack

| Layer        | Technology |
|-------------|------------|
| **Frontend** | [Streamlit](https://streamlit.io/) — interactive dashboards and apps |
| **Deep Learning** | [TensorFlow](https://www.tensorflow.org/) — LSTM, CNN, and custom models |
| **NLP**      | BERT-based models (e.g. Transformers / TensorFlow) |
| **Clustering** | K-Means (e.g. scikit-learn) |
| **Language** | Python 3.x |

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Equimaster-Ai.git
   cd Equimaster-Ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run src/app.py
   ```
   *(Adjust the path if your Streamlit entrypoint lives elsewhere.)*

---

## Project Structure

```
Equimaster-Ai/
├── src/          # Python scripts and app code
├── data/         # Stock CSVs and datasets (git-ignored)
├── models/       # Trained model weights
├── notebooks/    # Experiments and exploration
└── tests/        # System checks and tests
```

---

## License

MIT (or your chosen license).
