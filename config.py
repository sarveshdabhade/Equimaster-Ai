"""Centralized configuration - single source of truth for ticker lists and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_VERSION = 6
PROVIDER_SYMBOLS = {"LTIM.NS": "LTM.NS", "NSEI": "^NSEI", "NIFTY 50": "^NSEI"}


def provider_symbol(ticker: str) -> str:
    return PROVIDER_SYMBOLS.get(ticker, ticker)


def get_project_root() -> Path:
    """Return the repository root as an absolute path."""
    return PROJECT_ROOT


def resolve_app_path(*parts: str) -> str:
    """Resolve a path relative to the project root, not the current working directory."""
    return str(PROJECT_ROOT.joinpath(*parts))


def ensure_runtime_directories() -> None:
    """Create all runtime directories used by the app and training pipeline."""
    directories = [
        "data",
        "data/processed",
        "data/raw",
        "data/train_data",
        "data/news_cache",
        "images",
        "logs",
        "models",
    ]
    for rel_path in directories:
        Path(resolve_app_path(rel_path)).mkdir(parents=True, exist_ok=True)


# ─── NIFTY 50 Constituents (NSE .NS tickers) ───
# Checked against the NSE Indices constituent CSV on 2026-10-26; refresh after index rebalances.
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BSE.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS",
    "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INFY.NS", "INDIGO.NS", "JSWSTEEL.NS", "JIOFIN.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "MAXHEALTH.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS",
    "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS",
    "TMPV.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "TRENT.NS", "ULTRACEMCO.NS",
]

# ─── Bank Nifty Constituents ───
BANK_NIFTY_TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "AUBANK.NS",
]

# Combined NIFTY 50 + Bank Nifty (no duplicates, Nifty 50 first)
ALL_DOWNLOAD_TICKERS = list(dict.fromkeys(NIFTY_50_TICKERS + BANK_NIFTY_TICKERS + ["LICI.NS"]))

# ─── Index tickers for the scrolling ticker strip ───
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY 100": "^CNX100",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
}

# ─── Sensex 30 Constituents (for the Sensex ticker) ───
SENSEX_30_TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ITC.NS": "ITC Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "M&M.NS": "Mahindra & Mahindra",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
    "TECHM.NS": "Tech Mahindra",
    "NESTLEIND.NS": "Nestle India",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC Ltd",
    "INDUSINDBK.NS": "IndusInd Bank",
    "TATASTEEL.NS": "Tata Steel",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIPORTS.NS": "Adani Ports",
    "JSWSTEEL.NS": "JSW Steel",
    "DRREDDY.NS": "Dr. Reddy's Labs",
}

# ─── Major stocks for Market Movers widget ───
MARKET_MOVERS_TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "BAJFINANCE.NS": "Bajaj Finance",
    "LT.NS": "Larsen & Toubro",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "HCLTECH.NS": "HCL Technologies",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TITAN.NS": "Titan Company",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro Limited",
    "NESTLEIND.NS": "Nestle India",
    "POWERGRID.NS": "Power Grid Corporation",
    "NTPC.NS": "NTPC Limited",
    "ONGC.NS": "Oil & Natural Gas Corp",
    "COALINDIA.NS": "Coal India",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
}

# ─── Update data tickers (includes ^NSEI index) ───
UPDATE_TICKERS = ["^NSEI"] + ALL_DOWNLOAD_TICKERS

# ─── Shared constants ───
WINDOW_SIZE = 60
CLOSE_COL_IDX = 1  # default; overridden dynamically where possible
HORIZONS = [1, 5, 22, 63]
HORIZON_LABELS = {1: "Day", 5: "Week", 22: "Month", 63: "Quarter"}


def horizon_description(horizon: int) -> str:
    """Return the UI-friendly forecast period and its model session count."""
    session_word = "session" if horizon == 1 else "sessions"
    return f"{HORIZON_LABELS[horizon]} ({horizon} trading {session_word})"


HORIZON_CHOICES_DESCRIPTION = ", ".join(horizon_description(horizon) for horizon in HORIZONS)


FORECAST_TARGET_COLUMN = "Close"
FORECAST_TARGET_KIND = "future_close"
FORECAST_CUTOFF = "after_close"
FORECAST_BASELINE = "persistence"
FORECAST_EVALUATION_METRICS = ("mae", "rmse", "mape", "lstm_directional_accuracy")

# News/RAG settings. API keys are read from the environment at runtime.
NEWS_API_KEY_ENV = "NEWS_API_KEY"
MODEL_STORAGE_PATH_ENV = "EQUIMASTER_MODEL_PATH"
NEWS_CACHE_PATH_ENV = "EQUIMASTER_NEWS_CACHE_PATH"
FUSION_SHADOW_VERSION_ENV = "EQUIMASTER_FUSION_SHADOW_VERSION"
NEWS_CACHE_TTL_MINUTES = 4
NEWS_MAX_ARTICLES = 20
# Keep a full month of company coverage for the research panel and batch archive.
NEWS_RECENCY_DAYS = 30
NEWS_RETRY_ATTEMPTS = 3
TRUSTED_NEWS_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "thehindubusinessline.com",
    "business-today.in",
    "ndtvprofit.com",
    "finance.yahoo.com",
    "indiashippingnews.com",
    "etinfra.com",
    "newindianexpress.com",
    "theweek.in",
    "timesofindia.indiatimes.com",
    "rediff.com",
    "samco.in",
    "scanx.trade",
    "marketsmojo.com",
    "barandbench.com",
    "etedge-insights.com",
    "investmentguruindia.com",
    "economictimes.com",
    "marketwatch.com",
    "thehindu.com",
    "cnbctv18.com",
    "zeebiz.com",
    "autocarpro.in",
    "businessline.com",
]
NEWS_MODEL_NAME = "ProsusAI/finbert"
RAG_QUERY_PATTERNS = {
    "recent sentiment": "recent sentiment earnings outlook",
    "negative catalysts": "negative catalysts risk concern downgrade",
    "bullish momentum": "bullish momentum growth upgrade demand",
    "risk events": "risk events volatility loss pressure uncertainty",
}
