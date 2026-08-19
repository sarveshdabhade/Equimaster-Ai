"""Centralized configuration — single source of truth for ticker lists and constants."""

# ─── NIFTY 50 Constituents (NSE .NS tickers) ───
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

# ─── Bank Nifty Constituents ───
BANK_NIFTY_TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "AUBANK.NS",
]

# Combined NIFTY 50 + Bank Nifty (no duplicates, Nifty 50 first)
ALL_DOWNLOAD_TICKERS = list(dict.fromkeys(NIFTY_50_TICKERS + BANK_NIFTY_TICKERS))

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
UPDATE_TICKERS = ["^NSEI"] + NIFTY_50_TICKERS

# ─── Shared constants ───
WINDOW_SIZE = 60
CLOSE_COL_IDX = 1  # default; overridden dynamically where possible
HORIZONS = [1, 5, 22, 88]
