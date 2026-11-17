"""Compact, point-in-time fundamental snapshots for forward model research."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from config import PROJECT_ROOT

HISTORY_PATH = PROJECT_ROOT / "data" / "fundamental_history.sqlite3"
SNAPSHOT_FEATURES = (
    "market_cap", "pe", "forward_pe", "pb", "ps", "ev_ebitda", "peg",
    "roe", "roa", "roce", "roic", "operating_margin", "net_margin",
    "revenue_growth", "earnings_growth", "quarterly_earnings_growth",
    "sales_cagr_5", "profit_cagr_5", "eps_cagr_5", "debt_equity",
    "net_debt_ebitda", "interest_cover", "current_ratio", "quick_ratio",
    "fcf", "fcf_margin", "cfo_profit", "dividend_yield", "payout",
    "promoters", "fii", "dii", "nim", "gnpa", "nnpa", "casa",
    "capital_adequacy", "beta", "price",
)


def _connect() -> sqlite3.Connection:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_PATH, timeout=30)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_snapshots (
            ticker TEXT NOT NULL,
            snapshot_month TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            source_as_of TEXT,
            sector TEXT,
            industry TEXT,
            financial_sector INTEGER NOT NULL,
            features_json TEXT NOT NULL,
            PRIMARY KEY (ticker, snapshot_month)
        )
    """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamental_snapshots_month "
        "ON fundamental_snapshots(snapshot_month, ticker)"
    )
    return connection


def archive_snapshot(details: Mapping[str, Any], observed_at: datetime | None = None) -> bool:
    """Keep the latest verified observation per company/month; never backdate it."""
    ticker = str(details.get("Ticker") or "").strip().upper()
    metrics = details.get("metrics")
    if not ticker or not isinstance(metrics, Mapping):
        return False
    observed_at = observed_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_at = observed_at.astimezone(timezone.utc)
    features = {}
    for key in SNAPSHOT_FEATURES:
        metric = metrics.get(key)
        value = metric.get("value") if isinstance(metric, Mapping) else None
        if isinstance(value, bool):
            value = None
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        features[key] = value if value is not None and value == value and abs(value) != float("inf") else None
    source_as_of = str(details.get("As of") or "")
    month = observed_at.strftime("%Y-%m")
    with closing(_connect()) as connection:
        with connection:
            connection.execute("""
                INSERT INTO fundamental_snapshots
                    (ticker, snapshot_month, observed_at, source_as_of, sector, industry,
                     financial_sector, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, snapshot_month) DO UPDATE SET
                    observed_at=excluded.observed_at,
                    source_as_of=excluded.source_as_of,
                    sector=excluded.sector,
                    industry=excluded.industry,
                    financial_sector=excluded.financial_sector,
                    features_json=excluded.features_json
                WHERE excluded.observed_at >= fundamental_snapshots.observed_at
            """, (
                ticker, month, observed_at.isoformat(), source_as_of,
                details.get("Sector"), details.get("Industry"),
                int(bool(details.get("financial_sector"))),
                json.dumps(features, sort_keys=True, separators=(",", ":")),
            ))
    return True


def load_snapshots() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with closing(_connect()) as connection:
        rows = connection.execute("""
            SELECT ticker, snapshot_month, observed_at, source_as_of, sector,
                   industry, financial_sector, features_json
            FROM fundamental_snapshots
            ORDER BY snapshot_month, ticker
        """).fetchall()
    return [{
        "ticker": row[0], "snapshot_month": row[1], "observed_at": row[2],
        "source_as_of": row[3], "sector": row[4], "industry": row[5],
        "financial_sector": bool(row[6]), "features": json.loads(row[7]),
    } for row in rows]
