import json
from difflib import SequenceMatcher
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
import html
import logging
from functools import lru_cache

import requests

try:
    import yfinance as yf
except Exception:
    yf = None

from config import (
    NEWS_API_KEY_ENV,
    NEWS_CACHE_TTL_HOURS,
    NEWS_RETRY_ATTEMPTS,
    NEWS_RECENCY_DAYS,
    MARKET_MOVERS_TICKERS,
    TRUSTED_NEWS_DOMAINS,
    provider_symbol,
    SENSEX_30_TICKERS,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
NEWS_CACHE_DIR = ROOT_DIR / "data" / "news_cache"
NEWS_ARCHIVE_DIR = ROOT_DIR / "data" / "news_archive"
NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

POSITIVE_WORDS = [
    "rally", "surge", "upgrade", "beat", "strong", "growth", "profit", "expansion",
    "demand", "outperform", "gains", "acceleration", "positive", "bullish", "momentum",
    "breakout", "recovery", "upbeat", "support", "confidence", "resilience", "earnings",
    "expand", "expands", "expanded", "expanding", "grow", "grows", "grew", "growing",
    "increase", "increases", "increased", "increasing", "rise", "rises", "rose", "risen",
    "improve", "improves", "improved", "improving",
]
NEGATIVE_WORDS = [
    "slump", "drop", "downgrade", "miss", "weak", "loss", "pressure", "decline",
    "risk", "selloff", "concern", "bearish", "volatility", "uncertainty", "downturn",
    "recession", "liquidity", "headwind", "worry", "setback", "shortfall",
    "fall", "falls", "fell", "fallen", "drop", "drops", "dropped", "dropping",
    "declines", "declined", "weakens", "weakened", "weakening", "misses", "missed",
]
FINANCE_TERMS = {
    "stock", "share", "shares", "market", "earnings", "revenue", "profit", "loss",
    "guidance", "investor", "analyst", "dividend", "quarter", "ipo", "merger", "acquisition",
    "trading", "equity", "valuation", "margin", "demand", "sector", "finance", "financial",
}


def _normalize_ticker(symbol: str) -> str:
    return (symbol or "").strip().upper().replace(" ", "")


def _normalize_timestamp(value) -> str | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            timestamp = datetime.fromtimestamp(value, timezone.utc)
        else:
            text = str(value).strip()
            try:
                timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                timestamp = parsedate_to_datetime(text)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.isoformat()
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def canonicalize_article_url(url: str) -> str:
    """Remove tracking-only URL differences while retaining the original URL elsewhere."""
    try:
        parsed = urlparse(str(url or "").strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.lower().removeprefix("www.")
        port = parsed.port
        netloc = host if port is None or (scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
        tracking_keys = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in tracking_keys
        ]
        path = parsed.path.rstrip("/") or "/"
        return urlunparse((scheme, netloc, path, "", urlencode(sorted(query)), ""))
    except ValueError:
        return ""


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Deduplicate by canonical URL or same-day headline while preserving provenance."""
    results = []
    url_index = {}
    title_index = {}
    for original in articles:
        article = dict(original)
        original_url = str(article.get("url") or "").strip()
        canonical_url = article.get("canonical_url") or canonicalize_article_url(original_url)
        if canonical_url:
            article["canonical_url"] = canonical_url
        title = re.sub(r"[^a-z0-9]+", " ", str(article.get("title") or "").lower()).strip()
        published = _normalize_timestamp(article.get("published_at"))
        article["published_at"] = published
        article["observed_at"] = _normalize_timestamp(article.get("observed_at"))
        date_key = published[:10] if published else ""
        title_key = (_normalize_ticker(str(article.get("ticker") or "")), date_key)
        duplicate_idx = url_index.get(canonical_url) if canonical_url else None
        if duplicate_idx is None and title and date_key:
            for prior_title, prior_idx in title_index.get(title_key, []):
                if SequenceMatcher(None, title, prior_title).ratio() >= 0.94:
                    duplicate_idx = prior_idx
                    break
        if duplicate_idx is None and title and not date_key:
            duplicate_idx = title_index.get(title_key, {}).get(title)

        if duplicate_idx is not None:
            existing = results[duplicate_idx]
            sources = set(existing.get("duplicate_sources", []))
            sources.update((str(existing.get("source") or "Unknown"), str(article.get("source") or "Unknown")))
            existing["duplicate_sources"] = sorted(sources)
            urls = set(existing.get("duplicate_urls", []))
            urls.update(candidate for candidate in (existing.get("url"), original_url) if candidate)
            existing["duplicate_urls"] = sorted(urls)
            if canonical_url:
                url_index[canonical_url] = duplicate_idx
            continue

        result_idx = len(results)
        results.append(article)
        if canonical_url:
            url_index[canonical_url] = result_idx
        if title:
            if date_key:
                title_index.setdefault(title_key, []).append((title, result_idx))
            else:
                title_index.setdefault(title_key, {})[title] = result_idx
    return results


def normalize_article(article: dict, ticker: str) -> dict | None:
    """Normalize provider output into the shared news item contract."""
    title = re.sub(r"\s+", " ", str(article.get("title") or "")).strip()
    if not title or title.lower() in {"removed", "[removed]"}:
        return None
    summary = re.sub(r"\s+", " ", str(article.get("summary") or "")).strip()
    content = re.sub(r"\s+", " ", str(article.get("content") or summary or title)).strip()
    published_at = _normalize_timestamp(article.get("published_at"))
    observed_at = _normalize_timestamp(article.get("observed_at"))
    url = str(article.get("url") or "").strip()
    publisher_url = str(article.get("publisher_url") or "").strip()
    return {
        "ticker": _normalize_ticker(ticker),
        "title": title,
        "summary": summary,
        "text": " ".join(part for part in (title, summary, content) if part),
        "content": content,
        "source": str(article.get("source") or "Unknown").strip(),
        "published_at": published_at,
        "url": url,
        "canonical_url": canonicalize_article_url(url),
        "observed_at": observed_at,
        **({"publisher_url": publisher_url} if publisher_url else {}),
    }


def _published_timestamp(article: dict) -> float:
    try:
        value = _normalize_timestamp(article.get("published_at"))
        if value is None:
            return 0.0
        published = datetime.fromisoformat(value)
        return published.timestamp()
    except (TypeError, ValueError, OSError):
        return 0.0


def _is_trusted_article(article: dict) -> bool:
    """Accept only articles linked to a recognized publisher."""
    url = str(article.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return False

    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    trusted_domains = tuple(TRUSTED_NEWS_DOMAINS)

    def is_trusted_host(host):
        host = (urlparse(str(host or "")).hostname or "").lower().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in trusted_domains)

    if is_trusted_host(url):
        return True
    return hostname == "news.google.com" and is_trusted_host(article.get("publisher_url"))


def is_finance_relevant(article: dict, ticker: str) -> bool:
    text = str(article.get("text") or " ".join(str(article.get(k) or "") for k in ("title", "summary", "content"))).lower()
    symbol = ticker.upper().removesuffix(".NS").removesuffix(".BO")
    if symbol in ("NSEI", "^NSEI", "NIFTY50"):
        return bool(FINANCE_TERMS.intersection(re.findall(r"[a-z]+", text)))
    normalized_text = f" {re.sub(r'[^a-z0-9]+', ' ', text).strip()} "
    ticker_aliases = [symbol, provider_symbol(ticker.upper()).removesuffix(".NS").removesuffix(".BO")]
    company_aliases = _company_name_aliases(ticker.upper())

    def matches(alias):
        normalized_alias = re.sub(r"[^a-z0-9]+", " ", str(alias).lower()).strip()
        return bool(normalized_alias and f" {normalized_alias} " in normalized_text)

    if any(matches(alias) for alias in company_aliases if alias):
        return True
    if any(matches(alias) for alias in ticker_aliases if alias):
        symbol_text = re.sub(r"[^a-z0-9]", "", symbol)
        if len(symbol_text) <= 4:
            return bool(FINANCE_TERMS.intersection(re.findall(r"[a-z]+", text)))
        return True
    return False


NEWS_ALIASES = {
    "ADANIENT.NS": ["Adani Enterprises"], "ADANIPORTS.NS": ["Adani Ports"],
    "APOLLOHOSP.NS": ["Apollo Hospitals"], "AUBANK.NS": ["AU Small Finance"],
    "BAJAJ-AUTO.NS": ["Bajaj Auto"], "BAJAJFINSV.NS": ["Bajaj Finserv"],
    "BANDHANBNK.NS": ["Bandhan Bank"], "BANKBARODA.NS": ["Bank of Baroda"],
    "BPCL.NS": ["Bharat Petroleum"], "BRITANNIA.NS": ["Britannia"],
    "COALINDIA.NS": ["Coal India"],
    "DIVISLAB.NS": ["Divi's Laboratories", "Divis Laboratories", "Divi's Labs", "Divis Labs"],
    "DRREDDY.NS": ["Dr Reddy", "Dr. Reddy"],
    "EICHERMOT.NS": ["Eicher Motors", "Royal Enfield"],
    "FEDERALBNK.NS": ["Federal Bank"], "HDFCLIFE.NS": ["HDFC Life"],
    "GRASIM.NS": ["Grasim Industries", "Grasim"],
    "HEROMOTOCO.NS": ["Hero MotoCorp"], "HINDALCO.NS": ["Hindalco Industries", "Hindalco"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"], "IDFCFIRSTB.NS": ["IDFC First"],
    "ITC.NS": ["ITC Limited", "ITC Ltd"],
    "LICI.NS": ["Life Insurance Corporation"], "LTIM.NS": ["LTM Limited", "LTIMindtree"],
    "ONGC.NS": ["Oil and Natural Gas Corporation", "Oil & Natural Gas Corporation", "ONGC"],
    "POWERGRID.NS": ["Power Grid Corporation of India", "Power Grid", "PGCIL"],
    "SBILIFE.NS": ["SBI Life"], "SHRIRAMFIN.NS": ["Shriram Finance"],
    "SBIN.NS": ["State Bank of India"], "TATACONSUM.NS": ["Tata Consumer", "Tata Consumer Products"],
    "TITAN.NS": ["Titan Company"], "ULTRACEMCO.NS": ["UltraTech Cement", "Ultratech"],
    "M&M.NS": ["Mahindra & Mahindra"],
}

_RSS_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "LiveMint": "https://www.livemint.com/rss/markets",
    "Hindu BusinessLine": "https://www.thehindubusinessline.com/feeder/default.rss",
}
_RSS_FEED_CACHE: dict[str, tuple[float, list[dict]]] = {}
_RSS_CACHE_SECONDS = 300
_GOOGLE_NEWS_CACHE: dict[str, tuple[float, list[dict]]] = {}


@lru_cache(maxsize=128)
def _company_name_aliases(ticker: str) -> tuple[str, ...]:
    aliases = [
        SENSEX_30_TICKERS.get(ticker, ""),
        MARKET_MOVERS_TICKERS.get(ticker, ""),
        *NEWS_ALIASES.get(ticker, []),
    ]
    details_path = ROOT_DIR / "data" / "fundamentals" / f"{ticker}.json"
    try:
        details = json.loads(details_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        details = {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load company-name metadata for news matching: %s", ticker)
        details = {}
    company = str(details.get("Company") or "").strip()
    if company:
        aliases.append(company)
        aliases.append(re.sub(r"\b(limited|ltd|incorporated|inc)\b", "", company, flags=re.IGNORECASE).strip())
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _company_search_name(ticker: str) -> str:
    aliases = _company_name_aliases(ticker)
    if not aliases:
        return ""
    return re.sub(
        r"\b(limited|ltd|incorporated|inc)\b\.?$", "", aliases[-1], flags=re.IGNORECASE
    ).strip()


def _google_news_query(ticker: str) -> str:
    symbol = ticker.removesuffix(".NS").removesuffix(".BO")
    aliases = [
        re.sub(r"\b(limited|ltd|incorporated|inc)\b\.?$", "", alias, flags=re.IGNORECASE).strip()
        for alias in _company_name_aliases(ticker)
    ]
    terms = [f'"{symbol} stock"']
    terms.extend(f'"{alias}"' for alias in dict.fromkeys(aliases) if alias)
    return f'{" OR ".join(terms)} when:{NEWS_RECENCY_DAYS}d'


def _matches_recent_window(article: dict) -> bool:
    published_at = _normalize_timestamp(article.get("published_at"))
    if not published_at:
        return True
    published = datetime.fromisoformat(published_at)
    now = datetime.now(timezone.utc)
    return now.timestamp() - NEWS_RECENCY_DAYS * 86400 <= published.timestamp() <= now.timestamp()


def _rss_articles(ticker: str, max_articles: int):
    """Merge fresh, direct-link articles from available financial publishers."""
    records = []
    for source, url in _RSS_FEEDS.items():
        fetched_at, cached_records = _RSS_FEED_CACHE.get(source, (0.0, []))
        if time.time() - fetched_at > _RSS_CACHE_SECONDS:
            try:
                response = requests.get(
                    url, timeout=8, headers={"User-Agent": "Equimaster-AI/1.0 news reader"}
                )
                response.raise_for_status()
                feed = ET.fromstring(response.content)
                feed_records = []
                items = feed.findall(".//item")
                if not items:
                    items = feed.findall(".//{*}entry")
                for item in items:
                    fields = {
                        child.tag.rsplit("}", 1)[-1].lower(): child
                        for child in item
                    }
                    title = fields.get("title")
                    summary = fields.get("description")
                    if summary is None:
                        summary = fields.get("summary")
                    if summary is None:
                        summary = fields.get("content")
                    published = fields.get("pubdate")
                    if published is None:
                        published = fields.get("published")
                    if published is None:
                        published = fields.get("updated")
                    link = fields.get("link")
                    link_url = ""
                    if link is not None:
                        link_url = link.attrib.get("href") or link.text or ""
                    published_value = published.text if published is not None else ""
                    if published_value:
                        try:
                            published_value = parsedate_to_datetime(published_value).isoformat()
                        except (TypeError, ValueError, OverflowError):
                            published_value = _normalize_timestamp(published_value) or ""
                    summary_text = (summary.text or "") if summary is not None else ""
                    summary_text = re.sub(r"<[^>]+>", " ", html.unescape(summary_text))
                    feed_records.append({
                        "title": html.unescape(title.text or "") if title is not None else "",
                        "summary": re.sub(r"\s+", " ", summary_text).strip(),
                        "published_at": published_value,
                        "source": source,
                        "url": link_url.strip(),
                    })
                _RSS_FEED_CACHE[source] = (time.time(), feed_records)
                cached_records = feed_records
            except (OSError, ValueError, ET.ParseError, requests.RequestException) as exc:
                logger.warning("Publisher RSS fetch failed for %s (%s)", source, type(exc).__name__)
                _RSS_FEED_CACHE[source] = (time.time(), [])
                cached_records = []
        for record in cached_records:
            article = normalize_article(record, ticker)
            if (article and _is_trusted_article(article) and _matches_recent_window(article)
                    and is_finance_relevant(article, ticker)):
                records.append(article)
    return sorted(deduplicate_articles(records), key=_published_timestamp, reverse=True)[:max_articles]


def _google_news_articles(ticker: str, max_articles: int):
    """Search Google News RSS, retaining only results attributed to trusted publishers."""
    cached_at, records = _GOOGLE_NEWS_CACHE.get(ticker, (0.0, []))
    if time.time() - cached_at > _RSS_CACHE_SECONDS:
        query = _google_news_query(ticker)
        try:
            response = requests.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
                timeout=8,
                headers={"User-Agent": "Equimaster-AI/1.0 news reader"},
            )
            response.raise_for_status()
            feed = ET.fromstring(response.content)
            records = []
            for item in feed.findall(".//item"):
                fields = {
                    child.tag.rsplit("}", 1)[-1].lower(): child
                    for child in item
                }
                title = fields.get("title")
                summary = fields.get("description")
                published = fields.get("pubdate")
                source = fields.get("source")
                link = fields.get("link")
                record = normalize_article({
                    "title": html.unescape(title.text or "") if title is not None else "",
                    "summary": html.unescape(summary.text or "") if summary is not None else "",
                    "published_at": published.text if published is not None else None,
                    "source": source.text if source is not None else "",
                    "publisher_url": source.attrib.get("url", "") if source is not None else "",
                    "url": link.text if link is not None else "",
                }, ticker)
                if (record and _is_trusted_article(record) and _matches_recent_window(record)
                        and is_finance_relevant(record, ticker)):
                    records.append(record)
            _GOOGLE_NEWS_CACHE[ticker] = (time.time(), records)
        except (OSError, ValueError, ET.ParseError, requests.RequestException) as exc:
            logger.warning("Google News RSS fetch failed for %s (%s)", ticker, type(exc).__name__)
            _GOOGLE_NEWS_CACHE[ticker] = (time.time(), [])
            records = []
    return sorted(deduplicate_articles(records), key=_published_timestamp, reverse=True)[:max_articles]


def relevance_details(article: dict, ticker: str) -> dict:
    text = str(article.get("text") or "").lower()
    ticker_name = _normalize_ticker(ticker).lower().replace(".ns", "").replace(".bo", "")
    tokens = set(re.findall(r"[a-z]+", text))
    ticker_match = ticker_name in text
    finance_hits = len(FINANCE_TERMS.intersection(tokens))
    score = min(1.0, (0.65 if ticker_match else 0.0) + min(0.35, finance_hits * 0.05))
    level = "High" if score >= 0.75 else "Medium" if score >= 0.35 else "Low"
    return {"relevance_score": round(score, 3), "relevance_level": level}


def _cache_path_for_ticker(ticker: str) -> Path:
    safe = _normalize_ticker(ticker).replace(".", "_")
    return NEWS_CACHE_DIR / f"{safe}.json"


def _read_cache(ticker: str):
    path = _cache_path_for_ticker(ticker)
    if not path.exists():
        return []
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    if age_hours > NEWS_CACHE_TTL_HOURS:
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        normalized = [normalize_article(item, ticker) for item in payload]
        filtered = [
            item for item in normalized
            if item and _is_trusted_article(item) and _matches_recent_window(item)
            and is_finance_relevant(item, ticker)
        ]
        return deduplicate_articles(filtered)
    except (OSError, ValueError, TypeError):
        return []


def _write_cache(ticker: str, articles):
    observed = datetime.now(timezone.utc)
    records = [{**article, "observed_at": observed.isoformat()} for article in articles]
    try:
        with _cache_path_for_ticker(ticker).open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Failed to write current news cache for %s", ticker)
    try:
        archive = NEWS_ARCHIVE_DIR / ticker.replace(".", "_")
        archive.mkdir(parents=True, exist_ok=True)
        (archive / (observed.strftime("%Y%m%dT%H%M%S%fZ") + ".json")).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("Failed to archive point-in-time news observation for %s", ticker)
    return records


def load_archived_articles(ticker: str, as_of: datetime) -> list[dict]:
    """Load only evidence published and observed by the requested UTC cutoff."""
    cutoff_text = _normalize_timestamp(as_of)
    if cutoff_text is None:
        raise ValueError("A valid as_of timestamp is required for archived news")
    cutoff = datetime.fromisoformat(cutoff_text)
    archive_dir = NEWS_ARCHIVE_DIR / _normalize_ticker(ticker).replace(".", "_")
    if not archive_dir.exists():
        return []

    archived = []
    for path in sorted(archive_dir.glob("*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except (OSError, ValueError):
            logger.exception("Failed to read archived news snapshot %s", path)
            continue
        if not isinstance(records, list):
            logger.warning("Ignoring malformed archived news snapshot %s", path)
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            article = normalize_article(record, ticker)
            if article is None:
                continue
            published_text = article.get("published_at")
            observed_text = article.get("observed_at")
            if not published_text or not observed_text:
                continue
            published = datetime.fromisoformat(published_text)
            observed = datetime.fromisoformat(observed_text)
            if published > cutoff or observed > cutoff:
                continue
            if not _is_trusted_article(article) or not is_finance_relevant(article, ticker):
                continue
            article["duplicate_sources"] = list(record.get("duplicate_sources") or [])
            article["duplicate_urls"] = list(record.get("duplicate_urls") or [])
            archived.append(article)
    return deduplicate_articles(archived)


def _newsapi_articles(ticker: str, max_articles: int, api_key: str | None = None):
    api_key = api_key or os.getenv(NEWS_API_KEY_ENV)
    if not api_key:
        return []

    query_ticker = ticker.replace(".NS", "").replace(".BO", "")
    company_name = _company_search_name(ticker)
    query = f'"{query_ticker}"'
    if company_name:
        query += f' OR "{company_name}"'
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "domains": ",".join(TRUSTED_NEWS_DOMAINS),
        "apiKey": api_key,
    }
    for attempt in range(NEWS_RETRY_ATTEMPTS):
        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            response.raise_for_status()
            normalized = []
            for item in (response.json().get("articles") or []):
                normalized_item = normalize_article({
                    "title": item.get("title"),
                    "summary": item.get("description") or item.get("content"),
                    "content": item.get("content") or item.get("description"),
                    "source": (item.get("source") or {}).get("name", "News API"),
                    "published_at": item.get("publishedAt"),
                    "url": item.get("url"),
                }, ticker)
                if (normalized_item and _is_trusted_article(normalized_item)
                        and _matches_recent_window(normalized_item)
                        and is_finance_relevant(normalized_item, ticker)):
                    normalized.append(normalized_item)
            return deduplicate_articles(normalized)[:max_articles]
        except (OSError, ValueError, TypeError, requests.RequestException) as exc:
            if attempt == NEWS_RETRY_ATTEMPTS - 1:
                logger.warning("NewsAPI request failed for %s (%s)", ticker, type(exc).__name__)
            if attempt < NEWS_RETRY_ATTEMPTS - 1:
                time.sleep(2**attempt)
    return []


def _yahoo_articles(ticker: str, max_articles: int):
    """Fetch recent ticker-specific news from Yahoo Finance."""
    if yf is None:
        return []
    try:
        items = yf.Ticker(provider_symbol(ticker)).news or []
        normalized = []
        for item in items:
            content = item.get("content") or item
            title = content.get("title") or item.get("title")
            provider = content.get("provider") or item.get("provider") or {}
            canonical = content.get("canonicalUrl") or item.get("link") or {}
            published = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, timezone.utc).isoformat()
            normalized_item = normalize_article({
                "title": title,
                "summary": content.get("summary") or content.get("description") or "",
                "content": content.get("summary") or content.get("description") or title,
                "source": provider.get("displayName") if isinstance(provider, dict) else provider,
                "published_at": published,
                "url": canonical.get("url") if isinstance(canonical, dict) else canonical,
            }, ticker)
            if (normalized_item and _is_trusted_article(normalized_item)
                    and _matches_recent_window(normalized_item)
                    and is_finance_relevant(normalized_item, ticker)):
                normalized.append(normalized_item)
        return sorted(deduplicate_articles(normalized), key=_published_timestamp, reverse=True)[:max_articles]
    except Exception as exc:
        logger.warning("Yahoo Finance news fetch failed for %s (%s)", ticker, type(exc).__name__)
        return []


def fetch_ticker_articles(
    ticker: str,
    max_articles: int = 10,
    force_refresh: bool = False,
    api_key: str | None = None,
):
    ticker = _normalize_ticker(ticker)
    if not ticker or max_articles < 1:
        return []

    cached = _read_cache(ticker)
    if cached and not force_refresh:
        return sorted(cached, key=_published_timestamp, reverse=True)[:max_articles]

    # Merge every configured live source; a partial provider response must not
    # prevent newer articles from the other sources from being considered.
    live_articles = []
    if api_key is None:
        live_articles.extend(_newsapi_articles(ticker, max_articles))
    else:
        live_articles.extend(_newsapi_articles(ticker, max_articles, api_key))
    live_articles.extend(_yahoo_articles(ticker, max_articles))
    live_articles.extend(_rss_articles(ticker, max_articles))
    live_articles.extend(_google_news_articles(ticker, max_articles))
    source_articles = live_articles or cached
    articles = sorted(deduplicate_articles(source_articles), key=_published_timestamp, reverse=True)[:max_articles]
    if live_articles:
        articles = _write_cache(ticker, articles)
    return articles


def clear_news_cache(ticker: str | None = None):
    if ticker is None:
        for path in NEWS_CACHE_DIR.glob("*.json"):
            path.unlink(missing_ok=True)
        return
    _cache_path_for_ticker(ticker).unlink(missing_ok=True)
