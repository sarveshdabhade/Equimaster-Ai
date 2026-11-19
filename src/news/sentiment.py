from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

from .fetcher import (
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    fetch_ticker_articles,
    load_archived_articles,
    relevance_details,
)
from .retriever import NewsRetriever
from config import NEWS_RECENCY_DAYS, RAG_QUERY_PATTERNS
from src.llm.evidence import synthesize_retrieved_evidence
from src.llm.provider import LLMProviderError, create_llm_provider

torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

_FINBERT_NAME = "ProsusAI/finbert"
_FINBERT_TOKENIZER = None
_FINBERT_MODEL = None
_FINBERT_LOAD_FAILED = False
_POSITIVE_PHRASES = (
    "beats estimates", "beat estimates", "profit rises", "profit growth", "record profit",
    "raises guidance", "guidance raised", "order win", "wins contract", "upgrade to buy",
    "strong results", "margin expands", "debt reduced", "debt reduction",
)
_NEGATIVE_PHRASES = (
    "misses estimates", "missed estimates", "profit falls", "profit warning", "cuts guidance",
    "guidance cut", "downgrade to sell", "weak results", "loss widens", "debt default",
    "accounting fraud", "regulatory probe", "order cancelled", "order canceled",
    "called off", "deal falls through", "deal collapses",
    "guidance withdrawn", "withdraws guidance",
)
_NEGATORS = {"not", "no", "never", "neither", "without", "hardly", "barely"}
_INTENSIFIERS = {"sharply", "strongly", "steep", "severely", "significantly", "record", "major"}
_POSITIVE_VARIANTS = {
    "jumps", "jumped", "jumping", "rallies", "rallied", "surges", "surged",
    "gains", "gained", "soars", "soared", "climbs", "climbed",
}
_NEGATIVE_VARIANTS = {
    "sell", "selling", "slides", "slid", "sliding", "slip", "slips", "slipped", "sinks", "sank",
    "tumbles", "tumbled", "plunges", "plunged", "crashes", "crashed",
    "sells", "sold",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _visible_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(str(value or ""))
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _sentiment_text(article: dict) -> str:
    source = str(article.get("source") or "").strip()
    title = _visible_text(article.get("title") or "")
    body = _visible_text(article.get("summary") or article.get("content") or "")
    if source:
        attribution = re.compile(rf"\s*[-|–—]\s*{re.escape(source)}\s*$", re.IGNORECASE)
        title = attribution.sub("", title).strip()
        body = attribution.sub("", body).strip()
        body = re.sub(rf"\s+{re.escape(source)}\s*$", "", body, flags=re.IGNORECASE).strip()
    if body.casefold().rstrip(". ") == title.casefold().rstrip(". "):
        body = ""
    return ". ".join(part for part in (title, body) if part)


def _load_finbert():
    global _FINBERT_TOKENIZER, _FINBERT_MODEL, _FINBERT_LOAD_FAILED, torch, AutoTokenizer, AutoModelForSequenceClassification
    if os.getenv("EQUIMASTER_FINBERT", "0") != "1":
        return False
    if _FINBERT_MODEL is not None:
        return True
    if _FINBERT_LOAD_FAILED:
        return False
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        local_only = os.getenv("EQUIMASTER_ALLOW_MODEL_DOWNLOADS", "0") != "1"
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(_FINBERT_NAME, local_files_only=local_only)
        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(_FINBERT_NAME, local_files_only=local_only)
        _FINBERT_MODEL.eval()
    except Exception:
        _FINBERT_TOKENIZER = None
        _FINBERT_MODEL = None
        _FINBERT_LOAD_FAILED = True
    return _FINBERT_MODEL is not None


def _finbert_score(text: str) -> float | None:
    if not _load_finbert():
        return None
    try:
        inputs = _FINBERT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probabilities = torch.softmax(_FINBERT_MODEL(**inputs).logits, dim=-1)[0]
        labels = {str(label).lower(): float(probabilities[index]) for index, label in _FINBERT_MODEL.config.id2label.items()}
        return labels.get("positive", 0.0) - labels.get("negative", 0.0)
    except Exception:
        return None


def _keyword_score(text: str) -> float:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z]+", lowered)
    positive_terms = (set(POSITIVE_WORDS) - {"earnings"}) | _POSITIVE_VARIANTS | {"buy"}
    negative_terms = set(NEGATIVE_WORDS) | _NEGATIVE_VARIANTS
    total = 0.0
    for index, token in enumerate(tokens):
        polarity = 1.0 if token in positive_terms else -1.0 if token in negative_terms else 0.0
        if not polarity:
            continue
        context = tokens[max(0, index - 3):index]
        if any(word in _NEGATORS for word in context):
            polarity *= -1
        if any(word in _INTENSIFIERS for word in context[-2:]):
            polarity *= 1.35
        total += polarity
    total += 1.6 * sum(lowered.count(phrase) for phrase in _POSITIVE_PHRASES)
    total -= 1.6 * sum(lowered.count(phrase) for phrase in _NEGATIVE_PHRASES)
    if not total:
        return 0.0
    length_scale = math.sqrt(max(1.0, len(tokens) / 10.0))
    return math.tanh((total / length_scale) * 0.65)


def _article_freshness_weight(article: dict, now: datetime) -> float:
    """Give current catalysts more influence while retaining month-long context."""
    try:
        published = datetime.fromisoformat(str(article.get("published_at", "")).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - published.astimezone(timezone.utc)).total_seconds() / 86400)
    except (TypeError, ValueError):
        return 0.35
    return 0.35 + 0.65 * math.exp(-age_days / 12.0)


def _article_is_in_window(article: dict, recency_days: int | None, as_of=None) -> bool:
    try:
        published = datetime.fromisoformat(str(article.get("published_at", "")).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        published = published.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False
    cutoff_time = as_of or datetime.now(timezone.utc)
    if cutoff_time.tzinfo is None:
        cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)
    cutoff_time = cutoff_time.astimezone(timezone.utc)
    if published > cutoff_time:
        return False
    if recency_days is not None and published < cutoff_time - timedelta(days=max(0, recency_days)):
        return False
    if as_of is not None:
        try:
            observed = datetime.fromisoformat(str(article.get("observed_at", "")).replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            if observed.astimezone(timezone.utc) > cutoff_time:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _impact_details(text: str, score: float, relevance: float) -> dict[str, Any]:
    lowered = text.lower()
    high_impact_terms = (
        "earnings", "guidance", "merger", "acquisition", "fraud", "regulator", "ban",
        "lawsuit", "default", "dividend", "downgrade", "upgrade", "profit warning",
    )
    medium_impact_terms = (
        "revenue", "margin", "demand", "expansion", "competition", "analyst", "outlook",
        "volatility", "headwind", "order", "contract",
    )
    high_hits = sum(term in lowered for term in high_impact_terms)
    medium_hits = sum(term in lowered for term in medium_impact_terms)
    magnitude = abs(float(score))
    impact_score = min(1.0, 0.35 * min(1.0, high_hits / 2) + 0.2 * min(1.0, medium_hits / 3) + 0.3 * magnitude + 0.15 * relevance)
    level = "High" if impact_score >= 0.55 else "Medium" if impact_score >= 0.28 else "Low"
    return {"impact_score": round(impact_score, 3), "impact_level": level}


def analyze_ticker_sentiment(
    ticker: str,
    max_articles: int = 8,
    query_type: str = "recent sentiment",
    recency_days: int | None = NEWS_RECENCY_DAYS,
    as_of=None,
    articles=None,
    include_llm: bool = False,
) -> dict[str, Any]:
    ticker = (ticker or "").strip().upper()
    if as_of is not None and articles is None:
        articles = load_archived_articles(ticker, as_of)
        if not articles:
            raise ValueError(f"No archived timestamped news is available for {ticker} at the requested cutoff")
    else:
        articles = articles if articles is not None else fetch_ticker_articles(ticker, max_articles=max_articles)
    articles = [
        article for article in articles
        if _article_is_in_window(article, recency_days, as_of)
    ]
    if not articles:
        return {"ticker": ticker, "overall_score": 0.0, "label": "Neutral", "confidence": 0.0,
                "articles": [], "evidence": [], "drivers": [], "summary": f"No market news available for {ticker}.",
                "status": "no_evidence",
                "llm_synthesis": None,
                "llm_synthesis_status": "no_evidence" if include_llm else "not_requested"}

    query = RAG_QUERY_PATTERNS.get(query_type, query_type)
    retrieved = NewsRetriever(top_k=max_articles).retrieve(
        articles, f"{ticker} {query}", recency_days=recency_days, as_of=as_of
    )
    relevance_by_title = {}
    for chunk in retrieved:
        relevance_by_title[chunk["title"]] = max(relevance_by_title.get(chunk["title"], 0.0), chunk["relevance"])

    scored = []
    now = datetime.now(timezone.utc)
    for article in articles:
        title = str(article.get("title") or "").strip()
        text = _sentiment_text(article)
        relevance_info = relevance_details(article, ticker)
        relevance = max(
            relevance_by_title.get(title, 0.0),
            float(relevance_info.get("relevance_score", 0.0)),
            0.35,  # fetched articles already passed the ticker/company match filter
        )
        score = _finbert_score(text)
        backend = "finbert" if score is not None else "keyword"
        if score is None:
            score = _keyword_score(text)
        # Widen small but directional model outputs; preserve the sign and cap
        # extreme scores so a single headline cannot overwhelm the full feed.
        score = max(-1.0, min(1.0, float(score) * (1.15 + 0.35 * relevance)))
        source_name = str(article.get("source", "")).lower()
        source_weight = 1.1 if any(name in source_name for name in ("reuters", "bloomberg", "financial times")) else 1.0
        impact_info = _impact_details(text, score, relevance)
        freshness_weight = _article_freshness_weight(article, now)
        duplicate_sources = set(article.get("duplicate_sources") or [])
        corroboration_weight = min(1.2, 1.0 + 0.04 * len(duplicate_sources))
        article_weight = (
            source_weight
            * corroboration_weight
            * freshness_weight
            * (0.55 + 0.75 * relevance)
            * (0.8 + 0.55 * impact_info["impact_score"])
        )
        scored.append({
            **article,
            "score": round(score, 3),
            "backend": backend,
            "sentiment": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral",
            "confidence": round(min(0.99, 0.45 + abs(score) * 0.4 + relevance * 0.15), 3),
            "source_weight": source_weight,
            "relevance": round(relevance, 3),
            "freshness_weight": round(freshness_weight, 3),
            "corroboration_weight": round(corroboration_weight, 3),
            "article_weight": round(article_weight, 3),
            **relevance_info,
            **impact_info,
            "text": text,
        })

    total_weight = sum(item["article_weight"] for item in scored)
    overall = (
        max(-1.0, min(1.0, sum(item["score"] * item["article_weight"] for item in scored) / total_weight))
        if total_weight else 0.0
    )
    positive_weight = sum(item["article_weight"] for item in scored if item["score"] > 0.1)
    negative_weight = sum(item["article_weight"] for item in scored if item["score"] < -0.1)
    positive = sum(item["score"] > 0.1 for item in scored)
    negative = sum(item["score"] < -0.1 for item in scored)
    directional_weight = positive_weight + negative_weight
    positive_ratio = positive_weight / directional_weight if directional_weight else 0.0
    negative_ratio = negative_weight / directional_weight if directional_weight else 0.0
    if overall <= -0.28 and negative_ratio >= 0.55:
        label = "Risk-off"
    elif overall >= 0.12 and positive_ratio >= 0.48:
        label = "Bullish"
    elif overall <= -0.12 and negative_ratio >= 0.48:
        label = "Bearish"
    elif positive and negative:
        label = "Mixed"
    elif overall >= 0.12:
        label = "Bullish"
    elif overall <= -0.12:
        label = "Bearish"
    else:
        label = "Neutral"
    drivers = []
    if positive:
        drivers.append(f"{positive} positive catalyst{'s' if positive != 1 else ''}")
    if negative:
        drivers.append(f"{negative} negative catalyst{'s' if negative != 1 else ''}")
    if not drivers:
        drivers.append("limited fresh catalysts")
    evidence = [{key: item.get(key) for key in (
        "ticker", "title", "source", "published_at", "score", "sentiment",
        "confidence", "source_weight", "relevance", "summary", "text", "url",
        "relevance_score", "relevance_level", "impact_score", "impact_level",
    )} for item in scored]
    directional_agreement = max(positive_ratio, negative_ratio) if directional_weight else 0.0
    evidence_coverage = min(0.08, len(scored) * 0.01)
    confidence = min(0.92, 0.35 + abs(overall) * 0.4 + directional_agreement * 0.17 + evidence_coverage) if scored else 0.0
    directional_articles = [item for item in scored if abs(item["score"]) > 0.1]
    top_catalyst = max(
        directional_articles,
        key=lambda item: abs(item["score"]) * item["article_weight"],
        default=None,
    )
    headline_note = f" Strongest headline: {top_catalyst['title']}" if top_catalyst else ""
    summary = (
        f"{ticker} news sentiment is {label.lower()} ({overall:+.2f}) across {len(scored)} verified articles, "
        f"with {', '.join(drivers)}.{headline_note}"
    )
    llm_synthesis = None
    llm_synthesis_status = "not_requested"
    llm_synthesis_error = None
    if include_llm:
        try:
            provider = create_llm_provider()
            if provider is None:
                llm_synthesis_status = "disabled"
            elif not retrieved:
                llm_synthesis_status = "no_retrieved_evidence"
            else:
                cutoff = as_of or datetime.now(timezone.utc)
                llm_synthesis = synthesize_retrieved_evidence(
                    ticker, retrieved, cutoff, provider=provider
                )
                llm_synthesis_status = "generated"
                cited_ids = set(llm_synthesis["summary_evidence_ids"])
                for claim in llm_synthesis["claims"]:
                    cited_ids.update(claim["evidence_ids"])
                sources = {
                    item["chunk_id"]: {
                        "title": item.get("title", ""),
                        "source": item.get("source", "Unknown"),
                        "url": item.get("canonical_url") or item.get("url") or "",
                        "published_at": item.get("published_at"),
                        "observed_at": item.get("observed_at"),
                    }
                    for item in retrieved
                    if item.get("chunk_id") in cited_ids
                }
                llm_synthesis["citation_sources"] = sources
        except (LLMProviderError, ValueError) as exc:
            llm_synthesis_status = "error"
            llm_synthesis_error = str(exc)

    result = {"ticker": ticker, "overall_score": round(overall, 3), "label": label,
            "confidence": round(confidence, 3), "articles": scored, "evidence": evidence,
            "status": "available" if evidence else "no_evidence",
            "drivers": drivers, "summary": summary,
            "llm_synthesis": llm_synthesis, "llm_synthesis_status": llm_synthesis_status}
    if llm_synthesis_error:
        result["llm_synthesis_error"] = llm_synthesis_error
    return result
