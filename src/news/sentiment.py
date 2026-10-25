from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

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
    tokens = re.findall(r"[a-z]+", text.lower())
    positive_terms = set(POSITIVE_WORDS)
    negative_terms = set(NEGATIVE_WORDS)
    positive = sum(token in positive_terms for token in tokens)
    negative = sum(token in negative_terms for token in tokens)
    total = max(1.0, math.sqrt(len(tokens) / 8.0))
    return max(-1.0, min(1.0, (positive - negative) / total))


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
    for article in articles:
        if article.get("title", "") not in relevance_by_title:
            continue
        text = " ".join(article.get(field, "") for field in ("title", "summary", "content"))
        relevance = relevance_by_title.get(article.get("title", ""), 0.0)
        score = _finbert_score(text)
        backend = "finbert" if score is not None else "keyword"
        if score is None:
            score = _keyword_score(text)
        score = max(-1.0, min(1.0, score * (0.7 + relevance)))
        source_name = str(article.get("source", "")).lower()
        source_weight = 1.1 if any(name in source_name for name in ("reuters", "bloomberg", "financial times")) else 1.0
        relevance_info = relevance_details(article, ticker)
        impact_info = _impact_details(text, score, relevance)
        scored.append({
            **article,
            "score": round(score, 3),
            "backend": backend,
            "sentiment": "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral",
            "confidence": round(min(0.99, 0.5 + abs(score) * 0.5 + relevance * 0.2), 3),
            "source_weight": source_weight,
            "relevance": round(relevance, 3),
            **relevance_info,
            **impact_info,
            "text": text,
        })

    weighted = [item["score"] * item["source_weight"] * (0.55 + item["relevance"] * 1.2) for item in scored]
    overall = max(-1.0, min(1.0, float(np.mean(weighted)))) if weighted else 0.0
    positive = sum(item["score"] > 0 for item in scored)
    negative = sum(item["score"] < 0 for item in scored)
    positive_ratio = positive / len(scored) if scored else 0.0
    negative_ratio = negative / len(scored) if scored else 0.0
    if overall < -0.25 and negative_ratio >= 0.6:
        label = "Risk-off"
    elif overall > 0.25 and positive_ratio >= 0.6:
        label = "Bullish"
    elif overall < -0.25:
        label = "Bearish"
    elif positive and negative:
        label = "Mixed"
    else:
        label = "Neutral"
    drivers = []
    if positive:
        drivers.append("earnings and demand tailwinds")
    if negative:
        drivers.append("execution and risk concerns")
    if not drivers:
        drivers.append("limited fresh catalysts")
    evidence = [{key: item.get(key) for key in (
        "ticker", "title", "source", "published_at", "score", "sentiment",
        "confidence", "source_weight", "relevance", "summary", "text", "url",
        "relevance_score", "relevance_level", "impact_score", "impact_level",
    )} for item in scored]
    confidence = min(0.96, 0.5 + abs(overall) * 0.7) if scored else 0.0
    summary = f"{ticker} sentiment is {label.lower()} with a composite score of {overall:.2f}, driven by {', '.join(drivers)}."
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
