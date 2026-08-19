import math
from typing import List, Dict, Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - fallback path when torch is unavailable
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - fallback path when transformers is unavailable
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from .fetcher import POSITIVE_WORDS, NEGATIVE_WORDS, fetch_ticker_articles
from .retriever import NewsRetriever


_FINBERT_MODEL = "ProsusAI/finbert"
_FINBERT_TOKENIZER = None
_FINBERT_MODEL_OBJ = None


def _safe_score(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _word_frequency(text: str, words: List[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(word.lower()) for word in words)


def _load_finbert_if_available():
    global _FINBERT_TOKENIZER, _FINBERT_MODEL_OBJ
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        return False
    if _FINBERT_MODEL_OBJ is not None and _FINBERT_TOKENIZER is not None:
        return True
    try:
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        _FINBERT_MODEL_OBJ = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
        _FINBERT_MODEL_OBJ.eval()
        return True
    except Exception:
        _FINBERT_TOKENIZER = None
        _FINBERT_MODEL_OBJ = None
        return False


def _finbert_score(text: str) -> float:
    if not _load_finbert_if_available():
        return 0.0

    try:
        inputs = _FINBERT_TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = _FINBERT_MODEL_OBJ(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        if hasattr(_FINBERT_MODEL_OBJ.config, "id2label"):
            scores = {
                str(_FINBERT_MODEL_OBJ.config.id2label.get(idx, idx)).lower(): float(prob)
                for idx, prob in enumerate(probs)
            }
        else:
            scores = {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
            }
        dominant = max(scores, key=scores.get)
        if dominant == "positive":
            return 1.0 * scores.get("positive", 0.0)
        if dominant == "negative":
            return -1.0 * scores.get("negative", 0.0)
        return 0.0
    except Exception:
        return 0.0


def analyze_ticker_sentiment(ticker: str, max_articles: int = 8) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    articles = fetch_ticker_articles(ticker, max_articles=max_articles)
    if not articles:
        return {
            "ticker": ticker,
            "overall_score": 0.0,
            "label": "Neutral",
            "confidence": 0.0,
            "articles": [],
            "evidence": [],
            "summary": f"No market news available for {ticker} right now.",
        }

    query = f"{ticker} stock market sentiment earnings outlook risk opportunity"
    retriever = NewsRetriever(top_k=max_articles)
    retrieved_chunks = retriever.retrieve(articles, query)

    scored_articles: List[Dict[str, Any]] = []
    for article in articles:
        text = " ".join([
            article.get("title", ""),
            article.get("summary", ""),
            article.get("content", ""),
        ])
        positive_hits = _word_frequency(text, POSITIVE_WORDS)
        negative_hits = _word_frequency(text, NEGATIVE_WORDS)
        raw_score = (positive_hits * 1.0) - (negative_hits * 1.0)
        article_relevance = 0.0
        for chunk in retrieved_chunks:
            if chunk.get("title") == article.get("title"):
                article_relevance = max(article_relevance, float(chunk.get("relevance", 0.0)))

        finbert_bias = _finbert_score(text)
        if finbert_bias != 0.0:
            score = _safe_score(finbert_bias * (0.7 + article_relevance))
        else:
            adjusted = raw_score * (0.7 + article_relevance)
            score = _safe_score(adjusted / max(1.0, math.sqrt(len(text.split()) / 8.0)))

        scored_articles.append({
            "title": article.get("title", "").strip(),
            "summary": article.get("summary", "").strip(),
            "source": article.get("source", "Unknown"),
            "published_at": article.get("published_at"),
            "url": article.get("url") or "",
            "score": round(score, 3),
            "relevance": round(article_relevance, 3),
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "text": text,
        })

    weighted_scores = []
    for item in scored_articles:
        weight = 0.55 + max(0.0, item["relevance"]) * 1.2
        weighted_scores.append(item["score"] * weight)

    overall_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
    overall_score = _safe_score(overall_score)

    if overall_score > 0.25:
        label = "Bullish"
    elif overall_score < -0.25:
        label = "Bearish"
    else:
        label = "Neutral"

    confidence = min(0.96, 0.5 + abs(overall_score) * 0.7)

    top_articles = sorted(scored_articles, key=lambda item: (item["score"], item["relevance"]), reverse=True)[:max_articles]
    evidence = [
        {
            "title": item["title"],
            "source": item["source"],
            "score": item["score"],
            "relevance": item["relevance"],
            "summary": item["summary"],
            "url": item["url"],
        }
        for item in top_articles
    ]

    positive_count = sum(1 for item in top_articles if item["score"] > 0)
    negative_count = sum(1 for item in top_articles if item["score"] < 0)
    neutral_count = len(top_articles) - positive_count - negative_count

    drivers = []
    if positive_count > 0:
        drivers.append("demand and earnings tailwinds")
    if negative_count > 0:
        drivers.append("execution and risk concerns")
    if neutral_count > 0:
        drivers.append("mixed market positioning")

    if not drivers:
        drivers = ["limited fresh catalysts"]

    summary = (
        f"{ticker} sentiment is {label.lower()} with a composite score of {overall_score:.2f}. "
        f"The leading evidence is driven by {', '.join(drivers[:2])}, while the broader news flow suggests "
        f"{('positive momentum' if label == 'Bullish' else 'negative pressure' if label == 'Bearish' else 'mixed market tone')} in the short term."
    )

    return {
        "ticker": ticker,
        "overall_score": round(overall_score, 3),
        "label": label,
        "confidence": round(confidence, 3),
        "articles": top_articles,
        "evidence": evidence,
        "summary": summary,
        "drivers": drivers,
    }
