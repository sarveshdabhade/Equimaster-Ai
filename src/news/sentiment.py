import math
from typing import List, Dict, Any

import numpy as np

from .fetcher import POSITIVE_WORDS, NEGATIVE_WORDS, fetch_ticker_articles
from .retriever import NewsRetriever


def _safe_score(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _word_frequency(text: str, words: List[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(word.lower()) for word in words)


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

    summary = (
        f"{ticker} sentiment is {label.lower()} with a composite score of {overall_score:.2f}. "
        f"Recent retrieved news suggests {('positive momentum' if label == 'Bullish' else 'negative pressure' if label == 'Bearish' else 'mixed market tone')} in the short term."
    )

    return {
        "ticker": ticker,
        "overall_score": round(overall_score, 3),
        "label": label,
        "confidence": round(confidence, 3),
        "articles": top_articles,
        "evidence": evidence,
        "summary": summary,
    }
