"""Evaluate labeled sentiment and retrieval datasets without fetching future news."""
from collections import Counter
from datetime import datetime
import math
import numpy as np
from src.news.fetcher import canonicalize_article_url
from src.news.sentiment import _finbert_score, _keyword_score
from src.news.retriever import NewsRetriever

LABELS = ("negative", "neutral", "positive")


def _canonical_evidence_url(value):
    text = str(value or "").strip()
    return canonicalize_article_url(text) or text


def _retrieval_metrics(hits, relevant_urls, k):
    relevant = {_canonical_evidence_url(url) for url in relevant_urls if str(url or "").strip()}
    unique_hits = []
    seen = set()
    for position, hit in enumerate(hits):
        url = _canonical_evidence_url(hit.get("canonical_url") or hit.get("url"))
        identity = ("url", url) if url else (
            "article", hit.get("article_id") or hit.get("chunk_id") or position
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique_hits.append(url)

    ranked = unique_hits[:k]
    relevant_ranks = [rank for rank, url in enumerate(ranked, start=1) if url and url in relevant]
    hit_count = len(relevant_ranks)
    recall = hit_count / len(relevant) if relevant else 0.0
    precision = hit_count / k
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": reciprocal_rank,
        "ndcg_at_k": ndcg,
    }


def evaluate_news_dataset(payload):
    examples = payload["sentiment"]
    if not examples:
        raise ValueError("At least one independently labeled sentiment example is required")
    pairs, confidence = [], []
    backends = Counter()
    for row in examples:
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown sentiment label {row['label']}")
        score = _finbert_score(row["text"])
        backends["keyword" if score is None else "finbert"] += 1
        if score is None:
            score = _keyword_score(row["text"])
        predicted = "positive" if score > .15 else "negative" if score < -.15 else "neutral"
        correct = int(predicted == row["label"])
        pairs.append((row["label"], predicted))
        confidence.append((min(.99, .5 + abs(score) * .5), correct))
    per_class = {}
    for label in LABELS:
        tp = sum(a == b == label for a, b in pairs)
        precision = tp / max(1, sum(b == label for a, b in pairs))
        recall = tp / max(1, sum(a == label for a, b in pairs))
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    ece = 0.
    for lower in np.arange(0, 1, .1):
        group = [(c, y) for c, y in confidence if lower <= c < lower + .1]
        if group:
            ece += len(group) / len(confidence) * abs(np.mean([x[0] for x in group]) - np.mean([x[1] for x in group]))
    retrieval = []
    for row in payload.get("retrieval", []):
        k = int(row.get("k", 3))
        if k < 1:
            raise ValueError("Retrieval evaluation k must be a positive integer")
        as_of = datetime.fromisoformat(row["as_of"])
        hits = NewsRetriever(top_k=k).retrieve(
            row["articles"], row["query"], as_of=as_of, recency_days=None
        )
        metrics = _retrieval_metrics(hits, row["relevant_urls"], k)
        retrieval.append({"query": row["query"], "k": k, **metrics})
    kind = payload.get("kind", "unspecified")
    counts = Counter(a for a, b in pairs)
    return {"dataset_kind": kind, "provenance": payload.get("provenance"), "sample_count": len(pairs),
            "backend_counts": dict(backends), "class_counts": dict(counts),
            "majority_class_baseline_accuracy": max(counts.values()) / len(pairs),
            "balanced_accuracy": float(np.mean([v["recall"] for v in per_class.values()])),
            "accuracy": sum(a == b for a, b in pairs) / len(pairs), "per_class": per_class,
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "heuristic_confidence_ece": float(ece),
            "retrieval_precision_at_k": float(np.mean([item["precision_at_k"] for item in retrieval])) if retrieval else None,
            "retrieval_recall_at_k": float(np.mean([item["recall_at_k"] for item in retrieval])) if retrieval else None,
            "retrieval_mrr": float(np.mean([item["mrr"] for item in retrieval])) if retrieval else None,
            "retrieval_ndcg_at_k": float(np.mean([item["ndcg_at_k"] for item in retrieval])) if retrieval else None,
            "retrieval_per_query": retrieval,
            "retrieval_query_count": len(retrieval),
            "limitation": ("Synthetic smoke data verifies behavior, not real-world accuracy." if kind == "synthetic_smoke"
                           else "Sentence-level benchmark results do not establish Indian-market price-signal accuracy or calibrated confidence.")}
