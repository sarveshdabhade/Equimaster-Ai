"""Export a timestamped, deduplicated batch for independent news annotation.

Only publisher-linked headlines and short summaries are exported. Labels are
left blank; this tool never creates or guesses sentiment annotations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import os
import sys
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import ALL_DOWNLOAD_TICKERS, NEWS_RECENCY_DAYS


def _timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def build_annotation_rows(
    archive_root: Path,
    tickers: list[str],
    since: datetime,
    until: datetime,
) -> list[dict[str, str]]:
    """Return one blank-label row per ticker and canonical article URL."""
    collected: dict[tuple[str, str], dict[str, str]] = {}
    for ticker in tickers:
        directory = archive_root / ticker.replace(".", "_")
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for article in payload:
                published = _timestamp(article.get("published_at"))
                observed = _timestamp(article.get("observed_at"))
                url = str(article.get("canonical_url") or article.get("url") or "").strip()
                canonical = _canonical_url(url)
                title = " ".join(str(article.get("title") or "").split())
                if (not canonical or not title or published is None or observed is None
                        or published < since or published > until or observed > until):
                    continue
                key = (ticker, canonical)
                previous = collected.get(key)
                if previous is not None:
                    prior_observed = _timestamp(previous["observed_at"])
                    if prior_observed is not None and observed >= prior_observed:
                        continue
                row_id = hashlib.sha256(f"{ticker}\n{canonical}".encode("utf-8")).hexdigest()[:20]
                collected[key] = {
                    "id": row_id,
                    "ticker": ticker,
                    "title": title,
                    "summary": " ".join(str(article.get("summary") or "").split())[:1000],
                    "source": str(article.get("source") or ""),
                    "published_at": published.isoformat(),
                    "observed_at": observed.isoformat(),
                    "url": url,
                    "sentiment_label": "",
                    "annotator_id": "",
                    "confidence": "",
                    "adjudication_notes": "",
                }
    return sorted(collected.values(), key=lambda row: (row["published_at"], row["ticker"], row["id"]))


def export_batch(output: Path, days: int = NEWS_RECENCY_DAYS, as_of: datetime | None = None) -> dict:
    if days < 1:
        raise ValueError("days must be positive")
    until = as_of or datetime.now(timezone.utc)
    until = until.astimezone(timezone.utc) if until.tzinfo else until.replace(tzinfo=timezone.utc)
    since = until - timedelta(days=days)
    rows = build_annotation_rows(ROOT / "data" / "news_archive", ALL_DOWNLOAD_TICKERS, since, until)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "ticker", "title", "summary", "source", "published_at", "observed_at", "url",
              "sentiment_label", "annotator_id", "confidence", "adjudication_notes"]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    manifest = {
        "created_at": until.isoformat(),
        "window_days": days,
        "published_from": since.isoformat(),
        "published_through": until.isoformat(),
        "ticker_count": len({row["ticker"] for row in rows}),
        "row_count": len(rows),
        "unique_article_count": len({row["id"] for row in rows}),
        "label_status": "unlabeled; independent human annotation required",
        "source_policy": "verified archived articles with publication and observation timestamps",
        "redistribution_note": "Headlines and short summaries are retained with publisher URLs for local annotation. Review source-specific terms before redistribution.",
        "csv": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=NEWS_RECENCY_DAYS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "benchmarks" / f"indian_market_news_annotations_{datetime.now(timezone.utc):%Y%m%d}.csv")
    args = parser.parse_args()
    print(json.dumps(export_batch(args.output, args.days), indent=2))


if __name__ == "__main__":
    main()
