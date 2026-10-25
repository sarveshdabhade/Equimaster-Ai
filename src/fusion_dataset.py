"""Validation contract for point-in-time LSTM and news-fusion evaluation rows."""
from __future__ import annotations

from datetime import date, datetime, timezone
import math
from typing import Any, Mapping, Sequence

from config import HORIZON_CHOICES_DESCRIPTION, HORIZONS

FUSION_DATASET_SCHEMA_VERSION = 1
_LLM_STANCES = {"bearish", "bullish", "mixed", "neutral"}


def normalize_as_of(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("as_of must be an ISO timestamp") from exc
    else:
        raise ValueError("as_of must be an ISO timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_target_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError as exc:
                raise ValueError("target_date must be an ISO date or timestamp") from exc
    raise ValueError("target_date must be an ISO date or timestamp")


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def validate_fusion_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one forecast/evidence row for later fusion tests."""
    if not isinstance(record, Mapping):
        raise ValueError("Each fusion record must be an object")
    required = {
        "schema_version", "ticker", "as_of", "horizon", "target_date",
        "current_close", "lstm_predicted_close", "actual_close",
        "news_sentiment_score", "llm_stance", "llm_evidence_ids",
        "llm_model_id", "lstm_model_id", "evidence",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Fusion record is missing required fields: {', '.join(sorted(missing))}")
    unexpected = set(record) - required
    if unexpected:
        raise ValueError(f"Fusion record contains unsupported fields: {', '.join(sorted(unexpected))}")
    if type(record["schema_version"]) is not int or record["schema_version"] != FUSION_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported fusion dataset schema version: {record['schema_version']}")

    ticker = str(record["ticker"] or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    horizon = record["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of: {HORIZON_CHOICES_DESCRIPTION}")

    as_of = normalize_as_of(record["as_of"])
    target_date = normalize_target_date(record["target_date"])
    if target_date <= as_of.date():
        raise ValueError("target_date must be later than the forecast origin date")

    current_close = _finite_number(record["current_close"], "current_close", positive=True)
    predicted_close = _finite_number(record["lstm_predicted_close"], "lstm_predicted_close", positive=True)
    actual_close = _finite_number(record["actual_close"], "actual_close", positive=True)
    sentiment_score = _finite_number(record["news_sentiment_score"], "news_sentiment_score")
    if not -1.0 <= sentiment_score <= 1.0:
        raise ValueError("news_sentiment_score must be in the range -1..1")

    stance = record["llm_stance"]
    if not isinstance(stance, str) or stance not in _LLM_STANCES:
        raise ValueError("llm_stance must be bullish, bearish, neutral, or mixed")
    llm_model_id = str(record["llm_model_id"] or "").strip()
    lstm_model_id = str(record["lstm_model_id"] or "").strip()
    if not llm_model_id or not lstm_model_id:
        raise ValueError("llm_model_id and lstm_model_id are required for reproducibility")

    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must contain timestamped retrieved chunks")
    normalized_evidence = []
    evidence_ids = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("Each evidence record must be an object")
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in evidence_ids:
            raise ValueError("Evidence IDs must be non-empty and unique within a sample")
        published = normalize_as_of(item.get("published_at"))
        observed = normalize_as_of(item.get("observed_at"))
        if published > as_of or observed > as_of:
            raise ValueError(f"Evidence {evidence_id} was not available at the forecast origin")
        evidence_ids.add(evidence_id)
        normalized_evidence.append({
            "evidence_id": evidence_id,
            "published_at": published.isoformat(),
            "observed_at": observed.isoformat(),
        })

    cited_ids = record["llm_evidence_ids"]
    if not isinstance(cited_ids, list) or not cited_ids:
        raise ValueError("llm_evidence_ids must contain cited retrieved evidence IDs")
    normalized_citations = list(dict.fromkeys(
        str(value).strip() for value in cited_ids if isinstance(value, str) and value.strip()
    ))
    if len(normalized_citations) != len(cited_ids) or not set(normalized_citations).issubset(evidence_ids):
        raise ValueError("llm_evidence_ids must reference supplied evidence records")

    return {
        "schema_version": FUSION_DATASET_SCHEMA_VERSION,
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "horizon": horizon,
        "target_date": target_date.isoformat(),
        "current_close": current_close,
        "lstm_predicted_close": predicted_close,
        "actual_close": actual_close,
        "news_sentiment_score": sentiment_score,
        "llm_stance": stance,
        "llm_evidence_ids": normalized_citations,
        "llm_model_id": llm_model_id,
        "lstm_model_id": lstm_model_id,
        "evidence": normalized_evidence,
    }


def validate_fusion_dataset(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and chronologically order unique ticker/origin/horizon samples."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("At least one fusion dataset record is required")
    normalized = [validate_fusion_record(record) for record in records]
    keys = [(row["ticker"], row["as_of"], row["horizon"]) for row in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Fusion dataset contains duplicate ticker/as_of/horizon records")
    return sorted(normalized, key=lambda row: (row["ticker"], row["horizon"], row["as_of"]))


def llm_feature_row_from_sentiment(
    ticker: str,
    as_of: datetime | str,
    sentiment_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a successful sentiment result into cutoff-auditable LLM features."""
    synthesis = sentiment_result.get("llm_synthesis")
    if sentiment_result.get("llm_synthesis_status") != "generated" or not isinstance(synthesis, Mapping):
        raise ValueError("A successfully generated LLM synthesis is required")
    cited_ids = list(synthesis.get("summary_evidence_ids") or [])
    for claim in synthesis.get("claims") or []:
        cited_ids.extend(claim.get("evidence_ids") or [])
    cited_ids = list(dict.fromkeys(cited_ids))
    sources = synthesis.get("citation_sources")
    if not isinstance(sources, Mapping) or not cited_ids:
        raise ValueError("LLM synthesis must retain its cited evidence provenance")
    evidence = []
    for evidence_id in cited_ids:
        source = sources.get(evidence_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"Cited evidence {evidence_id} is missing provenance")
        evidence.append({
            "evidence_id": evidence_id,
            "published_at": source.get("published_at"),
            "observed_at": source.get("observed_at"),
        })
    return {
        "ticker": str(ticker or "").strip().upper(),
        "as_of": normalize_as_of(as_of).isoformat(),
        "news_sentiment_score": sentiment_result.get("overall_score"),
        "llm_stance": synthesis.get("stance"),
        "llm_evidence_ids": cited_ids,
        "llm_model_id": synthesis.get("llm_model_id"),
        "evidence": evidence,
    }


def build_fusion_dataset(
    forecast_rows: Sequence[Mapping[str, Any]],
    llm_feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join each LSTM backtest row to news/LLM features from the same origin."""
    if not forecast_rows:
        raise ValueError("At least one LSTM forecast row is required")

    features_by_origin = {}
    for index, feature in enumerate(llm_feature_rows):
        if not isinstance(feature, Mapping):
            raise ValueError(f"LLM feature row {index} must be an object")
        ticker = str(feature.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError(f"LLM feature row {index} requires a ticker")
        origin = normalize_as_of(feature.get("as_of")).isoformat()
        key = (ticker, origin)
        if key in features_by_origin:
            raise ValueError(f"Duplicate LLM feature row for {ticker} at {origin}")
        features_by_origin[key] = feature

    records = []
    for index, forecast in enumerate(forecast_rows):
        if not isinstance(forecast, Mapping):
            raise ValueError(f"LSTM forecast row {index} must be an object")
        ticker = str(forecast.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError(f"LSTM forecast row {index} requires a ticker")
        origin = normalize_as_of(forecast.get("as_of")).isoformat()
        feature = features_by_origin.get((ticker, origin))
        if feature is None:
            raise ValueError(f"Missing point-in-time LLM features for {ticker} at {origin}")
        predicted_close = forecast.get("lstm_predicted_close", forecast.get("predicted_close"))
        records.append({
            "schema_version": FUSION_DATASET_SCHEMA_VERSION,
            "ticker": ticker,
            "as_of": origin,
            "horizon": forecast.get("horizon"),
            "target_date": forecast.get("target_date"),
            "current_close": forecast.get("current_close"),
            "lstm_predicted_close": predicted_close,
            "actual_close": forecast.get("actual_close"),
            "news_sentiment_score": feature.get("news_sentiment_score"),
            "llm_stance": feature.get("llm_stance"),
            "llm_evidence_ids": feature.get("llm_evidence_ids"),
            "llm_model_id": feature.get("llm_model_id"),
            "lstm_model_id": forecast.get("lstm_model_id"),
            "evidence": feature.get("evidence"),
        })
    return validate_fusion_dataset(records)
