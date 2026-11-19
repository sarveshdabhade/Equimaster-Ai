"""Transparent peer and cash-flow valuation scenarios for company snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from statistics import median
from typing import Any, Mapping


_DCF_WEIGHT = 0.60
_PEER_WEIGHT = 0.40
_SCENARIO_RANGE = "Scenario dispersion, not a confidence interval or validated forecast"
_MONEY_KEYS = {
    "price", "market_cap", "eps", "forward_eps", "book_value", "revenue", "profit",
    "ebitda", "net_debt", "cash", "debt", "cfo", "capex", "fcf", "fcfe", "net_borrowing",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _metric(details: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    metrics = details.get("metrics")
    value = metrics.get(key) if isinstance(metrics, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _value(details: Mapping[str, Any], key: str, currency: str | None = None) -> float | None:
    metric = _metric(details, key)
    value = _number(metric.get("value"))
    if value is None:
        return None
    if key in _MONEY_KEYS:
        metric_currency = metric.get("currency")
        if not currency or metric_currency != currency or metric_currency == "Unverified currency":
            return None
    return value


def _snapshot_is_recent(details: Mapping[str, Any]) -> bool:
    stamp = details.get("As of")
    if not stamp:
        return False
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600
        return -24 <= age_hours <= 72
    except (TypeError, ValueError, OverflowError):
        return False


def _universe_values(universe: Any):
    if isinstance(universe, Mapping):
        return list(universe.values())
    try:
        return list(universe or ())
    except TypeError:
        return []


def _matching_peers(
    details: Mapping[str, Any], universe: Any, metric_key: str, currency: str | None, scope: str,
) -> list[tuple[Mapping[str, Any], float]]:
    subject_metric = _metric(details, metric_key)
    if not subject_metric:
        # The subject need not have a reported market multiple to be valued.
        # Match the peer multiple to the underlying target financial anchor.
        anchor_key = {
            "pe": "eps", "pb": "book_value", "ev_ebitda": "ebitda",
            "ev_sales": "revenue", "ps": "revenue",
        }.get(metric_key)
        subject_metric = _metric(details, anchor_key) if anchor_key else {}
    subject_ticker = str(details.get("Ticker") or "").upper()
    subject_sector = details.get("Sector")
    subject_industry = details.get("Industry")
    if not subject_metric or not subject_industry:
        return []
    matches = []
    for peer in _universe_values(universe):
        if not isinstance(peer, Mapping) or not _snapshot_is_recent(peer):
            continue
        peer_ticker = str(peer.get("Ticker") or "").upper()
        if not peer_ticker or peer_ticker == subject_ticker:
            continue
        if (peer.get("Currency") or peer.get("Financial currency")) != currency:
            continue
        if bool(peer.get("financial_sector")) != bool(details.get("financial_sector")):
            continue
        if scope == "industry":
            if peer.get("Industry") != subject_industry:
                continue
        elif not subject_sector or peer.get("Sector") != subject_sector:
            continue
        metric = _metric(peer, metric_key)
        # Keep comparisons on a common source, period, and reporting basis.
        if any(metric.get(field) != subject_metric.get(field) for field in ("source", "period", "basis")):
            continue
        value = _number(metric.get("value"))
        if value is None or value <= 0:
            continue
        if metric_key in _MONEY_KEYS and metric.get("currency") != currency:
            continue
        matches.append((peer, value))
    return matches


def _peer_scope(details: Mapping[str, Any], universe: Any, metric_key: str, currency: str | None):
    industry = _matching_peers(details, universe, metric_key, currency, "industry")
    sector = _matching_peers(details, universe, metric_key, currency, "sector")
    # Same-industry evidence is preferred. A sector fallback needs at least two
    # observations; one same-industry peer can still support an explicit scenario.
    if len(industry) >= 2:
        return industry, "industry"
    if len(sector) >= 2:
        return sector, "sector"
    if industry:
        return industry, "industry"
    return [], "unavailable"


def _component(metric_key: str, method: str, value: float, multiple: float | None = None,
               peers: list[tuple[Mapping[str, Any], float]] | None = None) -> dict[str, Any]:
    result = {"metric_key": metric_key, "method": method, "value": value}
    if multiple is not None:
        result["multiple"] = multiple
    if peers is not None:
        result["peer_tickers"] = [str(peer.get("Ticker")) for peer, _ in peers]
        result["peer_values"] = [peer_value for _, peer_value in peers]
    return result


def _peer_scenario(
    details: Mapping[str, Any], universe: Any, currency: str | None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    financial = bool(details.get("financial_sector"))
    missing: list[str] = []

    if financial:
        book = _value(details, "book_value", currency)
        if book is not None and book > 0:
            peers, scope = _peer_scope(details, universe, "pb", currency)
            if peers:
                mult = median(value for _, value in peers)
                return ([_component("pb", "Peer P/B", mult * book, mult, peers)], scope, missing)
        eps = _value(details, "eps", currency)
        if eps is not None and eps > 0:
            peers, scope = _peer_scope(details, universe, "pe", currency)
            if peers:
                mult = median(value for _, value in peers)
                return ([_component("pe", "Peer P/E fallback", mult * eps, mult, peers)], scope, missing)
        missing.extend(["Positive currency-verified book value or EPS", "At least one compatible same-industry peer multiple"])
        return [], "unavailable", missing

    ebitda = _value(details, "ebitda", currency)
    net_debt = _value(details, "net_debt", currency)
    shares = _value(details, "shares")
    if ebitda is not None and ebitda > 0 and net_debt is not None and shares is not None and shares > 0:
        peers, scope = _peer_scope(details, universe, "ev_ebitda", currency)
        if peers:
            values = [((multiple * ebitda - net_debt) / shares, peer) for peer, multiple in peers]
            values = [(value, peer) for value, peer in values if math.isfinite(value) and value > 0]
            if values:
                multiple = median(_number(_metric(peer, "ev_ebitda").get("value")) or 0 for _, peer in values)
                component = _component("ev_ebitda", "Peer EV/EBITDA", median(value for value, _ in values), multiple,
                                       [(peer, value) for value, peer in values])
                component["scenario_values"] = [value for value, _ in values]
                return [component], scope, missing

    eps = _value(details, "eps", currency)
    if eps is not None and eps > 0:
        peers, scope = _peer_scope(details, universe, "pe", currency)
        if peers:
            multiples = [value for _, value in peers]
            return ([_component("pe", "Peer P/E", median(multiples) * eps, median(multiples), peers)], scope, missing)

    revenue = _value(details, "revenue", currency)
    if revenue is not None and revenue > 0 and shares is not None and shares > 0:
        if net_debt is not None:
            peers, scope = _peer_scope(details, universe, "ev_sales", currency)
            if peers:
                values = [((multiple * revenue - net_debt) / shares, peer) for peer, multiple in peers]
                values = [(value, peer) for value, peer in values if math.isfinite(value) and value > 0]
                if values:
                    mult = median(value for _, value in peers)
                    component = _component("ev_sales", "Peer EV/Sales", median(value for value, _ in values), mult,
                                           [(peer, value) for value, peer in values])
                    component["scenario_values"] = [value for value, _ in values]
                    return [component], scope, missing
        peers, scope = _peer_scope(details, universe, "ps", currency)
        if peers:
            multiples = [value for _, value in peers]
            return ([_component("ps", "Peer P/S", median(multiples) * revenue / shares, median(multiples), peers)], scope, missing)

    assumptions = details.get("valuation_assumptions")
    multiple = _number(assumptions.get("earnings_multiple")) if isinstance(assumptions, Mapping) else None
    if eps is not None and eps > 0 and multiple is not None and 0 < multiple <= 100:
        assumption_peer = ({"Ticker": "Explicit valuation assumption"}, multiple)
        return ([_component("pe", "Explicit earnings-multiple scenario", eps * multiple, multiple,
                            [assumption_peer])], "assumption", missing)

    if ebitda is None or ebitda <= 0:
        missing.append("Positive currency-verified EBITDA, EPS, or revenue anchor")
    if shares is None or shares <= 0:
        missing.append("Positive share count for total-company revenue or EV scenarios")
    missing.append("Compatible same-industry peers or an explicit valuation multiple")
    return [], "unavailable", missing


def _supplied_model(model: Any, currency: str | None, name: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(model, Mapping) or not model.get("available"):
        return None, None
    value = _number(model.get("value"))
    model_currency = model.get("currency")
    if value is None or value <= 0:
        return None, f"{name} estimate is non-finite or non-positive."
    if not currency or model_currency != currency:
        return None, f"{name} currency ({model_currency or 'unverified'}) does not match share-price currency ({currency or 'unverified'})."
    return {"metric_key": name, "method": str(model.get("method") or name), "value": value,
            "peer_count": model.get("peer_count")}, None


def _unavailable(currency: str | None, missing: list[str], limitations: list[str], peer_scope: str = "unavailable"):
    return {
        "available": False, "value": None, "low": None, "high": None,
        "coverage": "unavailable", "confidence": "low", "method": "Unavailable",
        "currency": currency, "peer_scope": peer_scope, "peer_count": 0,
        "basis": "No supported valuation anchor", "range_kind": _SCENARIO_RANGE,
        "assumptions": [], "missing_inputs": list(dict.fromkeys(missing)),
        "limitations": list(dict.fromkeys(limitations)), "components": [],
    }


def estimate_stock_valuation(
    details: Mapping[str, Any],
    companies: Any,
    *,
    current_price: float | None = None,
    dcf: Mapping[str, Any] | None = None,
    strict_peer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate per-share value from supplied models or comparable fundamentals.

    The range describes scenario dispersion and is not a statistical interval.
    The subject's current price and reported valuation multiples are never used
    to create the fair-value anchor.
    """
    if not isinstance(details, Mapping):
        return _unavailable(None, ["Company fundamentals snapshot"], ["Input snapshot is invalid."])
    currency = details.get("Currency") or details.get("Financial currency")
    if not isinstance(currency, str) or not currency.strip() or currency == "Unverified currency":
        currency = None
    currency = currency.strip() if currency else None

    dcf_component, dcf_error = _supplied_model(dcf, currency, "DCF")
    peer_component, peer_error = _supplied_model(strict_peer, currency, "Strict peer")
    limitations = [error for error in (dcf_error, peer_error) if error]

    if dcf_component and peer_component:
        value = _DCF_WEIGHT * dcf_component["value"] + _PEER_WEIGHT * peer_component["value"]
        components = [dcf_component, peer_component]
        coverage, confidence, peer_scope = "full", "high", "industry"
        peer_count = int(peer_component.get("peer_count") or 0)
        method = f"60% {dcf_component['method']} / 40% {peer_component['method']}"
    elif dcf_component or peer_component:
        component = dcf_component or peer_component
        value = component["value"]
        components = [component]
        peer_count = int(component.get("peer_count") or 0)
        coverage = "partial"
        confidence = "medium" if dcf_component or peer_count >= 2 else "low"
        peer_scope = "industry" if peer_count else "model"
        method = component["method"]
        limitations.append("Only one supported valuation method is available; the result is a partial scenario.")
    else:
        components, peer_scope, missing = _peer_scenario(details, companies, currency)
        if not components:
            if not currency:
                missing.insert(0, "Verified share-price currency")
            if limitations:
                limitations.append("Supplied valuation methods were rejected; no compatible fallback anchor was available.")
            return _unavailable(currency, missing, limitations, peer_scope)
        component = components[0]
        value = component["value"]
        peer_count = len(component.get("peer_tickers", []))
        if peer_scope == "sector" or peer_scope == "assumption" or peer_count == 1:
            coverage, confidence = "scenario", "low"
        else:
            coverage = "partial"
            confidence = "high" if peer_count >= 5 else "medium"
        method = component["method"]
        if peer_scope == "sector":
            limitations.append("Same-industry peers were insufficient; the estimate uses sector peers as a lower-confidence scenario.")
        if peer_scope == "assumption":
            limitations.append("An explicit user-supplied multiple is used; this scenario is not peer-validated.")

    scenario_values = []
    for component in components:
        scenario_values.extend(_number(item) for item in component.get("scenario_values", []) if _number(item) is not None)
        if component.get("value") is not None:
            scenario_values.append(float(component["value"]))
    scenario_values = [item for item in scenario_values if item is not None and math.isfinite(item) and item > 0]
    if scenario_values:
        low, high = min([value, *scenario_values]), max([value, *scenario_values])
    else:
        low = high = value
    if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
        low, high = value * 0.80, value * 1.20
        limitations.append("Scenario bounds use a transparent +/-20% sensitivity because only one point estimate is available.")
    if not all(math.isfinite(item) and item > 0 for item in (value, low, high)):
        return _unavailable(currency, ["Finite positive valuation output"], limitations)

    assumptions = []
    snapshot_assumptions = details.get("valuation_assumptions")
    if isinstance(snapshot_assumptions, Mapping):
        assumptions.extend(f"{key}={item}" for key, item in snapshot_assumptions.items())
    current = _number(current_price)
    if current is None or current <= 0:
        current = _value(details, "price", currency)
    return {
        "available": True, "value": value, "low": low, "high": high,
        "coverage": coverage, "confidence": confidence, "method": method,
        "currency": currency, "peer_scope": peer_scope, "peer_count": peer_count,
        "basis": "Per-share scenario using verified snapshot inputs",
        "range_kind": _SCENARIO_RANGE, "assumptions": assumptions,
        "missing_inputs": [], "limitations": list(dict.fromkeys(limitations)),
        "components": components, "current_price": current,
    }
