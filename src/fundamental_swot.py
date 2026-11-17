"""Explainable fundamental SWOT, sector-aware fair value, and action policy."""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping

import pandas as pd

from src.fundamental_ml import predict_excess_return
from src.fundamentals import peer_reference
from src.fundamental_catalog import RANGES, SPECS

_DEFAULT_ASSUMPTIONS = {"risk_free_rate": 0.065, "equity_risk_premium": 0.055, "terminal_growth": 0.04}
_DCF_WEIGHT = 0.60
_PEER_WEIGHT = 0.40
_VALUE_BAND = 0.15
_ALPHA_BAND = 0.05


def load_valuation_assumptions(ticker: str, company_beta: float | None = None) -> dict[str, float | None]:
    """Use fixed research defaults and the beta reported for the company."""
    result = dict(_DEFAULT_ASSUMPTIONS)
    try:
        beta = float(company_beta) if company_beta is not None else None
    except (TypeError, ValueError):
        beta = None
    result["beta"] = beta if beta is not None and math.isfinite(beta) else None
    return result


def _value(details: Mapping[str, Any], key: str) -> float | None:
    metric = details.get("metrics", {}).get(key, {})
    if not isinstance(metric, Mapping):
        return None
    try:
        result = float(metric.get("value"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _financial_dcf(details: Mapping[str, Any], cost_of_equity: float, terminal_growth: float) -> dict[str, Any]:
    book_value = _value(details, "book_value")
    roe_percent = _value(details, "roe")
    payout_percent = _value(details, "payout")
    if book_value is None or book_value <= 0 or roe_percent is None:
        return {"available": False, "reason": "Positive book value and reported ROE are required for residual-income valuation."}
    book_currency = details.get("metrics", {}).get("book_value", {}).get("currency")
    quote_currency = details.get("Currency") or "INR"
    if book_currency in (None, "Unverified currency") or book_currency != quote_currency:
        return {"available": False, "reason": f"Book value currency ({book_currency or 'unverified'}) must match the share-price currency ({quote_currency})."}
    if cost_of_equity <= terminal_growth + 0.01:
        return {"available": False, "reason": "Cost of equity must exceed terminal growth by at least 1 percentage point."}
    roe = roe_percent / 100.0
    retention = 0.5 if payout_percent is None else 1.0 - min(1.0, max(0.0, payout_percent / 100.0))
    book = book_value
    premium = 0.0
    for year in range(1, 6):
        projected_roe = roe + (cost_of_equity - roe) * (year / 5.0)
        earnings = projected_roe * book
        residual_income = earnings - cost_of_equity * book
        premium += residual_income / ((1.0 + cost_of_equity) ** year)
        book += earnings * retention
    value = book_value + premium
    if not math.isfinite(value) or value <= 0:
        return {"available": False, "reason": "Residual-income valuation produced a non-positive value."}
    return {"available": True, "value": value, "method": "5-year residual income", "currency": quote_currency,
            "reason": "Book value plus five years of residual income; ROE fades to cost of equity."}


def _industrial_dcf(details: Mapping[str, Any], cost_of_equity: float, terminal_growth: float,
                    forecast_growth: float) -> dict[str, Any]:
    cash_flow = _value(details, "fcfe")
    fcf = _value(details, "fcf")
    cfo = _value(details, "cfo")
    capex = _value(details, "capex")
    net_borrowing = _value(details, "net_borrowing")
    shares = _value(details, "shares")
    if cash_flow is None:
        if cfo is not None and capex is not None and net_borrowing is None:
            reason = "Same-period net borrowing was not reported or could not be currency-verified, so FCFE cannot be calculated."
        else:
            reason = "FCFE requires same-period operating cash flow, capital expenditure, and reported net borrowing; one or more inputs are unavailable."
        if fcf is not None and fcf <= 0:
            fcf_metric = details.get("metrics", {}).get("fcf", {})
            fcf_currency = fcf_metric.get("currency") or details.get("Currency") or "currency units"
            fcf_period = fcf_metric.get("period") or "latest annual period"
            reason += f" CFO less capex was negative ({fcf:,.0f} {fcf_currency}, {fcf_period})."
        if shares is None or shares <= 0:
            reason += " A positive share count is also required."
        return {"available": False, "reason": reason}
    if cash_flow <= 0:
        return {"available": False, "reason": f"Annual FCFE is non-positive ({cash_flow:,.0f}); a positive cash flow is required for this DCF."}
    if shares is None or shares <= 0:
        return {"available": False, "reason": "A positive share count is required to convert equity value to value per share."}
    cash_currency = details.get("metrics", {}).get("fcfe", {}).get("currency")
    quote_currency = details.get("Currency") or "INR"
    if cash_currency in (None, "Unverified currency") or cash_currency != quote_currency:
        return {"available": False, "reason": f"FCFE currency ({cash_currency or 'unverified'}) must match the share-price currency ({quote_currency})."}
    if cost_of_equity <= terminal_growth + 0.01:
        return {"available": False, "reason": "Cost of equity must exceed terminal growth by at least 1 percentage point."}
    rates = [forecast_growth + (terminal_growth - forecast_growth) * (year / 4.0)
             for year in range(5)]
    present_value = 0.0
    projected = cash_flow
    for index, growth in enumerate(rates, start=1):
        projected *= 1.0 + growth
        present_value += projected / ((1.0 + cost_of_equity) ** index)
    terminal_value = projected * (1.0 + terminal_growth) / (cost_of_equity - terminal_growth)
    present_value += terminal_value / ((1.0 + cost_of_equity) ** 5)
    value = present_value / shares
    if not math.isfinite(value) or value <= 0:
        return {"available": False, "reason": "Equity DCF produced a non-positive value."}
    period = details.get("metrics", {}).get("fcfe", {}).get("period")
    return {"available": True, "value": value, "method": "5-year FCFE DCF", "currency": cash_currency,
            "cash_flow_period": period, "forecast_growth": forecast_growth,
            "reason": "Verified annual FCFE grows from the blended sales-growth estimate, then fades to terminal growth."}


def _peer_value(details: Mapping[str, Any], peers, financial_sector: bool) -> dict[str, Any]:
    if financial_sector:
        peer_multiples = peer_reference(peers, details, "pb")
        book_value = _value(details, "book_value")
        book_currency = details.get("metrics", {}).get("book_value", {}).get("currency")
        quote_currency = details.get("Currency") or "INR"
        if (len(peer_multiples) >= 2 and book_value is not None and book_value > 0
                and book_currency == quote_currency):
            multiple = median(peer_multiples)
            return {"available": True, "value": multiple * book_value, "method": "Peer P/B", "multiple": multiple, "currency": quote_currency,
                    "peer_count": len(peer_multiples), "reason": "Selected same-industry peers with matching source, period, and reporting basis."}
        peer_multiples = peer_reference(peers, details, "pe")
        eps = _value(details, "eps")
        eps_currency = details.get("metrics", {}).get("eps", {}).get("currency")
        if (len(peer_multiples) >= 2 and eps is not None and eps > 0
                and eps_currency == (details.get("Currency") or "INR")):
            multiple = median(peer_multiples)
            return {"available": True, "value": multiple * eps, "method": "Peer P/E fallback", "multiple": multiple, "currency": quote_currency,
                    "peer_count": len(peer_multiples), "reason": "P/B unavailable; using matched same-industry P/E peers."}
        return {"available": False, "reason": (
            f"P/B: {len(peer_reference(peers, details, 'pb'))} matched peers; at least two, positive book value, and matching share-price currency are required. "
            f"P/E fallback: {len(peer_multiples)} matched peers; at least two, positive EPS, and matching share-price currency are required."
        )}

    peer_multiples = peer_reference(peers, details, "ev_ebitda")
    ebitda, net_debt, shares = (_value(details, key) for key in ("ebitda", "net_debt", "shares"))
    ebitda_currency = details.get("metrics", {}).get("ebitda", {}).get("currency")
    debt_currency = details.get("metrics", {}).get("net_debt", {}).get("currency")
    quote_currency = details.get("Currency") or "INR"
    if (len(peer_multiples) >= 2 and ebitda is not None and ebitda > 0 and net_debt is not None
            and shares is not None and shares > 0 and ebitda_currency == debt_currency
            and ebitda_currency == quote_currency and ebitda_currency not in (None, "Unverified currency")):
        multiple = median(peer_multiples)
        value = (multiple * ebitda - net_debt) / shares
        if math.isfinite(value) and value > 0:
            return {"available": True, "value": value, "method": "Peer EV/EBITDA", "multiple": multiple, "currency": quote_currency,
                    "peer_count": len(peer_multiples), "reason": "Matched peer multiple; net debt and EBITDA currencies verified."}
    peer_multiples = peer_reference(peers, details, "pe")
    eps = _value(details, "eps")
    eps_currency = details.get("metrics", {}).get("eps", {}).get("currency")
    if (len(peer_multiples) >= 2 and eps is not None and eps > 0
            and eps_currency == quote_currency):
        multiple = median(peer_multiples)
        return {"available": True, "value": multiple * eps, "method": "Peer P/E fallback", "multiple": multiple, "currency": quote_currency,
                "peer_count": len(peer_multiples), "reason": "EV/EBITDA unavailable; using matched same-industry P/E peers."}
    return {"available": False, "reason": (
        f"EV/EBITDA: {len(peer_reference(peers, details, 'ev_ebitda'))} matched peers; at least two, positive EBITDA, "
        "net debt, shares, and currencies matching the share price are required. "
        f"P/E fallback: {len(peer_multiples)} matched peers; at least two, positive EPS, and matching share-price currency are required. "
        "Peer matching also requires the same reported industry, source, period, and basis."
    )}


def _relative_factors(details: Mapping[str, Any], peers, keys: tuple[tuple[str, bool], ...], bucket: str,
                      family: str = "Fundamental risk") -> list[dict[str, Any]]:
    factors = []
    for key, higher_is_better in keys:
        subject = _value(details, key)
        peer_values = peer_reference(peers, details, key)
        if subject is None or len(peer_values) < 2:
            continue
        peer_median = median(peer_values)
        scale = max(abs(peer_median), abs(subject) * 0.05, 1e-6)
        relative = (subject - peer_median) / scale
        if not higher_is_better:
            relative *= -1.0
        if relative >= 0.15:
            polarity, score = "positive", 1.0
        elif relative <= -0.15:
            polarity, score = "negative", -1.0
        else:
            continue
        spec = details.get("metrics", {}).get(key, {})
        factors.append({
            "bucket": bucket if (bucket == "Strengths" and score > 0) or (bucket == "Weaknesses" and score < 0) else "Weaknesses" if score < 0 else "Strengths",
            "family": family,
            "score": score,
            "title": spec.get("label", key),
            "evidence": f"{subject:.2f} vs selected-peer median {peer_median:.2f} ({relative:+.0%} on the favorable scale).",
            "source": "Fundamentals vs selected same-industry peers",
        })
    return factors


def _valuation_profitability_review(details: Mapping[str, Any], peers) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every available valuation/profitability metric and retain neutral context."""
    factors: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    metrics = details.get("metrics", {})
    financial = bool(details.get("financial_sector"))
    for key, spec in SPECS.items():
        category = spec["group"]
        sector_disclosure = financial and category == "Financial-sector disclosures"
        if category not in ("Valuation", "Profitability") and not sector_disclosure:
            continue
        metric = metrics.get(key, {})
        subject = _value(details, key)
        if subject is None:
            continue

        label = spec["label"]
        peer_values = peer_reference(peers, details, key)
        peer_median = median(peer_values) if len(peer_values) >= 2 else None
        score = 0.0
        status = "Context only"
        evidence = f"Reported value: {subject:.2f}. No directional scoring rule is appropriate."

        if key == "enterprise_value":
            status = "Scale context; not scored"
        elif peer_median is not None:
            ratio = (subject - peer_median) / max(abs(peer_median), abs(subject) * 0.05, 1e-6)
            lower_is_better = (
                (category == "Valuation" and key not in ("earnings_yield", "fcf_yield"))
                or key in ("gnpa", "nnpa")
            )
            if lower_is_better:
                ratio *= -1.0
            if ratio >= 0.15:
                score, status = 1.0, "Strength" if category == "Profitability" or key in ("earnings_yield", "fcf_yield") else "Potential value strength"
            elif ratio <= -0.15:
                score, status = -1.0, "Weakness" if category == "Profitability" or key in ("earnings_yield", "fcf_yield") else "Potential valuation concern"
            else:
                status = "In line with peers"
            evidence = f"{subject:.2f} vs matched same-industry peer median {peer_median:.2f} ({ratio:+.0%} on the favorable scale; n={len(peer_values)})."
        elif category == "Profitability":
            if key in ("roe", "roce", "roce_annual", "roic"):
                score = 1.0 if subject >= 15 else -1.0 if subject < 10 else 0.0
            elif key == "roa":
                score = 1.0 if subject >= 5 else -1.0 if subject < 2 else 0.0
            elif key in ("gross_margin", "operating_margin", "ebitda_margin", "net_margin", "fcf_margin"):
                score = 1.0 if subject >= 20 else -1.0 if subject < 10 else 0.0
            if score:
                status = "Strength" if score > 0 else "Weakness"
                evidence = f"{subject:.2f}%; application screening guide used because two matched peers were unavailable."
            else:
                status = "No clear signal"
                evidence = f"{subject:.2f}%; between the screening guide's strength and weakness thresholds."
        elif sector_disclosure:
            if key in ("gnpa", "nnpa"):
                score = -1.0 if subject >= 5 else 1.0 if subject <= 2 else 0.0
            elif key == "capital_adequacy":
                score = 1.0 if subject >= 15 else -1.0 if subject < 12 else 0.0
            if score:
                status = "Strength" if score > 0 else "Weakness"
                evidence = f"{subject:.2f}%; sector screening guide used because two matched peers were unavailable."
            else:
                status = "No clear signal"
                evidence = f"{subject:.2f}%; peer comparison is preferred for this sector metric."
        elif category == "Valuation":
            rule = spec.get("rule")
            if rule in RANGES:
                limits, labels, _ = RANGES[rule]
                assessment = labels[sum(subject >= threshold for threshold in limits)]
                if key in ("earnings_yield", "fcf_yield"):
                    score = 1.0 if assessment in ("Moderate cash yield", "High cash yield") else -1.0 if assessment == "Negative cash yield" else 0.0
                elif assessment in ("Low multiple", "Low vs growth"):
                    score = 1.0
                elif assessment in ("High multiple", "Very high multiple", "High vs growth"):
                    score = -1.0
            elif key in ("earnings_yield", "fcf_yield"):
                score = 1.0 if subject >= 6 else -1.0 if subject < 0 else 0.0
            elif key in ("pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales", "price_cfo", "price_fcf", "peg") and subject <= 0:
                status = "Not meaningful"
            if score:
                status = "Potential value strength" if score > 0 else "Potential valuation concern"
                evidence = f"{subject:.2f}{'%' if spec['unit'] == 'percent' else 'x'}; application screening guide used because two matched peers were unavailable."
            elif status != "Not meaningful":
                status = "No clear signal"
                evidence = f"{subject:.2f}{'%' if spec['unit'] == 'percent' else 'x'}; no material peer or screening signal."

        source = str(metric.get("source") or "Source unavailable")
        period = str(metric.get("period") or "Period unavailable")
        review.append({"Parameter": label, "Value": subject, "Peer median": peer_median,
                       "Unit": spec["unit"], "Review": status, "Period": period, "Source": source})
        if score:
            bucket = "Strengths" if score > 0 else "Weaknesses"
            factors.append({"bucket": bucket, "family": category, "score": score, "title": label,
                            "evidence": evidence, "source": source})
    return factors, review


def _news_factor(news: Mapping[str, Any] | None, positive: bool) -> dict[str, Any] | None:
    if not news or news.get("status") != "available":
        return None
    score = float(news.get("overall_score") or 0.0)
    if (positive and score < 0.20) or (not positive and score > -0.20):
        return None
    candidates = [item for item in news.get("evidence", [])
                  if (item.get("score", 0) > 0.1 if positive else item.get("score", 0) < -0.1)]
    top = max(candidates, key=lambda item: abs(float(item.get("score") or 0)), default=None)
    if not top:
        return None
    return {"bucket": "Opportunities" if positive else "Threats", "score": 1.0 if positive else -1.0,
            "family": "News",
            "title": "Positive verified news catalyst" if positive else "Negative verified news catalyst",
            "evidence": str(top.get("title") or "Company news") + f" ({top.get('source', 'verified publisher')}).",
            "source": "Recent verified company news"}


def calculate_fundamental_valuation(
    details: Mapping[str, Any],
    peer_frame,
    assumptions: Mapping[str, float],
    current_price: float | None = None,
    valuation_universe=None,
) -> dict[str, Any]:
    """Share the same automatic valuation across company pages and universe reports."""
    from src.fundamental_valuation import estimate_stock_valuation

    universe = {}
    if isinstance(valuation_universe, Mapping):
        candidates = valuation_universe.values()
    else:
        candidates = valuation_universe or ()
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("Ticker"):
            universe[candidate["Ticker"]] = candidate
    if peer_frame is not None and not peer_frame.empty and "_details" in peer_frame:
        for candidate in peer_frame["_details"]:
            if isinstance(candidate, Mapping) and candidate.get("Ticker"):
                universe[candidate["Ticker"]] = candidate
    universe[details.get("Ticker", "")] = details
    automatic_peers = pd.DataFrame([
        {"Ticker": stock, "_details": candidate} for stock, candidate in universe.items()
    ])
    financial = bool(details.get("financial_sector"))
    risk_free = float(assumptions["risk_free_rate"])
    risk_premium = float(assumptions["equity_risk_premium"])
    terminal_growth = float(assumptions["terminal_growth"])
    try:
        beta = float(assumptions.get("beta"))
    except (TypeError, ValueError):
        beta = None
    if beta is not None and not math.isfinite(beta):
        beta = None
    cost_of_equity = risk_free + beta * risk_premium if beta is not None else None
    sales_cagr = _value(details, "sales_cagr_5")
    recent_growth = _value(details, "revenue_growth")
    growth_inputs = [((sales_cagr / 100.0), 0.70) if sales_cagr is not None else None,
                     ((recent_growth / 100.0), 0.30) if recent_growth is not None else None]
    growth_inputs = [item for item in growth_inputs if item is not None]
    forecast_growth = sum(value * weight for value, weight in growth_inputs) / sum(weight for _, weight in growth_inputs) if growth_inputs else terminal_growth
    forecast_growth = min(0.25, max(-0.10, forecast_growth))

    dcf = ({"available": False, "reason": "Company beta is unavailable; cost of equity cannot be estimated."}
           if cost_of_equity is None else
           _financial_dcf(details, cost_of_equity, terminal_growth) if financial else
           _industrial_dcf(details, cost_of_equity, terminal_growth, forecast_growth))
    peer_value = _peer_value(details, automatic_peers, financial)
    try:
        current_price = float(current_price)
    except (TypeError, ValueError):
        current_price = None
    if current_price is None or not math.isfinite(current_price) or current_price <= 0:
        current_price = _value(details, "price")
    valuation = estimate_stock_valuation(
        {**details, "valuation_assumptions": dict(assumptions)}, universe,
        current_price=current_price, dcf=dcf, strict_peer=peer_value,
    )
    fair_value = valuation.get("value") if valuation.get("available") else None
    valuation_status = valuation.get("coverage", "unavailable")
    upside = fair_value / current_price - 1.0 if fair_value is not None and current_price else None
    value_status = (
        "Below estimate" if upside is not None and upside >= _VALUE_BAND else
        "Above estimate" if upside is not None and upside <= -_VALUE_BAND else
        "Near estimate" if upside is not None else "Unavailable"
    )
    return {
        "ticker": details.get("Ticker"), "current_price": current_price,
        "dcf": dcf, "peer_value": peer_value, "valuation": valuation,
        "dcf_weight": _DCF_WEIGHT, "peer_weight": _PEER_WEIGHT,
        "fair_value": fair_value, "valuation_status": valuation_status,
        "valuation_low": valuation.get("low"), "valuation_high": valuation.get("high"),
        "valuation_method": valuation.get("method"), "valuation_confidence": valuation.get("confidence"),
        "value_status": value_status, "upside": upside,
        "cost_of_equity": cost_of_equity, "forecast_growth": forecast_growth,
        "terminal_growth": terminal_growth,
    }


def analyze_fundamental_swot(
    ticker: str,
    details: Mapping[str, Any],
    peer_frame,
    current_price: float | None,
    news: Mapping[str, Any] | None,
    assumptions: Mapping[str, float],
    valuation_universe=None,
) -> dict[str, Any]:
    """Build evidence-backed SWOT, automatic valuation, and a governed action."""
    financial = bool(details.get("financial_sector"))
    value_context = calculate_fundamental_valuation(
        details, peer_frame, assumptions, current_price, valuation_universe,
    )
    fair_value = value_context["fair_value"]
    valuation_status = value_context["valuation_status"]
    upside = value_context["upside"]

    internal_keys = (() if financial else
        (("net_debt_ebitda", False), ("debt_equity", False), ("interest_cover", True), ("cfo_profit", True)))
    factors = _relative_factors(details, peer_frame, internal_keys, "Strengths", family="Balance sheet")
    metric_factors, metric_review = _valuation_profitability_review(details, peer_frame)
    factors.extend(metric_factors)
    sales_growth = _value(details, "revenue_growth")
    earnings_growth = _value(details, "earnings_growth")
    if sales_growth is not None:
        if sales_growth >= 5:
            factors.append({"bucket": "Opportunities", "family": "Growth", "score": 1.0, "title": "Revenue growth",
                            "evidence": f"Latest reported revenue growth is {sales_growth:.1f}%.", "source": "Published fundamentals"})
        elif sales_growth < 0:
            factors.append({"bucket": "Threats", "family": "Growth", "score": -1.0, "title": "Revenue contraction",
                            "evidence": f"Latest reported revenue growth is {sales_growth:.1f}%.", "source": "Published fundamentals"})
    if earnings_growth is not None and earnings_growth < 0:
        factors.append({"bucket": "Threats", "family": "Growth", "score": -1.0, "title": "Earnings contraction",
                        "evidence": f"Latest reported earnings growth is {earnings_growth:.1f}%.", "source": "Published fundamentals"})
    if upside is not None and upside >= _VALUE_BAND:
        factors.append({"bucket": "Opportunities", "family": "Valuation", "score": 1.0, "title": "Price below valuation estimate",
                        "evidence": f"The valuation estimate is {upside:+.1%} above the current price.", "source": value_context["valuation_method"] or "Fundamental valuation"})
    elif upside is not None and upside <= -_VALUE_BAND:
        factors.append({"bucket": "Threats", "family": "Valuation", "score": -1.0, "title": "Price above valuation estimate",
                        "evidence": f"The valuation estimate is {upside:+.1%} below the current price.", "source": value_context["valuation_method"] or "Fundamental valuation"})
    positive_news = _news_factor(news, True)
    negative_news = _news_factor(news, False)
    if positive_news:
        factors.append(positive_news)
    if negative_news:
        factors.append(negative_news)

    grouped = {key: [factor for factor in factors if factor["bucket"] == key]
               for key in ("Strengths", "Weaknesses", "Opportunities", "Threats")}
    family_scores: dict[str, list[float]] = {}
    for factor in factors:
        family_scores.setdefault(factor.get("family", factor["bucket"]), []).append(float(factor["score"]))
    family_means = [sum(values) / len(values) for values in family_scores.values()]
    swot_score = 100.0 * sum(family_means) / len(family_means) if family_means else None
    swot_label = (
        "Mostly positive evidence" if swot_score is not None and swot_score >= 30 else
        "Leaning positive" if swot_score is not None and swot_score >= 10 else
        "Mixed evidence" if swot_score is not None and swot_score > -10 else
        "Leaning negative" if swot_score is not None and swot_score > -30 else
        "Mostly negative evidence" if swot_score is not None else "Insufficient evidence"
    )

    features = {key: _value(details, key) for key in details.get("metrics", {})}
    features["financial_sector"] = financial
    ml = predict_excess_return(features)
    alpha = ml.get("expected_excess_return") if ml.get("available") else None
    full_valuation = valuation_status == "full"
    if alpha is None:
        action = "PENDING ML DATA"
        if valuation_status == "unavailable":
            value_note = "A valuation needs additional financial inputs."
        else:
            value_note = "The valuation estimate is available."
        action_reason = f"{value_note} Return recommendations remain pending until the 12-month model passes validation."
    elif alpha <= -_ALPHA_BAND:
        action = "SELL-TO-EXIT"
        action_reason = f"Validated model forecasts {alpha:+.1%} 12-month excess return versus NIFTY 50."
    elif full_valuation and upside is not None and upside <= -_VALUE_BAND:
        action = "SELL-TO-EXIT"
        action_reason = f"Full DCF/peer valuation indicates {upside:+.1%} downside to fair value."
    elif full_valuation and upside is not None and upside >= _VALUE_BAND and alpha >= _ALPHA_BAND:
        action = "BUY"
        action_reason = f"Full fair value indicates {upside:+.1%} upside and the model forecasts {alpha:+.1%} excess return."
    else:
        action = "HOLD"
        action_reason = ("A Buy requires both full DCF/peer coverage and the selected value/return thresholds."
                         if not full_valuation else "The selected Buy/Sell thresholds are not met.")

    return {
        **value_context,
        "ticker": ticker,
        "swot_score": swot_score,
        "swot_label": swot_label,
        "factor_count": len(factors),
        "score_family_count": len(family_scores),
        "swot": grouped,
        "valuation_profitability_review": metric_review,
        "ml": ml,
        "action": action,
        "action_reason": action_reason,
        "thresholds": {"value_band": _VALUE_BAND, "alpha_band": _ALPHA_BAND},
        "snapshot_as_of": details.get("As of"),
    }
