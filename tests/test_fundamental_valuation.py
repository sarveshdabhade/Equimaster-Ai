"""Valuation arithmetic, coverage fallbacks, and financial-input safeguards."""
from copy import deepcopy
from datetime import datetime, timezone
import math

import pytest

from src.fundamental_valuation import estimate_stock_valuation


_MONEY_KEYS = {
    "price", "market_cap", "eps", "forward_eps", "book_value", "revenue", "profit",
    "ebitda", "net_debt", "cash", "debt", "cfo", "capex", "fcf", "fcfe", "net_borrowing",
}


def company(ticker, values=None, *, industry="Industrial Machinery", sector="Industrials", financial=False):
    metrics = {}
    for key, value in (values or {}).items():
        metrics[key] = {
            "value": value, "source": "Yahoo Finance", "period": "TTM / latest reported",
            "basis": "Provider reported", "currency": "INR" if key in _MONEY_KEYS else None,
        }
    return {
        "Ticker": ticker, "Company": ticker, "Industry": industry, "Sector": sector,
        "Currency": "INR", "Financial currency": "INR", "financial_sector": financial,
        "As of": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
    }


def peer(ticker, *, pe=20, pb=2, ev_ebitda=8, ev_sales=2, ps=2, **identity):
    return company(ticker, {
        "price": 100, "shares": 1_000_000, "eps": 5, "book_value": 50,
        "revenue": 100_000_000, "profit": 5_000_000, "ebitda": 10_000_000,
        "net_debt": 0, "pe": pe, "pb": pb, "ev_ebitda": ev_ebitda,
        "ev_sales": ev_sales, "ps": ps, "roe": 10,
    }, **identity)


def universe(*companies):
    return {item["Ticker"]: item for item in companies}


def test_universe_report_uses_cashflow_valuation_without_peers(monkeypatch, tmp_path):
    import scripts.valuation_coverage as coverage
    subject = company("SINGLE.NS", {"beta": 1, "price": 100, "shares": 1_000_000,
                      "fcfe": 5_000_000, "cfo": 10_000_000, "capex": 5_000_000, "net_borrowing": 0})
    monkeypatch.setattr(coverage, "resolve_app_path", lambda *parts: str(tmp_path.joinpath(*parts)))
    monkeypatch.setattr(coverage, "read_cached_details", lambda ticker: subject if ticker == "SINGLE.NS" else None)
    report = coverage.build_coverage_report(["SINGLE.NS"])
    assert report["summary"]["complete"]
    assert report["summary"]["accounted_for"] == 1
    row = report["results"][0]
    assert row["method"] == "5-year FCFE DCF"
    assert row["dcf"]["available"]
    assert row["valuation_parameters"]["beta"] == 1
    assert row["assumptions"]


def test_strict_peer_valuation_preserves_verified_currency():
    import pandas as pd
    from src.fundamental_swot import calculate_fundamental_valuation, load_valuation_assumptions
    identity = {"industry": "Banks - Regional", "sector": "Financial Services", "financial": True}
    subject = company("BANK.NS", {"book_value": 50, "pb": 3, "price": 150}, **identity)
    companies = universe(subject, peer("A.NS", pb=1, **identity), peer("B.NS", pb=2, **identity))
    for details in companies.values():
        details["metrics"]["pb"]["basis"] = "Consolidated"
    frame = pd.DataFrame([{"Ticker": ticker, "_details": details} for ticker, details in companies.items()])
    result = calculate_fundamental_valuation(subject, frame, load_valuation_assumptions("BANK.NS"), valuation_universe=companies)
    assert result["peer_value"]["available"]
    assert result["peer_value"]["currency"] == "INR"
    assert result["fair_value"] == pytest.approx(75)


def assert_positive_scenario(result):
    assert result["available"]
    assert all(math.isfinite(float(result[key])) for key in ("value", "low", "high"))
    assert 0 < result["low"] <= result["value"] <= result["high"]
    assert result["confidence"] in {"high", "medium", "low"}
    assert result["coverage"] in {"full", "partial", "scenario"}
    assert "scenario" in result["range_kind"].lower()


def test_industry_earnings_value_uses_peer_median_and_excludes_subject():
    subject = company("TARGET.NS", {"eps": 5, "price": 100, "pe": 999})
    companies = universe(subject, peer("A.NS", pe=12), peer("B.NS", pe=20), peer("C.NS", pe=28))
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    earnings = [part for part in result["components"] if part.get("metric_key") == "pe"]
    assert earnings
    assert earnings[0]["value"] == pytest.approx(100)
    assert "TARGET.NS" not in earnings[0]["peer_tickers"]
    assert result["value"] == pytest.approx(100)


def test_target_price_and_own_multiple_do_not_anchor_fair_value():
    subject = company("TARGET.NS", {"eps": 5, "price": 100, "pe": 999})
    peers = [peer("A.NS", pe=12), peer("B.NS", pe=20), peer("C.NS", pe=28)]
    first = estimate_stock_valuation(subject, universe(subject, *peers), current_price=40)
    changed = deepcopy(subject)
    changed["metrics"]["price"]["value"] = 1_000
    changed["metrics"]["pe"]["value"] = 1
    second = estimate_stock_valuation(changed, universe(changed, *peers), current_price=400)
    assert_positive_scenario(first)
    assert_positive_scenario(second)
    assert first["value"] == pytest.approx(second["value"])
    assert first["low"] == pytest.approx(second["low"])
    assert first["high"] == pytest.approx(second["high"])


def test_sector_fallback_is_explicit_and_has_low_confidence():
    subject = company("TARGET.NS", {"eps": 5}, industry="Unique Business")
    companies = universe(subject, peer("A.NS", pe=12), peer("B.NS", pe=20), peer("C.NS", pe=28))
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    assert result["peer_scope"] == "sector"
    assert result["confidence"] == "low"
    assert result["coverage"] == "scenario"
    assert result["limitations"]


def test_negative_cash_flow_company_can_use_revenue_without_inventing_cash_flow():
    subject = company("GROWTH.NS", {
        "price": 200, "shares": 1_000_000, "eps": -2, "profit": -2_000_000,
        "fcf": -10_000_000, "revenue": 100_000_000, "net_debt": 0,
    }, industry="Internet Retail", sector="Consumer Cyclical")
    companies = universe(subject, *[
        peer(ticker, ev_sales=multiple, ps=multiple, industry="Specialty Retail", sector="Consumer Cyclical")
        for ticker, multiple in (("A.NS", 1), ("B.NS", 2), ("C.NS", 3))
    ])
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    assert any(part.get("metric_key") in {"ev_sales", "ps"} for part in result["components"])
    assert result["coverage"] == "scenario"
    assert result["confidence"] == "low"
    assert subject["metrics"]["fcf"]["value"] == -10_000_000
    assert "fcfe" not in subject["metrics"]


def test_bank_prefers_book_value_without_using_industrial_ev_multiples():
    identity = {"industry": "Banks - Regional", "sector": "Financial Services", "financial": True}
    subject = company("BANK.NS", {"book_value": 50, "eps": -1}, **identity)
    companies = universe(subject, peer("A.NS", pb=1, **identity), peer("B.NS", pb=1.5, **identity), peer("C.NS", pb=2, **identity))
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    assert result["value"] == pytest.approx(75)
    assert any(part.get("metric_key") == "pb" for part in result["components"])
    assert not any(part.get("metric_key") in {"ev_sales", "ev_ebitda"} for part in result["components"])


def test_ev_multiple_adds_net_cash_when_converting_to_equity():
    subject = company("TARGET.NS", {"shares": 1_000_000, "ebitda": 10_000_000, "net_debt": -5_000_000})
    companies = universe(subject, peer("A.NS", ev_ebitda=5), peer("B.NS", ev_ebitda=7), peer("C.NS", ev_ebitda=9))
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    assert result["value"] == pytest.approx(75)


def test_missing_net_debt_uses_price_sales_instead_of_assuming_zero_debt():
    subject = company("TARGET.NS", {"shares": 1_000_000, "revenue": 100_000_000})
    companies = universe(subject, peer("A.NS", ps=1), peer("B.NS", ps=2), peer("C.NS", ps=3))
    result = estimate_stock_valuation(subject, companies)
    assert_positive_scenario(result)
    assert any(part.get("metric_key") == "ps" for part in result["components"])
    assert not any(part.get("metric_key") == "ev_sales" for part in result["components"])


def test_monetary_input_in_another_currency_is_withheld():
    subject = company("TARGET.NS", {"eps": 5, "price": 100})
    subject["metrics"]["eps"]["currency"] = "USD"
    companies = universe(subject, peer("A.NS"), peer("B.NS"), peer("C.NS"))
    result = estimate_stock_valuation(subject, companies)
    assert not result["available"]
    assert result.get("value") is None
    assert result["missing_inputs"] or result["limitations"]


def test_missing_shares_does_not_turn_total_sales_into_per_share_value():
    subject = company("TARGET.NS", {"revenue": 100_000_000, "price": 100})
    companies = universe(subject, peer("A.NS"), peer("B.NS"), peer("C.NS"))
    result = estimate_stock_valuation(subject, companies)
    assert not result["available"]
    assert result.get("value") is None


def test_current_price_and_self_multiple_alone_are_not_a_valuation():
    subject = company("TARGET.NS", {"price": 123, "pe": 20, "pb": 2})
    result = estimate_stock_valuation(subject, universe(subject), current_price=123)
    assert not result["available"]
    assert result.get("value") is None
    assert result["missing_inputs"]


def test_no_peer_earnings_scenario_requires_an_explicit_multiple():
    subject = company("TARGET.NS", {"eps": 5, "price": 100, "pe": 20})
    result = estimate_stock_valuation(subject, universe(subject))
    assert not result["available"]
    subject["valuation_assumptions"] = {"earnings_multiple": 12}
    result = estimate_stock_valuation(subject, universe(subject))
    assert_positive_scenario(result)
    assert result["value"] == pytest.approx(60)
    assert result["coverage"] == "scenario"
    assert result["confidence"] == "low"
    assert result["assumptions"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), None, -5, 0])
def test_invalid_earnings_are_not_replaced_with_a_fabricated_value(invalid):
    subject = company("TARGET.NS", {"eps": invalid, "price": 100})
    result = estimate_stock_valuation(subject, universe(subject, peer("A.NS"), peer("B.NS"), peer("C.NS")))
    assert not result["available"]
    assert result.get("value") is None


def test_industry_single_peer_is_an_explicit_low_confidence_scenario():
    subject = company("TARGET.NS", {"eps": 5})
    result = estimate_stock_valuation(subject, universe(subject, peer("ONLY.NS", pe=20)))
    assert_positive_scenario(result)
    assert result["confidence"] == "low"
    assert result["coverage"] == "scenario"
    assert result["peer_count"] == 1


def test_supplied_complete_models_blend_with_existing_weights():
    subject = company("TARGET.NS", {"price": 100})
    dcf = {"available": True, "value": 100, "method": "Verified FCFE DCF", "currency": "INR"}
    strict = {"available": True, "value": 110, "method": "Peer P/E", "peer_count": 3, "currency": "INR"}
    result = estimate_stock_valuation(subject, universe(subject), dcf=dcf, strict_peer=strict)
    assert_positive_scenario(result)
    assert result["value"] == pytest.approx(104)
    assert result["coverage"] == "full"
    assert result["confidence"] == "high"


def test_foreign_currency_supplied_dcf_does_not_become_an_inr_estimate():
    subject = company("TARGET.NS", {"price": 100})
    dcf = {"available": True, "value": 100, "method": "FCFE DCF", "currency": "USD"}
    result = estimate_stock_valuation(subject, universe(subject), dcf=dcf)
    assert not result["available"]
    assert result.get("value") is None


def test_nonfinite_supplied_dcf_is_rejected():
    subject = company("TARGET.NS", {"price": 100})
    dcf = {"available": True, "value": float("inf"), "method": "FCFE DCF", "currency": "INR"}
    result = estimate_stock_valuation(subject, universe(subject), dcf=dcf)
    assert not result["available"]
    assert result.get("value") is None
