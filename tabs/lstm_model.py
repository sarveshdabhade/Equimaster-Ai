import json
import os
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import FUSION_SHADOW_VERSION_ENV, HORIZONS, HORIZON_LABELS, horizon_description
from components.layout import CHART_STYLE, CHART_TEMPLATE
from src.artifacts import resolve_artifact, canonical_ticker
from src.forecast import predict_forecast
from src.fusion_artifacts import load_fusion_artifact
from src.fusion_model import predict_fused_forecast
from src.market_context import build_market_context
from src.news.sentiment import analyze_ticker_sentiment
from src.prediction_contract import forecast_signal, combined_signal, classify_signal


def _run_fusion_shadow(df, forecast, artifact, ticker, horizon, shadow_version):
    if not shadow_version:
        return None
    if artifact["legacy"] or not artifact["metadata"].get("version"):
        return {"status": "unavailable", "detail": "A versioned LSTM artifact is required."}

    session_date = pd.Timestamp(df.index[-1]).date()
    as_of = datetime.combine(
        session_date,
        time(15, 30),
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )
    sentiment = analyze_ticker_sentiment(
        ticker,
        max_articles=5,
        as_of=as_of,
        include_llm=True,
    )
    synthesis = sentiment.get("llm_synthesis")
    if not synthesis or sentiment.get("llm_synthesis_status") != "generated":
        return {
            "status": "unavailable",
            "detail": sentiment.get("llm_synthesis_error")
            or f"Timestamped archived evidence/LLM synthesis unavailable ({sentiment.get('llm_synthesis_status', 'unknown')}).",
        }

    cited_ids = list(synthesis.get("summary_evidence_ids") or [])
    for claim in synthesis.get("claims", []):
        cited_ids.extend(claim.get("evidence_ids") or [])
    cited_ids = list(dict.fromkeys(cited_ids))
    sources = synthesis.get("citation_sources", {})
    evidence = [
        {"evidence_id": evidence_id, **sources[evidence_id]}
        for evidence_id in cited_ids
        if evidence_id in sources
    ]
    request = {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "horizon": horizon,
        "current_close": float(df["Close"].iloc[-1]),
        "lstm_predicted_close": float(forecast["prices"][-1]),
        "news_sentiment_score": float(sentiment["overall_score"]),
        "llm_stance": synthesis["stance"],
        "llm_evidence_ids": cited_ids,
        "llm_model_id": synthesis["llm_model_id"],
        "lstm_model_id": artifact["metadata"]["version"],
        "evidence": evidence,
    }
    fusion_artifact = load_fusion_artifact(shadow_version)
    prediction = predict_fused_forecast(fusion_artifact, request)
    return {
        **prediction,
        "status": "shadow_candidate",
        "inference_status": prediction["status"],
    }


def _forecast_chart(df, result, window_size, horizon_label):
    history = pd.to_numeric(df["Close"], errors="coerce").dropna().tail(window_size)
    if history.empty:
        raise ValueError("Historical closing prices are unavailable for the forecast chart.")

    history_x = np.linspace(0, 50, len(history))
    prices = np.asarray(result["prices"], dtype=float)
    forecast_x = np.linspace(50, 100, len(prices) + 1)
    forecast_y = np.concatenate(([float(history.iloc[-1])], prices))

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=history_x,
        y=history.to_numpy(dtype=float),
        name="Historical close",
        mode="lines",
        line={"color": "#91A4BD", "width": 2},
        hovertemplate="Historical close: INR %{y:,.2f}<extra></extra>",
    ))
    chart.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        name="Direct forecast",
        mode="lines+markers",
        line={"color": "#00C9FF", "width": 3},
        marker={"size": 6},
        hovertemplate="Forecast path: INR %{y:,.2f}<extra></extra>",
    ))

    if result["lower"] is not None:
        lower = np.asarray(result["lower"], dtype=float)
        upper = np.asarray(result["upper"], dtype=float)
        chart.add_trace(go.Scatter(
            x=forecast_x,
            y=np.concatenate(([forecast_y[0]], upper)),
            mode="lines",
            line={"width": 0},
            name="Upper interval",
            showlegend=True,
            hoverinfo="skip",
        ))
        chart.add_trace(go.Scatter(
            x=forecast_x,
            y=np.concatenate(([forecast_y[0]], lower)),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(0,201,255,.16)",
            name="Lower interval",
            showlegend=True,
            hoverinfo="skip",
        ))

    chart.add_vrect(x0=0, x1=50, fillcolor="rgba(145,164,189,.035)", line_width=0, layer="below")
    chart.add_vrect(x0=50, x1=100, fillcolor="rgba(0,201,255,.035)", line_width=0, layer="below")
    chart.add_vline(x=50, line_dash="dash", line_color="#718096", line_width=1)
    chart.add_annotation(x=25, y=1.08, yref="paper", text="HISTORICAL PRICE",
                         showarrow=False, font={"size": 11, "color": "#BAC7DA"})
    chart.add_annotation(x=75, y=1.08, yref="paper", text="DIRECT FORECAST",
                         showarrow=False, font={"size": 11, "color": "#7CE6FF"})
    chart.update_layout(
        xaxis={
            "title": "Trading-session timeline",
            "range": [0, 100],
            "tickmode": "array",
            "tickvals": [0, 25, 50, 75, 100],
            "ticktext": [
                f"-{len(history)} sessions",
                "Recent history",
                "Last close",
                "Mid-horizon",
                horizon_label,
            ],
        },
        yaxis_title="Close (INR)",
        template=CHART_TEMPLATE,
        **CHART_STYLE,
        height=430,
        margin={"l": 15, "r": 20, "t": 48, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.16, "x": 1, "xanchor": "right"},
    )
    return chart


def render_lstm_tab(df, ticker, MODEL_PATH=None, SCALER_PATH=None):
    st.subheader("Deep Learning Price Forecast")
    ticker = canonical_ticker(ticker)
    horizon = st.selectbox("Forecast Period", HORIZONS, format_func=HORIZON_LABELS.__getitem__)
    st.caption(
        "Day = 1 session · Week = 5 · Month = 22 · Quarter = 88 "
        "(current quarter model spans approximately four months)."
    )
    try:
        artifact = resolve_artifact(ticker, horizon)
    except FileNotFoundError:
        st.warning("LSTM Brain offline. Model files not found.")
        return
    metadata = artifact["metadata"]
    if artifact["legacy"]:
        st.warning("Legacy model: training dates and evaluation are unavailable. Retrain with the backend training pipeline.")
    else:
        st.caption(f"Model trained {metadata['trained_at']} | Training dataset through {metadata['data_end']}")
    shadow_version = os.getenv(FUSION_SHADOW_VERSION_ENV, "").strip()
    signature = (
        ticker, horizon, str(artifact["model"]),
        str(df.index[-1]) if df is not None and len(df) else None,
        shadow_version,
    )
    st.caption(
        "Direct forecast: the selected horizon's model predicts the full path in one pass. "
        "Historical holdout metrics below show measured accuracy for this ticker and period."
    )
    if st.button("Generate Direct Forecast", type="primary"):
        try:
            with st.spinner("Computing forecast and news context..."):
                forecast = predict_forecast(df, ticker, horizon)
                context = build_market_context(ticker, horizon)
            shadow = None
            if shadow_version:
                try:
                    shadow = _run_fusion_shadow(
                        df, forecast, artifact, ticker, horizon, shadow_version
                    )
                except Exception as exc:
                    shadow = {"status": "error", "detail": str(exc)}
            st.session_state["forecast_result"] = (signature, forecast, context, shadow)
        except Exception as exc:
            st.error(f"Unable to forecast: {exc}")
    stored = st.session_state.get("forecast_result")
    if stored and len(stored) == 4 and stored[0] == signature:
        _, result, context, shadow = stored
        prices = result["prices"]
        current = float(df["Close"].iloc[-1])
        predicted_return = float(prices[-1] / current - 1)
        sentiment = float(context.get("sentiment", {}).get("overall_score", 0))
        signal = combined_signal(predicted_return, sentiment)
        a, b, c = st.columns(3)
        a.metric(f"Predicted Close · {HORIZON_LABELS[horizon]}", f"INR {prices[-1]:,.2f}", f"{predicted_return:+.2%}")
        b.metric("Combined Signal", f"{signal:+.2f}", classify_signal(signal))
        c.metric("News Evidence", len(context.get("sentiment", {}).get("evidence", [])))
        if shadow_version:
            with st.expander("Experimental LLM + LSTM shadow forecast"):
                st.caption(
                    "Shadow-only: this candidate does not change the LSTM forecast, "
                    "combined signal, chart, or exported CSV."
                )
                if shadow and shadow.get("status") == "shadow_candidate":
                    fused_close = float(shadow["fused_predicted_close"])
                    st.metric(
                        f"Fusion candidate close · {HORIZON_LABELS[horizon]}",
                        f"INR {fused_close:,.2f}",
                        f"{fused_close / float(prices[-1]) - 1:+.2%} vs LSTM",
                    )
                    st.caption(
                        f"Fusion artifact {shadow.get('fusion_model_version')} · "
                        f"LLM {shadow.get('llm_model_id')} · "
                        f"LSTM {shadow.get('lstm_model_id')} · "
                        f"status: {shadow.get('status')}"
                    )
                elif shadow:
                    st.warning(
                        f"Shadow forecast unavailable ({shadow.get('status')}): "
                        f"{shadow.get('detail', 'No additional details.')}"
                    )
                else:
                    st.info("Shadow forecast has not been generated for this result.")
        steps = list(range(1, len(prices) + 1))
        st.caption(
            f"Selected period: {horizon_description(horizon)}. "
            "The left half shows recent closing prices; the right half shows the direct forecast."
        )
        chart = _forecast_chart(
            df,
            result,
            int(artifact["metadata"].get("window_size", 60)),
            HORIZON_LABELS[horizon],
        )
        table = pd.DataFrame({"Step": steps, "Forecast": prices})
        if result["lower"] is not None:
            chart.add_trace(go.Scatter(x=steps, y=result["upper"], mode="lines", line_width=0, name="Upper interval"))
            chart.add_trace(go.Scatter(x=steps, y=result["lower"], mode="lines", line_width=0,
                                      fill="tonexty", fillcolor="rgba(0,201,255,.18)", name="Lower interval"))
            table["Lower interval"] = result["lower"]
            table["Upper interval"] = result["upper"]
            st.caption("90% empirical validation intervals; test coverage is shown below. Coverage is not guaranteed.")
        else:
            st.caption("Calibrated prediction intervals are unavailable for this model.")
        st.plotly_chart(chart, width="stretch", theme=None)
        st.dataframe(table, hide_index=True)
        st.download_button("Export Forecast CSV", table.to_csv(index=False), f"{ticker}_forecast.csv", "text/csv")
    with st.expander(f"Direct Model Holdout Performance · {HORIZON_LABELS[horizon]}", expanded=True):
        entry = artifact["evaluation"]
        metrics = entry.get("metrics")
        if not metrics:
            st.info("No measured performance available; train a versioned model.")
        else:
            baseline = entry["baseline"]
            if metrics["mae"] >= baseline["mae"]:
                st.warning("This model did not beat the persistence baseline on held-out MAE.")
            a, b, c = st.columns(3)
            a.metric("Held-out directional accuracy", f"{metrics['lstm_directional_accuracy']:.1%}")
            b.metric("Held-out MAE", f"INR {metrics['mae']:,.2f}")
            c.metric("Held-out MAPE", f"{metrics['mape']:.2f}%")
            st.write(f"Persistence baseline MAE: INR {baseline['mae']:,.2f}; held-out samples: {metrics['sample_count']}.")
            st.write(f"Empirical interval test coverage: {entry['interval']['test_coverage']:.1%} (90% nominal target).")
            st.caption(
                f"Chronological held-out evaluation for {horizon_description(horizon)}; "
                "overlapping target dates were purged at fit/validation/test boundaries. "
                "Directional accuracy is a historical hit rate, not a probability or guarantee."
            )
            backtest = artifact["model"].parent / f"backtest_h{horizon}.csv"
            if backtest.exists():
                st.download_button("Export Historical Predictions", backtest.read_bytes(), backtest.name, "text/csv")
