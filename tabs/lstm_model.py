import json
import math
import os
from datetime import datetime, time, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import FUSION_SHADOW_VERSION_ENV, HORIZONS, HORIZON_LABELS
from components.layout import CHART_STYLE, CHART_TEMPLATE
from src.artifacts import resolve_artifact, canonical_ticker
from src.forecast import predict_forecast
from src.fusion_artifacts import load_fusion_artifact
from src.fusion_model import predict_fused_forecast
from src.market_context import build_market_context
from src.news.sentiment import analyze_ticker_sentiment
from src.prediction_fusion import fundamental_signal, fuse_forecast


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


def _forecast_chart(df, result, window_size, horizon, horizon_label):
    history = pd.to_numeric(df["Close"], errors="coerce").dropna().tail(window_size)
    if history.empty:
        raise ValueError("Historical closing prices are unavailable for the forecast chart.")

    history_x = np.arange(len(history), dtype=float)
    last_x = float(len(history) - 1)
    horizon_sessions = max(1, int(horizon))
    # Preserve the Day < Week < Month < Quarter ordering without leaving a
    # large empty region for the Quarter target.
    compressed_gap = 0.35 + 1.45 * math.log1p(horizon_sessions) / math.log1p(max(HORIZONS))
    target_x = last_x + compressed_gap
    actual_close = float(history.iloc[-1])
    final_prices = result.get("final_prices")
    estimate = float(np.asarray(final_prices if final_prices is not None else result["prices"], dtype=float)[-1])

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
        x=[last_x, target_x],
        y=[actual_close, estimate],
        name="Estimate connector",
        mode="lines",
        line={"color": "#8BE28B", "width": 2, "dash": "dot"},
        showlegend=False,
        hoverinfo="skip",
    ))
    chart.add_trace(go.Scatter(
        x=[target_x],
        y=[estimate],
        name=f"Final estimate · {horizon_label} target",
        mode="markers",
        marker={"color": "#8BE28B", "size": 13, "symbol": "diamond", "line": {"color": "#E5FFE5", "width": 1}},
        hovertemplate=f"{horizon_label} target estimate: INR %{{y:,.2f}}<extra></extra>",
    ))
    chart.add_vline(x=last_x, line_dash="dash", line_color="#718096", line_width=1)
    chart.add_annotation(x=0.02, xref="paper", y=1.08, yref="paper", text="RECENT PRICE HISTORY",
                         showarrow=False, xanchor="left", font={"size": 11, "color": "#BAC7DA"})
    chart.add_annotation(x=0.98, xref="paper", y=1.08, yref="paper", text=f"{horizon_label.upper()} TARGET",
                         showarrow=False, xanchor="right", font={"size": 11, "color": "#8BE28B"})
    chart.update_layout(
        xaxis={
            "title": "Recent trading sessions · horizon gap compressed for readability",
            "range": [0, target_x + 1.5],
            "tickmode": "array",
            "tickvals": [0, max(0, last_x // 2), last_x, target_x],
            "ticktext": [f"-{len(history)} sessions", "Recent history", "Last close", f"{horizon_label} target"],
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
    try:
        artifact = resolve_artifact(ticker, horizon)
    except FileNotFoundError:
        st.warning("LSTM Brain offline. Model files not found.")
        return
    if artifact["legacy"]:
        st.warning("Legacy model: training dates and evaluation are unavailable. Retrain with the backend training pipeline.")
    shadow_version = os.getenv(FUSION_SHADOW_VERSION_ENV, "").strip()
    signature = (
        ticker, horizon, str(artifact["model"]),
        str(df.index[-1]) if df is not None and len(df) else None,
        shadow_version,
    )
    if st.button("Generate Final Estimate", type="primary"):
        try:
            with st.spinner("Computing forecast and news context..."):
                forecast = predict_forecast(df, ticker, horizon)
                context = build_market_context(ticker, horizon, include_llm=True)
                fundamentals = fundamental_signal(ticker)
                final_estimate = fuse_forecast(
                    float(df["Close"].iloc[-1]), forecast["prices"], horizon,
                    df["Close"].to_numpy(), context.get("sentiment"), fundamentals,
                )
            shadow = None
            if shadow_version:
                try:
                    shadow = _run_fusion_shadow(
                        df, forecast, artifact, ticker, horizon, shadow_version
                    )
                except Exception as exc:
                    shadow = {"status": "error", "detail": str(exc)}
            st.session_state["forecast_result"] = (signature, forecast, context, shadow, final_estimate)
        except Exception as exc:
            st.error(f"Unable to forecast: {exc}")
    stored = st.session_state.get("forecast_result")
    if stored and len(stored) == 5 and stored[0] == signature:
        _, result, context, shadow, final_estimate = stored
        prices = result["prices"]
        result["final_prices"] = final_estimate["final_prices"]
        predicted_return = float(final_estimate["final_return"])
        a, b = st.columns(2)
        a.metric(f"Final Estimate - {HORIZON_LABELS[horizon]}", f"INR {final_estimate['final_predicted_close']:,.2f}", f"{predicted_return:+.2%}")
        b.metric("LSTM baseline", f"INR {prices[-1]:,.2f}",
                 f"{final_estimate['final_predicted_close'] / float(prices[-1]) - 1:+.2%} vs LSTM")
        if shadow_version:
            with st.expander("Experimental LLM + LSTM shadow forecast"):
                st.caption(
                    "This separately trained LLM + LSTM candidate does not change the final research estimate."
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
        chart = _forecast_chart(
            df,
            result,
            int(artifact["metadata"].get("window_size", 60)),
            horizon,
            HORIZON_LABELS[horizon],
        )
        table = pd.DataFrame({
            "Step": steps,
            "LSTM forecast": prices,
            "Final research estimate": final_estimate["final_prices"],
        })
        if result["lower"] is not None:
            table["LSTM lower empirical interval"] = result["lower"]
            table["LSTM upper empirical interval"] = result["upper"]
        st.plotly_chart(chart, width="stretch", theme=None)
        st.dataframe(table, hide_index=True)
        st.download_button("Export Forecast CSV", table.to_csv(index=False), f"{ticker}_forecast.csv", "text/csv")
