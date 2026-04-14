import os
import subprocess
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import json
import streamlit as st
import logging
import tensorflow as tf
import plotly.graph_objects as go
from src.preprocessor import add_technical_indicators
from streamlit_lottie import st_lottie
from utils import load_lottieurl


def render_lstm_tab(df, ticker, MODEL_PATH, SCALER_PATH, CLOSE_COL_IDX=1):
    st.markdown('<h2 class="section-title"><span class="accent">🧠 Deep Learning</span> Price Forecast</h2>', unsafe_allow_html=True)
    
    # Phase 2: Toggle for model mode + confidence level
    col_mode, col_conf = st.columns([1, 2])
    with col_mode:
        use_iterative = st.toggle("🚀 Iterative Multi-Step", value=True, help="Use single-step model iteratively for longer horizons (more accurate)")
    with col_conf:
        confidence_level = st.slider("Confidence Bands", 70, 95, 85, help="Wider bands = higher confidence interval (mocked variance)")
    
    # Layout: main chart + sidebar info
    colA, colB = st.columns([3, 1])

    with colA:
        # Enhanced micro animation
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            lottie = load_lottieurl("https://assets9.lottiefiles.com/temp/lf20_UKH0om.json")
            if lottie:
                st_lottie(lottie, height=72, key=f"lstm_{ticker}_anim")

            # Enhanced horizon with confidence
            horizon = st.selectbox(
                "Forecast Horizon", 
                [1, 5, 22, 88], 
                format_func=lambda x: f"{ {1: '1 Day', 5: '1 Week', 22: '1 Month', 88: '4 Months'}[x] } ({x} steps)",
                help="Number of prediction steps ahead"
            )

        # determine model path for selected horizon
        model_path_local = MODEL_PATH
        if horizon and horizon > 1:
            # prefer native Keras format, fall back to legacy h5
            candidate_keras = f"models/lstm_{ticker}_h{horizon}.keras"
            candidate_h5 = f"models/lstm_{ticker}_h{horizon}.h5"
            if os.path.exists(candidate_keras):
                model_path_local = candidate_keras
            else:
                model_path_local = candidate_h5

        # Ensure we have enough historical rows to build the 60-day window
        has_history = df is not None and len(df) >= 60

        if not has_history:
            st.warning("Not enough historical rows to form a 60-day window. Need at least 60 rows.")

        if has_history and model_path_local and os.path.exists(model_path_local) and SCALER_PATH and os.path.exists(SCALER_PATH):
            if st.button("Initialize LSTM Sequence", use_container_width=True):
                with st.spinner("Neural Network processing sequence..."):
                    try:
                        # Log which ticker/horizon we're building for (helps debug in multi-worker runs)
                        logger = logging.getLogger(__name__)
                        logger.info(f"Initialize LSTM for {ticker} horizon={horizon} model={model_path_local}")
                        st.info(f"Building model for {ticker} — Horizon: {horizon}")
                        model_mtime = os.path.getmtime(model_path_local) if os.path.exists(model_path_local) else None
                        scaler_mtime = os.path.getmtime(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

                        @st.cache_resource
                        def load_model_and_scaler(m_path, s_path, mtime, stime):
                            return tf.keras.models.load_model(m_path), joblib.load(s_path)

                        model, scaler = load_model_and_scaler(model_path_local, SCALER_PATH, model_mtime, scaler_mtime)

                        recent_data = df.tail(60).values

                        df_features = recent_data.shape[1]
                        scaler_features = getattr(scaler, 'n_features_in_', None)
                        model_input_dim = None
                        try:
                            model_input_dim = model.input_shape[2]
                        except Exception:
                            model_input_dim = None

                        if scaler_features is not None and scaler_features != df_features:
                            st.error(f"Feature mismatch: processed CSV has {df_features} columns but scaler expects {scaler_features}.")
                            st.info("This usually happens when indicators changed and the model needs retraining.")
                            if st.button(f"Retrain model for {ticker} now (single ticker)"):
                                with st.spinner("Retraining model for this ticker..."):
                                    subprocess.run([sys.executable, "scripts/quick_run.py", "--ticker", ticker, "--epochs", "3"], check=False)
                                    st.success("Retrain command completed — refresh to load new model.")
                            raise RuntimeError("Feature mismatch between processed data and scaler")

                        if model_input_dim is not None and model_input_dim != scaler_features:
                            st.error(f"Model expects {model_input_dim} features but scaler provides {scaler_features}.")
                            if st.button(f"Retrain model for {ticker} now (single ticker)"):
                                with st.spinner("Retraining model for this ticker..."):
                                    subprocess.run([sys.executable, "scripts/quick_run.py", "--ticker", ticker, "--epochs", "3"], check=False)
                                    st.success("Retrain command completed — refresh to load new model.")
                            raise RuntimeError("Feature mismatch between model and scaler")

                        recent_data_scaled = scaler.transform(recent_data)
                        n_features = scaler.n_features_in_

                        # If horizon > 1, prefer to build multi-step forecast by
                        # iterating the single-step (T+1) model so predictions
                        # are consistent prefixes across horizons. If the
                        # single-step model is missing, fall back to the
                        # horizon-specific model (if available).
                        preds = None

                        # prefer .keras model if present, fallback to .h5
                        single_model_keras = os.path.join('models', f'lstm_{ticker}.keras')
                        single_model_h5 = os.path.join('models', f'lstm_{ticker}.h5')
                        single_model_path = single_model_keras if os.path.exists(single_model_keras) else single_model_h5
                        horizon_model_path = model_path_local

                        if horizon > 1 and os.path.exists(single_model_path):
                            # iterative multi-step forecast using single-step model
                            # We'll update the full feature window each step by
                            # appending the predicted raw Close, recomputing
                            # technical indicators, then scaling the window.
                            single_model = tf.keras.models.load_model(single_model_path)

                            # Work with raw DataFrame window (last 60 rows)
                            raw_window = df.tail(60).copy()
                            step_preds = []

                            for _step in range(horizon):
                                # Build scaled input from current raw_window
                                try:
                                    window_vals = raw_window.values
                                    scaled_window = scaler.transform(window_vals)
                                except Exception:
                                    # if scaler can't transform (column mismatch), abort
                                    raise

                                X_in = np.expand_dims(scaled_window, axis=0)
                                p_scaled = single_model.predict(X_in)
                                p_val_scaled = p_scaled.flatten()[0]

                                # convert scaled prediction to raw price
                                dummy = np.zeros((1, n_features))
                                dummy[0, CLOSE_COL_IDX] = p_val_scaled
                                p_raw = scaler.inverse_transform(dummy)[0, CLOSE_COL_IDX]

                                step_preds.append(p_val_scaled)

                                # create a new raw row for next step: set price columns
                                last_row = raw_window.iloc[-1].to_dict()
                                new_row = last_row.copy()
                                # set numeric price fields to predicted raw close
                                for col in ['Adj Close', 'Open', 'High', 'Low', 'Close']:
                                    if col in raw_window.columns:
                                        new_row[col] = p_raw
                                # keep volume same as last
                                if 'Volume' in raw_window.columns:
                                    new_row['Volume'] = raw_window['Volume'].iloc[-1]
                                # append and recompute indicators on extended df
                                # create a new index for the appended row to avoid duplicate index labels
                                try:
                                    last_idx = raw_window.index[-1]
                                    if pd.api.types.is_datetime64_any_dtype(raw_window.index):
                                        new_idx = last_idx + pd.tseries.offsets.BDay(1)
                                    else:
                                        # numeric or other index types: pick next integer
                                        try:
                                            new_idx = last_idx + 1
                                        except Exception:
                                            new_idx = None
                                except Exception:
                                    new_idx = None

                                if new_idx is not None:
                                    new_row_df = pd.DataFrame([new_row], index=[new_idx])
                                    temp_df = pd.concat([raw_window, new_row_df])
                                else:
                                    # fallback: ignore index so pandas assigns a fresh RangeIndex
                                    temp_df = pd.concat([raw_window.reset_index(drop=True), pd.DataFrame([new_row])], ignore_index=True)

                                temp_df = add_technical_indicators(temp_df)

                                # Re-align columns to match the original processed DataFrame
                                # Deduplicate expected columns (preserve order) to avoid
                                # errors if the original DataFrame had repeated names.
                                expected_cols = list(dict.fromkeys(df.columns))

                                # If temp_df has duplicate column labels (can happen when
                                # indicators are concatenated repeatedly), drop duplicate
                                # columns keeping the first occurrence so reindex works.
                                if temp_df.columns.duplicated().any():
                                    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

                                temp_df = temp_df.reindex(columns=expected_cols)

                                # Fill any NaNs introduced by indicator calculation:
                                # forward-fill, then backfill, then finally use last_row values or 0
                                # use .ffill()/.bfill() to avoid pandas compatibility issues
                                temp_df = temp_df.ffill()
                                temp_df = temp_df.bfill()
                                for col in expected_cols:
                                    if temp_df[col].isnull().any():
                                        if col in last_row:
                                            temp_df[col] = temp_df[col].fillna(last_row[col])
                                        else:
                                            temp_df[col] = temp_df[col].fillna(0)

                                # take last 60 rows for the next iteration
                                raw_window = temp_df.tail(60).copy()

                            preds = np.array(step_preds)
                        else:
                            # use the loaded model (could be single-step or multi-output)
                            X_input = np.array([recent_data_scaled])
                            prediction_scaled = model.predict(X_input)
                            preds = prediction_scaled.flatten()

# Phase 2: Enhanced chart with confidence bands
                        # Mock confidence intervals (± std dev based on confidence_level)
                        conf_std = (100 - confidence_level) / 100 * 0.05  # Mock volatility factor
                        
                        # inverse transform predictions to price units
                        dummy = np.zeros((len(preds), n_features))
                        for i, p in enumerate(preds):
                            dummy[i, CLOSE_COL_IDX] = p
                        inv = scaler.inverse_transform(dummy)[:, CLOSE_COL_IDX]
                        
                        # Mock confidence bands
                        upper_band = inv * (1 + conf_std)
                        lower_band = inv * (1 - conf_std)

                        current_price = df['Close'].iloc[-1]

                        # Build forecast index
                        last_idx = None
                        try:
                            last_idx = pd.to_datetime(df.index[-1])
                        except:
                            last_idx = None

                        if last_idx is not None:
                            future_idx = pd.bdate_range(start=last_idx + pd.Timedelta(days=1), periods=len(inv))
                            hist_idx = df.index[-120:] if len(df) >= 120 else df.index
                        else:
                            future_idx = [f"T+{i+1}" for i in range(len(inv))]
                            hist_idx = df.index[-60:]

                        # Enhanced Plotly chart with bands
                        fig = go.Figure()
                        # Historical
                        fig.add_trace(go.Scatter(
                            x=hist_idx, y=df['Close'].tail(len(hist_idx)), 
                            mode='lines', name='Historical Close', 
                            line=dict(color='#61E5FF', width=3)
                        ))
                        
                        # Forecast line
                        fig.add_trace(go.Scatter(
                            x=future_idx, y=inv, mode='lines+markers', 
                            name=f'Forecast T+{horizon}', 
                            line=dict(color='#FFBE5C', width=3), 
                            marker=dict(size=8, color='#FFBE5C')
                        ))
                        
                        # Confidence bands (Phase 2)
                        fig.add_trace(go.Scatter(
                            x=future_idx, y=upper_band, fill=None,
                            line=dict(color='rgba(255,190,92,0.2)', width=0),
                            showlegend=False, hovertemplate=None
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_idx+[future_idx[-1]], y=upper_band+[lower_band[-1]],
                            fill='tonexty', fillcolor='rgba(255,190,92,0.15)',
                            line=dict(color='rgba(255,255,255,0)'), name=f'{confidence_level}% Confidence'
                        ))
                        
                        # Vertical divider
                        if last_idx:
                            fig.add_vline(x=last_idx, line_dash='dot', line_color='gray', name='Now')

                        fig.update_layout(
                            template='plotly_dark', 
                            height=580, 
                            margin=dict(l=20, r=20, t=30, b=40),
                            hovermode='x unified',
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )

                        # Interactive chart with zoom/reset
                        chart_container = st.container()
                        with chart_container:
                            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                        
                        # Reset zoom button
                        if st.button("🔄 Reset Chart View", key="reset_chart"):
                            st.cache_data.clear()
                            st.rerun()

                        # Primary forecast metric (enhanced)
                        pred_val = float(inv[-1]) if horizon > 1 else float(inv[0])
                        metric_label = f"🎯 Predicted Close T+{horizon}"
                        change = pred_val - current_price
                        change_pct = (change / current_price) * 100
                        st.metric(metric_label, f"₹{pred_val:,.2f}", f"{change:,.1f} ({change_pct:.1f}%)")

                        # Signal badge
                        signal_color = "🟢 BULLISH" if change > 0 else "🔴 BEARISH"
                        st.markdown(f"**LSTM Signal:** {signal_color} 🚀")

                        # Phase 2: Interactive forecast table
                        forecast_df = pd.DataFrame({
                            'Step': [f'T+{i+1}' for i in range(len(inv))],
                            'Forecast': [f'₹{v:,.1f}' for v in inv],
                            'Upper CI': [f'₹{u:,.1f}' for u in upper_band],
                            'Lower CI': [f'₹{l:,.1f}' for l in lower_band],
                            'Change %': [f'{((v - current_price)/current_price*100):+.1f}%' for v in inv]
                        })
                        
                        with st.expander(f"📋 Detailed Forecast Table ({len(inv)} steps)", expanded=False):
                            st.dataframe(
                                forecast_df, 
                                use_container_width=True, 
                                column_config={
                                    "Forecast": st.column_config.TextColumn("Forecast Price"),
                                    "Upper CI": st.column_config.TextColumn("Upper Confidence"),
                                    "Lower CI": st.column_config.TextColumn("Lower Confidence"),
                                    "Change %": st.column_config.NumberColumn("Δ%", format="%.1f%%")
                                },
                                hide_index=True
                            )
                            st.download_button(
                                "📥 Export CSV", 
                                forecast_df.to_csv(index=False), 
                                f"{ticker}_lstm_forecast.csv",
                                "text/csv"
                            )
                        
                        # Historical accuracy mock stats (Phase 2)
                        with st.expander("📈 Model Performance History", expanded=False):
                            st.markdown("""
                            **Backtest Metrics (Mock - Last 30 predictions):**
                            - **Hit Rate:** 67.8% (within ±2% of actual)
                            - **MAE:** ₹14.2 
                            - **MAPE:** 1.8%
                            - **Avg Horizon:** 22 days
                            """)
                            col_acc1, col_acc2, col_acc3 = st.columns(3)
                            with col_acc1:
                                st.metric("✅ Hit Rate", "67.8%", "↑ 2.1%")
                            with col_acc2:
                                st.metric("📏 MAE", "₹14.2")
                            with col_acc3:
                                st.metric("📊 MAPE", "1.8%")
                    except Exception as e:
                        st.error(f"Engine failed to compute: {e}")
                        st.text(traceback.format_exc())
        else:
            st.warning("LSTM Brain offline. Model files not found.")

    with colB:
        st.info("**Architecture:** Analyzes 60-day non-linear sequences of OHLCV and RSI data to output a continuous price probability.")
        st.markdown("---")
        st.info("Bulk training and model-conversion controls have been moved to a dedicated page: 'Training Manager' (see the app Pages/sidebar).")
