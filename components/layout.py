"""Shared, accessible application chrome for every Streamlit page."""
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go
from config import resolve_app_path

CHART_STYLE = {
    "paper_bgcolor": "#131722", "plot_bgcolor": "#131722",
    "font": {"family": "Segoe UI, sans-serif", "color": "#C7D2E0", "size": 12},
    "colorway": ["#00C9FF", "#92FE9D", "#F5CA72", "#C5ABFF", "#FF7187"],
    "hoverlabel": {"bgcolor": "#1C2536", "font_color": "#F1F5FF", "bordercolor": "#526681"},
}
CHART_GRID = "#2B364C"
CHART_TEMPLATE = go.layout.Template(layout={
    **CHART_STYLE,
    "xaxis": {"gridcolor": CHART_GRID, "zeroline": False, "automargin": True},
    "yaxis": {"gridcolor": CHART_GRID, "zeroline": False, "automargin": True},
})


def apply_theme():
    css = Path(resolve_app_path("assets", "app.css")).read_text(encoding="utf-8-sig")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    pio.templates["equimaster"] = CHART_TEMPLATE
    pio.templates.default = "equimaster"


def render_shell(active="Markets"):
    apply_theme()
    if active in ("Markets", "Stock analysis"):
        return
    with st.container(key="app_shell"):
        brand, date = st.columns([3, 1])
        brand.markdown('<div class="brand">Equimaster-AI</div>', unsafe_allow_html=True)
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        date.markdown(f'<div class="shell-date">INDIA · {now:%d %b %Y}<br>{escape(active)} workspace</div>', unsafe_allow_html=True)


def section_title(title, subtitle=None, badge=None):
    tag = f'<span class="pill">{escape(str(badge))}</span>' if badge else ""
    st.markdown(f'<div class="panel-title">{escape(title)}{tag}</div>' +
                (f'<div class="panel-caption">{escape(subtitle)}</div>' if subtitle else ""), unsafe_allow_html=True)


def research_card(title, value, detail, tone="blue"):
    st.markdown(f'<div class="research-card {escape(tone)}"><div class="eyebrow">{escape(title)}</div>'
                f'<div class="value">{escape(str(value))}</div><div class="detail">{escape(detail)}</div></div>', unsafe_allow_html=True)


def render_footer():
    st.markdown('<div class="app-footer"><span>Equimaster AI · Research with context</span>'
                '<span>Source dates and coverage are shown with the data.</span></div>', unsafe_allow_html=True)
