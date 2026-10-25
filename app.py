"""Equimaster market research dashboard."""
import logging
import streamlit as st
from config import ensure_runtime_directories, resolve_app_path
from components.layout import render_shell, render_footer
from views import render_home_view

ensure_runtime_directories()
st.set_page_config(page_title="Markets | Equimaster AI", page_icon="E", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(filename=resolve_app_path("logs", "app_errors.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
render_shell("Markets")
render_home_view()
render_footer()
