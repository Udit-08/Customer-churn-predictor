"""Look and feel: colours, CSS, Plotly style and small layout helpers."""
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#14233A"
TEAL = "#0F9B8E"
CORAL = "#E4604B"
AMBER = "#E8A33D"
BLUE = "#4C6FD1"
VIOLET = "#8662C7"
GREY = "#8A96A8"
LINE = "#E3E8F0"
PALETTE = [TEAL, CORAL, BLUE, AMBER, VIOLET, GREY]

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {NAVY}; }}
h3 {{ margin-top: .4rem; }}

/* KPI cards */
[data-testid="stMetric"] {{
    background: #fff; border: 1px solid {LINE}; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 2px 10px rgba(20,35,58,.04);
}}
[data-testid="stMetricLabel"] {{ overflow: visible; }}
[data-testid="stMetricLabel"] p {{ color: #6B7A90; font-size: .76rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: .03em; white-space: normal; overflow: visible;
    text-overflow: clip; line-height: 1.25; }}
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(20,35,58,.04); height: 100%; }}
.kpi .l {{ color: #6B7A90; font-size: .76rem; font-weight: 650; text-transform: uppercase;
    letter-spacing: .03em; line-height: 1.25; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 750; color: {NAVY}; line-height: 1.25; }}
.kpi .s {{ color: #6B7A90; font-size: .82rem; }}
.kpi .s.bad {{ color: #B83A27; }} .kpi .s.good {{ color: #0B7468; }}
[data-testid="stMetricValue"] {{ font-size: 1.7rem; font-weight: 750; color: {NAVY}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {LINE}; }}
.brand {{ font-size: 1.15rem; font-weight: 800; color: {NAVY}; line-height: 1.2; }}
.brand span {{ color: {TEAL}; }}
.brand-sub {{ color: #6B7A90; font-size: .8rem; margin-bottom: .6rem; }}

/* Page header */
.kicker {{ color: {TEAL}; font-size: .75rem; font-weight: 800; letter-spacing: .14em;
    text-transform: uppercase; margin-bottom: .1rem; }}
.page-title {{ font-size: 2rem; font-weight: 800; color: {NAVY}; line-height: 1.15; margin: 0; }}
.page-sub {{ color: #5B6B82; font-size: 1rem; margin: .35rem 0 1.2rem; max-width: 820px; }}

/* Cards */
.card {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 2px 10px rgba(20,35,58,.04); height: 100%; }}
.card h4 {{ margin: 0 0 .15rem; font-size: 1rem; color: {NAVY}; }}
.card .big {{ font-size: 1.9rem; font-weight: 800; color: {NAVY}; line-height: 1.1; }}
.card .muted, .muted {{ color: #6B7A90; font-size: .85rem; }}
.pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem;
    font-weight: 700; letter-spacing: .02em; }}
.pill-high {{ background: #FDE8E4; color: #B83A27; }}
.pill-mid {{ background: #FDF1DC; color: #A56A0A; }}
.pill-low {{ background: #DDF4F0; color: #0B7468; }}
.insight {{ background: #fff; border: 1px solid {LINE}; border-left: 4px solid {TEAL};
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; color: {NAVY}; }}
.insight b {{ color: {NAVY}; }}
.note {{ background: #EEF3FB; border-radius: 10px; padding: 10px 14px; color: #34496A;
    font-size: .88rem; }}
div[data-testid="stTabs"] button p {{ font-weight: 650; }}
</style>
"""


def setup_page():
    st.set_page_config(page_title="Churn Intelligence", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=NAVY, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, margin=dict(l=10, r=10, t=48, b=10),
        title=dict(font=dict(size=15, color=NAVY), x=0, xanchor="left"),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, title_font=dict(size=12)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, title_font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(font_size=12),
    )
    pio.templates["churn"] = template
    pio.templates.default = "churn"


def page_header(kicker, title, subtitle):
    st.markdown(
        f'<div class="kicker">{kicker}</div><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def card(title, big, sub="", pill=None, pill_class="pill-low"):
    pill_html = (
        f'<span class="pill {escape(str(pill_class))}">{escape(str(pill))}</span>'
        if pill else ""
    )
    st.markdown(
        f'<div class="card"><h4>{escape(str(title))}</h4><div class="big">{escape(str(big))}</div>'
        f'<div class="muted">{escape(str(sub))}</div>'
        f'<div style="margin-top:6px">{pill_html}</div></div>',
        unsafe_allow_html=True)


def kpi(column, label, value, sub="", tone=""):
    """A KPI tile with an optional second line (tone: '', 'bad' or 'good')."""
    column.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s {tone}">{sub or "&nbsp;"}</div></div>', unsafe_allow_html=True)


def money(value):
    return "$" + f"{value:,.0f}"


def show(fig, key=None):
    """Plotly chart at full width, working across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(data, **kwargs):
    """DataFrame at full width, working across Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)
