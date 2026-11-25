"""
theme.py - Centralized visual theming, custom CSS components, and Plotly templates for DataSentry.

Provides a unified, professional appearance across both Light and Dark mode,
mapping semantic colors to dataset health grades and rendering publication-ready charts.
"""

from typing import Dict, Any
import streamlit as st
import plotly.graph_objects as go
from ..config import QUALITY_GRADES, UI_CONFIG


def get_grade_color(score: float) -> str:
    """Return hex color corresponding to a 0-100 quality score."""
    if score >= 90.0:
        return QUALITY_GRADES["A"][1]
    elif score >= 80.0:
        return QUALITY_GRADES["B"][1]
    elif score >= 70.0:
        return QUALITY_GRADES["C"][1]
    elif score >= 60.0:
        return QUALITY_GRADES["D"][1]
    return QUALITY_GRADES["F"][1]


def inject_custom_css():
    """Inject modern, responsive CSS styling into the Streamlit application."""
    css = """
    <style>
        /* Modern Font and Global Resets */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Top Header & Titles */
        .app-hero-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
            color: #FFFFFF;
            padding: 24px 32px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 15px -3px rgba(15, 23, 42, 0.2);
        }
        .app-hero-header h1 {
            color: #FFFFFF !important;
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            margin: 0 0 6px 0 !important;
            letter-spacing: -0.02em;
        }
        .app-hero-header p {
            color: #94A3B8 !important;
            font-size: 1.0rem !important;
            margin: 0 !important;
            line-height: 1.5;
        }

        .section-header {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
            margin: 20px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 2px solid #F1F5F9;
            padding-bottom: 8px;
        }

        /* Cards and Containers */
        .stat-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
        }
        .stat-card:hover {
            box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.08);
            border-color: #CBD5E1;
        }
        .stat-card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .stat-card-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }
        .stat-card-delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 4px;
        }

        /* Badges */
        .grade-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .grade-badge-a { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
        .grade-badge-b { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
        .grade-badge-c { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
        .grade-badge-d { background: #FFEDD5; color: #9A3412; border: 1px solid #FDBA74; }
        .grade-badge-f { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }

        /* Dataset Selection Banner */
        .sample-dataset-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .sample-dataset-card h4 {
            margin: 0 0 6px 0;
            color: #1E293B;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .sample-dataset-card p {
            font-size: 0.85rem;
            color: #64748B;
            margin-bottom: 12px;
            line-height: 1.4;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        .sidebar-brand {
            padding: 10px 0 16px 0;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 16px;
        }
        .sidebar-brand h2 {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0;
        }
        .sidebar-brand p {
            font-size: 0.8rem;
            color: #64748B;
            margin: 2px 0 0 0;
        }
        .status-pill {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 14px;
        }

        /* Smooth buttons */
        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template() -> go.layout.Template:
    """Return unified, clean Plotly layout template matching DataSentry theme."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155"),
        title=dict(font=dict(size=15, color="#0F172A", family="Plus Jakarta Sans, sans-serif")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        colorway=["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899"],
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=11),
            title=dict(font=dict(color="#334155", size=12))
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=11),
            title=dict(font=dict(color="#334155", size=12))
        ),
        legend=dict(
            font=dict(size=11, color="#475569"),
            bordercolor="#E2E8F0",
            borderwidth=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    return template
