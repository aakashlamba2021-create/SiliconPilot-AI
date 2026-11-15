"""Shared, native Streamlit page header."""
from __future__ import annotations

import streamlit as st


def render_header(title: str, subtitle: str = "") -> None:
    with st.container(border=True):
        st.title(title, icon=":material/developer_board:")
        if subtitle:
            st.caption(subtitle)
