# app.py (맨 위를 이 블록으로 교체)
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")
# ↑ 반드시 첫 Streamlit 호출이어야 함. 위에 st.caption 등 아무 것도 두지 마세요.

import sys, platform, os, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

import numpy as np, cv2
from PIL import Image
from io import BytesIO

# vision 모듈 임포트 (이름 정확히!)
try:
    from vision import detect_pd_px, overlay_rgba, load_fixed_antena
except Exception as e:
    st.error(f"vision 임포트 실패: {e}")
    st.stop()

# Streamlit 버전 호환용 이미지 표시 헬퍼
def show_image(img_bgr, **kwargs):
    # 1) 최신 인자(use_container_width) 시도
    try:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True, **kwargs)
        return
    except TypeError:
        pass
    # 2) 구버전 인자(use_column_width)로 fallback
    try:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True, **kwargs)
        return
    except TypeError:
        pass
    # 3) 그래도 안 되면 기본 호출
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), **kwargs)

