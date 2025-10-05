import streamlit as st

st.set_page_config(page_title='얼굴 실측 결과', layout='centered')

# 쿼리 읽기 (?pd=..&cheek=..&nosechin=..)
q = st.query_params
pd = float(q.get('pd') if 'pd' in q else None
cheek = float(q.get('cheek') if 'cheek' in q else None
nosechin = float(q.get('nosechin') if 'nosechin' in q else None

st.title('🧍 얼굴 실측 결과 (from iPhone)')
col1, col2, col3 = st.columns(3)
if pd is not None: col1.metric('PD (mm)', f'{pd:.3f}')
if cheek is not None: col2.metric('광대 폭 (mm)', f'{cheek:.3f}')
if nosechin is not None: col3.metric('코–턱 (mm)', f'{nosechin:.3f}')

st.divider()
st.caption('아이폰에서 측정 후 화면을 한 번 탭하면 이 페이지가 자동으로 열립니다.')
