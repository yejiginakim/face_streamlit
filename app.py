import streamlit as st

# ✅ 가장 먼저, 단 한 번만
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

import numpy as np, cv2
from PIL import Image
from io import BytesIO

from vision import (
    detect_pd_px, overlay_rgba, load_fixed_antena,
)

# ---------- URL 쿼리에서 PD(mm) 받기 ----------
params = st.query_params
def fget(k, default=None):
    try:
        v = params.get(k, None)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default

# 아이폰에서 pd_mm 또는 pd 로 들어오게 했음
PD_MM = fget("pd_mm", fget("pd", None))

st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커)")

with st.sidebar:
    st.subheader("PD (mm)")
    if PD_MM is not None:
        st.metric("PD (from iPhone)", f"{PD_MM:.3f}")
    PD_MM = st.number_input("PD (mm) 직접 입력 가능", value=PD_MM or 0.0,
                            step=0.1, format="%.3f") or None

    st.subheader("미세 조정(옵션)")
    dx = st.slider("수평 오프셋(px)", -200, 200, 0)
    dy = st.slider("수직 오프셋(px)", -200, 200, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, 1.0)

colL, colR = st.columns([1, 1])

with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg", "jpeg", "png"])

with colR:
    st.markdown("### 2) 결과/수치")
    st.caption("프레임: Antena_01.png  /  A=52.7, DBL=20, TOTAL=145.1 (B 미사용)")

# ---------- 프레임 PNG 로드 ----------
fg, dims = load_fixed_antena()
if fg is None:
    st.error("frames/images/Antena_01.png 을 RGBA/AVIF(PIL)로 읽을 수 없어요. 경로/포맷을 확인해 주세요.")
    st.stop()

A, DBL, TOTAL = dims
GCD = A + DBL                          # mm
k = (TOTAL / GCD) if GCD else 2.0      # TOTAL/GCD 비율(대개 ~2.0)

# ---------- 얼굴 업로드 처리 ----------
if not img_file:
    st.info("얼굴 사진을 업로드해 주세요.")
    st.stop()

file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if face_bgr is None:
    st.error("얼굴 이미지를 읽을 수 없습니다.")
    st.stop()

# ---------- PD_px / 각도 / 중점 ----------
pd_px, angle_deg, mid = detect_pd_px(face_bgr)
if pd_px is None:
    st.error("얼굴/눈 검출 실패. 정면, 밝은 조명에서 다시 시도해 주세요.")
    st.stop()

st.write(f"**PD_px**: {pd_px:.2f} px  /  **angle**: {angle_deg:.2f}°  /  **mid**: {tuple(round(v,1) for v in mid)}")

# ---------- 스케일 계산 (GCD 앵커) ----------
if PD_MM:
    px_per_mm = pd_px / PD_MM
    st.write(f"**px_per_mm**: {px_per_mm:.4f}")
    target_total_px = (GCD * px_per_mm) * k  # GCD→전체 폭 변환
else:
    st.warning("PD(mm)가 없어 정확 스케일을 계산할 수 없어요. iPhone 링크로 열거나 사이드바에 PD(mm)를 입력해 주세요.")
    target_total_px = pd_px * k              # 근사: 얼굴 PD_px 기반

# ---------- 리사이즈/회전/배치 ----------
h0, w0 = fg.shape[:2]
scale = (target_total_px / w0) * scale_mult
new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg, new_size, interpolation=cv2.INTER_LINEAR)

M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), angle_deg, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M,
    (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0)
)

gx = int(mid[0] - fg_rot.shape[1] / 2) + dx
gy = int(mid[1] - fg_rot.shape[0] / 2) + dy   # B 미사용 → dy로만 수직 조정
out = overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)

st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="합성 결과", use_container_width=True)

# ---------- 검증 로그 ----------
if PD_MM:
    ratio1 = target_total_px / pd_px
    ratio2 = TOTAL / PD_MM
    st.caption(f"검증: target_total_px/pd_px = {ratio1:.3f}  vs  TOTAL/PD_MM = {ratio2:.3f}")

# ---------- 다운로드 ----------
buf = BytesIO()
Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                   file_name="Antena_01_result.png", mime="image/png")

