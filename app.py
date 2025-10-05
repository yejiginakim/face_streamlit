# ---------- 반드시 최상단, 1회 ----------
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# ---------- 기본 표시 ----------
import sys, platform, os, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# ---------- 안전 임포트 ----------
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

import vision  # 위에서 만든 vision.py

# ---------- 유틸: 이미지 표시(버전 호환) ----------
def show_image_bgr(img_bgr, **kwargs):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        st.image(rgb, **kwargs)          # 최신 streamlit
    except TypeError:
        st.image(rgb)                    # 구버전 호환

# ---------- 사이드바 ----------
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커)")

with st.sidebar:
    # iOS에서 ?pd_mm=~~ 로 들어오면 자동 반영
    params = st.query_params
    def fget_float(k, default=None):
        try:
            v = params.get(k, None)
            return float(v) if v not in (None, "") else default
        except Exception:
            return default

    pd_from_url = fget_float("pd_mm", fget_float("pd", None))
    PD_MM = st.number_input("PD (mm)", value=pd_from_url or 63.0, step=0.1, format="%.3f")

    st.markdown("---")
    st.subheader("프레임 옵션")
    white_frame = st.checkbox("프린지 제거(dematte): 프레임이 밝거나 흰색", value=True)
    apply_occ   = st.checkbox("가림(윗눈꺼풀/코등 소프트 마스크)", value=True)

    st.markdown("---")
    st.subheader("미세 조정")
    dx = st.slider("수평 오프셋(px)", -250, 250, 0)
    dy = st.slider("수직 오프셋(px)", -250, 250, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.80, 1.20, 1.00)

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"])

with colR:
    st.markdown("### 2) 결과/수치")

# ---------- 프레임 로드 ----------
fg_bgra, dims = vision.load_fixed_antena()
if fg_bgra is None or dims is None:
    st.error("프레임 이미지를 읽을 수 없어요. 경로/포맷을 확인해 주세요.")
    st.code(f"""
exists(frames)={os.path.isdir('frames')}
exists(frames/images)={os.path.isdir('frames/images')}
glob Antena_01.*={glob.glob('frames/images/Antena_01.*')}
    """, language="text")
    st.stop()

A, DBL, TOTAL = dims
GCD = A + DBL
k_ratio = (TOTAL / GCD) if GCD else 2.0
st.caption(f"A={A}, DBL={DBL}, TOTAL={TOTAL} → GCD={GCD}, k=TOTAL/GCD={k_ratio:.3f}")

# 프린지 보정(프레임이 밝거나 흴 때 권장)
# 프린지 보정(프레임이 밝거나 흴 때 권장)
if white_frame and hasattr(vision, "dematte_any_color"):
    fg_bgra = vision.dematte_any_color(fg_bgra, matte_color=(255, 255, 255))
elif white_frame:
    st.warning("프린지 제거 함수( dematte_any_color )가 vision.py에 없어서 건너뜀")


# ---------- 얼굴 업로드 필요 ----------
if not img_file:
    st.info("얼굴 사진을 업로드하면 합성을 시작합니다.")
    st.stop()

# ---------- 얼굴 로드 + PD/각도 ----------
file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if face_bgr is None:
    st.error("얼굴 이미지를 읽을 수 없어요.")
    st.stop()

pd_px, angle_deg, mid = vision.detect_pd_px(face_bgr)
if pd_px is None:
    st.error("눈/얼굴을 찾지 못했어요. 정면, 밝은 조명에서 다시 시도해 주세요.")
    st.stop()

st.write(f"**PD_px**: {pd_px:.2f} px  /  **roll**: {angle_deg:.2f}°  /  **mid**: {tuple(round(v,1) for v in mid)}")

# ---------- 스케일 ----------
px_per_mm = (pd_px / PD_MM) if PD_MM else None
if px_per_mm:
    st.write(f"**px_per_mm**: {px_per_mm:.4f}")
    target_total_px = (GCD * px_per_mm) * k_ratio
else:
    st.warning("PD(mm)가 없어 근사 스케일로 합성합니다. (TOTAL/GCD 비례)")
    target_total_px = pd_px * k_ratio

# ---------- 프레임 리사이즈/회전 ----------
h0, w0 = fg_bgra.shape[:2]
scale = (target_total_px / w0) * scale_mult
new_size = (max(1, int(w0*scale)), max(1, int(h0*scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

M = cv2.getRotationMatrix2D((fg_scaled.shape[1]/2, fg_scaled.shape[0]/2), angle_deg, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

# ---------- 위치(브릿지 자동 오프셋) ----------
nb = vision.nose_bridge_point(face_bgr)
auto_up = int(((mid[1] - nb[1]) if nb is not None else 0) * 0.35)  # 0.25~0.45 조절
gx = int(mid[0] - fg_rot.shape[1]/2) + dx
gy = int(mid[1] - fg_rot.shape[0]/2) + dy - auto_up

# ---------- 합성 ----------
out = vision.overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)

# (선택) 가림 마스크 적용
if apply_occ:
    occ = vision.build_occlusion_mask(face_bgr)  # 0~1
    if occ is not None:
        comp = out.astype(np.float32)
        base = face_bgr.astype(np.float32)
        # 얼굴 쪽(occ)이 1일수록 원본을 더 보이게 (소프트 가림)
        comp = comp*(1 - occ[...,None]) + base*(occ[...,None])
        out = comp.astype(np.uint8)

# ---------- 출력 ----------
show_image_bgr(out, caption="합성 결과")

# 다운로드
buf = BytesIO()
Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                   file_name="Antena_01_result.png", mime="image/png")
