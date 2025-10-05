import streamlit as st
import numpy as np, cv2
from PIL import Image
from io import BytesIO

# -------------------- 페이지 설정(한 번만) --------------------
st.set_page_config(page_title='얼굴 실측 + 합성', layout='wide')

# -------------------- URL 쿼리 파라미터 --------------------
params = st.query_params

def fget(name, default=None):
    """안전한 float 파싱 (빈값/잘못된 값 대비)"""
    try:
        val = params.get(name, None)
        return float(val) if val not in (None, "") else default
    except Exception:
        return default

# iOS에서 오는 실측 mm (pd_mm 권장, 호환: pd)
PD_MM       = fget('pd_mm', fget('pd', None))
CHEEK_MM    = fget('cheek_mm', fget('cheek', None))
NOSECHIN_MM = fget('nosechin_mm', fget('nosechin', None))

st.title("🧍→🕶️ iPhone ARKit 실측 + MediaPipe 합성")

# -------------------- 사이드바 --------------------
with st.sidebar:
    st.subheader("프레임 치수 (mm)")
    A = st.number_input("A (렌즈 가로)", value=57.0, step=0.1)
    DBL = st.number_input("DBL (브리지)", value=18.0, step=0.1)
    TOTAL = st.number_input("총 가로폭", value=150.0, step=0.1)
    B = st.number_input("B (렌즈 세로, 옵션)", value=44.7, step=0.1)
    st.caption("GCD = A + DBL = 렌즈 중심 간 거리 (Frame PD)")

    # iPhone 없이도 사용 가능하도록 PD(mm) 수동 입력 옵션
    if PD_MM is None:
        st.markdown("### iPhone 없이 PD 입력")
        PD_MM = st.number_input("PD (mm)", min_value=0.0, step=0.1, format="%.3f") or None

    st.markdown("### 미세 조정(옵션)")
    dx = st.slider("수평 오프셋(px)", -200, 200, 0)
    dy = st.slider("수직 오프셋(px)", -200, 200, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, 1.0)

# -------------------- 좌/우 영역 --------------------
colL, colR = st.columns([1,1])

with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진(아이폰에서 저장한 사진)", type=["jpg","jpeg","png"])
    st.markdown("### 2) 선글라스 PNG 업로드 (투명배경)")
    png_file = st.file_uploader("프레임 PNG", type=["png"])

with colR:
    st.markdown("### 3) 결과/수치")
    m1, m2, m3 = st.columns(3)
    if PD_MM is not None:       m1.metric('PD (mm)', f'{PD_MM:.3f}')
    if CHEEK_MM is not None:    m2.metric('광대 폭 (mm)', f'{CHEEK_MM:.3f}')
    if NOSECHIN_MM is not None: m3.metric('코–턱 (mm)', f'{NOSECHIN_MM:.3f}')

# -------------------- MediaPipe 준비 (캐시) --------------------
@st.cache_resource
def create_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,   # iris 사용
        max_num_faces=1
    )

def detect_pd_px(bgr):
    """눈 중심으로 PD_px, 눈선 각도, 중점(mid) 계산"""
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0].landmark
    # eyelid 4점 평균 중심 (안정적)
    L_ids = [33,133,159,145]
    R_ids = [362,263,386,374]
    L = np.array([[lm[i].x*w, lm[i].y*h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x*w, lm[i].y*h] for i in R_ids], np.float32).mean(axis=0)
    pd_px = float(np.hypot(*(R-L)))
    angle = float(np.degrees(np.arctan2(R[1]-L[1], R[0]-L[0])))
    mid = ((L[0]+R[0])/2.0, (L[1]+R[1])/2.0)
    return pd_px, angle, mid

# -------------------- 합성 유틸 --------------------
def overlay_rgba(bg_bgr, fg_rgba, x, y):
    """bg_bgr(BGR) 위에 fg_rgba(RGBA)를 (x,y) 좌상단 기준으로 합성"""
    H,W = bg_bgr.shape[:2]
    h,w = fg_rgba.shape[:2]
    x0,y0 = max(x,0), max(y,0)
    x1,y1 = min(x+w, W), min(y+h, H)
    if x0>=x1 or y0>=y1:
        return bg_bgr
    fg_cut = fg_rgba[y0-y:y1-y, x0-x:x1-x]
    alpha = (fg_cut[:, :, 3:4].astype(np.float32)/255.0)
    bg_roi = bg_bgr[y0:y1, x0:x1, :].astype(np.float32)
    fg_rgb = fg_cut[:, :, :3].astype(np.float32)
    out = fg_rgb*alpha + bg_roi*(1-alpha)
    bg_bgr[y0:y1, x0:x1, :] = out.astype(np.uint8)
    return bg_bgr

# -------------------- 실행 --------------------
if img_file and png_file:
    # 얼굴 이미지
    face_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    face_bgr = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
    if face_bgr is None:
        st.error("얼굴 이미지를 읽을 수 없습니다.")
    else:
        res = detect_pd_px(face_bgr)
        if res[0] is None:
            st.error("얼굴/눈을 찾지 못했어요. 정면, 밝은 조명에서 다시 시도해 주세요.")
        else:
            pd_px, angle_deg, mid = res
            st.write(f"**PD_px**: {pd_px:.2f} px / **angle**: {angle_deg:.2f}° / **mid**: {tuple(round(v,1) for v in mid)}")

            # px/mm
            if PD_MM is None:
                st.warning("URL/사이드바에 PD(mm)가 없어 정확 스케일을 계산할 수 없어요. iPhone에서 롱프레스로 들어오거나, 사이드바에서 PD(mm)를 입력하세요.")
            px_per_mm = (pd_px / PD_MM) if PD_MM else None
            if px_per_mm:
                st.write(f"**px_per_mm**: {px_per_mm:.4f}")

            # 선글라스 PNG
            png_bytes = np.asarray(bytearray(png_file.read()), dtype=np.uint8)
            fg = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)
            if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
                st.error("PNG는 RGBA(투명배경)여야 합니다. (알파 채널 필요)")
            else:
                # 목표 스케일 (기본: 전체 가로폭)
                GCD = A + DBL  # mm
                if px_per_mm:
                    target_total_px = TOTAL * px_per_mm
                else:
                    # 근사: PD_px × (TOTAL/GCD)
                    target_total_px = pd_px * (TOTAL / GCD)

                h0, w0 = fg.shape[:2]
                scale = (target_total_px / w0) * scale_mult
                new_size = (max(1, int(w0*scale)), max(1, int(h0*scale)))
                fg_scaled = cv2.resize(fg, new_size, interpolation=cv2.INTER_LINEAR)

                # 회전 (투명 경계 유지)
                M = cv2.getRotationMatrix2D(
                    (fg_scaled.shape[1]/2, fg_scaled.shape[0]/2),
                    angle_deg, 1.0
                )
                fg_rot = cv2.warpAffine(
                    fg_scaled, M,
                    (fg_scaled.shape[1], fg_scaled.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0,0,0,0)
                )

                # 위치: 브리지 중심(가로 중앙)을 mid에 정렬 + 세로 오프셋(B의 0.35배 위)
                if px_per_mm:
                    px_per_mm_y = px_per_mm
                else:
                    # 근사 px/mm (PD_mm이 없을 때 GCD 사용)
                    px_per_mm_y = pd_px / GCD

                offset_y = int(- (B * 0.35) * px_per_mm_y)
                gx = int(mid[0] - fg_rot.shape[1]/2) + dx
                gy = int(mid[1] - fg_rot.shape[0]/2 + offset_y) + dy

                out = overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="합성 결과", use_container_width=True)

                # 검증용 로그(스케일 맞는지 대략 비교)
                if PD_MM and px_per_mm:
                    ratio1 = target_total_px / pd_px
                    ratio2 = TOTAL / PD_MM
                    st.caption(f"검증: target_total_px/pd_px = {ratio1:.3f}  vs  TOTAL/PD_MM = {ratio2:.3f}")

                # 다운로드 버튼
                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                st.download_button(
                    "결과 PNG 다운로드",
                    data=buf.getvalue(),
                    file_name="glasses_result.png",
                    mime="image/png"
                )

elif img_file and not png_file:
    st.info("선글라스 PNG를 업로드하면 합성이 완성됩니다. (투명배경 RGBA 권장)")
elif png_file and not img_file:
    st.info("얼굴 사진을 업로드해 주세요(아이폰에서 롱프레스로 저장).")
else:
    if PD_MM is not None or CHEEK_MM is not None or NOSECHIN_MM is not None:
        st.caption("아이폰에서 측정한 실측 mm 값이 쿼리로 전달되었습니다. 사진/PNG 업로드 시 정확 스케일 합성이 진행됩니다.")
    else:
        st.info("왼쪽에 얼굴 사진과 선글라스 PNG를 업로드하면 여기 결과가 표시됩니다.\n아이폰에서 롱프레스로 들어오면 `pd_mm`이 자동으로 붙어요.")

