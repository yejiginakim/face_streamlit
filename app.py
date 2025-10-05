import streamlit as st
import numpy as np, cv2
from PIL import Image
from io import BytesIO

# ---------- 설정 ----------
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# 고정 프레임 정보
A = 52.7
DBL = 20.0
TOTAL = 145.1
PNG_PATH = "frames/images/Antena_01.png"   # ← 이 경로에 파일 넣어두기

# URL 쿼리에서 PD(mm) 받기 (pd_mm 우선, pd도 허용)
params = st.query_params
def fget(k, default=None):
    try:
        v = params.get(k, None)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default
PD_MM = fget("pd_mm", fget("pd", None))

st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커)")

with st.sidebar:
    st.subheader("PD(mm)")
    if PD_MM is not None:
        st.metric("PD (from iPhone)", f"{PD_MM:.3f}")
    PD_MM = st.number_input("PD (mm) 직접 입력 가능", value=PD_MM or 0.0, step=0.1, format="%.3f") or None

    st.subheader("미세 조정(옵션)")
    dx = st.slider("수평 오프셋(px)", -200, 200, 0)
    dy = st.slider("수직 오프셋(px)", -200, 200, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, 1.0)

colL, colR = st.columns([1,1])

with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"])

with colR:
    st.markdown("### 2) 결과/수치")
    st.caption(f"Antena_01 치수 ▶ A={A}, DBL={DBL}, TOTAL={TOTAL}  (B 미사용)")

# ---------- MediaPipe (캐시) ----------
@st.cache_resource
def create_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

def detect_pd_px(bgr):
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks: return None, None, None
    lm = res.multi_face_landmarks[0].landmark
    L_ids = [33,133,159,145]; R_ids = [362,263,386,374]
    L = np.array([[lm[i].x*w, lm[i].y*h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x*w, lm[i].y*h] for i in R_ids], np.float32).mean(axis=0)
    pd_px = float(np.hypot(*(R-L)))
    angle = float(np.degrees(np.arctan2(R[1]-L[1], R[0]-L[0])))
    mid = ((L[0]+R[0])/2.0, (L[1]+R[1])/2.0)
    return pd_px, angle, mid

def overlay_rgba(bg_bgr, fg_rgba, x, y):
    H,W = bg_bgr.shape[:2]; h,w = fg_rgba.shape[:2]
    x0,y0 = max(x,0), max(y,0); x1,y1 = min(x+w, W), min(y+h, H)
    if x0>=x1 or y0>=y1: return bg_bgr
    fg_cut = fg_rgba[y0-y:y1-y, x0-x:x1-x]
    alpha = (fg_cut[:, :, 3:4].astype(np.float32)/255.0)
    bg_roi = bg_bgr[y0:y1, x0:x1, :].astype(np.float32)
    fg_rgb = fg_cut[:, :, :3].astype(np.float32)
    bg_bgr[y0:y1, x0:x1, :] = (fg_rgb*alpha + bg_roi*(1-alpha)).astype(np.uint8)
    return bg_bgr

# ---------- PNG 로더 (RGBA 보장) ----------
def load_frame_png(path):
    fg = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA 기대
    if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
        # OpenCV가 못 읽거나 알파 없으면 PIL로 재시도
        try:
            pil = Image.open(path).convert("RGBA")
            fg = cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)
        except Exception:
            return None
    return fg

# ---------- 실행 ----------
if not img_file:
    st.info("얼굴 사진을 업로드해 주세요.")
else:
    # 얼굴 로드
    face_bgr = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if face_bgr is None:
        st.error("얼굴 이미지를 읽을 수 없습니다.")
    else:
        pd_px, angle_deg, mid = detect_pd_px(face_bgr)
        if pd_px is None:
            st.error("얼굴/눈 검출 실패. 정면, 밝은 조명에서 다시 시도해주세요.")
        else:
            st.write(f"**PD_px**: {pd_px:.2f} px / **angle**: {angle_deg:.2f}° / **mid**: {tuple(round(v,1) for v in mid)}")

            # px/mm
            if not PD_MM:
                st.warning("PD(mm)가 없어 정확 스케일을 계산할 수 없어요. iPhone 링크 또는 사이드바에서 PD(mm) 입력.")
                px_per_mm = None
            else:
                px_per_mm = pd_px / PD_MM
                st.write(f"**px_per_mm**: {px_per_mm:.4f}")

            # 프레임 PNG
            fg = load_frame_png(PNG_PATH)
            if fg is None:
                st.error(f"프레임 이미지를 읽을 수 없습니다: {PNG_PATH} (PNG RGBA 필요)")
            else:
                # --- 스케일 계산 (GCD 앵커) ---
                GCD = A + DBL
                k = (TOTAL / GCD) if GCD else 2.0
                if px_per_mm:
                    target_total_px = (GCD * px_per_mm) * k
                else:
                    target_total_px = pd_px * k  # 근사

                h0, w0 = fg.shape[:2]
                scale = (target_total_px / w0) * scale_mult
                new_size = (max(1, int(w0*scale)), max(1, int(h0*scale)))
                fg_scaled = cv2.resize(fg, new_size, interpolation=cv2.INTER_LINEAR)

                # 회전(투명 유지)
                M = cv2.getRotationMatrix2D((fg_scaled.shape[1]/2, fg_scaled.shape[0]/2), angle_deg, 1.0)
                fg_rot = cv2.warpAffine(
                    fg_scaled, M,
                    (fg_scaled.shape[1], fg_scaled.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0,0,0,0)
                )

                # 위치: 브리지 중심(가로 중앙)을 눈 중점에 정렬
                gx = int(mid[0] - fg_rot.shape[1]/2) + dx
                gy = int(mid[1] - fg_rot.shape[0]/2) + dy  # B미사용 → 기본 0 + dy로만 조정

                out = overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="합성 결과", use_container_width=True)

                # 검증 로그
                if PD_MM and px_per_mm:
                    ratio1 = target_total_px / pd_px
                    ratio2 = TOTAL / PD_MM
                    st.caption(f"검증: target_total_px/pd_px = {ratio1:.3f}  vs  TOTAL/PD_MM = {ratio2:.3f}")

                # 다운로드
                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                                   file_name="Antena_01_result.png", mime="image/png")
