import os
os.environ["OPENCV_HEADLESS"] = "1"   # ← cv2 임포트 ‘전에’ 있어야 함

import numpy as np, cv2



# ---------- 반드시 최상단 1회 ----------
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# ---------- 기본 설정/임포트 ----------
import os, pathlib, sys, platform, glob
import numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download

from faceshape import FaceShapeModel, decide_rule_vs_top2
from metrics import compute_metrics_bgr

# (선택) TF 로그 줄이기
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# ---------- vision 임포트만 지연(없어도 UI 뜨게) ----------
err_msgs = []
try:
    import vision  # vision.py에 detect_pd_px / load_fixed_antena / overlay_rgba 필요
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

# ---------- HF Hub에서 모델/클래스 경로 확보 ----------
REPO_ID = "gina728/faceshape1"
MODEL_FILENAME = "faceshape_best.keras"   # HF에 올린 정확한 파일명
CLASSES_PATH = "models/classes.txt"       # 레포에 이 이름으로 커밋해두는 걸 권장

@st.cache_resource
def get_model_path():
    local = pathlib.Path("models") / MODEL_FILENAME
    if local.exists():
        return str(local)
    return hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, repo_type="model")

@st.cache_resource
def load_faceshape_model():
    model_path = get_model_path()
    if not os.path.isfile(CLASSES_PATH):
        raise FileNotFoundError(
            f"classes not found: {CLASSES_PATH}  (레포에 models/classes.txt로 커밋하세요)"
        )
    return FaceShapeModel(model_path, CLASSES_PATH, img_size=(224, 224))

try:
    faceshape_model = load_faceshape_model()
    st.caption(f"Loaded model from: {get_model_path()}")
    st.caption(f"Classes path: {CLASSES_PATH}")
except Exception as e:
    st.error(f"얼굴형 모델 로드 실패: {e}")
    st.stop()

# ---------- 유틸: 이미지 표시 ----------
def show_image_bgr(img_bgr, **kwargs):
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, use_container_width=True, **kwargs)
    except Exception as e:
        st.error(f"이미지 표시 중 오류: {e}")

# ---------- 사이드바 / 입력 UI ----------
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 안전모드")

with st.sidebar:
    st.subheader("📱 iPhone/URL 측정값")

    # 안전 쿼리 파서
    def _qget(name):
        v = st.query_params.get(name)
        return v[0] if isinstance(v, list) else v

    def _qfloat(name):
        v = _qget(name)
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    def _qbool(name, default=False):
        v = _qget(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "on")

    PD_MM_raw       = _qfloat("pd_mm") or _qfloat("pd")
    CHEEK_MM_raw    = _qfloat("cheek_mm") or _qfloat("cheek")
    NOSECHIN_MM_raw = _qfloat("nosechin_mm") or _qfloat("nosechin")

    use_phone_default = _qbool("use_phone", default=False)
    use_phone = st.checkbox("iPhone/URL 측정값 사용", value=use_phone_default, key="use_phone_ck")

    DEFAULT_CHEEK_MM = 150.0
    DEFAULT_PD_MM    = None

    # 얼굴폭(mm)
    if use_phone and (CHEEK_MM_raw is not None):
        CHEEK_MM = CHEEK_MM_raw
        st.success(f"📏 iPhone 얼굴 폭: {CHEEK_MM:.1f} mm")
    else:
        CHEEK_MM = st.number_input("얼굴 폭(mm)", value=DEFAULT_CHEEK_MM, step=0.5)

    # PD(mm)
    if use_phone and (PD_MM_raw is not None):
        PD_MM = PD_MM_raw
        st.write(f"👁️ PD(mm): {PD_MM:.1f} (iPhone)")
    else:
        pd_in = st.number_input("PD(mm) (옵션, 비워도 됨)", value=0.0, step=0.1, format="%.1f")
        PD_MM = pd_in if pd_in > 0 else DEFAULT_PD_MM

    NOSECHIN_MM = NOSECHIN_MM_raw if (use_phone and NOSECHIN_MM_raw is not None) else None

    # 하드 클램프
    if CHEEK_MM is not None:
        CHEEK_MM = float(min(max(CHEEK_MM, 100.0), 220.0))
    if PD_MM is not None:
        PD_MM = float(min(max(PD_MM, 45.0), 75.0))

    # 폰값 미사용이면 강제로 None
    if not use_phone:
        PD_MM = None
        NOSECHIN_MM = None

    st.divider()
    st.subheader("미세 조정")
    dx = st.slider("수평 오프셋(px)", -200, 200, 0)
    dy = st.slider("수직 오프셋(px)", -200, 200, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, 1.0)
    st.caption("iPhone/URL 측정값 사용: " + ("ON" if use_phone else "OFF"))

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"])

with colR:
    st.markdown("### 카테고리 선택 ")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요')
    use_kind   = st.multiselect('분류', ['fashion', 'sports'],    placeholder='선택하세요')

st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind

disabled = not (use_gender and use_kind)
run = st.button('실행', disabled=disabled)
if disabled:
    st.warning('성별과 분류에서 각각 최소 1개 이상 선택하세요.')
elif run and err_msgs:
    st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
    st.code("\n".join(err_msgs), language="text")

if err_msgs:
    st.info("vision 모듈 문제를 해결해야 합성이 진행됩니다. (vision.py 확인)")
    st.stop()

st.divider()

# ---------- 프레임 로드 ----------
@st.cache_resource
def _load_antena():
    return vision.load_fixed_antena()

try:
    fg_bgra, dims = _load_antena()
except Exception as e:
    st.error(f"프레임 로드 호출 실패: {e}")
    st.stop()

if fg_bgra is None or dims is None:
    st.error("프레임 이미지를 읽을 수 없어요. 경로/포맷을 확인해 주세요.")
    st.code(f"""
exists(frames)={os.path.isdir('frames')}
exists(frames/images)={os.path.isdir('frames/images')}
list(frames/images)[:10]={os.listdir('frames/images')[:10] if os.path.isdir('frames/images') else 'N/A'}
glob Antena_01.*={glob.glob('frames/images/SF191SKN_004_61 .*')}
    """, language="text")
    st.stop()

A, DBL, TOTAL = dims
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0
st.caption(f"프레임 치수 A={A}, DBL={DBL}, TOTAL={TOTAL} (GCD={GCD}, k=TOTAL/GCD={k:.3f})")

# ---------- 얼굴 이미지 ----------
if not img_file:
    st.info("얼굴 사진을 업로드하면 합성을 시작합니다.")
    st.stop()

try:
    file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
    face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if face_bgr is None:
        raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
except Exception as e:
    st.error(f"얼굴 이미지 로드 실패: {e}")
    st.stop()

# ---------- 얼굴형 추론 (HF 모델 사용) ----------
ar, jaw, cw, jw = compute_metrics_bgr(face_bgr)  # MediaPipe 지표
pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
probs = faceshape_model.predict_probs(pil_img)
_, final_label, explain = decide_rule_vs_top2(
    probs, faceshape_model.class_names, ar=ar, jaw_deg=jaw, cw=cw, jw=jw
)
st.session_state["faceshape_label"] = final_label

# ---------- PD/자세/스케일/합성 ----------
pd_px   = None
mid     = (0, 0)
eye_roll_deg = 0.0
PD_SRC  = None  # 'iphone' | 'manual' | 'mediapipe' | None

if (PD_MM is not None) and (PD_MM > 0):
    PD_SRC = "manual"
elif use_phone and (PD_MM_raw is not None):
    PD_SRC = "iphone"
    PD_MM  = PD_MM_raw
else:
    try:
        pd_px, eye_roll_deg, mid = vision.detect_pd_px(face_bgr)
        if pd_px is None:
            raise RuntimeError("눈 검출 실패")
        PD_SRC = "mediapipe"
    except Exception as e:
        PD_SRC = None
        st.error(f"MediaPipe PD 계산 실패: {e}")
        st.stop()

yaw = pitch = roll = None
if hasattr(vision, "head_pose_ypr"):
    try:
        yaw, pitch, roll = vision.head_pose_ypr(face_bgr)  # ° 단위
    except Exception:
        yaw = pitch = roll = None
if roll is None:
    roll = eye_roll_deg

if PD_SRC == "mediapipe":
    st.write(
        f"**PD_px**: {pd_px:.2f} px  /  "
        f"**roll**: {roll:.2f}°{' (eye-line)' if yaw is None else ''}  /  "
        f"**mid**: {tuple(round(v,1) for v in mid)}"
    )
elif PD_SRC in ("iphone", "manual"):
    tag = "iPhone 측정값" if PD_SRC == "iphone" else "수동 입력"
    st.write(f"**PD(mm)**: {PD_MM:.2f} mm ({tag})  /  **roll**: {roll:.2f}°")
else:
    st.warning("PD 소스를 확인할 수 없습니다.")

# 스케일 & 위치 계산
h_face, w_face = face_bgr.shape[:2]
fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)
h0, w0 = fg_bgra.shape[:2]

GCD2PD_CAL = 0.92
target_GCD_px = pd_px if pd_px is not None else (
    (PD_MM / (CHEEK_MM / max(w_face, 1e-6))) if PD_MM else None
)

frame_GCD_px0 = w0 / max(k, 1e-6)
if target_GCD_px is not None:
    target_total_px = (target_GCD_px * GCD2PD_CAL) * k
else:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

Cw_px = None
if hasattr(vision, "cheek_width_px"):
    try:
        Cw_px = vision.cheek_width_px(face_bgr)
    except Exception:
        Cw_px = None
if Cw_px is None and (cw is not None):
    Cw_px = float(cw)

min_w = 0.60 * w_face
max_w = 0.95 * w_face
if Cw_px is not None:
    min_w = max(min_w, 0.70 * Cw_px)
    max_w = min(max_w, 0.98 * Cw_px)

target_total_px = float(np.clip(target_total_px, min_w, max_w))
scale = (target_total_px / max(w0, 1)) * float(scale_mult)
scale = float(np.clip(scale, 0.35, 2.2))

new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

pitch_deg = pitch if pitch is not None else 0.0
pitch_dy  = int(pitch_deg * 0.8)

if mid == (0, 0):
    gx = int(face_bgr.shape[1] * 0.5 - fg_rot.shape[1] * 0.5) + dx
    gy = int(face_bgr.shape[0] * 0.45 - fg_rot.shape[0] * 0.5) + dy
else:
    anchor = 0.50
    gx = int(mid[0] - fg_rot.shape[1] * anchor) + dx
    gy = int(mid[1] - fg_rot.shape[0] * 0.50) + dy + pitch_dy

# 합성 (여백 확보)
h_bg, w_bg = face_bgr.shape[:2]
margin_x, margin_y = 300, 150
bg_expanded = cv2.copyMakeBorder(
    face_bgr, margin_y, margin_y, margin_x, margin_x,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
gx_expanded = gx + margin_x
gy_expanded = gy + margin_y

out = vision.overlay_rgba(bg_expanded, fg_rot, gx_expanded, gy_expanded)
show_image_bgr(out, caption="합성 결과")

# 다운로드
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                       file_name="SF191SKN_004_61.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

# (옵션) 얼굴형 기반 추천
if final_label := st.session_state.get("faceshape_label"):
    rec = None
    if final_label == "Oval":
        rec = "대부분의 프레임 OK (aviator/wayfarer/스퀘어/원형)"
    elif final_label == "Round":
        rec = "각진 프레임 추천 (스퀘어/레트로 스퀘어)"
    elif final_label == "Square":
        rec = "곡선형 프레임 추천 (원형/오벌/보스턴)"
    elif final_label == "Oblong":
        rec = "세로를 낮추고 가로가 긴 타입 (wayfarer/클럽마스터)"
    elif final_label == "Heart":
        rec = "하부가 살짝 넓은 오벌/보스턴, 얇은 메탈 림"
    st.markdown("### 얼굴형 판별 결과")
    st.success(f"당신은 **{final_label}형**입니다.")
    with st.expander("판별 근거(디버그)"):
        st.write({
            "AR": None if ar is None else round(float(ar), 4),
            "jaw_deg": None if jaw is None else round(float(jaw), 2),
            "Cw": None if cw is None else round(float(cw), 2),
            "Jw": None if jw is None else round(float(jw), 2),
            "explain": explain
        })
    if rec:
        st.info(f"👓 얼굴형({final_label}) 추천: {rec}")

