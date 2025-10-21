# ---------- 반드시 최상단 1회 ----------
# --- Keras 백엔드 고정: 반드시 모든 import 이전 ---


import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # (선택) TF 로그 줄이기
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # (선택) CPU 강제


import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# ---------- 기본 진단 캡션 ----------
import sys, platform, os, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# ---------- 유틸: 이미지 표시(버전 호환) ----------
def show_image_bgr(img_bgr, **kwargs):
    try:
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            st.image(rgb, use_container_width=True, **kwargs)
        except TypeError:
            try:
                st.image(rgb, use_column_width=True, **kwargs)
            except TypeError:
                st.image(rgb, **kwargs)
    except Exception as e:
        st.error(f"이미지 표시 중 오류: {e}")

# ---------- 지연 임포트: 실패해도 UI는 뜨게 ----------
cv2 = np = Image = None
vision = None
err_msgs = []

try:
    import numpy as np
except Exception as e:
    err_msgs.append(f"numpy import 실패: {e}")

try:
    import cv2
except Exception as e:
    err_msgs.append(f"opencv(cv2) import 실패: {e}")

try:
    from PIL import Image
except Exception as e:
    err_msgs.append(f"Pillow import 실패: {e}")

try:
    import vision  # vision.py 에 detect_pd_px / load_fixed_antena / overlay_rgba 있어야 함
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

# ▶ 추가: 얼굴형 추론을 위한 모듈들
try:
    from faceshape import FaceShapeModel, decide_rule_vs_top2
except Exception as e:
    err_msgs.append(f"faceshape 모듈 임포트 실패: {e}")

try:
    from metrics import compute_metrics_bgr
except Exception as e:
    err_msgs.append(f"metrics 모듈 임포트 실패: {e}")

# ---------- 사이드바 / 입력 UI는 무조건 출력 ----------
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 안전모드")

with st.sidebar:
    st.subheader("📱 iPhone/URL 측정값")

    # ---- 안전한 쿼리 파서 ----
    def _qget(name):
        v = st.query_params.get(name)
        if isinstance(v, list):  # 다중 값일 때 첫 번째
            v = v[0]
        return v

    def _qfloat(name):
        v = _qget(name)
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    def _qbool(name, default=False):
        # FIX: 월러스(:=) 제거
        v = _qget(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "on")

    # ---- 쿼리 원본(raw) 값만 일단 읽기 ----
    PD_MM_raw       = _qfloat("pd_mm") or _qfloat("pd")
    CHEEK_MM_raw    = _qfloat("cheek_mm") or _qfloat("cheek")
    NOSECHIN_MM_raw = _qfloat("nosechin_mm") or _qfloat("nosechin")

    # ✅ 기본값: 쿼리 플래그로 제어 (없으면 False)
    use_phone_default = _qbool("use_phone", default=False)
    use_phone = st.checkbox("iPhone/URL 측정값 사용", value=use_phone_default, key="use_phone_ck")

    DEFAULT_CHEEK_MM = 150.0
    DEFAULT_PD_MM    = None  # None이면 나중에 PD_px 사용

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
        PD_MM = pd_in if pd_in > 0 else DEFAULT_PD_MM  # 0 → None

    # 기타
    NOSECHIN_MM = NOSECHIN_MM_raw if (use_phone and NOSECHIN_MM_raw is not None) else None

    # ✅ 하드 클램프
    if CHEEK_MM is not None:
        CHEEK_MM = float(min(max(CHEEK_MM, 100.0), 220.0))
    if PD_MM is not None:
        PD_MM = float(min(max(PD_MM, 45.0), 75.0))

    # ✅ 핵심: 폰값 미사용이면 강제로 None
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
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요')

# 예: 플래그로 사용
is_female = 'female' in use_gender
is_male   = 'male'   in use_gender
is_unisex = 'unisex' in use_gender
is_fashion = 'fashion' in use_kind
is_sports  = 'sports'  in use_kind

# 예: 세션에 저장(다른 페이지/콜백에서도 사용)
st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind

# 5) 실행 버튼: 두 그룹 모두 최소 1개 선택돼야 활성화
disabled = not (use_gender and use_kind)
run = st.button('실행', disabled=disabled)
if disabled:
    st.warning('성별과 분류에서 각각 최소 1개 이상 선택하세요.')
elif run:
    st.success(f'실행! 성별={use_gender}, 분류={use_kind}')
    if err_msgs:
        st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
        st.code("\n".join(err_msgs), language="text")

# ---------- 임포트 실패 시, 여기서 멈추지 말고 안내만 ----------
if err_msgs:
    st.info("위 임포트 문제를 해결해야 합성이 진행됩니다. (requirements.txt / OpenCV headless / vision.py / faceshape.py / metrics.py 확인)")
    st.stop()

st.divider()

# ---------- 프레임 로드 ----------
try:
    fg_bgra, dims = vision.load_fixed_antena()
except Exception as e:
    st.error(f"프레임 로드 호출 실패: {e}")
    dims = None
    fg_bgra = None

if fg_bgra is None or dims is None:
    st.error("프레임 이미지를 읽을 수 없어요. 경로/포맷을 확인해 주세요.")
    st.code(f"""
exists(frames)={os.path.isdir('frames')}
exists(frames/images)={os.path.isdir('frames/images')}
list(frames/images)[:10]={os.listdir('frames/images')[:10] if os.path.isdir('frames/images') else 'N/A'}
glob Antena_01.*={glob.glob('frames/images/SF191SKN_004_61.*')}
    """, language="text")
    st.stop()

A, DBL, TOTAL = dims
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0
st.caption(f"프레임 치수 A={A}, DBL={DBL}, TOTAL={TOTAL} (GCD={GCD}, k=TOTAL/GCD={k:.3f})")

# ---------- 얼굴 이미지 업로드 필요 ----------
if not img_file:
    st.info("얼굴 사진을 업로드하면 합성을 시작합니다.")
    st.stop()

# ---------- 얼굴 이미지 읽기 ----------
try:
    file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
    face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if face_bgr is None:
        raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
except Exception as e:
    st.error(f"얼굴 이미지 로드 실패: {e}")
    st.stop()

# ============================
# ▶ 얼굴형 inference (모델 + 규칙 결합) — 단일 블록
# ============================
MODEL_PATH   = "models/faceshape_efficientnetB4_best_20251018_223855.keras"  # ✅ 네 파일명으로 고정
CLASSES_PATH = "models/classes.txt"
IMG_SIZE     = (224, 224)

@st.cache_resource
def _load_faceshape():
    return FaceShapeModel(MODEL_PATH, CLASSES_PATH, img_size=IMG_SIZE)

def _is_lfs_pointer(path:str)->bool:
    """모델 파일이 Git LFS 포인터인지 빠르게 판별"""
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) > 2048:  # 2KB 넘으면 포인터 아님
            return False
        with open(path, "rb") as f:
            head = f.read(256)
        return (b"git-lfs" in head) or (b"github.com/spec" in head)
    except Exception:
        return False

faceshape_model = None
if not os.path.isfile(MODEL_PATH):
    st.warning("※ 얼굴형 모델(.keras)이 없습니다. (models/*.keras 필요)")
elif not os.path.isfile(CLASSES_PATH):
    st.warning("※ classes.txt 파일이 없습니다. (models/classes.txt 필요)")
elif _is_lfs_pointer(MODEL_PATH):
    st.error("모델 파일이 Git LFS 포인터로 보입니다. Releases/S3 등에서 **실제 바이너리**를 받아오세요.")
else:
    try:
        faceshape_model = _load_faceshape()
    except Exception as e:
        st.error("얼굴형 모델 로드 실패 — 모델 없이 추천은 스킵하고 합성만 진행합니다.")
        st.exception(e)
        faceshape_model = None

final_label = None
if faceshape_model is not None:
    try:
        # 1) 모델 확률
        pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        probs = faceshape_model.predict_probs(pil_img)  # (C,)
        classes = faceshape_model.class_names

        # 2) (선택) MediaPipe 지표 — 실패해도 None 반환
        try:
            ar, jaw, cw, jw = compute_metrics_bgr(face_bgr)
        except Exception:
            ar = jaw = cw = jw = None

        # 3) decide_rule_vs_top2 로 최종 선택
        idx, label, reason = decide_rule_vs_top2(
            probs, classes, ar=ar, jaw_deg=jaw, cw=cw, jw=jw
        )
        final_label = label

        # 4) UI
        st.subheader("최종 얼굴형 (모델+규칙)")
        st.success(final_label)

        with st.expander("얼굴형 디버그"):
            order = np.argsort(-probs)
            st.write("모델 상위 확률:")
            for i in order[:min(5, len(probs))]:
                st.write(f"- {classes[i]:7s}: {probs[i]:.4f}")
            st.write("지표:", {
                "AR": None if ar is None else round(float(ar), 4),
                "jaw_deg": None if jaw is None else round(float(jaw), 2),
                "Cw": None if cw is None else round(float(cw), 2),
                "Jw": None if jw is None else round(float(jw), 2),
            })
            st.caption(reason)
    except Exception as e:
        st.warning("얼굴형 추론 중 경고가 발생했습니다. 아래 상세를 확인하세요.")
        st.exception(e)

# 다운스트림에서 쓰기 쉽게 세션에 저장
st.session_state["faceshape_label"] = final_label

# ============================
# PD/자세/스케일/합성
# ============================

# ============================
# PD 계산 (iPhone/수동/MediaPipe) + 출력
# ============================
pd_px   = None
mid     = (0, 0)
eye_roll_deg = 0.0
PD_SRC  = None  # 'iphone' | 'manual' | 'mediapipe' | None

# 우선순위:
# 1) 수동 입력 PD_MM(양수면) -> 'manual'
# 2) use_phone 켠 상태 + PD_MM_raw(URl/iPhone 값 존재) -> 'iphone'
# 3) 둘 다 없으면 MediaPipe로 pd_px 계산 -> 'mediapipe'
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

# 2) (있으면) 3축 자세 → 없으면 roll은 눈선 기반
yaw = pitch = roll = None
if hasattr(vision, "head_pose_ypr"):
    try:
        yaw, pitch, roll = vision.head_pose_ypr(face_bgr)  # ° 단위
    except Exception:
        yaw = pitch = roll = None
if roll is None:
    roll = eye_roll_deg

# ✅ PD 표시(소스별로 문구 다르게)
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

# 3) 프레임 PNG 클린업(흰 배경 제거 + 여백 트림)
fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)

# ========= 스케일 & 위치 계산 =========
h_face, w_face = face_bgr.shape[:2]

# 프레임의 초기 치수
h0, w0 = fg_bgra.shape[:2]

# 1) PD/볼폭 정보 준비
#    - PD(px) 우선 사용
#    - PD(mm)만 있으면 mm->px 변환
#    - 보정계수(GCD→PD)로 과대/과소를 1차 조절
GCD2PD_CAL = 0.92

target_GCD_px = None
if pd_px is not None:
    target_GCD_px = pd_px
elif PD_MM:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_GCD_px = PD_MM / max(mm_per_px, 1e-6)

# 볼폭(cheek width) 픽셀 – 하드 클램프 기준
Cw_px = vision.cheek_width_px(face_bgr)  # None일 수 있음

# 프레임 파일에서 GCD(px) 추정: width / k   (k = TOTAL / (A+DBL))
frame_GCD_px0 = w0 / max(k, 1e-6)

# 2) 목표 폭(px) 산출
if target_GCD_px is not None:
    # GCD 정합 기반 폭
    target_GCD_px *= GCD2PD_CAL
    target_total_px = target_GCD_px * k     # TOTAL = k * GCD
else:
    # PD 정보가 전혀 없으면 TOTAL(mm)로 후퇴
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

# 3) 하드 클램프: 얼굴/볼폭 대비 과대 방지
#    - 프레임 총폭이 얼굴폭의 0.60~0.95배 범위로
#    - Cw(px)가 있으면 0.70~0.98배 추가 제한
min_w = 0.60 * w_face
max_w = 0.95 * w_face
if Cw_px is not None:
    min_w = max(min_w, 0.70 * Cw_px)
    max_w = min(max_w, 0.98 * Cw_px)

target_total_px = float(np.clip(target_total_px, min_w, max_w))

# 4) 스케일 계산 + 안전 클립
scale = (target_total_px / max(w0, 1)) * float(scale_mult)
scale = float(np.clip(scale, 0.35, 2.2))

# 5) 리사이즈
new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

# 6) 회전(roll 반대)
M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

# 7) 위치 – PD 중점 기준(있으면), 없으면 중앙
pitch_deg = pitch if 'pitch' in locals() and pitch is not None else 0.0
pitch_dy  = int(pitch_deg * 0.8)

if mid == (0, 0):  # iPhone 모드(중점 없음)
    gx = int(face_bgr.shape[1] * 0.5 - fg_rot.shape[1] * 0.5) + dx
    gy = int(face_bgr.shape[0] * 0.45 - fg_rot.shape[0] * 0.5) + dy
else:               # MediaPipe 모드(눈 중점 기준)
    anchor = 0.50   # 렌즈 중앙 기준 정렬(0.45~0.55로 미세조정)
    gx = int(mid[0] - fg_rot.shape[1] * anchor) + dx
    gy = int(mid[1] - fg_rot.shape[0] * 0.50) + dy + pitch_dy

# 8) 합성 전: 여백 확보 (잘림 방지)
h_bg, w_bg = face_bgr.shape[:2]
margin_x, margin_y = 300, 150
bg_expanded = cv2.copyMakeBorder(
    face_bgr, margin_y, margin_y, margin_x, margin_x,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)

# 위치 좌표도 margin 보정
gx_expanded = gx + margin_x
gy_expanded = gy + margin_y

# 9) 합성
out = vision.overlay_rgba(bg_expanded, fg_rot, gx_expanded, gy_expanded)
show_image_bgr(out, caption="합성 결과")

# ---------- 다운로드 ----------
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                       file_name="SF191SKN_004_61.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

# ========= (옵션) 얼굴형 기반 추천 예시 =========
if final_label:
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

    if rec:
        st.info(f"👓 얼굴형({final_label}) 추천: {rec}")

