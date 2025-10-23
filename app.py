# =============================
# 0) 백엔드/로그 환경변수 먼저 고정
# =============================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # Keras 3 백엔드 -> TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"              # (선택) TF 로그 억제
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # (선택) CPU 강제

# =============================
# 1) 핵심 라이브러리 임포트 & 버전 확인
# =============================
import numpy as np
import tensorflow as tf
import keras
import cv2
import PIL

print("NumPy:", np.__version__)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("cv2:", cv2.__version__)
print("Pillow:", PIL.__version__)

# Streamlit
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# 시스템 정보
import sys, platform, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# =============================
# 2) faceshape / vision / metrics 임포트 (이 시점!)
# =============================
err_msgs = []

try:
    # faceshape는 KERAS_BACKEND 고정 후에 임포트해야 안전
    from faceshape import (
        FaceShapeModel,
        apply_rules,
        decide_rule_vs_top2,   # 쓰지 않으려면 임포트 안 해도 됨
        topk_from_probs,
        top2_strings,
    )
except Exception as e:
    err_msgs.append(f"faceshape 임포트 실패: {e}")

try:
    import vision  # vision.py: detect_pd_px / load_fixed_antena / overlay_rgba / ...
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

try:
    from metrics import compute_metrics_bgr
except Exception as e:
    err_msgs.append(f"metrics 임포트 실패: {e}")

# =============================
# 3) 유틸: 이미지 표시
# =============================
def show_image_bgr(img_bgr, **kwargs):
    try:
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

# =============================
# 4) UI
# =============================
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 안전모드")

with st.sidebar:
    st.subheader("📱 iPhone/URL 측정값")

    def _qget(name):
        v = st.query_params.get(name)
        if isinstance(v, list):
            v = v[0]
        return v

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

    if use_phone and (CHEEK_MM_raw is not None):
        CHEEK_MM = CHEEK_MM_raw
        st.success(f"📏 iPhone 얼굴 폭: {CHEEK_MM:.1f} mm")
    else:
        CHEEK_MM = st.number_input("얼굴 폭(mm)", value=DEFAULT_CHEEK_MM, step=0.5)

    if use_phone and (PD_MM_raw is not None):
        PD_MM = PD_MM_raw
        st.write(f"👁️ PD(mm): {PD_MM:.1f} (iPhone)")
    else:
        pd_in = st.number_input("PD(mm) (옵션, 비워도 됨)", value=0.0, step=0.1, format="%.1f")
        PD_MM = pd_in if pd_in > 0 else DEFAULT_PD_MM

    NOSECHIN_MM = NOSECHIN_MM_raw if (use_phone and NOSECHIN_MM_raw is not None) else None

    if CHEEK_MM is not None:
        CHEEK_MM = float(min(max(CHEEK_MM, 100.0), 220.0))
    if PD_MM is not None:
        PD_MM = float(min(max(PD_MM, 45.0), 75.0))

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

is_female = 'female' in use_gender
is_male   = 'male'   in use_gender
is_unisex = 'unisex' in use_gender
is_fashion = 'fashion' in use_kind
is_sports  = 'sports'  in use_kind

st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind

disabled = not (use_gender and use_kind)
run = st.button('실행', disabled=disabled)
if disabled:
    st.warning('성별과 분류에서 각각 최소 1개 이상 선택하세요.')
elif run:
    st.success(f'실행! 성별={use_gender}, 분류={use_kind}')
    if err_msgs:
        st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
        st.code("\n".join(err_msgs), language="text")

if err_msgs:
    st.info("위 임포트 문제를 해결해야 합성이 진행됩니다. (requirements.txt / OpenCV headless / vision.py / faceshape.py / metrics.py 확인)")
    st.stop()

st.divider()
# 실행 버튼 누르기 전에는 아무 것도 진행하지 않음
if not run:
    st.info("성별/분류를 선택하고 '실행'을 누르면 얼굴형 분석과 추천이 시작됩니다.")
    st.stop()




# =============================
# 6) 얼굴 이미지 업로드
# =============================
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

# =============================
# 7) 얼굴형 모델 로드
# =============================
MODEL_PATH   = "models/faceshape_efficientnetB4_best_20251018_223855.keras"
CLASSES_PATH = "models/classes_20251018_223855.txt"
IMG_SIZE     = (224, 224)

@st.cache_resource
def _load_faceshape():
    return FaceShapeModel(MODEL_PATH, CLASSES_PATH, img_size=IMG_SIZE)

def _is_lfs_pointer(path:str)->bool:
    try:
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) > 2048:
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
    st.warning("※ classes.txt 파일이 없습니다. (models/classes*.txt 필요)")
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
        # (A) 모델 확률 (보정 없이 Top-2만 표시)
        pil_img = PIL.Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        probs = faceshape_model.predict_probs(pil_img)                # ← faceshape_model 사용
        top2_raw = topk_from_probs(probs, faceshape_model.class_names)    # 원본
        labels_raw = top2_strings(top2_raw)

        st.subheader("모델 Top-2 (원본)")
        st.write(" / ".join(labels_raw))

        # (B) (선택) MediaPipe 지표
        try:
            ar, jaw, cw, jw = compute_metrics_bgr(face_bgr)
        except Exception:
            ar = jaw = cw = jw = None

        # (C) 규칙 보정 + 재랭킹 🔧
        if any(v is not None for v in (ar, jaw, cw, jw)):
            adj = apply_rules(                                      # 🔧 보정 실행
            probs, faceshape_model.class_names,
            ar=ar, jaw_deg=jaw, cw=cw, jw=jw
            )
            probs_adj = adj['rule_probs']
            top2_adj  = topk_from_probs(probs_adj, faceshape_model.class_names)
            labels_adj = top2_strings(top2_adj)

            st.subheader("모델 Top-2 (규칙 보정 후)")               # 🔧 보정 결과 표시
            st.write(" / ".join(labels_adj))
            final_label = adj['rule_label']                          # 🔧 최종 라벨은 보정 결과
            reason = "rules+model"
        else:
            # 지표가 없으면 모델 원본 유지
            idx, final_label, reason = decide_rule_vs_top2(probs, faceshape_model.class_names)
            st.info("지표 없음 → 보정 미적용 (model-top1)")

        with st.expander("얼굴형 디버그"):
            order = np.argsort(-probs)
            st.write("모델 상위 확률(원본):")
            for i in order[:min(5, len(probs))]:
                st.write(f"- {faceshape_model.class_names[i]:7s}: {probs[i]:.4f}")
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

st.session_state["faceshape_label"] = final_label



# =============================
# 5) 프레임 로드 (엑셀 카탈로그 · 최소 규칙 + sports시 shield)
# =============================
import os, random
import pandas as pd

EXCEL_PATH = "sg_df.xlsx"  # 카탈로그 경로

# 6개 모양 고정
SHAPES6 = {"Round","Rectangular","Trapezoid","Aviator","Cat-eye","Shield"}

# 얼굴형 → 최소 추천 모양(우선순위)
FRAME_RULES_ORDERED = {
    "Oval":   ["Trapezoid","Rectangular"],
    "Round":  ["Rectangular"],
    "Square": ["Round"],
    "Oblong": ["Rectangular","Trapezoid"],
    "Heart":  ["Cat-eye","Round"],
}
MAX_SHAPES_PER_FACE = 1   # 너무 많지 않게 1개만 (원하면 2로)

def _norm(x):
    return (x or "").strip().lower()

# 0) 엑셀 로드
try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    st.error(f"엑셀 카탈로그 로드 실패: {e}")
    st.stop()

# 1) 필수 컬럼 체크
need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm"]
for c in need_cols:
    if c not in df.columns:
        st.error(f"엑셀에 '{c}' 컬럼이 없습니다.")
        st.stop()

# 2) 전처리/검증

df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()   # fashion/sports
df["sex"]     = df["sex"].astype(str).str.strip().str.lower()       # male/female/unisex
for c in ["lens_mm","bridge_mm","total_mm"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

bad = df.loc[~df["shape"].isin(SHAPES6), ["product_id","brand","shape"]]
if len(bad) > 0:
    st.error("shape 값은 round/rectangular/trapezoid/aviator/cat-eye/shield 만 허용됩니다.")
    st.dataframe(bad)
    st.stop()

# 3) 성별/분류 1차 필터
gset = {_norm(g) for g in use_gender}
kset = {_norm(k) for k in use_kind}

f = pd.Series([True] * len(df))
if gset:
    f &= df["sex"].isin(gset) | (df["sex"] == "unisex")
if kset:
    f &= df["purpose"].isin(kset)

cand = df[f].copy()

# 4) 얼굴형 최소 규칙 + sports면 shield 조건부 추가
label = st.session_state.get("faceshape_label", final_label)

if label in FRAME_RULES_ORDERED:
    ok_shapes = list(FRAME_RULES_ORDERED[label][:MAX_SHAPES_PER_FACE])

    # sports 선택 시, 일부 얼굴형에 한해 shield 추가 (최대 2유형로 제한)
    if 'sports' in kset and 'shield' not in ok_shapes:
        if label in ('Oval','Round','Oblong'):   # 필요하면 얼굴형 추가 가능
            ok_shapes.append('shield')
            ok_shapes = ok_shapes[:2]

    pool = cand[cand["shape"].isin(set(ok_shapes))]
    if len(pool) > 0:
        cand = pool

if len(cand) == 0:
    st.error("조건(성별/분류/얼굴형)에 맞는 프레임을 찾지 못했습니다.")
    st.stop()

# 5) 후보 중 랜덤 1개 선택
row = cand.sample(1, random_state=random.randint(0,10_000)).iloc[0].to_dict()

# 6) 이미지 경로 결정: image_path 우선, 없으면 product_id.* 탐색
# === 이미지 경로 해결(이미지 폴더 없음 버전) ===
import os, glob

FRAME_ROOT = "frame"  # 최상위 폴더만 사용

# shape 값 -> 실제 폴더명 매핑 (엑셀은 소문자/하이픈, 폴더는 대문자/언더스코어)
SHAPE_DIR_MAP = {
    "aviator":     "Aviator",
    "cat-eye":     "Cat_eye",
    "rectangular": "Rectangular",
    "round":       "Round",
    "shield":      "Shield",
    "trapezoid":   "Trapezoid",
}
EXTS = (".png", ".webp", ".avif", ".jpg", ".jpeg")

def _resolve_image(row: dict) -> str | None:
    """
    우선순위:
      1) row['image_path']가 실제 존재
      2) frame/{ShapeDir}/{product_id}.*
      3) frame/**/{product_id}.*  (최후 수단: 재귀 탐색)
    """
    # 0) 명시 경로 우선
    p = (row.get("image_path") or "").strip()
    if p and os.path.exists(p):
        return p

    pid = str(row.get("product_id", "")).strip()
    if not pid:
        return None

    # 1) shape 폴더에서 찾기
    shape_val = str(row.get("shape", "")).strip().lower()
    shape_dir = SHAPE_DIR_MAP.get(shape_val)
    if shape_dir:
        base = os.path.join(FRAME_ROOT, shape_dir, pid)
        for ext in EXTS:
            cp = base + ext
            if os.path.exists(cp):
                return cp

    # 2) 최후 수단: 전역 재귀 탐색
    pattern = os.path.join(FRAME_ROOT, "**", pid + ".*")
    for cp in glob.glob(pattern, recursive=True):
        if os.path.splitext(cp)[1].lower() in EXTS and os.path.isfile(cp):
            return cp

    return None



img_path = _resolve_image(row)
if not img_path:
    st.error(
        f"이미지를 찾지 못했습니다: frame/<Aviator|Cat_eye|Rectangular|Round|Shield|Trapezoid>/{row['product_id']}.[png|webp|avif|jpg|jpeg]"
    )
    st.stop()

# 7) 프레임 이미지 로드 (BGRA 보장)
fg_bgra = vision.ensure_bgra(img_path)
if fg_bgra is None:
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}")
    st.stop()

# 8) 치수 세팅
A, DBL, TOTAL = float(row["lens"]), float(row["bridc"]), float(row["total_r"])
dims = (A, DBL, TOTAL)
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0

st.caption(
    f"선택 프레임: {row.get('brand','')} / {row.get('product_id','')}  · "
    f"shape={row.get('shape','?')} · A={A}, DBL={DBL}, TOTAL={TOTAL} "
    f"(GCD={GCD}, k=TOTAL/GCD={k:.3f})"
)


# =============================
# 8) PD/자세/스케일/합성
# =============================
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

fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)

h_face, w_face = face_bgr.shape[:2]
h0, w0 = fg_bgra.shape[:2]

GCD2PD_CAL = 0.92
target_GCD_px = None
if pd_px is not None:
    target_GCD_px = pd_px
elif PD_MM:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_GCD_px = PD_MM / max(mm_per_px, 1e-6)

Cw_px = vision.cheek_width_px(face_bgr)  # None일 수 있음
frame_GCD_px0 = w0 / max(k, 1e-6)

if target_GCD_px is not None:
    target_GCD_px *= GCD2PD_CAL
    target_total_px = target_GCD_px * k
else:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

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

try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    PIL.Image.fromarray(rgb).save(buf, format="PNG")
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                       file_name="SF191SKN_004_61.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

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

