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
import random
import pandas as pd
from pathlib import Path

print("NumPy:", np.__version__)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("cv2:", cv2.__version__)
print("Pillow:", PIL.__version__)

# Streamlit
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성", layout="wide")

# 시스템 정보
import sys, platform, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# =============================
# 2) faceshape / vision / metrics 임포트 (이 시점!)
# =============================
err_msgs = []
try:
    from faceshape import (
        FaceShapeModel,
        apply_rules,
        decide_rule_vs_top2,
        topk_from_probs,
        top2_strings,
    )
except Exception as e:
    err_msgs.append(f"faceshape 임포트 실패: {e}")

try:
    import vision  # detect_pd_px / head_pose_ypr / ensure_bgra / ...
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

def _norm(x):
    return (x or "").strip().lower()

# =============================
# 4) UI
# =============================
st.title("🧍→🕶️ 선글라스 합성 (얼굴형 기반 추천)")

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

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"])

with colR:
    st.markdown("### 카테고리 선택 ")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요')
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요')

st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind

disabled = not (use_gender and use_kind)
clicked = st.button('실행', disabled=disabled)
if clicked:
    st.session_state['started'] = True

if err_msgs:
    st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
    st.code("\n".join(err_msgs), language="text")

st.divider()
if not st.session_state.get('started', False):
    st.info("성별/분류를 선택하고 '실행'을 누르면 얼굴형 분석과 추천이 시작됩니다.")
    st.stop()

# =============================
# 6) 얼굴 이미지 업로드 (세션에 보관)
# =============================
if not img_file and 'face_bytes' not in st.session_state:
    st.info("얼굴 사진을 업로드하면 시작합니다.")
    st.stop()

if img_file is not None:
    st.session_state['face_bytes'] = img_file.getvalue()

file_bytes = np.frombuffer(st.session_state['face_bytes'], dtype=np.uint8)
face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if face_bgr is None:
    st.error("얼굴 이미지 디코딩 실패(파일 형식/손상 확인).")
    st.stop()

# =============================
# 7) 얼굴형 모델 로드 → 얼굴형 추론
# =============================
MODEL_PATH   = "models/faceshape_efficientnetB4_best_20251018_223855.keras"
CLASSES_PATH = "models/classes_20251018_223855.txt"
IMG_SIZE     = (224, 224)

@st.cache_resource
def _load_faceshape():
    return FaceShapeModel(MODEL_PATH, CLASSES_PATH, img_size=IMG_SIZE)

def _is_lfs_pointer(path:str)->bool:
    try:
        if not os.path.isfile(path): return False
        if os.path.getsize(path) > 2048: return False
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
    st.error("모델 파일이 Git LFS 포인터 같습니다. 실제 바이너리를 배치하세요.")
else:
    try:
        faceshape_model = _load_faceshape()
    except Exception as e:
        st.error("얼굴형 모델 로드 실패 — 모델 없이 추천은 스킵합니다.")
        st.exception(e)
        faceshape_model = None

final_label = None
if faceshape_model is not None:
    try:
        pil_img = PIL.Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        probs = faceshape_model.predict_probs(pil_img)
        top2_raw = topk_from_probs(probs, faceshape_model.class_names)
        labels_raw = top2_strings(top2_raw)
        st.subheader("모델 Top-2 (원본)")
        st.write(" / ".join(labels_raw))

        # (선택) MediaPipe 지표
        try:
            ar, jaw, cw, jw = compute_metrics_bgr(face_bgr)
        except Exception:
            ar = jaw = cw = jw = None

        if any(v is not None for v in (ar, jaw, cw, jw)):
            adj = apply_rules(probs, faceshape_model.class_names, ar=ar, jaw_deg=jaw, cw=cw, jw=jw)
            probs_adj = adj['rule_probs']
            top2_adj  = topk_from_probs(probs_adj, faceshape_model.class_names)
            labels_adj = top2_strings(top2_adj)
            st.subheader("모델 Top-2 (규칙 보정 후)")
            st.write(" / ".join(labels_adj))
            final_label = adj['rule_label']
        else:
            _, final_label, _ = decide_rule_vs_top2(probs, faceshape_model.class_names)
            st.info("지표 없음 → 보정 미적용 (model-top1)")
    except Exception as e:
        st.warning("얼굴형 추론 중 경고가 발생했습니다.")
        st.exception(e)

st.session_state["faceshape_label"] = final_label

# =============================
# 5) 프레임 로드 (엑셀 · 최소 규칙 + sports 시 shield)
# =============================
ROOT = Path.cwd()
EXCEL_PATH = ROOT / "sunglass_df_test.xlsx"   # ← 네 엑셀 경로

SHAPES6 = {"round","rectangular","trapezoid","aviator","cat_eye","shield"}

FRAME_RULES_ORDERED = {
    "Oval":   ["trapezoid","rectangular"],
    "Round":  ["rectangular"],
    "Square": ["round"],
    "Oblong": ["rectangular","trapezoid"],
    "Heart":  ["cat-eye","round"],
}
MAX_SHAPES_PER_FACE = 1   # 필요시 2로

# 0) 엑셀 로드
try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    st.error(f"엑셀 카탈로그 로드 실패: {e}")
    st.stop()

# 1) 필수 컬럼 체크 (lens_mm / bridge_mm / total_mm 사용)
need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm"]
for c in need_cols:
    if c not in df.columns:
        st.error(f"엑셀에 '{c}' 컬럼이 없습니다.")
        st.stop()

# 2) 전처리/검증
df["shape"]   = df["shape"].astype(str).str.strip().str.lower()
df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()   # fashion/sports
df["sex"]     = df["sex"].astype(str).str.strip().str.lower()       # male/female/unisex
for c in ["lens_mm","bridge_mm","total_mm"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

bad = df.loc[~df["shape"].isin(SHAPES6), ["product_id","brand","shape"]]
if len(bad) > 0:
    st.error("shape 값은 round/rectangular/trapezoid/aviator/cat-eye/shield 만 허용됩니다.")
    st.dataframe(bad); st.stop()

# 3) 성별/분류 필터
gset = {_norm(g) for g in st.session_state['use_gender']}
kset = {_norm(k) for k in st.session_state['use_kind']}

f = pd.Series([True] * len(df))
if gset:
    f &= df["sex"].isin(gset) | (df["sex"] == "unisex")
if kset:
    f &= df["purpose"].isin(kset)

cand = df[f].copy()

# 4) 얼굴형 최소 규칙 + sports면 shield 추가(제한)
label = st.session_state.get("faceshape_label", final_label)
if label in FRAME_RULES_ORDERED:
    ok_shapes = list(FRAME_RULES_ORDERED[label][:MAX_SHAPES_PER_FACE])
    if 'sports' in kset and 'shield' not in ok_shapes:
        if label in ('Oval','Round','Oblong'):   # 필요시 추가 확장 가능
            ok_shapes.append('shield')
            ok_shapes = ok_shapes[:2]
    pool = cand[cand["shape"].isin(set(ok_shapes))]
    if len(pool) > 0:
        cand = pool

if len(cand) == 0:
    st.error("조건(성별/분류/얼굴형)에 맞는 프레임을 찾지 못했습니다.")
    st.stop()

# 5) 후보 중 랜덤 1개 선택
row = cand.sample(1, random_state=random.randint(0, 10_000)).iloc[0].to_dict()

# 6) 이미지 경로 결정
def _resolve_image(row: dict):
    p = (row.get("image_path") or "").strip()
    if p and os.path.exists(p):
        return p
    pid = str(row.get("product_id", "")).strip()
    base = os.path.join("frames","images", pid)
    for ext in (".png",".webp",".avif",".jpg",".jpeg"):
        cp = base + ext
        if os.path.exists(cp):
            return cp
    return None

img_path = _resolve_image(row)
if not img_path:
    st.error(f"이미지 파일을 찾을 수 없습니다: frames/images/{row.get('product_id','?')}.[png|webp|avif|jpg]")
    st.stop()

# 7) 프레임 이미지 로드 (BGRA 보장)
def _ensure_bgra_fallback(p: str):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

try:
    fg_bgra = vision.ensure_bgra(img_path)
    if fg_bgra is None:
        raise RuntimeError("vision.ensure_bgra returned None")
except Exception:
    fg_bgra = _ensure_bgra_fallback(img_path)

if fg_bgra is None:
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}")
    st.stop()

# 8) 치수 세팅
A, DBL, TOTAL = float(row["lens_mm"]), float(row["bridge_mm"]), float(row["total_mm"])
if any(pd.isna([A, DBL, TOTAL])):
    st.error("선택된 프레임의 치수(lens_mm/bridge_mm/total_mm)가 비어 있습니다.")
    st.stop()

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

# 8-1) PD 소스 결정 (실패해도 중단하지 않음)
if (PD_MM is not None) and (PD_MM > 0):
    PD_SRC = "manual"
elif use_phone and (PD_MM_raw is not None):
    PD_SRC = "iphone"; PD_MM = PD_MM_raw
else:
    try:
        pd_px, eye_roll_deg, mid = vision.detect_pd_px(face_bgr)
        PD_SRC = "mediapipe" if pd_px is not None else None
        if PD_SRC is None:
            st.warning("눈 검출 실패 → PD 없이 진행합니다.")
    except Exception as e:
        PD_SRC = None
        st.warning(f"MediaPipe PD 계산 실패({e}) → PD 없이 진행합니다.")

# 8-2) 머리자세(없어도 진행)
yaw = pitch = roll = None
if hasattr(vision, "head_pose_ypr"):
    try:
        yaw, pitch, roll = vision.head_pose_ypr(face_bgr)  # ° 단위
    except Exception:
        yaw = pitch = roll = None
if roll is None:
    roll = eye_roll_deg

# 8-3) 디버그 표기
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
    st.caption("PD 미사용: 프레임 총폭과 얼굴 폭으로 스케일 맞춥니다.")

# 8-4) 프레임 전처리
fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)

h_face, w_face = face_bgr.shape[:2]
h0, w0 = fg_bgra.shape[:2]

# 8-5) 목표 스케일 계산 (PD 있으면 GCD 기반, 없으면 TOTAL↔CHEEK_MM 기반)
GCD2PD_CAL = 0.92
target_GCD_px = None
if pd_px is not None:
    target_GCD_px = pd_px
elif PD_MM:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_GCD_px = PD_MM / max(mm_per_px, 1e-6)

Cw_px = vision.cheek_width_px(face_bgr)  # None일 수 있음

if target_GCD_px is not None:
    target_GCD_px *= GCD2PD_CAL
    target_total_px = target_GCD_px * k  # k = TOTAL/GCD
else:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

# 안전 범위 클램프
min_w = 0.60 * w_face
max_w = 0.95 * w_face
if Cw_px is not None:
    min_w = max(min_w, 0.70 * Cw_px)
    max_w = min(max_w, 0.98 * Cw_px)
target_total_px = float(np.clip(target_total_px, min_w, max_w))

# 8-6) 리사이즈/회전
scale = (target_total_px / max(w0, 1)) * float(scale_mult)
scale = float(np.clip(scale, 0.35, 2.2))

new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

# 8-7) 위치 앵커
pitch_deg = pitch if pitch is not None else 0.0
pitch_dy  = int(pitch_deg * 0.8)

if mid == (0, 0):
    gx = int(face_bgr.shape[1] * 0.5 - fg_rot.shape[1] * 0.5) + dx
    gy = int(face_bgr.shape[0] * 0.45 - fg_rot.shape[0] * 0.5) + dy
else:
    anchor = 0.50
    gx = int(mid[0] - fg_rot.shape[1] * anchor) + dx
    gy = int(mid[1] - fg_rot.shape[0] * 0.50) + dy + pitch_dy

# 8-8) 합성
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

# 8-9) 다운로드
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    PIL.Image.fromarray(rgb).save(buf, format="PNG")
    b = str(row.get("brand", "frame")).strip().replace(" ", "_")
    pid = str(row.get("product_id", "")).strip()
    fname = f"{b}_{pid}.png" if pid else f"{b}.png"
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(), file_name=fname, mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

# =============================
# 9) (선택) 얼굴형 텍스트 추천
# =============================
if final_label:
    rec = None
    if final_label == "Oval":
        rec = "대부분의 프레임 OK (rectangular/trapezoid/aviator)"
    elif final_label == "Round":
        rec = "각진 프레임 추천 (rectangular/wayfarer 계열)"
    elif final_label == "Square":
        rec = "곡선형 프레임 추천 (round/aviator)"
    elif final_label == "Oblong":
        rec = "가로 비중 높은 타입 (rectangular/trapezoid)"
    elif final_label == "Heart":
        rec = "상단 가벼운 실루엣 (cat-eye/round)"
    if rec:
        st.info(f"👓 얼굴형({final_label}) 추천: {rec}")

# (옵션) 다시 시작 버튼
with st.sidebar:
    if st.button("🔄 다시 시작"):
        for k in ("started","face_bytes"):
            st.session_state.pop(k, None)
        st.rerun()

