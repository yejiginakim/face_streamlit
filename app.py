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
    import vision  # vision.py: detect_pd_px / head_pose_ypr / overlay_rgba / ...
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
# 4) UI & 상태
# =============================
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 라이브 스케일")

# 세션 상태
st.session_state.setdefault("started", False)      # 시작 여부 (한 번만 누르면 계속 유지)
st.session_state.setdefault("chosen_row", None)    # 추천된 프레임(row dict) 고정
st.session_state.setdefault("faceshape_label", None)

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
    st.subheader("미세 조정 (변경 시 즉시 적용)")
    dx = st.slider("수평 오프셋(px)", -200, 200, st.session_state.get("dx", 0), key="dx")
    dy = st.slider("수직 오프셋(px)", -200, 200, st.session_state.get("dy", 0), key="dy")
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, st.session_state.get("scale_mult", 1.0), key="scale_mult")

    st.caption("iPhone/URL 측정값 사용: " + ("ON" if use_phone else "OFF"))

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"], key="face_file")

with colR:
    st.markdown("### 카테고리 선택 ")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요', key="gender_ms")
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요', key="kind_ms")

is_ready_to_start = bool(use_gender and use_kind and img_file)

start_btn = st.button('시작/추천 고정', disabled=not is_ready_to_start)
reset_btn = st.button('다시 추천')

if start_btn:
    st.session_state.started = True
    st.success(f"시작! 성별={use_gender}, 분류={use_kind} — 이후 슬라이더는 즉시 반영됩니다.")

if reset_btn:
    st.session_state.chosen_row = None
    st.toast("프레임을 새로 추천합니다.")

if err_msgs:
    st.info("위 임포트 문제를 해결해야 합성이 진행됩니다. (requirements.txt / OpenCV headless / vision.py / faceshape.py / metrics.py 확인)")
    st.stop()

st.divider()

# 시작 조건: 버튼을 한번 누르거나(고정) / 혹은 업로드+선택이 모두 준비되면 자동 진행
if not (st.session_state.started or is_ready_to_start):
    st.info("얼굴 사진 업로드 · 성별/분류 선택 후 **시작/추천 고정**을 누르세요. 이후 슬라이더는 즉시 적용됩니다.")
    st.stop()

# =============================
# 6) 얼굴 이미지 업로드 처리
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

final_label = st.session_state.get("faceshape_label")
if faceshape_model is not None:
    try:
        pil_img = PIL.Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        probs = faceshape_model.predict_probs(pil_img)
        top2_raw = topk_from_probs(probs, faceshape_model.class_names)
        labels_raw = top2_strings(top2_raw)
        st.subheader("모델 Top-2 (원본)")
        st.write(" / ".join(labels_raw))

        try:
            ar, jaw, cw, jw = compute_metrics_bgr(face_bgr)
        except Exception:
            ar = jaw = cw = jw = None

        if any(v is not None for v in (ar, jaw, cw, jw)):
            adj = apply_rules(
                probs, faceshape_model.class_names,
                ar=ar, jaw_deg=jaw, cw=cw, jw=jw
            )
            probs_adj = adj['rule_probs']
            top2_adj  = topk_from_probs(probs_adj, faceshape_model.class_names)
            labels_adj = top2_strings(top2_adj)
            st.subheader("모델 Top-2 (규칙 보정 후)")
            st.write(" / ".join(labels_adj))
            final_label = adj['rule_label']
            st.session_state.faceshape_label = final_label
            reason = "rules+model"
        else:
            _, final_label, reason = decide_rule_vs_top2(probs, faceshape_model.class_names)
            st.session_state.faceshape_label = final_label
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

# =============================
# 5) 프레임 로드 (엑셀 카탈로그 · 최소 규칙 + sports시 shield)
# =============================
import pandas as pd

EXCEL_PATH = "sg_df.xlsx"  # 카탈로그 경로 (앱 폴더 권장)

# 6개 모양 고정 (소문자 통일)
SHAPES6 = {"round","rectangular","trapezoid","aviator","cat-eye","shield"}

# 얼굴형 → 최소 추천 모양(우선순위)
FRAME_RULES_ORDERED = {
    "Oval":   ["trapezoid","rectangular"],
    "Round":  ["rectangular"],
    "Square": ["round"],
    "Oblong": ["rectangular","trapezoid"],
    "Heart":  ["cat-eye","round"],
}
MAX_SHAPES_PER_FACE = 1

def _norm(x):
    return (x or "").strip().lower()

def normalize_shape(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower().replace("_","-")
    t = " ".join(t.split()).replace(" ", "-")
    syn = {
        "cateye":"cat-eye", "cat":"cat-eye",
        "rectangle":"rectangular", "rect":"rectangular",
        "wayfarer":"trapezoid", "trap":"trapezoid",
        "wrap":"shield", "wraparound":"shield",
        "circle":"round", "pilot":"aviator",
    }
    return syn.get(t, t)

try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    st.error(f"엑셀 카탈로그 로드 실패: {e}")
    st.stop()

need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm","image_path"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"엑셀에 누락된 컬럼: {missing}")
    st.stop()

# 전처리
df["product_id"] = df["product_id"].astype(str).str.strip()
df["shape"]   = df["shape"].astype(str).map(normalize_shape)
df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()
df["sex"]     = df["sex"].astype(str).str.strip().str.lower()
for c in ["lens_mm","bridge_mm","total_mm"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

bad = df.loc[~df["shape"].isin(SHAPES6), ["product_id","brand","shape"]]
if len(bad) > 0:
    st.error("shape 값은 round/rectangular/trapezoid/aviator/cat-eye/shield 만 허용됩니다. (아래 미일치 항목 확인)")
    st.dataframe(bad.head(50))
    st.stop()

# 필터
gset = {_norm(g) for g in use_gender}
kset = {_norm(k) for k in use_kind}

f = pd.Series([True] * len(df))
if gset:
    f &= df["sex"].isin(gset) | (df["sex"] == "unisex")
if kset:
    f &= df["purpose"].isin(kset)

cand = df[f].copy()

# 얼굴형 규칙
label = st.session_state.get("faceshape_label", None)
if label in FRAME_RULES_ORDERED:
    ok_shapes = [s.lower() for s in FRAME_RULES_ORDERED[label][:MAX_SHAPES_PER_FACE]]
    if 'sports' in kset and 'shield' not in ok_shapes and label in ('Oval','Round','Oblong'):
        ok_shapes.append('shield')
        ok_shapes = ok_shapes[:2]
    pool = cand[cand["shape"].isin(set(ok_shapes))]
    if len(pool) > 0:
        cand = pool

if len(cand) == 0:
    st.error("조건(성별/분류/얼굴형)에 맞는 프레임을 찾지 못했습니다.")
    st.stop()

# 프레임 고정(세션 유지) — 다시 추천 버튼 누르기 전까지 유지
if st.session_state.chosen_row is None:
    row = cand.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0].to_dict()
    st.session_state.chosen_row = row
else:
    row = st.session_state.chosen_row

# =============================
# 6) 이미지 경로 해결 (images 폴더 없이)
# =============================
import glob as _glob

FRAME_ROOT = "frame"
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
    p = (row.get("image_path") or "").strip()
    if p and os.path.exists(p):
        return p
    pid = str(row.get("product_id", "")).strip()
    if not pid:
        return None
    shape_val = str(row.get("shape", "")).strip().lower()
    shape_dir = SHAPE_DIR_MAP.get(shape_val)
    if shape_dir:
        base = os.path.join(FRAME_ROOT, shape_dir, pid)
        for ext in EXTS:
            cp = base + ext
            if os.path.exists(cp):
                return cp
    pattern = os.path.join(FRAME_ROOT, "**", pid + ".*")
    for cp in _glob.glob(pattern, recursive=True):
        if os.path.splitext(cp)[1].lower() in EXTS and os.path.isfile(cp):
            return cp
    return None

img_path = _resolve_image(row)
if not img_path:
    st.error(
        f"이미지를 찾지 못했습니다: frame/<Aviator|Cat_eye|Rectangular|Round|Shield|Trapezoid>/{row['product_id']}.[png|webp|avif|jpg|jpeg]"
    )
    st.stop()

# =============================
# 7) 프레임 이미지 로드 및 치수 세팅
# =============================
fg_bgra = vision.ensure_bgra(img_path)
if fg_bgra is None:
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}")
    st.stop()

A     = float(row["lens_mm"])      # 렌즈 가로(mm)
DBL   = float(row["bridge_mm"])    # 브리지(mm)
TOTAL = float(row["total_mm"])     # 전체 가로(mm)
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0

st.caption(
    f"선택 프레임: {row.get('brand','')} / {row.get('product_id','')}  · "
    f"shape={row.get('shape','?')} · A={A}, DBL={DBL}, TOTAL={TOTAL} "
    f"(GCD={GCD}, k=TOTAL/GCD={k:.3f})"
)

# =============================
# 8) PD/자세/스케일/합성 — 슬라이더
