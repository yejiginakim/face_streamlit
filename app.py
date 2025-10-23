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
    import vision  # vision.py: detect_pd_px / head_pose_ypr / overlay_rgba / ...
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

try:
    from metrics import compute_metrics_bgr
except Exception as e:
    err_msgs.append(f"metrics 임포트 실패: {e}")

# =============================
# 3) 유틸
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
# 4) UI & 세션 상태
# =============================
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 잠금 후 슬라이더만")

# 세션 키 초기화 (무거운 단계는 '잠금' 때만 실행)
for k, v in {
    "locked": False,
    "faceshape_label": None,
    "row": None,                 # 추천된 프레임 row(dict)
    "face_bgr": None,            # 원본 얼굴 이미지 (BGR)
    "fg_bgra": None,             # 전처리된 프레임 PNG (BGRA)
    "mid": (0, 0),
    "roll": 0.0,
    "pitch": 0.0,
    "k_ratio": 2.0,
    "TOTAL_mm": None,
    "CHEEK_MM": 150.0,
    "PD_MM_raw": None,
}.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.subheader("📱 iPhone/URL 측정값 (잠금 시 1회만 반영)")

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

    use_phone_default = _qbool("use_phone", default=False)
    use_phone = st.checkbox("iPhone/URL 측정값 사용", value=use_phone_default, key="use_phone_ck")

    PD_MM_raw       = _qfloat("pd_mm") or _qfloat("pd")
    CHEEK_MM_raw    = _qfloat("cheek_mm") or _qfloat("cheek")
    NOSECHIN_MM_raw = _qfloat("nosechin_mm") or _qfloat("nosechin")

    DEFAULT_CHEEK_MM = st.session_state.CHEEK_MM or 150.0

    if use_phone and (CHEEK_MM_raw is not None):
        CHEEK_MM = CHEEK_MM_raw
    else:
        CHEEK_MM = st.number_input("얼굴 폭(mm)", value=float(DEFAULT_CHEEK_MM), step=0.5)

    if use_phone and (PD_MM_raw is not None):
        PD_MM = PD_MM_raw
    else:
        pd_in = st.number_input("PD(mm) (옵션)", value=0.0, step=0.1, format="%.1f")
        PD_MM = pd_in if pd_in > 0 else None

    if CHEEK_MM is not None:
        CHEEK_MM = float(min(max(CHEEK_MM, 100.0), 220.0))
    if PD_MM is not None:
        PD_MM = float(min(max(PD_MM, 45.0), 75.0))

    st.caption("※ 위 값들은 '잠금'을 누를 때 딱 1회만 반영됩니다.")

    st.divider()
    st.subheader("🎚️ 스케일/오프셋 (합성은 고정, 오버레이만 갱신)")
    # 슬라이더는 항상 즉시 적용 — 하지만 오버레이만 다시 그린다
    dx = st.slider("수평 오프셋(px)", -400, 400, 0, key="dx")
    dy = st.slider("수직 오프셋(px)", -400, 400, 0, key="dy")
    scale_mult = st.slider("스케일(배)", 0.5, 2.0, 1.0, key="scale_mult")

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드 (잠금 시 고정)")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"], key="face_file")
with colR:
    st.markdown("### 2) 카테고리 선택 (잠금 시 고정)")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요', key="gender_ms")
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요', key="kind_ms")

# 제어 버튼: 잠금/해제
lock = st.button('🔒 잠금(한 번만 무거운 계산)')
unlock = st.button('🔓 잠금 해제(다시 준비)')
if unlock:
    for k in ["locked","row","face_bgr","fg_bgra","mid","roll","pitch","k_ratio","TOTAL_mm","faceshape_label"]:
        st.session_state[k] = None if k not in ("locked",) else False
    st.rerun()

# =============================
# 잠금 시 1회만: 무거운 단계 실행 → 세션에 저장
# =============================
if lock:
    # 0) 필수 체크
    if not (img_file and use_gender and use_kind):
        st.error("사진 업로드, 성별/분류 선택이 필요합니다.")
        st.stop()

    # 1) 얼굴 이미지 고정
    try:
        file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
        face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if face_bgr is None:
            raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
        st.session_state.face_bgr = face_bgr
    except Exception as e:
        st.error(f"얼굴 이미지 로드 실패: {e}")
        st.stop()

    # 2) 얼굴형 추론 (있으면)
    MODEL_PATH   = "models/faceshape_efficientnetB4_best_20251018_223855.keras"
    CLASSES_PATH = "models/classes_20251018_223855.txt"
    IMG_SIZE     = (224, 224)

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
    if os.path.isfile(MODEL_PATH) and os.path.isfile(CLASSES_PATH) and not _is_lfs_pointer(MODEL_PATH):
        try:
            @st.cache_resource
            def _load_faceshape():
                return FaceShapeModel(MODEL_PATH, CLASSES_PATH, img_size=IMG_SIZE)
            faceshape_model = _load_faceshape()
        except Exception:
            faceshape_model = None

    final_label = None
    if faceshape_model is not None:
        try:
            pil_img = PIL.Image.fromarray(cv2.cvtColor(st.session_state.face_bgr, cv2.COLOR_BGR2RGB))
            probs = faceshape_model.predict_probs(pil_img)
            try:
                ar, jaw, cw, jw = compute_metrics_bgr(st.session_state.face_bgr)
            except Exception:
                ar = jaw = cw = jw = None
            if any(v is not None for v in (ar, jaw, cw, jw)):
                adj = apply_rules(probs, faceshape_model.class_names, ar=ar, jaw_deg=jaw, cw=cw, jw=jw)
                final_label = adj['rule_label']
            else:
                _, final_label, _ = decide_rule_vs_top2(probs, faceshape_model.class_names)
        except Exception:
            final_label = None
    st.session_state.faceshape_label = final_label

    # 3) 카탈로그 로드 & 프레임 1회 선택
    import pandas as pd
    EXCEL_PATH = "sg_df.xlsx"
    SHAPES6 = {"round","rectangular","trapezoid","aviator","cat-eye","shield"}
    def _norm(x):
        return (x or "").strip().lower()

    def normalize_shape(s: str) -> str:
        if not isinstance(s, str):
            return ""
        t = s.strip().lower().replace("_","-")
        t = " ".join(t.split()).replace(" ", "-")
        syn = {"cateye":"cat-eye","rectangle":"rectangular","rect":"rectangular","wayfarer":"trapezoid","wrap":"shield","circle":"round","pilot":"aviator"}
        return syn.get(t, t)

    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        st.error(f"엑셀 카탈로그 로드 실패: {e}")
        st.stop()

    need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm","image_path"]
    for c in need_cols:
        if c not in df.columns:
            st.error(f"엑셀에 '{c}' 컬럼이 없습니다.")
            st.stop()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["shape"]   = df["shape"].astype(str).map(normalize_shape)
    df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()
    df["sex"]     = df["sex"].astype(str).str.strip().str.lower()
    for c in ["lens_mm","bridge_mm","total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df.loc[~df["shape"].isin(SHAPES6), ["product_id","brand","shape"]]
    if len(bad) > 0:
        st.error("shape 값은 round/rectangular/trapezoid/aviator/cat-eye/shield 만 허용됩니다.")
        st.dataframe(bad.head(50))
        st.stop()

    gset = {_norm(g) for g in use_gender}
    kset = {_norm(k) for k in use_kind}
    f = pd.Series([True] * len(df))
    if gset:
        f &= df["sex"].isin(gset) | (df["sex"] == "unisex")
    if kset:
        f &= df["purpose"].isin(kset)

    cand = df[f].copy()

    FRAME_RULES_ORDERED = {
        "Oval":   ["trapezoid","rectangular"],
        "Round":  ["rectangular"],
        "Square": ["round"],
        "Oblong": ["rectangular","trapezoid"],
        "Heart":  ["cat-eye","round"],
    }
    MAX_SHAPES_PER_FACE = 1

    label = st.session_state.faceshape_label
    if label in FRAME_RULES_ORDERED:
        ok_shapes = [s for s in FRAME_RULES_ORDERED[label][:MAX_SHAPES_PER_FACE]]
        if 'sports' in kset and 'shield' not in ok_shapes and label in ('Oval','Round','Oblong'):
            ok_shapes.append('shield')
            ok_shapes = ok_shapes[:2]
        pool = cand[cand["shape"].isin(set(ok_shapes))]
        if len(pool) > 0:
            cand = pool

    if len(cand) == 0:
        st.error("조건(성별/분류/얼굴형)에 맞는 프레임을 찾지 못했습니다.")
        st.stop()

    row = cand.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0].to_dict()
    st.session_state.row = row

    # 4) 프레임 이미지 로드 & 전처리 (한 번만)
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

    fg_bgra = vision.ensure_bgra(img_path)
    if fg_bgra is None:
        st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}")
        st.stop()

    # 전처리 (한 번만)
    fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
    fg_bgra = vision.trim_transparent(fg_bgra, pad=8)
    st.session_state.fg_bgra = fg_bgra

    # 치수/비율 (한 번만)
    A     = float(row["lens_mm"])      # 렌즈 가로(mm)
    DBL   = float(row["bridge_mm"])    # 브리지(mm)
    TOTAL = float(row["total_mm"])     # 전체 가로(mm)
    GCD = A + DBL
    k = (TOTAL / GCD) if GCD else 2.0
    st.session_state.k_ratio = float(k)
    st.session_state.TOTAL_mm = float(TOTAL)
    st.session_state.CHEEK_MM = float(CHEEK_MM)
    st.session_state.PD_MM_raw = float(PD_MM) if PD_MM is not None else None

    # PD/자세 1회 계산
    pd_px = None
    mid = (0, 0)
    eye_roll_deg = 0.0
    try:
        pd_px, eye_roll_deg, mid = vision.detect_pd_px(st.session_state.face_bgr)
    except Exception:
        pass

    yaw = pitch = roll = None
    if hasattr(vision, "head_pose_ypr"):
        try:
            yaw, pitch, roll = vision.head_pose_ypr(st.session_state.face_bgr)
        except Exception:
            yaw = pitch = roll = None
    if roll is None:
        roll = eye_roll_deg

    st.session_state.mid = mid
    st.session_state.roll = float(roll or 0.0)
    st.session_state.pitch = float(pitch or 0.0)

    st.session_state.locked = True
    st.success("🔒 잠금 완료 — 이제 슬라이더만 움직여도 합성은 고정, 선글라스만 이동/스케일 조정됩니다.")

# =============================
# 잠금 이후: 오버레이만 갱신 (가벼운 경로)
# =============================
if not st.session_state.locked:
    st.info("사진/카테고리 선택 후 **잠금**을 누르세요. 이후엔 슬라이더만으로 오버레이를 바꿉니다.")
    st.stop()

# 가벼운 합성 경로(슬라이더 변경 때마다 실행되지만, 무거운 단계는 건드리지 않음)
face_bgr = st.session_state.face_bgr
fg_bgra  = st.session_state.fg_bgra
mid      = st.session_state.mid or (0, 0)
roll     = float(st.session_state.roll or 0.0)
pitch    = float(st.session_state.pitch or 0.0)
CHEEK_MM = float(st.session_state.CHEEK_MM or 150.0)
PD_MM    = st.session_state.PD_MM_raw
k        = float(st.session_state.k_ratio or 2.0)
TOTAL    = float(st.session_state.TOTAL_mm or 140.0)

h_face, w_face = face_bgr.shape[:2]
h0, w0 = fg_bgra.shape[:2]

# 목표 폭 계산(잠금 시 저장된 치수만 사용)
if PD_MM is not None:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = (PD_MM / (0.92)) * k / max(mm_per_px, 1e-6)
else:
    mm_per_px = CHEEK_MM / max(w_face, 1e-6)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

min_w = 0.60 * w_face
max_w = 0.95 * w_face
from math import isfinite
if not isfinite(target_total_px):
    target_total_px = 0.8 * w_face

target_total_px = float(np.clip(target_total_px, min_w, max_w))

# 사용자가 주는 scale_mult만 반영
scale = (target_total_px / max(w0, 1)) * float(st.session_state.scale_mult)
scale = float(np.clip(scale, 0.35, 2.2))

# 리사이즈/회전만 수행 (빠름)
new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)
M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

pitch_dy  = int((pitch or 0.0) * 0.8)
if mid == (0, 0):
    gx = int(face_bgr.shape[1] * 0.5 - fg_rot.shape[1] * 0.5) + st.session_state.dx
    gy = int(face_bgr.shape[0] * 0.45 - fg_rot.shape[0] * 0.5) + st.session_state.dy
else:
    anchor = 0.50
    gx = int(mid[0] - fg_rot.shape[1] * anchor) + st.session_state.dx
    gy = int(mid[1] - fg_rot.shape[0] * 0.50) + st.session_state.dy + pitch_dy

margin_x, margin_y = 300, 150
bg_expanded = cv2.copyMakeBorder(
    face_bgr, margin_y, margin_y, margin_x, margin_x,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)

gx_expanded = gx + margin_x
gy_expanded = gy + margin_y

out = vision.overlay_rgba(bg_expanded, fg_rot, gx_expanded, gy_expanded)
show_image_bgr(out, caption="합성 결과 — 프레임/탐지 고정, 슬라이더만 반영")

try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    PIL.Image.fromarray(rgb).save(buf, format="PNG")
    file_name = (st.session_state.row or {}).get('product_id', 'frame')
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(), file_name=f"{file_name}_result.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

