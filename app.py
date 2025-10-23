# =============================
# 0) 백엔드/로그 환경변수 먼저 고정
# =============================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # Keras 3 -> TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"              # TF 로그 억제
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU 강제

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
import sys, platform, glob, hashlib
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

def file_md5(b: bytes)->str:
    return hashlib.md5(b).hexdigest()

# =============================
# 4) UI & 세션 상태
# =============================
st.title("🧍→🕶️ Antena_01 합성 — 실시간 추천/선택/슬라이더")

# 세션 기본값(무거운 건 이미지/필터 바뀔 때만 갱신)
defaults = {
    "img_key": None,            # 업로드 이미지 해시
    "face_bgr": None,
    "faceshape_label": None,
    "mid": (0, 0),
    "roll": 0.0,
    "pitch": 0.0,
    "CHEEK_MM": 150.0,
    "PD_MM_raw": None,
    "recs": [],                 # 추천 4개 dict 리스트
    "selected_pid": None,       # 현재 선택된 product_id
    "fg_bgra": None,            # 현재 선택 프레임 이미지(BGRA)
    "k_ratio": 2.0,
    "TOTAL_mm": None,
    # 슬라이더
    "dx": 0, "dy": 0, "scale_mult": 1.0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.subheader("📱 iPhone/URL 측정값")
    def _qget(name):
        v = st.query_params.get(name)
        if isinstance(v, list): v = v[0]
        return v
    def _qfloat(name):
        v = _qget(name)
        try: return float(v) if v not in (None, "", "None") else None
        except: return None
    def _qbool(name, default=False):
        v = _qget(name)
        if v is None: return default
        return str(v).lower() in ("1", "true", "yes", "on")

    use_phone_default = _qbool("use_phone", default=False)
    use_phone = st.checkbox("iPhone/URL 측정값 사용", value=use_phone_default, key="use_phone_ck")

    PD_MM_raw_q       = _qfloat("pd_mm") or _qfloat("pd")
    CHEEK_MM_raw_q    = _qfloat("cheek_mm") or _qfloat("cheek")
    NOSECHIN_MM_raw_q = _qfloat("nosechin_mm") or _qfloat("nosechin")

    DEFAULT_CHEEK_MM = st.session_state.CHEEK_MM or 150.0
    if use_phone and (CHEEK_MM_raw_q is not None):
        CHEEK_MM = CHEEK_MM_raw_q
    else:
        CHEEK_MM = st.number_input("얼굴 폭(mm)", value=float(DEFAULT_CHEEK_MM), step=0.5)

    if use_phone and (PD_MM_raw_q is not None):
        PD_MM = PD_MM_raw_q
    else:
        pd_in = st.number_input("PD(mm) (옵션)", value=0.0, step=0.1, format="%.1f")
        PD_MM = pd_in if pd_in > 0 else None

    if CHEEK_MM is not None:
        CHEEK_MM = float(min(max(CHEEK_MM, 100.0), 220.0))
    if PD_MM is not None:
        PD_MM = float(min(max(PD_MM, 45.0), 75.0))

    # 세션 저장(항상 최신 보유)
    st.session_state.CHEEK_MM = float(CHEEK_MM)
    st.session_state.PD_MM_raw = float(PD_MM) if PD_MM is not None else None

    st.divider()
    st.subheader("🎚️ 스케일/오프셋 (오버레이만 갱신)")
    st.session_state.dx = st.slider("수평 오프셋(px)", -400, 400, st.session_state.dx, key="dx_sl")
    st.session_state.dy = st.slider("수직 오프셋(px)", -400, 400, st.session_state.dy, key="dy_sl")
    st.session_state.scale_mult = st.slider("스케일(배)", 0.5, 2.0, st.session_state.scale_mult, key="scale_sl")

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"], key="face_file")
with colR:
    st.markdown("### 2) 카테고리 선택")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요', key="gender_ms")
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요', key="kind_ms")

if err_msgs:
    st.warning("초기 임포트 경고가 있습니다. 아래 로그를 확인하세요.")
    st.code("\n".join(err_msgs), language="text")

# =============================
# 추천/정규화 규칙
# =============================
def normalize_shape(s: str) -> str:
    """엑셀 shape을 6종으로 매핑"""
    if not isinstance(s, str): return ""
    t = s.strip().lower().replace("_","-")
    t = " ".join(t.split()).replace(" ", "-")
    syn = {
        "round":"round", "circle":"round", "boston":"round", "panto":"round", "oval":"round",
        "rect":"rectangular", "rectangle":"rectangular", "rectangular":"rectangular", "square":"rectangular",
        "flat-top":"rectangular", "clubmaster":"rectangular",
        "trapezoid":"trapezoid", "wayfarer":"trapezoid", "browline":"trapezoid", "geometric":"trapezoid",
        "aviator":"aviator", "pilot":"aviator", "teardrop":"aviator",
        "cat-eye":"cat-eye", "cateye":"cat-eye", "cats-eye":"cat-eye", "butterfly":"cat-eye",
        "shield":"shield", "wrap":"shield", "wraparound":"shield", "mask":"shield", "visor":"shield",
    }
    return syn.get(t, t)

# 얼굴형 → 기본 추천 모양(2개)
BASE_SHAPES_BY_FACE = {
    "Oval":   ["trapezoid", "rectangular"],
    "Round":  ["rectangular", "trapezoid"],
    "Square": ["round", "trapezoid"],
    "Oblong": ["rectangular", "trapezoid"],
    "Heart":  ["cat-eye", "round"],
}
def get_shape_targets(face_label: str | None, kinds: set[str]) -> list[str]:
    base = BASE_SHAPES_BY_FACE.get(face_label, ["rectangular", "trapezoid"])
    base = list(base)
    # 스포츠 선택 시 shield 우선 포함
    if "sports" in kinds:
        if face_label in ("Oval","Round","Oblong","Square"):
            if "shield" in base: base.remove("shield")
            base.insert(0, "shield")
        else:
            if "shield" not in base: base.append("shield")
    # 유니크 2개로 제한
    uniq = []
    for s in base:
        if s not in uniq:
            uniq.append(s)
    return uniq[:2]

def pick_from_pool(pool_df, n, seed=None):
    if len(pool_df) <= 0: return []
    take = min(n, len(pool_df))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(pool_df.index.to_numpy(), size=take, replace=False)
    return pool_df.loc[idxs].to_dict(orient="records")

# =============================
# 5) 이미지/필터 변화를 감지해 무거운 단계 ‘필요 시’만 재계산
# =============================
def need_refresh(img_bytes: bytes, genders: list, kinds: list)->bool:
    new_key = None if img_bytes is None else file_md5(img_bytes)
    changed = False
    if st.session_state.img_key != new_key:
        changed = True
    # 필터(성별/분류) 바뀌면 추천만 다시
    prev_filters = (tuple(st.session_state.get("_genders", [])), tuple(st.session_state.get("_kinds", [])))
    new_filters  = (tuple(genders or []), tuple(kinds or []))
    if prev_filters != new_filters:
        changed = True
    st.session_state.img_key = new_key
    st.session_state._genders, st.session_state._kinds = new_filters
    return changed

# 업로드 & 필터 체크
if not img_file:
    st.info("정면 얼굴 사진을 업로드하고, 성별/분류를 선택하세요.")
    st.stop()

if not (use_gender and use_kind):
    st.warning("성별/분류에서 각각 최소 1개 이상 선택하세요.")
    st.stop()

# 이미지 바이트/해시
img_bytes = img_file.getvalue()
refresh = need_refresh(img_bytes, use_gender, use_kind)

# 무거운 단계(얼굴형/PD/추천) — 필요한 경우에만
if refresh:
    # 1) 얼굴 이미지 로드
    try:
        file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
        face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if face_bgr is None:
            raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
        st.session_state.face_bgr = face_bgr
    except Exception as e:
        st.error(f"얼굴 이미지 로드 실패: {e}")
        st.stop()

    # 2) 얼굴형 추론(있으면)
    MODEL_PATH   = "models/faceshape_efficientnetB4_best_20251018_223855.keras"
    CLASSES_PATH = "models/classes_20251018_223855.txt"
    IMG_SIZE     = (224, 224)
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

    # 3) 카탈로그 로드 & 추천 4개 구성
    import pandas as pd
    EXCEL_PATH = "sg_df.xlsx"
    SHAPES6 = {"round","rectangular","trapezoid","aviator","cat-eye","shield"}
    def _norm(x): return (x or "").strip().lower()

    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        st.error(f"엑셀 카탈로그 로드 실패: {e}")
        st.stop()

    # 필수 컬럼(이미지 경로는 optional)
    need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm"]
    for c in need_cols:
        if c not in df.columns:
            st.error(f"엑셀에 '{c}' 컬럼이 없습니다.")
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

    # 얼굴형 -> 타깃 모양 2개
    target_shapes = get_shape_targets(st.session_state.faceshape_label, kset)

    # 사용자 안내: 얼굴형과 추천 모양
    face_tag = st.session_state.faceshape_label or "모델 미탑재/판별불가"
    st.info(f"🧠 얼굴형 결과: **{face_tag}** → 추천 모양: **{', '.join(target_shapes)}**")

    # 각 모양당 2개 선별 → 최대 4개
    recs = []
    seed = int(np.random.randint(0, 1_000_000))
    for i, shp in enumerate(target_shapes):
        pool = cand[cand["shape"] == shp]
        if "sports" in kset:
            pool_s = pool[pool["purpose"] == "sports"]
            chosen = pick_from_pool(pool_s, 2, seed + i*10)
            if len(chosen) < 2:
                pool_f = pool[pool["purpose"] == "fashion"]
                chosen += pick_from_pool(
                    pool_f[~pool_f["product_id"].isin([c["product_id"] for c in chosen])],
                    2 - len(chosen),
                    seed + i*10 + 1
                )
        else:
            pool_f = pool[pool["purpose"] == "fashion"]
            chosen = pick_from_pool(pool_f, 2, seed + i*10)
            if len(chosen) < 2:
                pool_s = pool[pool["purpose"] == "sports"]
                chosen += pick_from_pool(
                    pool_s[~pool_s["product_id"].isin([c["product_id"] for c in chosen])],
                    2 - len(chosen),
                    seed + i*10 + 1
                )
        recs.extend(chosen)

    if len(recs) < 4:
        already = set([r["product_id"] for r in recs])
        remain_pool = cand[~cand["product_id"].isin(already)]
        recs += pick_from_pool(remain_pool, 4 - len(recs), seed + 999)

    recs = recs[:4]
    if len(recs) == 0:
        st.error("추천할 프레임을 찾지 못했습니다. 필터를 완화해 보세요.")
        st.stop()

    st.session_state.recs = recs
    # 이전 선택 무효화
    st.session_state.selected_pid = None
    st.session_state.fg_bgra = None

# =============================
# 6) 추천 4개 단일 선택 → 즉시 합성
# =============================
recs = st.session_state.recs or []
pretty_items = []
for i, r in enumerate(recs):
    label_face = st.session_state.faceshape_label or "Unknown"
    pretty_items.append(
        f"{i+1}) [{r.get('purpose','?')}] {r.get('brand','?')} / {r.get('product_id','?')} · {r.get('shape','?')} · {int(r.get('total_mm',0))}mm · Face:{label_face}"
    )

st.markdown("### 😀 얼굴형/카테고리에 맞춰 추천 4개입니다. 하나를 선택하면 바로 합성합니다.")
selected_label = st.selectbox("추천 중 1개 선택", options=pretty_items, index=0 if pretty_items else None)

# 선택된 row
if not recs:
    st.stop()
sel_idx = pretty_items.index(selected_label)
row = recs[sel_idx]
st.session_state.selected_pid = row.get("product_id")

# 프레임 이미지 경로 해석( image_path 없어도 product_id로 탐색 )
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
    p = (row.get("image_path") or "").strip() if "image_path" in row else ""
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

# 프레임 이미지 로드/전처리(가벼움)
fg_bgra = vision.ensure_bgra(img_path)
if fg_bgra is None:
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}")
    st.stop()
fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)
st.session_state.fg_bgra = fg_bgra

# 선택 프레임 치수/비율
A     = float(row["lens_mm"])
DBL   = float(row["bridge_mm"])
TOTAL = float(row["total_mm"])
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0
st.session_state.k_ratio = float(k)
st.session_state.TOTAL_mm = float(TOTAL)

# =============================
# 7) 합성(슬라이더만 반영)
# =============================
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

# 목표 폭 계산
mm_per_px = CHEEK_MM / max(w_face, 1e-6)
if PD_MM is not None:
    # PD(mm)->GCD(px)->TOTAL(px) (보정계수 0.92 역산)
    target_total_px = (PD_MM / (0.92)) * k / max(mm_per_px, 1e-6)
else:
    target_total_px = TOTAL / max(mm_per_px, 1e-6)

min_w = 0.60 * w_face
max_w = 0.95 * w_face
from math import isfinite
if not isfinite(target_total_px):
    target_total_px = 0.8 * w_face
target_total_px = float(np.clip(target_total_px, min_w, max_w))

# 슬라이더 스케일 반영
scale = (target_total_px / max(w0, 1)) * float(st.session_state.scale_mult)
scale = float(np.clip(scale, 0.35, 2.2))

# 리사이즈/회전
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

# 여백 붙여 안전 합성
margin_x, margin_y = 300, 150
bg_expanded = cv2.copyMakeBorder(
    face_bgr, margin_y, margin_y, margin_x, margin_x,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
gx_expanded = gx + margin_x
gy_expanded = gy + margin_y

out = vision.overlay_rgba(bg_expanded, fg_rot, gx_expanded, gy_expanded)
show_image_bgr(
    out,
    caption=f"합성 — 선택: {row.get('brand','?')} / {row.get('product_id','?')} · {row.get('shape','?')} · Face:{st.session_state.faceshape_label or 'Unknown'}"
)

# 다운로드
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    PIL.Image.fromarray(rgb).save(buf, format="PNG")
    file_name = row.get('product_id', 'frame')
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(), file_name=f"{file_name}_result.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")

