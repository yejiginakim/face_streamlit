# =============================
# 0) 환경변수
# =============================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# =============================
# 1) 라이브러리
# =============================
import numpy as np
import tensorflow as tf
import keras
import cv2
import PIL
import streamlit as st

print("NumPy:", np.__version__)
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("cv2:", cv2.__version__)
print("Pillow:", PIL.__version__)

import sys, platform, glob, hashlib

st.set_page_config(page_title="Antena_01 — PD→선글라스 합성", layout="wide")
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# =============================
# 2) 모듈 임포트
# =============================
err_msgs = []
try:
    from faceshape import FaceShapeModel, apply_rules, decide_rule_vs_top2
except Exception as e:
    err_msgs.append(f"faceshape 임포트 실패: {e}")

try:
    import vision  # detect_pd_px / cheek_width_px / head_pose_ypr / ensure_bgra / ...
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

try:
    from metrics import compute_metrics_bgr
except Exception as e:
    err_msgs.append(f"metrics 임포트 실패: {e}")

# =============================
# 3) 유틸
# =============================
# =============================
# 3) 유틸  (호환 가능한 st.image 래퍼)
# =============================
import inspect

def show_image_bgr(img_bgr, **kwargs):
    """
    Streamlit 버전에 따라 use_container_width / use_column_width를 자동 선택.
    """
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"이미지 RGB 변환 오류: {e}")
        return

    # st.image 시그니처에서 지원 파라미터 확인
    try:
        params = set(inspect.signature(st.image).parameters.keys())
    except Exception:
        params = set()

    try:
        if "use_container_width" in params:
            st.image(rgb, use_container_width=True, **kwargs)
        elif "use_column_width" in params:
            st.image(rgb, use_column_width=True, **kwargs)
        else:
            st.image(rgb, **kwargs)
    except TypeError:
        # 혹시 남아 있는 호환성 이슈 대비: 강제 기본 호출
        try:
            st.image(rgb, **kwargs)
        except Exception as e:
            st.error(f"이미지 표시 중 오류: {e}")

def _is_lfs_pointer(path:str)->bool:
    try:
        if not os.path.isfile(path): return False
        if os.path.getsize(path) > 2048: return False
        with open(path, "rb") as f:
            head = f.read(256)
        return (b"git-lfs" in head) or (b"github.com/spec" in head)
    except Exception:
        return False

def file_md5(b: bytes)->str:
    return hashlib.md5(b).hexdigest()

# nose-chin 길이 폴백
def _nose_chin_length_px_fallback(bgr):
    try:
        fm = vision.create_facemesh()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        nose = np.array([lm[1].x * w,   lm[1].y * h],  dtype=np.float32)
        chin = np.array([lm[152].x * w, lm[152].y * h], dtype=np.float32)
        return float(np.linalg.norm(chin - nose))
    except Exception:
        return None

def nose_chin_length_px_safe(bgr):
    if hasattr(vision, "nose_chin_length_px"):
        try:
            return vision.nose_chin_length_px(bgr)
        except Exception:
            pass
    return _nose_chin_length_px_fallback(bgr)

# =============================
# 4) 상태
# =============================
st.title("🧍→🕶️ Antena_01 — Top-2×2 추천 + 자동 스케일")

defaults = {
    "img_key": None,
    "face_bgr": None,
    "faceshape_label": None,
    "top2_labels": [],
    # 탐지
    "mid": (0,0),
    "roll": 0.0,
    "pitch": 0.0,
    "PD_px_auto": None,
    "Cw_px_auto": None,
    "NC_px_auto": None,
    # 추천
    "recs": [],
    "selected_pid": None,
    "fg_bgra": None,
    "k_ratio": 2.0,
    "TOTAL_mm": None,
    # 조작
    "dx": 0,
    "dy": 0,
    "scale_mult": 1.0,
}
for k,v in defaults.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.subheader("🎚️ 위치/스케일(미세조정)")
    st.session_state.dx = st.slider("수평 오프셋(px)", -400, 400, st.session_state.dx)
    st.session_state.dy = st.slider("수직 오프셋(px)", -400, 400, st.session_state.dy)
    st.session_state.scale_mult = st.slider("스케일(배)", 0.2, 2.0, float(st.session_state.scale_mult), 0.01,
                                        help="자동 스케일 기준에서 크게/작게 폭넓게 조정")

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"], key="face_file")
with colR:
    st.markdown("### 2) 카테고리 선택")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder='선택하세요', key="gender_ms")
    use_kind = st.multiselect('분류', ['fashion', 'sports'], placeholder='선택하세요', key="kind_ms")

if err_msgs:
    st.warning("초기 임포트 경고")
    st.code("\n".join(err_msgs))

# =============================
# 5) 추천 규칙
# =============================
def normalize_shape(s: str) -> str:
    if not isinstance(s, str): return ""
    t = s.strip().lower().replace("_","-")
    t = " ".join(t.split()).replace(" ", "-")
    syn = {
        "round":"round","circle":"round","boston":"round","panto":"round","oval":"round",
        "rect":"rectangular","rectangle":"rectangular","square":"rectangular","flat-top":"rectangular","clubmaster":"rectangular",
        "trapezoid":"trapezoid","wayfarer":"trapezoid","browline":"trapezoid","geometric":"trapezoid",
        "aviator":"aviator","pilot":"aviator","teardrop":"aviator",
        "cat-eye":"cat-eye","cateye":"cat-eye","cats-eye":"cat-eye","butterfly":"cat-eye",
        "shield":"shield","wrap":"shield","wraparound":"shield","mask":"shield","visor":"shield",
    }
    return syn.get(t, t)

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
    if "sports" in kinds:
        if face_label in ("Oval","Round","Oblong","Square"):
            if "shield" in base: base.remove("shield")
            base.insert(0, "shield")
        else:
            if "shield" not in base: base.append("shield")
    # 유니크 2개
    out=[]
    for s in base:
        if s not in out: out.append(s)
    return out[:2]

def pick_from_pool(pool_df, n, seed=None):
    if len(pool_df) <= 0: return []
    take = min(n, len(pool_df))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(pool_df.index.to_numpy(), size=take, replace=False)
    return pool_df.loc[idxs].to_dict(orient="records")

# =============================
# 6) 변화 감지
# =============================
def need_refresh(img_bytes: bytes, genders: list, kinds: list)->bool:
    new_key = None if img_bytes is None else file_md5(img_bytes)
    changed = (st.session_state.img_key != new_key) or \
              (tuple(st.session_state.get("_genders", [])) != tuple(genders or [])) or \
              (tuple(st.session_state.get("_kinds", [])) != tuple(kinds or []))
    st.session_state.img_key = new_key
    st.session_state._genders = genders or []
    st.session_state._kinds = kinds or []
    return changed

# 필수 체크
if not img_file:
    st.info("정면 얼굴 사진을 업로드하고, 성별/분류를 선택하세요.")
    st.stop()
if not (use_gender and use_kind):
    st.warning("성별/분류에서 각각 최소 1개 이상 선택하세요.")
    st.stop()

img_bytes = img_file.getvalue()
refresh = need_refresh(img_bytes, use_gender, use_kind)

# =============================
# 7) 무거운 단계 (얼굴형/탐지/추천)
# =============================
if refresh:
    # 얼굴 고정
    try:
        file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
        face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if face_bgr is None: raise RuntimeError("OpenCV decode 실패")
        st.session_state.face_bgr = face_bgr
    except Exception as e:
        st.error(f"얼굴 이미지 로드 실패: {e}"); st.stop()

    # 얼굴형 Top-2
    MODEL_PATH = "models/faceshape_efficientnetB4_best_20251018_223855.keras"
    CLASSES_PATH = "models/classes_20251018_223855.txt"
    faceshape_model = None
    if os.path.isfile(MODEL_PATH) and os.path.isfile(CLASSES_PATH) and not _is_lfs_pointer(MODEL_PATH):
        @st.cache_resource
        def _load_faceshape():
            return FaceShapeModel(MODEL_PATH, CLASSES_PATH, img_size=(224,224))
        try:
            faceshape_model = _load_faceshape()
        except Exception:
            faceshape_model = None

    final_label, top2_labels = None, []
    probs_use = None
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
                probs_use = adj['rule_probs']; final_label = adj['rule_label']
            else:
                probs_use = probs
                _, final_label, _ = decide_rule_vs_top2(probs, faceshape_model.class_names)
            idxs = np.argsort(-probs_use)[:2]
            top2_labels = [faceshape_model.class_names[i] for i in idxs]
        except Exception:
            pass
    st.session_state.faceshape_label = final_label
    if final_label and final_label not in top2_labels:
        top2_labels = [final_label] + top2_labels
    # dedup
    st.session_state.top2_labels = []
    for l in top2_labels:
        if l not in st.session_state.top2_labels:
            st.session_state.top2_labels.append(l)
        if len(st.session_state.top2_labels) == 2: break

    # 카탈로그 → 2*2 추천
    import pandas as pd
    EXCEL_PATH = "sg_df.xlsx"
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        st.error(f"엑셀 카탈로그 로드 실패: {e}"); st.stop()

    need_cols = ["product_id","brand","shape","purpose","sex","lens_mm","bridge_mm","total_mm"]
    for c in need_cols:
        if c not in df.columns:
            st.error(f"엑셀에 '{c}' 컬럼이 없습니다."); st.stop()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["shape"]   = df["shape"].astype(str).map(normalize_shape)
    df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()
    df["sex"]     = df["sex"].astype(str).str.strip().str.lower()
    for c in ["lens_mm","bridge_mm","total_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    SHAPES6 = {"round","rectangular","trapezoid","aviator","cat-eye","shield"}
    bad = df.loc[~df["shape"].isin(SHAPES6), ["product_id","brand","shape"]]
    if len(bad) > 0:
        st.error("shape 값은 round/rectangular/trapezoid/aviator/cat-eye/shield 만 허용됩니다.")
        st.dataframe(bad.head(30)); st.stop()

    gset = set([g.strip().lower() for g in use_gender])
    kset = set([k.strip().lower() for k in use_kind])

    f = (df["sex"].isin(gset) | (df["sex"]=="unisex")) & (df["purpose"].isin(kset))
    cand = df[f].copy()

    # 안내
    target_map = {}
    tl = st.session_state.top2_labels or [st.session_state.faceshape_label]
    for lbl in tl:
        target_map[lbl or "Unknown"] = get_shape_targets(lbl, kset)
    st.info("🧠 얼굴형 Top-2 추천: " + " / ".join([f"{k}: {', '.join(v)}" for k,v in target_map.items()]))

    recs = []
    seed = int(np.random.randint(0, 1_000_000))
    tl = st.session_state.top2_labels or [None]
    for li, face_lbl in enumerate(tl):
        shapes = get_shape_targets(face_lbl, kset)
        wanted = 2
        chosen = []
        def _pick(shape_name, take, seed_base):
            pool = cand[cand["shape"] == shape_name]
            # 우선 목적 맞는 쪽
            pool_main = pool[pool["purpose"].isin(kset)]
            C = pick_from_pool(pool_main, take, seed_base)
            if len(C) < take:
                # 다른 목적에서 보강
                pool_sub = pool[~pool.index.isin([cand[cand["product_id"]==c["product_id"]].index[0] for c in C])]
                C += pick_from_pool(pool_sub, take - len(C), seed_base+1)
            return C
        if len(shapes)>=1: chosen += _pick(shapes[0], wanted, seed+li*100)
        if len(chosen)<wanted and len(shapes)>=2: chosen += _pick(shapes[1], wanted-len(chosen), seed+li*100+50)
        for r in chosen[:wanted]:
            r["face_for"] = face_lbl
        recs += chosen[:wanted]

    if len(recs) < 4:
        already = set([r["product_id"] for r in recs])
        remain = cand[~cand["product_id"].isin(already)]
        extra = pick_from_pool(remain, 4 - len(recs), seed+999)
        for r in extra: r["face_for"] = st.session_state.faceshape_label
        recs += extra

    st.session_state.recs = recs[:4]

    # 탐지 (PD/볼폭/코↔턱/자세)
    try:
        pd_px, eye_roll_deg, mid = vision.detect_pd_px(st.session_state.face_bgr)
    except Exception:
        pd_px, eye_roll_deg, mid = None, 0.0, (0,0)

    try: Cw_px = vision.cheek_width_px(st.session_state.face_bgr)
    except Exception: Cw_px = None
    NC_px = nose_chin_length_px_safe(st.session_state.face_bgr)

    yaw = pitch = roll = None
    if hasattr(vision, "head_pose_ypr"):
        try:
            yaw, pitch, roll = vision.head_pose_ypr(st.session_state.face_bgr)
        except Exception:
            yaw = pitch = roll = None
    if roll is None: roll = eye_roll_deg

    st.session_state.mid   = mid
    st.session_state.roll  = float(roll or 0.0)
    st.session_state.pitch = float(pitch or 0.0)
    st.session_state.PD_px_auto = pd_px
    st.session_state.Cw_px_auto = Cw_px
    st.session_state.NC_px_auto = NC_px

# =============================
# 8) 추천 중 하나 선택
# =============================
recs = st.session_state.recs or []
if not recs:
    st.error("추천할 프레임을 못 찾았어요."); st.stop()

pretty = [f"{i+1}) [{r.get('purpose','?')}] {r.get('brand','?')} / {r.get('product_id','?')} · {r.get('shape','?')} · {int(r.get('total_mm',0))}mm · FaceFor:{r.get('face_for') or 'Unknown'}"
          for i,r in enumerate(recs)]
st.markdown("### 😀 추천 4개 중 선택하면 즉시 합성합니다.")
sel_label = st.selectbox("추천 (1개)", options=pretty, index=0)
row = recs[pretty.index(sel_label)]
st.session_state.selected_pid = row.get("product_id")

# =============================
# 9) 프레임 이미지 로드/전처리
# =============================
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
    if p and os.path.exists(p): return p
    pid = str(row.get("product_id","")).strip()
    if not pid: return None
    shape_dir = SHAPE_DIR_MAP.get(str(row.get("shape","")).strip().lower())
    if shape_dir:
        base = os.path.join(FRAME_ROOT, shape_dir, pid)
        for ext in EXTS:
            cp = base + ext
            if os.path.exists(cp): return cp
    pattern = os.path.join(FRAME_ROOT, "**", pid + ".*")
    for cp in glob.glob(pattern, recursive=True):
        if os.path.splitext(cp)[1].lower() in EXTS and os.path.isfile(cp):
            return cp
    return None

img_path = _resolve_image(row)
if not img_path:
    st.error(f"프레임 파일을 찾지 못했습니다: {row.get('product_id')}"); st.stop()

fg_bgra = vision.ensure_bgra(img_path)
if fg_bgra is None:
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}"); st.stop()

fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)
st.session_state.fg_bgra = fg_bgra

# 프레임 치수/비율
A, DBL, TOTAL = float(row["lens_mm"]), float(row["bridge_mm"]), float(row["total_mm"])
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0
st.session_state.k_ratio = float(k)
st.session_state.TOTAL_mm = float(TOTAL)

# =============================
# =============================
# =============================
# 10) 합성 — PD 정합 + 얼굴높이 캡
# =============================
face_bgr = st.session_state.face_bgr
fg_bgra  = st.session_state.fg_bgra
mid      = st.session_state.mid or (0,0)
roll     = float(st.session_state.roll or 0.0)
pitch    = float(st.session_state.pitch or 0.0)

# 탐지값
PD_px = st.session_state.PD_px_auto           # 양쪽 동공 거리(px)
Cw_px = st.session_state.Cw_px_auto           # 볼폭(px) — 폴백용
NC_px = st.session_state.NC_px_auto           # 코끝↔턱 길이(px)

# 프레임 스펙/원본 크기
k      = float(st.session_state.k_ratio or 2.0)   # TOTAL/GCD
TOTAL  = float(st.session_state.TOTAL_mm or 140.0)
GCD    = TOTAL / k if k else (float(row["lens_mm"]) + float(row["bridge_mm"]))
h_face, w_face = face_bgr.shape[:2]
h0, w0 = fg_bgra.shape[:2]

# 1) PD로 'GCD(px)' 정확히 맞추기
scale = None
if PD_px is not None and PD_px > 1 and TOTAL > 0 and GCD > 0 and w0 > 0:
    GCD2PD = 0.92  # PD ≈ 0.92 * GCD
    gcd_px_target = PD_px / GCD2PD  # 우리가 맞추고 싶은 GCD(px)
    gcd_px_in_image = w0 * (GCD / TOTAL)  # 트리밍된 프레임 이미지의 현재 GCD(px)
    scale = float(gcd_px_target / max(gcd_px_in_image, 1e-6))

# 2) PD가 없거나 실패하면 볼폭 기반 폴백
if scale is None:
    ALPHA = 0.80  # TOTAL ≈ ALPHA * cheek_width
    total_target_px = (Cw_px * ALPHA) if (Cw_px and Cw_px > 1) else (0.60 * w_face)
    scale = float(total_target_px / max(w0, 1))

# 3) 높이 캡: 선글라스 높이 ≤ 얼굴길이(코↔턱) × 0.65 (없으면 얼굴세로 0.40)
max_h = (0.65 * NC_px) if (NC_px and NC_px > 1) else (0.40 * h_face)
scale = min(scale, max_h / max(h0, 1))

# 4) 사용자 미세조정
scale *= float(st.session_state.scale_mult)
scale = float(np.clip(scale, 0.10, 2.50))

# ---- 리사이즈/회전/배치/합성 (이하는 기존과 동일) ----
new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)
M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

pitch_dy = int((pitch or 0.0) * 0.8)
if mid == (0, 0):
    gx = int(face_bgr.shape[1] * 0.5 - fg_rot.shape[1] * 0.5) + st.session_state.dx
    gy = int(face_bgr.shape[0] * 0.45 - fg_rot.shape[0] * 0.5) + st.session_state.dy
else:
    gx = int(mid[0] - fg_rot.shape[1] * 0.5) + st.session_state.dx
    gy = int(mid[1] - fg_rot.shape[0] * 0.5) + st.session_state.dy + pitch_dy

margin_x, margin_y = 300, 150
bg_expanded = cv2.copyMakeBorder(
    face_bgr, margin_y, margin_y, margin_x, margin_x,
    cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
gx_e = gx + margin_x
gy_e = gy + margin_y

out = vision.overlay_rgba(bg_expanded, fg_rot, gx_e, gy_e)
show_image_bgr(
    out,
    caption=f"합성 — {row.get('brand','?')} / {row.get('product_id','?')} · {row.get('shape','?')} · FaceFor:{row.get('face_for') or 'Unknown'}"
)
