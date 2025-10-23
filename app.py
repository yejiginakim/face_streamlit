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
# 2) faceshape / vision / metrics 임포트
# =============================
err_msgs = []
try:
    from faceshape import (
        FaceShapeModel,
        apply_rules,
        decide_rule_vs_top2,
    )
except Exception as e:
    err_msgs.append(f"faceshape 임포트 실패: {e}")

try:
    import vision  # detect_pd_px / cheek_width_px / overlay_rgba / (nose_chin_length_px 0/1)
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

# nose-chin 길이 폴백 (vision에 함수가 없을 때)
def _nose_chin_length_px_fallback(bgr):
    try:
        fm = vision.create_facemesh()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        # mediapipe indices: nose tip=1, chin=152
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
# 4) UI & 세션 상태
# =============================
st.title("🧍→🕶️ Antena_01 합성 — 실시간 추천(Top-2×2) + 자동스케일 + 슬라이더")

defaults = {
    "img_key": None,            # 업로드 이미지 해시
    "face_bgr": None,
    "faceshape_label": None,    # 최종 1위 라벨(백업)
    "top2_labels": [],          # 얼굴형 Top-2
    # 탐지 결과
    "mid": (0, 0),
    "roll": 0.0,
    "pitch": 0.0,
    "PD_px_auto": None,
    "Cw_px_auto": None,
    "NC_px_auto": None,
    # 치수 입력(옵션)
    "PD_MM_raw": None,          # 사용자가 넣은 PD(mm)
    "CHEEK_MM": None,           # 얼굴폭(mm) — 미사용 가능
    # 추천/선택
    "recs": [],                 # 추천 4개(dict): face_for 필드 포함
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
    st.subheader("📱 치수 입력(선택)")
    use_phone = st.checkbox("PD(mm) 직접 입력", value=False, help="체크하면 아래 PD(mm)를 사용하고, 안하면 자동 PD_px로 스케일합니다.")
    if use_phone:
        pd_in = st.number_input("PD(mm)", value=62.0, step=0.1, format="%.1f")
        st.session_state.PD_MM_raw = float(pd_in)
    else:
        st.session_state.PD_MM_raw = None

    # (참고용) 얼굴폭(mm)은 더이상 필수 아님 — 미입력 가능
    cheek_opt = st.text_input("얼굴 폭(mm, 옵션)", value="", placeholder="입력 안해도 됨")
    try:
        st.session_state.CHEEK_MM = float(cheek_opt) if cheek_opt.strip() else None
    except:
        st.session_state.CHEEK_MM = None

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
# 5) 이미지/필터 변화 감지 → 필요 시만 재계산
# =============================
def need_refresh(img_bytes: bytes, genders: list, kinds: list)->bool:
    new_key = None if img_bytes is None else file_md5(img_bytes)
    changed = False
    if st.session_state.img_key != new_key:
        changed = True
    prev_filters = (tuple(st.session_state.get("_genders", [])), tuple(st.session_state.get("_kinds", [])))
    new_filters  = (tuple(genders or []), tuple(kinds or []))
    if prev_filters != new_filters:
        changed = True
    st.session_state.img_key = new_key
    st.session_state._genders, st.session_state._kinds = new_filters
    return changed

# 필수 입력 체크
if not img_file:
    st.info("정면 얼굴 사진을 업로드하고, 성별/분류를 선택하세요.")
    st.stop()
if not (use_gender and use_kind):
    st.warning("성별/분류에서 각각 최소 1개 이상 선택하세요.")
    st.stop()

# 이미지 바이트/해시
img_bytes = img_file.getvalue()
refresh = need_refresh(img_bytes, use_gender, use_kind)

# =============================
# 6) 무거운 단계(얼굴형/탐지/추천) — 필요한 경우만
# =============================
if refresh:
    # 1) 얼굴 이미지
    try:
        file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
        face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if face_bgr is None:
            raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
        st.session_state.face_bgr = face_bgr
    except Exception as e:
        st.error(f"얼굴 이미지 로드 실패: {e}")
        st.stop()

    # 2) 얼굴형 추론 (Top-2까지)
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
    top2_labels = []
    probs_use = None
    if faceshape_model is not None:
        try:
            pil_img = PIL.Image.fromarray(cv2.cvtColor(st.session_state.face_bgr, cv2.COLOR_BGR2RGB))
            probs = faceshape_model.predict_probs(pil_img)
            # 규칙 지표(있으면 보정 확률 사용)
            try:
                ar, jaw, cw, jw = compute_metrics_bgr(st.session_state.face_bgr)
            except Exception:
                ar = jaw = cw = jw = None
            if any(v is not None for v in (ar, jaw, cw, jw)):
                adj = apply_rules(probs, faceshape_model.class_names, ar=ar, jaw_deg=jaw, cw=cw, jw=jw)
                probs_use = adj['rule_probs']
                final_label = adj['rule_label']
            else:
                probs_use = probs
                _, final_label, _ = decide_rule_vs_top2(probs, faceshape_model.class_names)

            # Top-2 라벨
            idxs = np.argsort(-probs_use)[:2]
            top2_labels = [faceshape_model.class_names[i] for i in idxs]
        except Exception:
            final_label = None
            top2_labels = []
    # 백업/중복 제거
    st.session_state.faceshape_label = final_label
    if final_label and final_label not in top2_labels:
        top2_labels = [final_label] + top2_labels
    st.session_state.top2_labels = [lbl for i,lbl in enumerate(top2_labels) if top2_labels.index(lbl) == i][:2]

    # 3) 카탈로그 로드 & 추천 4개 구성 (Top-2 × 각 2개)
    import pandas as pd
    EXCEL_PATH = "sg_df.xlsx"
    SHAPES6 = {"round","rectangular","trapezoid","aviator","cat-eye","shield"}
    def _norm(x): return (x or "").strip().lower()

    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        st.error(f"엑셀 카탈로그 로드 실패: {e}")
        st.stop()

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
        st.dataframe(bad.head(50)); st.stop()

    gset = {_norm(g) for g in use_gender}
    kset = {_norm(k) for k in use_kind}
    f = pd.Series([True] * len(df))
    if gset:
        f &= df["sex"].isin(gset) | (df["sex"] == "unisex")
    if kset:
        f &= df["purpose"].isin(kset)
    cand = df[f].copy()

    # 사용자 안내: 얼굴형 Top-2 및 각 추천 모양
    target_map = {}
    for lbl in (st.session_state.top2_labels or [st.session_state.faceshape_label]):
        target_map[lbl or "Unknown"] = get_shape_targets(lbl, kset)
    info_txt = " / ".join([f"{lbl}: {', '.join(shps)}" for lbl, shps in target_map.items()])
    st.info(f"🧠 얼굴형 Top-2 → 추천 모양: {info_txt}")

    # 각 얼굴형에서 2개씩 뽑기
    recs = []
    seed = int(np.random.randint(0, 1_000_000))
    for li, face_lbl in enumerate(st.session_state.top2_labels or [None]):
        shapes = get_shape_targets(face_lbl, kset)
        wanted = 2
        chosen = []

        def _pick(shape_name, take, seed_base):
            pool = cand[cand["shape"] == shape_name]
            if "sports" in kset:
                pool_s = pool[pool["purpose"] == "sports"]
                C = pick_from_pool(pool_s, take, seed_base)
                if len(C) < take:
                    pool_f = pool[pool["purpose"] == "fashion"]
                    C += pick_from_pool(
                        pool_f[~pool_f["product_id"].isin([c["product_id"] for c in C])],
                        take - len(C), seed_base+1
                    )
            else:
                pool_f = pool[pool["purpose"] == "fashion"]
                C = pick_from_pool(pool_f, take, seed_base)
                if len(C) < take:
                    pool_s = pool[pool["purpose"] == "sports"]
                    C += pick_from_pool(
                        pool_s[~pool_s["product_id"].isin([c["product_id"] for c in C])],
                        take - len(C), seed_base+1
                    )
            return C

        if len(shapes) >= 1:
            chosen += _pick(shapes[0], wanted, seed + li*100)
        if len(chosen) < wanted and len(shapes) >= 2:
            chosen += _pick(shapes[1], wanted - len(chosen), seed + li*100 + 50)

        for r in chosen[:wanted]:
            r["face_for"] = face_lbl
        recs += chosen[:wanted]

    # 4개 못 채우면 보충
    if len(recs) < 4:
        already = set([r["product_id"] for r in recs])
        remain_pool = cand[~cand["product_id"].isin(already)]
        extra = pick_from_pool(remain_pool, 4 - len(recs), seed + 999)
        for r in extra:
            r["face_for"] = st.session_state.faceshape_label
        recs += extra
    st.session_state.recs = recs[:4]

    # 4) 탐지(중심/자세/지표) - 한 번만 저장
    try:
        pd_px, eye_roll_deg, mid = vision.detect_pd_px(st.session_state.face_bgr)
    except Exception:
        pd_px = None; eye_roll_deg = 0.0; mid = (0,0)
    Cw_px = None
    try:
        Cw_px = vision.cheek_width_px(st.session_state.face_bgr)
    except Exception:
        Cw_px = None
    NC_px = nose_chin_length_px_safe(st.session_state.face_bgr)

    yaw = pitch = roll = None
    if hasattr(vision, "head_pose_ypr"):
        try:
            yaw, pitch, roll = vision.head_pose_ypr(st.session_state.face_bgr)
        except Exception:
            yaw = pitch = roll = None
    if roll is None:
        roll = eye_roll_deg

    st.session_state.mid        = mid
    st.session_state.roll       = float(roll or 0.0)
    st.session_state.pitch      = float(pitch or 0.0)
    st.session_state.PD_px_auto = pd_px
    st.session_state.Cw_px_auto = Cw_px
    st.session_state.NC_px_auto = NC_px

# =============================
# 7) 추천 4개 단일 선택 → 즉시 합성
# =============================
recs = st.session_state.recs or []
if not recs:
    st.error("추천할 프레임을 찾지 못했습니다."); st.stop()

pretty_items = []
for i, r in enumerate(recs):
    pretty_items.append(
        f"{i+1}) [{r.get('purpose','?')}] {r.get('brand','?')} / {r.get('product_id','?')} · {r.get('shape','?')} · {int(r.get('total_mm',0))}mm · FaceFor:{r.get('face_for') or 'Unknown'}"
    )
st.markdown("### 😀 얼굴형 Top-2 각각에서 2개씩 추천했습니다. 하나를 선택하면 바로 합성합니다.")
selected_label = st.selectbox("추천 중 1개 선택", options=pretty_items, index=0)

sel_idx = pretty_items.index(selected_label)
row = recs[sel_idx]
st.session_state.selected_pid = row.get("product_id")

# =============================
# 8) 프레임 이미지 로드/전처리 (image_path 없어도 product_id.* 탐색)
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
    # 최후 수단: 재귀 탐색
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
    st.error(f"프레임 이미지를 읽을 수 없습니다: {img_path}"); st.stop()

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
# 9) 합성 — 자동 스케일(기본값 無), 폭/높이 캡, 슬라이더 반영
# =============================
face_bgr = st.session_state.face_bgr
fg_bgra  = st.session_state.fg_bgra
mid      = st.session_state.mid or (0, 0)
roll     = float(st.session_state.roll or 0.0)
pitch    = float(st.session_state.pitch or 0.0)
PD_MM    = st.session_state.PD_MM_raw
PD_px    = st.session_state.PD_px_auto
Cw_px    = st.session_state.Cw_px_auto
NC_px    = st.session_state.NC_px_auto
k        = float(st.session_state.k_ratio or 2.0)
TOTAL    = float(st.session_state.TOTAL_mm or 140.0)

h_face, w_face = face_bgr.shape[:2]
h0, w0 = fg_bgra.shape[:2]

# -------- 목표 총가로(px) 결정 (CHEEK_MM 없어도 동작) --------
GCD2PD_CAL = 0.92  # GCD(px) ≈ PD(px) / 0.92  ↔  TOTAL(px) = GCD(px)*k

target_total_px = None

if PD_MM is not None and PD_px is not None and PD_px > 1:
    # PD(mm)와 PD_px를 직접 매칭 → mm_per_px 산출
    mm_per_px = PD_MM / (PD_px / GCD2PD_CAL)
    target_total_px = TOTAL / max(mm_per_px, 1e-6)
elif PD_px is not None and PD_px > 1:
    # 전적으로 자동 PD_px로 스케일
    target_total_px = PD_px * GCD2PD_CAL * k
elif Cw_px is not None:
    target_total_px = 0.9 * Cw_px  # 볼폭의 90%
else:
    target_total_px = 0.8 * w_face  # 최후 수단

# 폭 캡: 볼폭/화면폭
if Cw_px is not None:
    target_total_px = min(target_total_px, 0.95 * Cw_px)
target_total_px = float(np.clip(target_total_px, 0.45 * w_face, 0.95 * w_face))

# -------- 스케일 계산 (폭 기준 + 높이 캡) --------
scale_by_width = target_total_px / max(w0, 1)

if NC_px is not None and NC_px > 1:
    max_h = 0.80 * NC_px
    scale_by_height = max_h / max(h0, 1)
    scale = min(scale_by_width, scale_by_height)
else:
    scale = scale_by_width

# 사용자 미세 조정
scale *= float(st.session_state.scale_mult)
scale = float(np.clip(scale, 0.35, 2.0))

# 리사이즈/회전
new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)
M = cv2.getRotationMatrix2D((fg_scaled.shape[1] / 2, fg_scaled.shape[0] / 2), -roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

# 위치
pitch_dy = int((pitch or 0.0) * 0.8)
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
    caption=f"합성 — 선택: {row.get('brand','?')} / {row.get('product_id','?')} · {row.get('shape','?')} · FaceFor:{row.get('face_for') or 'Unknown'}"
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

