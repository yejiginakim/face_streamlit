# vision.py
import os
import numpy as np
import cv2
from functools import lru_cache
from PIL import Image
try:
    import pillow_avif_plugin  # AVIF까지 읽기 (있으면 자동등록)
except Exception:
    pass

# 고정 프레임 경로/치수 (파일/숫자만 바꿔 쓰면 됨)
FRAMES_DIR  = "frames/images"
ANTENA_FILE = "Antena_01.png"      # frames/images/Antena_01.png
A_FIXED, DBL_FIXED, TOTAL_FIXED = 52.7, 20.0, 145.1  # mm (A, DBL, TOTAL)

# ---------------- FaceMesh ----------------
@lru_cache(maxsize=1)
def create_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,   # iris 사용
        max_num_faces=1
    )

def np_landmarks(bgr: np.ndarray):
    """(N,2) float32 랜드마크(px). 실패 시 None"""
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x*w, p.y*h] for p in lm], dtype=np.float32)

def detect_pd_px(bgr: np.ndarray):
    """
    PD(px) = 좌우 '홍채 중심(iris center)' 간 거리
    roll(deg) = 눈선 기울기
    mid = 두 중심의 중점
    """
    pts = np_landmarks(bgr)
    if pts is None:
        return None, None, None

    # MediaPipe iris: L 468~471, R 473~476
    Lc = pts[[468,469,470,471]].mean(axis=0)
    Rc = pts[[473,474,475,476]].mean(axis=0)

    pd_px = float(np.hypot(*(Rc - Lc)))
    roll  = float(np.degrees(np.arctan2(Rc[1]-Lc[1], Rc[0]-Lc[0])))
    mid   = ((Lc[0]+Rc[0])/2.0, (Lc[1]+Rc[1])/2.0)
    return pd_px, roll, mid

def nose_bridge_point(bgr: np.ndarray):
    """코등(168) 좌표(px). 실패 시 None"""
    pts = np_landmarks(bgr)
    return None if pts is None else pts[168]

# ---------------- 이미지 로드/정리 ----------------
def ensure_bgra(path: str):
    """
    파일을 BGRA(HxWx4)로 로드. 실패 시 None
    - OpenCV 실패 → PIL로 재시도(AVIF/WEBP/PNG 등)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        if img.shape[2] == 4:
            return img
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.full_like(b, 255)
            return cv2.merge([b, g, r, a])
    try:
        pil = Image.open(path).convert("RGBA")
        rgba = np.array(pil)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception:
        return None

def load_fixed_antena():
    """
    frames/images/Antena_01.* 를 찾아 BGRA로 로드
    반환: (fg_bgra, (A, DBL, TOTAL)) / 실패 시 (None, None)
    """
    primary = os.path.join(FRAMES_DIR, ANTENA_FILE)
    cand = primary if os.path.exists(primary) else None

    if cand is None and os.path.isdir(FRAMES_DIR):
        base = os.path.splitext(ANTENA_FILE)[0].lower()
        for fn in os.listdir(FRAMES_DIR):
            name, ext = os.path.splitext(fn)
            if name.lower() == base and ext.lower() in (".png", ".avif", ".webp", ".jpg", ".jpeg"):
                cand = os.path.join(FRAMES_DIR, fn)
                break
    if cand is None:
        return None, None

    fg = ensure_bgra(cand)
    if fg is None:
        return None, None
    return fg, (A_FIXED, DBL_FIXED, TOTAL_FIXED)

# ---------------- 프린지(매트) 보정 ----------------
def dematte_any_color(fg_bgra: np.ndarray, matte_color=None) -> np.ndarray:
    """
    가장자리 매트색 프린지 제거 (프레임이 흰/밝아도 안전)
    matte_color=(255,255,255)로 고정하거나, None이면 자동추정
    """
    if fg_bgra is None or fg_bgra.shape[2] != 4:
        return fg_bgra
    b,g,r,a = cv2.split(fg_bgra)
    a_f = a.astype(np.float32)/255.0
    eps = 1e-6

    if matte_color is None:
        m = 8
        border = np.concatenate([fg_bgra[:m,:,:], fg_bgra[-m:,:,:],
                                 fg_bgra[:, :m,:], fg_bgra[:, -m:,:]], axis=0)
        mb, mg, mr, _ = [np.median(c) for c in cv2.split(border)]
    else:
        mb, mg, mr = matte_color

    b = np.clip((b.astype(np.float32) - (1-a_f)*mb) / np.maximum(a_f, eps), 0, 255)
    g = np.clip((g.astype(np.float32) - (1-a_f)*mg) / np.maximum(a_f, eps), 0, 255)
    r = np.clip((r.astype(np.float32) - (1-a_f)*mr) / np.maximum(a_f, eps), 0, 255)

    a = cv2.erode(a, np.ones((3,3), np.uint8), iterations=1)
    a = cv2.GaussianBlur(a, (5,5), 1)

    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), a])

# ---------------- 가림(occlusion) ----------------
def build_occlusion_mask(bgr: np.ndarray, blur=13, sigma=6) -> np.ndarray:
    """
    윗눈꺼풀/눈썹/코등 기반으로 얼굴이 위에 오도록 0~1 소프트 마스크 생성
    """
    pts = np_landmarks(bgr)
    if pts is None:
        return None
    H, W = bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)

    def poly(idx):
        cv2.fillPoly(mask, [pts[idx].astype(np.int32)], 255)

    # 대략적 영역(필요시 조정 가능)
    left_upper = [33, 246, 161, 160, 159, 158, 157, 173, 133]
    right_upper= [362, 398, 384, 385, 386, 387, 388, 466, 263]
    left_brow  = [70,63,105,66,107]
    right_brow = [336,296,334,293,300]
    nose_bridge= [6,168,197,195,5]

    poly(left_upper); poly(right_upper)
    poly(left_brow);  poly(right_brow)
    poly(nose_bridge)

    mask = cv2.GaussianBlur(mask, (blur, blur), sigma)
    mask = np.clip(mask.astype(np.float32)/255.0 * 1.15, 0, 1)
    return mask

# ---------------- 합성 ----------------
def overlay_rgba(bg_bgr: np.ndarray, fg_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """배경 BGR 위에 전경 BGRA를 (x,y) 상단 기준 알파합성"""
    H, W = bg_bgr.shape[:2]
    h, w = fg_bgra.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return bg_bgr
    fg_cut = fg_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha  = (fg_cut[:, :, 3:4].astype(np.float32) / 255.0)
    bg_roi = bg_bgr[y0:y1, x0:x1, :].astype(np.float32)
    fg_rgb = fg_cut[:, :, :3].astype(np.float32)
    out    = fg_rgb*alpha + bg_roi*(1-alpha)
    bg_bgr[y0:y1, x0:x1, :] = out.astype(np.uint8)
    return bg_bgr
