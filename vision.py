# vision.py
import os
import numpy as np
import cv2
from functools import lru_cache
from PIL import Image
try:
    import pillow_avif_plugin  # AVIF를 PIL로 읽을 수 있게(있으면 자동등록)
except Exception:
    pass

# 고정 프레임 경로/치수
FRAMES_DIR  = "frames/images"
ANTENA_FILE = "Antena_01.png"
A_FIXED, DBL_FIXED, TOTAL_FIXED = 52.7, 20.0, 145.1  # mm

@lru_cache(maxsize=1)
def create_facemesh():
    """MediaPipe FaceMesh 인스턴스(캐시)."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

def detect_pd_px(bgr: np.ndarray):
    """
    눈 중심 기반 PD_px, 눈선 각도(°), 중점(mid)을 계산.
    반환: (pd_px, angle_deg, (mx, my)) / 실패 시 (None, None, None)
    """
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0].landmark

    # 양쪽 눈의 안정된 4점 평균 중심
    L_ids = [33, 133, 159, 145]
    R_ids = [362, 263, 386, 374]
    L = np.array([[lm[i].x * w, lm[i].y * h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x * w, lm[i].y * h] for i in R_ids], np.float32).mean(axis=0)

    pd_px = float(np.hypot(*(R - L)))
    angle_deg = float(np.degrees(np.arctan2(R[1] - L[1], R[0] - L[0])))
    mid = ((L[0] + R[0]) / 2.0, (L[1] + R[1]) / 2.0)
    return pd_px, angle_deg, mid

def ensure_bgra(path: str):
    """
    파일 경로에서 BGRA(ndarray, HxWx4)를 보장해 로드.
    - OpenCV가 3채널로 읽으면 불투명 알파 추가
    - OpenCV가 실패하면 PIL로 재시도(AVIF/PNG/WEBP 등)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        if img.shape[2] == 4:
            return img  # BGRA
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.full_like(b, 255)
            return cv2.merge([b, g, r, a])
    try:
        pil = Image.open(path).convert("RGBA")   # RGBA
        rgba = np.array(pil)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception:
        return None

def load_fixed_antena():
    """
    frames/images 아래에서 Antena_01.* 를 찾아 BGRA로 로드.
    반환: (fg_bgra, (A, DBL, TOTAL)) / 실패 시 (None, None)
    """
    # 우선 .png 고정 경로 시도
    primary = os.path.join(FRAMES_DIR, ANTENA_FILE)
    cand = primary if os.path.exists(primary) else None
    # 없으면 확장자 가변 탐색
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

def overlay_rgba(bg_bgr: np.ndarray, fg_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """배경 BGR 위에 전경 BGRA를 (x,y) 상단 기준 알파합성."""
    H, W = bg_bgr.shape[:2]
    h, w = fg_bgra.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return bg_bgr
    fg_cut = fg_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = (fg_cut[:, :, 3:4].astype(np.float32) / 255.0)
    bg_roi = bg_bgr[y0:y1, x0:x1, :].astype(np.float32)
    fg_rgb = fg_cut[:, :, :3].astype(np.float32)
    out = fg_rgb * alpha + bg_roi * (1 - alpha)
    bg_bgr[y0:y1, x0:x1, :] = out.astype(np.uint8)
    return bg_bgr

