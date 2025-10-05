# vision.py (추가/확인)
import os, numpy as np, cv2
from PIL import Image
try:
    import pillow_avif_plugin
except Exception:
    pass

FRAMES_DIR = "frames/Images"
ANTENA_FILE = "Antena_01.png"
A_FIXED, DBL_FIXED, TOTAL_FIXED = 52.7, 20.0, 145.1

def ensure_bgra(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        if img.shape[2] == 4:
            return img  # BGRA
        if img.shape[2] == 3:
            b,g,r = cv2.split(img)
            a = np.full_like(b, 255)
            return cv2.merge([b,g,r,a])
    # OpenCV 실패 → PIL로
    try:
        pil = Image.open(path).convert("RGBA")
        rgba = np.array(pil)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception:
        return None

def load_fixed_antena():
    """
    Antena_01.* 을 frames/images에서 찾아 BGRA로 로드.
    반환: (fg_bgra, (A, DBL, TOTAL)) / 실패 시 (None, None)
    """
    # 우선 png를 시도
    primary = os.path.join(FRAMES_DIR, ANTENA_FILE)
    cand = primary if os.path.exists(primary) else None
    if cand is None:
        # 확장자 가변 탐색(.png/.avif/.webp/.jpg/.jpeg)
        if os.path.isdir(FRAMES_DIR):
            for fn in os.listdir(FRAMES_DIR):
                name, ext = os.path.splitext(fn)
                if name.lower() == os.path.splitext(ANTENA_FILE)[0].lower() and \
                   ext.lower() in [".png", ".avif", ".webp", ".jpg", ".jpeg"]:
                    cand = os.path.join(FRAMES_DIR, fn); break
    if cand is None:
        return None, None

    fg = ensure_bgra(cand)
    if fg is None:
        return None, None
    return fg, (A_FIXED, DBL_FIXED, TOTAL_FIXED)

