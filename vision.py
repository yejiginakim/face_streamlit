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
ANTENA_FILE = "SF191SKN_004_61.png"
A_FIXED, DBL_FIXED, TOTAL_FIXED = 61.0, 17.0, 148.0  # mm

@lru_cache(maxsize=1)
def create_facemesh():
    """MediaPipe FaceMesh 인스턴스(캐시)."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

def _euler_from_rotation_matrix(R):
    # ZYX 회전(roll around Z, pitch around X, yaw around Y) convention
    # OpenCV 좌표계 기준으로 안정적인 분해
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])   # around X
        yaw   = np.arctan2(-R[2,0], sy)      # around Y
        roll  = np.arctan2(R[1,0], R[0,0])   # around Z
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw   = np.arctan2(-R[2,0], sy)
        roll  = 0.0
    return np.degrees([yaw, pitch, roll])  # (yaw, pitch, roll) in deg


def head_pose_ypr(bgr):
    """
    Mediapipe FaceMesh(world) + solvePnP로 (yaw, pitch, roll) 각도(°) 반환.
    실패 시 (None, None, None)
    """
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None

    lm2d = res.multi_face_landmarks[0].landmark
    lm3d_all = getattr(res, "multi_face_world_landmarks", None)
    if not lm3d_all:
        return None, None, None
    lm3d = lm3d_all[0].landmark

    # 키 포인트 인덱스 (mediapipe facemesh 468)
    idx = {
        "nose": 1,       # 코끝
        "chin": 152,     # 턱
        "l_eye": 33,     # 왼 눈 바깥
        "r_eye": 263,    # 오른 눈 바깥
        "l_mouth": 61,   # 입 왼쪽
        "r_mouth": 291,  # 입 오른쪽
    }

    # 2D (픽셀)
    img_pts = np.array([
        [lm2d[idx["nose"]].x * w,   lm2d[idx["nose"]].y * h],
        [lm2d[idx["chin"]].x * w,   lm2d[idx["chin"]].y * h],
        [lm2d[idx["l_eye"]].x * w,  lm2d[idx["l_eye"]].y * h],
        [lm2d[idx["r_eye"]].x * w,  lm2d[idx["r_eye"]].y * h],
        [lm2d[idx["l_mouth"]].x * w,lm2d[idx["l_mouth"]].y * h],
        [lm2d[idx["r_mouth"]].x * w,lm2d[idx["r_mouth"]].y * h],
    ], dtype=np.float32)

    # 3D (mediapipe world coords: meters 단위, 얼굴 중심 좌표계)
    obj_pts = np.array([
        [lm3d[idx["nose"]].x,   lm3d[idx["nose"]].y,   lm3d[idx["nose"]].z],
        [lm3d[idx["chin"]].x,   lm3d[idx["chin"]].y,   lm3d[idx["chin"]].z],
        [lm3d[idx["l_eye"]].x,  lm3d[idx["l_eye"]].y,  lm3d[idx["l_eye"]].z],
        [lm3d[idx["r_eye"]].x,  lm3d[idx["r_eye"]].y,  lm3d[idx["r_eye"]].z],
        [lm3d[idx["l_mouth"]].x,lm3d[idx["l_mouth"]].y,lm3d[idx["l_mouth"]].z],
        [lm3d[idx["r_mouth"]].x,lm3d[idx["r_mouth"]].y,lm3d[idx["r_mouth"]].z],
    ], dtype=np.float32)

    # 카메라 내부 파라미터(대략값): fx=fy=width, cx=width/2, cy=height/2
    fx = fy = w * 1.0
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    dist = np.zeros(5)  # 왜곡 무시

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_from_rotation_matrix(R)
    return float(yaw), float(pitch), float(roll)


def head_roll_angle(bgr):
    """
    간단 롤(roll)만: 양 눈 중심 선의 기울기(°).
    """
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    L_ids = [33,133,159,145]
    R_ids = [362,263,386,374]
    L = np.array([[lm[i].x*w, lm[i].y*h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x*w, lm[i].y*h] for i in R_ids], np.float32).mean(axis=0)
    return float(np.degrees(np.arctan2(R[1]-L[1], R[0]-L[0])))



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

def remove_white_to_alpha(bgra: np.ndarray, thr: int = 240) -> np.ndarray:
    """
    거의 흰색(>=thr) 배경을 투명 알파로 바꿔줌.
    thr를 230~250 사이로 조절 가능.
    """
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return bgra
    b, g, r, a = cv2.split(bgra)
    white = (b >= thr) & (g >= thr) & (r >= thr)
    a[white] = 0
    return cv2.merge([b, g, r, a])

def trim_transparent(bgra: np.ndarray, pad: int = 6) -> np.ndarray:
    """
    알파가 0이 아닌 영역만 딱 맞게 잘라내고 가장자리로 pad 픽셀 여백을 남김.
    """
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return bgra
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bgra
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, bgra.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, bgra.shape[0])
    return bgra[y0:y1, x0:x1]

# vision.py
def cheek_width_px(bgr):
    """
    Mediapipe로 좌/우 볼 대표점(234, 454) 사이 픽셀거리(Cw_px) 반환.
    실패 시 None.
    """
    try:
        fm = create_facemesh()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        L = np.array([lm[234].x * w, lm[234].y * h], dtype=np.float32)
        R = np.array([lm[454].x * w, lm[454].y * h], dtype=np.float32)
        return float(np.linalg.norm(R - L))
    except Exception:
        return None
