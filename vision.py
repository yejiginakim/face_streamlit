# vision.py
import os
import glob
import numpy as np
import cv2
from functools import lru_cache
from PIL import Image
try:
    import pillow_avif_plugin  # AVIF 지원(있으면 자동 등록)
except Exception:
    pass


# =========================
# MediaPipe Facemesh (캐시)
# =========================
@lru_cache(maxsize=1)
def create_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )



# vision.py (아무 위치여도 되지만 이미지 유틸 위에 두는 걸 권장)

def eye_span_px(bgr):
    """
    Mediapipe FaceMesh로 좌/우 바깥 눈꼬리(33, 263) 사이 가로거리(px).
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
        L = np.array([lm[33].x * w,  lm[33].y * h], dtype=np.float32)   # left outer canthus
        R = np.array([lm[263].x * w, lm[263].y * h], dtype=np.float32)  # right outer canthus
        return float(np.linalg.norm(R - L))
    except Exception:
        return None


# =========================
# 얼굴 자세 / PD 계산
# =========================
def _euler_from_rotation_matrix(R):
    # ZYX (roll around Z, pitch around X, yaw around Y)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])   # X
        yaw   = np.arctan2(-R[2, 0], sy)       # Y
        roll  = np.arctan2(R[1, 0], R[0, 0])   # Z
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0.0
    return np.degrees([yaw, pitch, roll])  # (yaw, pitch, roll)


def head_pose_ypr(bgr):
    """
    (yaw, pitch, roll) [deg] 반환. 실패 시 (None, None, None)
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

    idx = {
        "nose": 1, "chin": 152,
        "l_eye": 33, "r_eye": 263,
        "l_mouth": 61, "r_mouth": 291,
    }

    img_pts = np.array([
        [lm2d[idx["nose"]].x * w,   lm2d[idx["nose"]].y * h],
        [lm2d[idx["chin"]].x * w,   lm2d[idx["chin"]].y * h],
        [lm2d[idx["l_eye"]].x * w,  lm2d[idx["l_eye"]].y * h],
        [lm2d[idx["r_eye"]].x * w,  lm2d[idx["r_eye"]].y * h],
        [lm2d[idx["l_mouth"]].x * w,lm2d[idx["l_mouth"]].y * h],
        [lm2d[idx["r_mouth"]].x * w,lm2d[idx["r_mouth"]].y * h],
    ], dtype=np.float32)

    obj_pts = np.array([
        [lm3d[idx["nose"]].x,   lm3d[idx["nose"]].y,   lm3d[idx["nose"]].z],
        [lm3d[idx["chin"]].x,   lm3d[idx["chin"]].y,   lm3d[idx["chin"]].z],
        [lm3d[idx["l_eye"]].x,  lm3d[idx["l_eye"]].y,  lm3d[idx["l_eye"]].z],
        [lm3d[idx["r_eye"]].x,  lm3d[idx["r_eye"]].y,  lm3d[idx["r_eye"]].z],
        [lm3d[idx["l_mouth"]].x,lm3d[idx["l_mouth"]].y,lm3d[idx["l_mouth"]].z],
        [lm3d[idx["r_mouth"]].x,lm3d[idx["r_mouth"]].y,lm3d[idx["r_mouth"]].z],
    ], dtype=np.float32)

    fx = fy = w * 1.0
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    dist = np.zeros(5)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_from_rotation_matrix(R)
    return float(yaw), float(pitch), float(roll)


def head_roll_angle(bgr):
    """눈 라인 기울기로 간단한 roll(°)"""
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
    (pd_px, eye_line_angle_deg, (mx,my)) / 실패 시 (None, None, None)
    """
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0].landmark

    L_ids = [33, 133, 159, 145]
    R_ids = [362, 263, 386, 374]
    L = np.array([[lm[i].x * w, lm[i].y * h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x * w, lm[i].y * h] for i in R_ids], np.float32).mean(axis=0)

    pd_px = float(np.hypot(*(R - L)))
    angle_deg = float(np.degrees(np.arctan2(R[1] - L[1], R[0] - L[0])))
    mid = ((L[0] + R[0]) / 2.0, (L[1] + R[1]) / 2.0)
    return pd_px, angle_deg, mid


def cheek_width_px(bgr):
    """
    좌/우 볼 대표점(234, 454) 거리(px). 실패 시 None.
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


# =========================
# 이미지 유틸
# =========================
def ensure_bgra(path: str):
    """
    파일에서 BGRA(ndarray, HxWx4) 보장 로드
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        if img.shape[2] == 4:
            return img
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.full_like(b, 255)
            return cv2.merge([b, g, r, a])
    # OpenCV 실패 시 PIL 경로
    try:
        pil = Image.open(path).convert("RGBA")
        rgba = np.array(pil)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception:
        return None


def remove_white_to_alpha(bgra: np.ndarray, thr: int = 240) -> np.ndarray:
    """거의 흰색(>=thr)을 투명 처리."""
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return bgra
    b, g, r, a = cv2.split(bgra)
    white = (b >= thr) & (g >= thr) & (r >= thr)
    a[white] = 0
    return cv2.merge([b, g, r, a])






def trim_transparent(bgra: np.ndarray, pad: int = 6) -> np.ndarray:
    """알파>0 영역만 크롭 + pad."""
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return bgra
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bgra
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, bgra.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, bgra.shape[0])
    return bgra[y0:y1, x0:x1]


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


# =========================
# (선택) 프레임 탐색 도우미
# =========================
# app.py에서 이미 _resolve_image로 경로를 찾고 있으니,
# vision 쪽에서도 필요하다면 아래 보조 함수를 사용할 수 있어.
SHAPE_DIR_MAP = {
    "aviator":     "Aviator",
    "cat-eye":     "Cat_eye",
    "rectangular": "Rectangular",
    "round":       "Round",
    "shield":      "Shield",
    "trapezoid":   "Trapezoid",
}
EXTS = (".png", ".webp", ".avif", ".jpg", ".jpeg")



def nose_chin_length_px(bgr):
    """
    Mediapipe로 코끝(1) ↔ 턱(152) 거리(px) 반환. 실패 시 None.
    """
    try:
        fm = create_facemesh()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        nose = np.array([lm[1].x * w,   lm[1].y * h], dtype=np.float32)
        chin = np.array([lm[152].x * w, lm[152].y * h], dtype=np.float32)
        return float(np.linalg.norm(chin - nose))
    except Exception:
        return None




def load_frame_image(image_path: str | None = None,
                     product_id: str | None = None,
                     shape: str | None = None,
                     frame_root: str = "frame"):
    """
    (옵션) 프레임 이미지 로더.
    1) image_path가 있으면 그대로 시도
    2) frame/{ShapeDir}/{product_id}.*
    3) frame/**/{product_id}.*
    성공 시 BGRA ndarray 반환, 실패 시 None
    """
    # 0) 명시 경로
    if image_path:
        p = image_path.strip()
        if os.path.exists(p):
            return ensure_bgra(p)

    if not product_id:
        return None

    # 1) shape 폴더에서
    if shape:
        shape_dir = SHAPE_DIR_MAP.get(str(shape).strip().lower())
        if shape_dir:
            base = os.path.join(frame_root, shape_dir, str(product_id).strip())
            for ext in EXTS:
                cp = base + ext
                if os.path.exists(cp):
                    return ensure_bgra(cp)

    # 2) 재귀 탐색
    pattern = os.path.join(frame_root, "**", str(product_id).strip() + ".*")
    for cp in glob.glob(pattern, recursive=True):
        if os.path.splitext(cp)[1].lower() in EXTS and os.path.isfile(cp):
            return ensure_bgra(cp)

    return None

