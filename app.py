# vision.py (추가 함수만)
import numpy as np
import cv2

def nose_chin_length_px(bgr):
    """
    Mediapipe Facemesh로 코끝(landmark 1) ↔ 턱(152) 픽셀 거리.
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
        nose = np.array([lm[1].x * w,   lm[1].y * h], dtype=np.float32)
        chin = np.array([lm[152].x * w, lm[152].y * h], dtype=np.float32)
        return float(np.linalg.norm(chin - nose))
    except Exception:
        return None

