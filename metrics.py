# metrics.py
import numpy as np, cv2, math
import mediapipe as mp

mp_face = mp.solutions.face_mesh
_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                         refine_landmarks=False, min_detection_confidence=0.5)

# 중요한 랜드마크 인덱스 (좌·우 볼, 좌·우 턱각, 이마 중앙, 턱끝)
_IDX = {'cheek_L':234, 'cheek_R':454, 'jaw_left':172, 'jaw_right':397, 'forehead':10, 'chin':152}

def _xy(lm, idx, w, h):
    p = lm[idx]; return np.array([p.x*w, p.y*h], dtype=np.float64)
def _dist(a,b): return float(np.linalg.norm(a-b))
def _ang(a,b,c):
    ba=a-b; bc=c-b
    d=(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    return math.degrees(math.acos(np.clip(np.dot(ba,bc)/d,-1.0,1.0)))

def compute_metrics_bgr(img_bgr):
    try:
        h,w = img_bgr.shape[:2]
        res = _mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None, None, None
        lm = res.multi_face_landmarks[0].landmark
        pCL=_xy(lm,_IDX['cheek_L'],w,h); pCR=_xy(lm,_IDX['cheek_R'],w,h)
        pJL=_xy(lm,_IDX['jaw_left'],w,h); pJR=_xy(lm,_IDX['jaw_right'],w,h)
        pF=_xy(lm,_IDX['forehead'],w,h);  pC=_xy(lm,_IDX['chin'],w,h)
        cw=_dist(pCL,pCR); jw=_dist(pJL,pJR); face_len=_dist(pF,pC)
        ar=face_len/max(cw,1e-6)
        jaw=( _ang(pCL,pJL,pC) + _ang(pCR,pJR,pC) )/2.0
        return ar, jaw, cw, jw
    except Exception:
        return None, None, None, None
