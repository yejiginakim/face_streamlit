# metrics.py
import numpy as np, cv2, math
#import mediapipe as mp

mp_face = mp.solutions.face_mesh
_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                         refine_landmarks=False, min_detection_confidence=0.5)

# 중요한 랜드마크 인덱스 (좌·우 볼, 좌·우 턱각, 이마 중앙, 턱끝)
_IDX = {'cheek_L':234, 'cheek_R':454, 'jaw_left':172, 'jaw_right':397, 'forehead':10, 'chin':152}

# --- Face Oval(실루엣) 연결 상수: Mediapipe 내장 ---
from mediapipe.solutions.face_mesh import FACEMESH_FACE_OVAL

def _xy(lm, idx, w, h):
    p = lm[idx]; return np.array([p.x*w, p.y*h], dtype=np.float64)

def _dist(a,b):
    return float(np.linalg.norm(a-b))

def _ang(a,b,c):
    # 각도 ∠ABC (단위: deg)
    ba=a-b; bc=c-b
    d=(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    cosv = np.clip(np.dot(ba,bc)/d, -1.0, 1.0)
    return math.degrees(math.acos(cosv))

def _oval_indices():
    # FACEMESH_FACE_OVAL은 (i,j) 연결쌍 집합 → 고유 정점 집합으로 변환
    s=set()
    for i,j in FACEMESH_FACE_OVAL:
        s.add(i); s.add(j)
    return sorted(s)

def _width_profile_from_oval(oval_pts, bands=(0.30,0.50,0.75), delta_ratio=0.02):
    """
    oval_pts: [(x,y), ...] 이미지 좌표 (정면 정렬 전제 아님, 그래도 실무상 충분히 작동)
    bands: 상/중/하 비율 (y_min + r*H)
    delta_ratio: 해당 y 주변 허용폭(±delta)
    return: [w_top, w_mid, w_low]
    """
    if not oval_pts:
        return [float('nan')]*len(bands)

    ys = [p[1] for p in oval_pts]
    y_min, y_max = min(ys), max(ys)
    H = y_max - y_min if y_max>y_min else 1.0
    delta = H * float(delta_ratio)

    widths=[]
    for r in bands:
        y = y_min + float(r)*H
        xs = [p[0] for (x,y0) in oval_pts if abs(y0 - y) <= delta for p in [(x,y0)]]
        if xs:
            w = max(xs) - min(xs)
        else:
            w = float('nan')
        widths.append(float(w))
    return widths

def compute_metrics_bgr(img_bgr, extras=False, bands=(0.30,0.50,0.75), delta_ratio=0.02):
    """
    기본 반환(호출 호환): ar, jaw_deg, Cw, Jw
    extras=True 이면 (ar, jaw_deg, Cw, Jw, extra_dict) 를 반환:
        extra_dict = {
          'w_top':..., 'w_mid':..., 'w_low':..., 'ratio_low_mid':...,
          'Fw_approx': ...   # 상부 폭 근사치 (= w_top)
        }
    """
    try:
        h,w = img_bgr.shape[:2]
        res = _mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return (None, None, None, None) if not extras else (None, None, None, None, {})

        lm = res.multi_face_landmarks[0].landmark

        # --- 핵심 점들 ---
        pCL=_xy(lm,_IDX['cheek_L'],w,h); pCR=_xy(lm,_IDX['cheek_R'],w,h)
        pJL=_xy(lm,_IDX['jaw_left'],w,h); pJR=_xy(lm,_IDX['jaw_right'],w,h)
        pF=_xy(lm,_IDX['forehead'],w,h);  pC=_xy(lm,_IDX['chin'],w,h)

        # --- 기본 지표 ---
        cw=_dist(pCL,pCR)
        jw=_dist(pJL,pJR)
        face_len=_dist(pF,pC)
        ar=face_len/max(cw,1e-6)

        # 턱각: 양쪽 턱 코너의 내각 평균 (cheek–jaw–chin)
        jaw_left  = _ang(pCL, pJL, pC)
        jaw_right = _ang(pCR, pJR, pC)
        jaw_deg   = (jaw_left + jaw_right) / 2.0

        if not extras:
            return ar, jaw_deg, cw, jw

        # --- Face Oval 기반 폭 프로파일 ---
        ids_oval = _oval_indices()
        oval_pts = [tuple(_xy(lm,i,w,h)) for i in ids_oval]
        w_top, w_mid, w_low = _width_profile_from_oval(oval_pts, bands=bands, delta_ratio=delta_ratio)

        # 안정성: NaN 방지
        ratio_low_mid = float(w_low) / float(w_mid) if (w_mid and w_mid==w_mid and w_mid>1e-6) else float('nan')

        extra = {
            'w_top': float(w_top),
            'w_mid': float(w_mid),
            'w_low': float(w_low),
            'ratio_low_mid': float(ratio_low_mid),
            'Fw_approx': float(w_top)  # 상부 폭 근사(Forehead/temple 근처)
        }
        return ar, jaw_deg, cw, jw, extra

    except Exception:
        return (None, None, None, None) if not extras else (None, None, None, None, {})

