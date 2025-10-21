# metrics.py
import numpy as np, cv2, math

# ----------------- 기본 유틸 -----------------
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

def _width_profile_from_oval(oval_pts, bands=(0.30,0.50,0.75), delta_ratio=0.02):
    """
    oval_pts: [(x,y), ...]
    bands: y_min + r*H 에서 폭 측정
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
        xs = [x for (x,y0) in oval_pts if abs(y0 - y) <= delta]
        w = (max(xs) - min(xs)) if xs else float('nan')
        widths.append(float(w))
    return widths

# 중요 랜드마크(좌·우 볼, 좌·우 턱각, 이마 중앙, 턱끝)
_IDX = {'cheek_L':234, 'cheek_R':454, 'jaw_left':172, 'jaw_right':397, 'forehead':10, 'chin':152}

# ----------------- 핵심 함수 -----------------
def compute_metrics_bgr(img_bgr, extras=False, bands=(0.30,0.50,0.75), delta_ratio=0.02):
    """
    반환(호환):
      extras=False -> (ar, jaw_deg, Cw, Jw)
      extras=True  -> (ar, jaw_deg, Cw, Jw, extra_dict)
        extra_dict = {
          'w_top':..., 'w_mid':..., 'w_low':..., 'ratio_low_mid':...,
          'Fw_approx': ...
        }
    """
    # 0) mediapipe를 함수 내부에서 지연 임포트
    try:
        import mediapipe as mp
        from mediapipe.solutions.face_mesh import FACEMESH_FACE_OVAL
    except Exception as e:
        # 설치/환경 문제 시, 호출부에서 잡아 UI로 안내할 수 있게 예외를 던짐
        raise ImportError(
            f"mediapipe 임포트 실패: {e}. "
            f"requirements.txt에 'mediapipe==0.10.14'가 포함되어 있는지 확인하세요."
        )

    # 1) FaceMesh는 with-context로 생성/해제 (리소스 누수 방지)
    try:
        h,w = img_bgr.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5
        ) as mesh:
            res = mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

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
        ids_oval = sorted({i for (i, j) in FACEMESH_FACE_OVAL} | {j for (i, j) in FACEMESH_FACE_OVAL})
        oval_pts = [tuple(_xy(lm,i,w,h)) for i in ids_oval]
        w_top, w_mid, w_low = _width_profile_from_oval(oval_pts, bands=bands, delta_ratio=delta_ratio)

        ratio_low_mid = (
            float(w_low) / float(w_mid)
            if (w_mid and w_mid == w_mid and w_mid > 1e-6) else float('nan')
        )

        extra = {
            'w_top': float(w_top),
            'w_mid': float(w_mid),
            'w_low': float(w_low),
            'ratio_low_mid': float(ratio_low_mid),
            'Fw_approx': float(w_top),
        }
        return ar, jaw_deg, cw, jw, extra

    except ImportError:
        # 위에서 재던진 임포트 에러는 그대로 밖으로
        raise
    except Exception:
        # 기타 예외는 조용히 실패 처리
        return (None, None, None, None) if not extras else (None, None, None, None, {})

