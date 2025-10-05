# vision.py
import os, csv
import numpy as np
import cv2
import streamlit as st  # cache_resource 사용
# MediaPipe는 내부에서 import (env 초기화 비용 줄이기)

FRAMES_DIR  = "frames/images"
CSV_PATH    = "frames/frames.csv"

@st.cache_resource
def create_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

def detect_pd_px(bgr):
    """눈 중심 기반 PD_px, 눈선 각도, 중점(mid) 구하기"""
    fm = create_facemesh()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0].landmark
    L_ids = [33,133,159,145]
    R_ids = [362,263,386,374]
    L = np.array([[lm[i].x*w, lm[i].y*h] for i in L_ids], np.float32).mean(axis=0)
    R = np.array([[lm[i].x*w, lm[i].y*h] for i in R_ids], np.float32).mean(axis=0)
    pd_px  = float(np.hypot(*(R-L)))
    angle  = float(np.degrees(np.arctan2(R[1]-L[1], R[0]-L[0])))
    mid    = ((L[0]+R[0])/2.0, (L[1]+R[1])/2.0)
    return pd_px, angle, mid

def overlay_rgba(bg_bgr, fg_rgba, x, y):
    """bg_bgr(BGR)에 fg_rgba(RGBA)를 (x,y) 상단 기준 알파합성"""
    H,W = bg_bgr.shape[:2]
    h,w = fg_rgba.shape[:2]
    x0,y0 = max(x,0), max(y,0)
    x1,y1 = min(x+w, W), min(y+h, H)
    if x0>=x1 or y0>=y1:
        return bg_bgr
    fg_cut = fg_rgba[y0-y:y1-y, x0-x:x1-x]
    alpha  = (fg_cut[:, :, 3:4].astype(np.float32) / 255.0)
    bg_roi = bg_bgr[y0:y1, x0:x1, :].astype(np.float32)
    fg_rgb = fg_cut[:, :, :3].astype(np.float32)
    out    = fg_rgb*alpha + bg_roi*(1-alpha)
    bg_bgr[y0:y1, x0:x1, :] = out.astype(np.uint8)
    return bg_bgr

def load_frames_csv(csv_path=CSV_PATH):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "filename": r["filename"],
                    "A": float(r["A"]),
                    "DBL": float(r["DBL"]),
                    "TOTAL": float(r.get("TOTAL") or 0),
                    "B": float(r.get("B") or 0),
                })
            except Exception:
                continue
    return rows

def load_png_from_choice(picked_row, frames_dir=FRAMES_DIR):
    """CSV 행에서 filename으로 RGBA PNG 로드 & 치수 튜플 반환"""
    if not picked_row:
        return None, None
    path = os.path.join(frames_dir, picked_row["filename"])
    if not os.path.exists(path):
        return None, None
    png = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # RGBA 기대
    if png is None or png.ndim != 3 or png.shape[2] != 4:
        return None, None
    A = picked_row["A"]; DBL = picked_row["DBL"]
    TOTAL = picked_row["TOTAL"]; B = picked_row["B"]
    return png, (A, DBL, TOTAL, B)
