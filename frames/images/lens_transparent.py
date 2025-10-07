#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def to_rgba(path: Path):
    img = Image.open(path).convert('RGBA')
    arr = np.array(img)            # H,W,4 (RGBA)
    return arr

def ensure_foreground_alpha(arr):
    """알파가 전부 255인 경우(배경이 합성 안된 PNG) 대비:
       흰 배경 추정해서 전경 마스크 생성 후 외곽은 알파 0으로 만든다."""
    a = arr[...,3]
    if np.any(a < 255):
        return arr  # 이미 투명 배경
    rgb = arr[...,:3].astype(np.float32)/255.0
    # 밝은 배경(흰색) 가정: 매우 밝고 채도 낮은 픽셀을 배경으로
    mx, mn = rgb.max(-1), rgb.min(-1)
    val, sat = mx, (mx-mn)/(mx+1e-6)
    bg = (val > 0.98) & (sat < 0.05)
    # 외곽에서 안쪽으로 퍼뜨림
    bg_u8 = (bg.astype(np.uint8)*255)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    out = arr.copy()
    out[...,3] = np.where(bg_u8>0, 0, 255).astype(np.uint8)
    return out

def kmeans2(X, max_iters=25):
    # X: (N,3) float32 [0..1]
    v = X.mean(axis=1)
    c0 = X[np.argmin(v)]
    c1 = X[np.argmax(v)]
    C = np.stack([c0, c1], axis=0)
    for _ in range(max_iters):
        d0 = np.sum((X - C[0])**2, axis=1)
        d1 = np.sum((X - C[1])**2, axis=1)
        lab = (d1 < d0).astype(np.int8)
        newC = []
        for k in (0,1):
            pts = X[lab==k]
            newC.append(C[k] if len(pts)==0 else pts.mean(axis=0))
        newC = np.stack(newC, axis=0)
        if np.allclose(newC, C): break
        C = newC
    return lab, C

def decide_clusters(rgb, fg, lab_full):
    # HSV 근사
    mx, mn = rgb.max(-1), rgb.min(-1)
    V = mx
    S = (mx - mn) / (mx + 1e-6)

    # 엣지/테두리 링
    edges = cv2.Canny((V*255).astype(np.uint8), 50, 150) > 0
    fg_u8 = (fg.astype(np.uint8) * 255)
    dil = cv2.dilate(fg_u8, np.ones((3,3), np.uint8), iterations=2) > 0
    ero = cv2.erode (fg_u8, np.ones((3,3), np.uint8), iterations=2) > 0
    ring = dil & (~ero)

    scores = []
    for k in (0,1):
        mk = np.zeros_like(fg, dtype=bool)
        mk[fg] = (lab_full == k)
        if mk.sum()==0:
            scores.append(-1); continue
        area_ratio   = mk.sum()/fg.sum()
        meanV        = V[mk].mean()
        meanS        = S[mk].mean()
        edge_overlap = (edges & mk).sum() / mk.sum()
        border_ratio = (ring  & mk).sum() / mk.sum()
        dark = 1.0 - float(meanV)
        score = (0.45*dark +
                 0.25*edge_overlap +
                 0.15*border_ratio +
                 0.10*(1.0 - area_ratio) +
                 0.05*float(meanS))
        scores.append(score)
    frame_cluster = int(np.argmax(scores))
    lens_cluster  = 1 - frame_cluster
    return frame_cluster, lens_cluster

def build_lens_mask(arr):
    rgba = arr.copy()
    rgb = rgba[...,:3].astype(np.float32)/255.0
    a   = rgba[...,3]
    fg = a > 0

    X = rgb[fg].reshape(-1,3)
    if X.size == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)

    labels, centers = kmeans2(X, max_iters=25)
    lab_full = np.zeros(fg.sum(), dtype=np.int8); lab_full[:] = labels
    frame_cluster, lens_cluster = decide_clusters(rgb, fg, lab_full)

    lens_mask = np.zeros_like(fg, dtype=bool)
    lens_mask[fg] = (lab_full == lens_cluster)

    # 코받침(밝고 무채색) 제외
    mx, mn = rgb.max(-1), rgb.min(-1)
    val, sat = mx, (mx-mn)/(mx+1e-6)
    nosepad = (val > 0.85) & (sat < 0.15)
    lens_mask &= ~nosepad

    # 가장자리 살짝 수축
    lens_u8 = (lens_mask.astype(np.uint8)*255)
    lens_u8 = cv2.erode(lens_u8, np.ones((3,3), np.uint8), iterations=1)
    return lens_u8 > 0

def apply_alpha(arr, lens_mask, alpha_scale):
    out = arr.copy()
    a = out[...,3]
    a[lens_mask] = (a[lens_mask].astype(np.float32)*alpha_scale).astype(np.uint8)
    out[...,3] = a
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--alpha', type=float, default=0.55, help='렌즈 투명도 스케일(0~1)')
    args = ap.parse_args()

    src = to_rgba(Path(args.in_path))
    src = ensure_foreground_alpha(src)
    lens_mask = build_lens_mask(src)
    out = apply_alpha(src, lens_mask, args.alpha)

    Image.fromarray(out, mode='RGBA').save(args.out_path)
    h,w = lens_mask.shape
    print(f"✅ saved: {args.out_path}  (size: {w}x{h}, lens pixels: {int(lens_mask.sum())})")

if __name__ == '__main__':
    main()
