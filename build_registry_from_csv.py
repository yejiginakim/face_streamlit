# build_registry_from_csv.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

PNG_DIR  = Path('frames/images')
OUT_JSON = Path('glasses_registry.json')

def ensure_bgra(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        return cv2.merge([b, g, r, a])
    return img  # already BGRA

def auto_anchors_from_png(bgra):
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return None
    H, W = bgra.shape[:2]
    frame = (bgra[..., 3] > 0).astype(np.uint8)
    if frame.sum() < 1000:
        return None

    hsv = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    S, V = hsv[..., 1], hsv[..., 2]
    lens = ((S < 0.35) & (V > 0.15) & (V < 0.95)).astype(np.uint8)
    lens = cv2.bitwise_and(lens, frame)
    lens = cv2.morphologyEx(lens, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, lab, stats, cents = cv2.connectedComponentsWithStats(lens, 8)
    if n < 3:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    top2 = np.argsort(areas)[-2:][::-1] + 1
    comps = []
    for k in top2:
        x = stats[k, cv2.CC_STAT_LEFT]
        y = stats[k, cv2.CC_STAT_TOP]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]
        comps.append({'bbox': (x, y, w, h)})
    comps.sort(key=lambda d: d['bbox'][0])  # left → right
    left, right = comps

    y_left  = left['bbox'][1]  + left['bbox'][3] / 2.0
    y_right = right['bbox'][1] + right['bbox'][3] / 2.0
    y_ref = int(round((y_left + y_right) / 2.0))
    band = max(3, int(0.06 * H))
    band_mask = frame[max(0, y_ref - band):min(H, y_ref + band + 1), :]

    midx = int((left['bbox'][0] + left['bbox'][0] + left['bbox'][2] +
                right['bbox'][0] + right['bbox'][0] + right['bbox'][2]) / 4.0)

    xs_left = np.where(band_mask[:, :midx] > 0)[1]
    xs_right = np.where(band_mask[:, midx:] > 0)[1] + midx
    if xs_left.size == 0 or xs_right.size == 0:
        return None

    hxL = int(xs_left.min())
    hxR = int(xs_right.max())
    hy = int(y_ref)

    l_edge = left['bbox'][0] + left['bbox'][2]
    r_edge = right['bbox'][0]
    bx = int(round((l_edge + r_edge) / 2.0))
    by = hy

    def frac(x, y):
        return [float(x) / float(W), float(y) / float(H)]

    return {
        'left_hinge':  frac(hxL, hy),
        'right_hinge': frac(hxR, hy),
        'bridge':      frac(bx,  by)
    }

def hinge_mm(spec):
    cand = 2.0 * spec['lens'] + spec['bridge'] + spec.get('rim_allowance_mm', 4.0)
    if spec.get('total') is not None:
        return min(spec['total'], cand)
    return cand

def build_registry_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    for name in ['product_id', 'lens_mm', 'bridge_mm']:
        if name not in df.columns:
            raise ValueError(f'CSV에 {name} 컬럼이 필요해요.')

    items = []
    missing_png = []
    missing_anchor = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        pid = str(row['product_id']).strip()
        lens = float(row['lens_mm'])
        bridge = float(row['bridge_mm'])
        total = None
        if 'total_mm' in df.columns and pd.notna(row['total_mm']):
            total = float(row['total_mm'])

        png_path = PNG_DIR / f'{pid}.png'
        entry = {
            'model_id': pid,
            'image': f'{pid}.png',
            'spec_mm': {'lens': lens, 'bridge': bridge, 'total': total, 'rim_allowance_mm': 4.0},
            'anchors': None,
            'y_offset_ratio': 0.12
        }

        if not png_path.exists():
            missing_png.append(pid)
        else:
            bgra = ensure_bgra(png_path)
            anchors = auto_anchors_from_png(bgra)
            entry['anchors'] = anchors
            if anchors is None:
                missing_anchor.append(pid)

        items.append(entry)

    OUT_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'✅ 저장: {OUT_JSON} (총 {len(items)}개)')

    if missing_png:
        print('⚠️ PNG 없음(앞 10개만 표시):', missing_png[:10])
    if missing_anchor:
        print('⚠️ 앵커 자동 실패(앞 10개만 표시):', missing_anchor[:10])

if __name__ == '__main__':
    build_registry_from_csv('products.csv')  # 너희 CSV 경로
