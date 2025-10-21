# faceshape.py — Keras3 로더 + 기존 규칙 유지 + 모델우선 스위칭

import os
from typing import Tuple, List, Optional

import numpy as np
import keras            # ✅ Keras 3
from PIL import Image

# ===== 하이퍼 =====
TH_MARGIN = 0.10
TEMP_UNCERT = 1.60

TH_AR_OBLONG, TH_JAW_SOFT = 1.50, 130.0
TH_AR_ROUND,  TH_JAW_ROUND = 1.25, 135.0
TH_CWJW_LOW,  TH_CWJW_HIGH = 0.95, 1.10

ROUND_BOOST, SQUARE_PENALTY = 1.40, 0.85
P_RND_DOWN, P_SQR_DOWN, P_OBL_DOWN, P_OVAL_UP = 0.60, 0.90, 0.70, 1.35
DELTA_LOCK, VETO_STRICT = 0.06, True

# ===== 규칙 스위치 기준(모델 확률 우선) =====
# 규칙 결과가 top2 밖일 때만 고려하고, 아래 두 조건을 모두 만족해야 스위치:
SWITCH_MIN_GAIN  = 0.06   # 규칙 확률 - 모델 top1 확률(절대 이득)
SWITCH_MIN_RATIO = 1.10   # 규칙 확률 / 모델 top1 확률(비율 이득)


class FaceShapeModel:
    def __init__(self, model_path: str, classes_path: str, img_size: Tuple[int, int]=(224, 224)):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f"classes not found: {classes_path}")

        # ✅ Keras3 저장 모델 로드 (Random*, Normalization 포함해도 OK)
        self.model = keras.saving.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )
        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [ln.strip() for ln in f if ln.strip()]
        self.img_size = tuple(img_size)

    def _preprocess(self, pil: Image.Image) -> np.ndarray:
        pil = pil.convert("RGB").resize(self.img_size)
        x = np.asarray(pil, dtype=np.float32)   # 모델 내부에 Rescaling/Normalization 있다고 가정
        return x[None, ...]                     # (1,H,W,3)

    def predict_probs(self, pil: Image.Image) -> np.ndarray:
        x = self._preprocess(pil)
        y = self.model.predict(x, verbose=0)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        y = np.asarray(y, np.float64)
        s = float(y.sum())
        return (y / s) if s > 0 else y


# ---------- 규칙 유틸 (기존 로직 유지) ----------
def _softmax_temp(x, t=1.0, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, 1.0)
    z = x ** (1.0 / max(t, 1e-6))
    return z / np.clip(z.sum(), eps, None)

def apply_rules(probs, class_names, ar=None, jaw_deg=None, cw=None, jw=None):
    p = np.asarray(probs, dtype=np.float64); p /= np.clip(p.sum(), 1e-12, None)
    top1 = int(p.argmax()); top2 = int(np.argsort(p)[-2])
    margin = float(p[top1]-p[top2])

    if margin < TH_MARGIN:
        p = _softmax_temp(p, t=TEMP_UNCERT)
        top1 = int(p.argmax()); top2 = int(np.argsort(p)[-2])
        margin = float(p[top1]-p[top2])

    idx = {n:i for i,n in enumerate(class_names)}

    # Oblong → Oval veto
    if ('Oblong' in idx) and ('Oval' in idx) and class_names[top1]=='Oblong':
        if (ar is not None and ar < TH_AR_OBLONG) and (jaw_deg is not None and jaw_deg >= TH_JAW_SOFT):
            boost = 1.75 if margin < TH_MARGIN else 1.35
            p[idx['Oval']] *= boost
            p[idx['Oblong']] *= 0.85

            cwjw = None if (cw in (None,) or jw in (None,0)) else float(cw)/float(jw)
            round_evidence = (('Round' in idx) and
                              (ar is not None and ar <= TH_AR_ROUND) and
                              (jaw_deg is not None and jaw_deg >= TH_JAW_ROUND) and
                              (cwjw is not None and TH_CWJW_LOW <= cwjw <= TH_CWJW_HIGH))

            if 'Round' in idx and not round_evidence and int(p.argmax()) == idx['Round']:
                p[idx['Round']] *= P_RND_DOWN
                p[idx['Oval']]  *= 1.10

            if 'Square' in idx:
                p[idx['Square']] *= P_SQR_DOWN

            p /= np.clip(p.sum(), 1e-12, None)
            top_after = int(p.argmax())
            if VETO_STRICT and not round_evidence:
                delta = float(p[top_after] - p[idx['Oval']])
                if delta < DELTA_LOCK:
                    p[idx['Oval']] *= P_OVAL_UP
                    p[top_after]   *= (P_OBL_DOWN if class_names[top_after]=='Oblong' else 0.75)
                    p /= np.clip(p.sum(), 1e-12, None)

            top1 = int(p.argmax())

    # Round 보정
    cwjw_ratio = None if (cw in (None,) or jw in (None,0)) else float(cw)/float(jw)
    round_candidate = (ar is not None and ar <= TH_AR_ROUND) and \
                      (jaw_deg is not None and jaw_deg >= TH_JAW_ROUND) and \
                      (cwjw_ratio is None or (TH_CWJW_LOW <= cwjw_ratio <= TH_CWJW_HIGH))
    if round_candidate and 'Round' in idx and class_names[top1] != 'Round':
        if 'Square' in idx and class_names[top1]=='Square' and (jaw_deg is not None and jaw_deg >= TH_JAW_ROUND):
            p[idx['Square']] *= SQUARE_PENALTY
        p[idx['Round']] *= ROUND_BOOST
        p /= np.clip(p.sum(), 1e-12, None)
        top1 = int(p.argmax())

    return {'rule_idx': top1, 'rule_label': class_names[top1], 'rule_probs': p}


def decide_rule_vs_top2(
    probs: np.ndarray,
    class_names: List[str],
    ar: Optional[float]=None, jaw_deg: Optional[float]=None,
    cw: Optional[float]=None, jw: Optional[float]=None
) -> Tuple[int, str, str]:
    """
    모델 확률 우선 정책:
      - margin >= TH_MARGIN → 무조건 모델 top1 유지
      - margin <  TH_MARGIN →
          * 규칙 결과가 {top1, top2} 안이면 → 모델 top1 유지
          * 규칙 결과가 top2 밖이면 → '충분한 이득'이 있을 때만 스위치
              (gain_abs ≥ SWITCH_MIN_GAIN and gain_ratio ≥ SWITCH_MIN_RATIO)
    """
    p = np.asarray(probs, dtype=np.float64)
    p /= np.clip(p.sum(), 1e-12, None)

    order = np.argsort(-p)
    top1, top2 = int(order[0]), int(order[1])
    margin = float(p[top1] - p[top2])

    # 1) 모델이 충분히 확신 → 모델 top1 고정
    if margin >= TH_MARGIN:
        return top1, class_names[top1], f"confident(model): margin={margin:.3f} ≥ {TH_MARGIN}"

    # 2) 규칙 적용
    rr = apply_rules(p, class_names, ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw)
    rule_probs = rr['rule_probs']
    rule_idx   = int(np.argmax(rule_probs))
    rule_label = class_names[rule_idx]

    # 3) 규칙 결과가 top2 안이면 → 모델 top1 유지 (모델 우선)
    if rule_idx in (top1, top2):
        return top1, class_names[top1], "uncertain: rule∈top2 → keep model top1"

    # 4) 규칙 결과가 top2 밖인데, '충분한 이득'이 있을 때만 스위치
    gain_abs   = float(rule_probs[rule_idx] - p[top1])
    gain_ratio = float(rule_probs[rule_idx] / max(p[top1], 1e-12))

    if (gain_abs >= SWITCH_MIN_GAIN) and (gain_ratio >= SWITCH_MIN_RATIO):
        return rule_idx, rule_label, (
            f"uncertain: rule outside top2 → switch (gain={gain_abs:.3f}, ratio={gain_ratio:.2f}×)"
        )

    # 5) 그 외엔 모델 top1 유지
    return top1, class_names[top1], "uncertain: insufficient rule gain → keep model top1"


# ===== Top-K 유틸 =====
def topk_from_probs(probs: np.ndarray, class_names: List[str], k: int = 2):
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    order = np.argsort(-p)[:k]
    return [(int(i), class_names[int(i)], float(p[int(i)])) for i in order]

def top2_strings(items):
    # items: [(idx, label, prob), ...]
    return [f"{label} ({prob*100:.1f}%)" for _, label, prob in items]


__all__ = [
    "FaceShapeModel",
    "apply_rules",
    "decide_rule_vs_top2",
    "topk_from_probs",
    "top2_strings",
]

