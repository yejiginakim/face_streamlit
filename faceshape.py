# faceshape.py — Keras3 로더 + Hard VETO 스캐닝 + Top-K 유틸 (보정 가중 없음)

import os
from typing import Tuple, List, Optional, Dict
import numpy as np
import keras  # Keras 3
from PIL import Image


# =========================
# 1) 모델 로더
# =========================
class FaceShapeModel:
    def __init__(self, model_path: str, classes_path: str, img_size: Tuple[int, int]=(224, 224)):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'model not found: {model_path}')
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f'classes not found: {classes_path}')

        # Keras 3 저장 모델 로드 (Random*, Normalization 층 포함되어도 OK)
        self.model = keras.saving.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )

        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [ln.strip() for ln in f if ln.strip()]

        self.img_size = tuple(img_size)

    def _preprocess(self, pil: Image.Image) -> np.ndarray:
        pil = pil.convert('RGB').resize(self.img_size)
        x = np.asarray(pil, dtype=np.float32)  # 모델 안에 Rescaling/Normalization 있다고 가정
        return x[None, ...]                    # (1,H,W,3)

    def predict_probs(self, pil: Image.Image) -> np.ndarray:
        x = self._preprocess(pil)
        y = self.model.predict(x, verbose=0)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        y = np.asarray(y, np.float64)
        s = float(y.sum())
        return (y / s) if s > 0 else y


# =========================
# 2) Top-K 유틸
# =========================
def topk_from_probs(probs: np.ndarray, class_names: List[str], k: int = 2):
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    order = np.argsort(-p)[:k]
    return [(int(i), class_names[int(i)], float(p[int(i)])) for i in order]


def top2_strings(items):
    # items: [(idx, label, prob), ...]
    return [f'{label} ({prob*100:.1f}%)' for _, label, prob in items]


# =========================
# 3) Hard VETO 로직
# =========================
def _veto_check(
    label: str,
    *,
    ar: Optional[float]=None,          # aspect ratio: 세로/가로
    jaw_deg: Optional[float]=None,     # 턱각(도)
    cw: Optional[float]=None,          # cheek width
    jw: Optional[float]=None,          # jaw width
    # ---- 임계값(“너무 아니면 제외”) ----
    square_gap_hard: float = 0.15,     # Square: |Cw-Jw|/max(Cw,eps) > 0.15 → 제외
    oblong_ar_min: float   = 1.35,     # Oblong: AR < 1.35 → 제외
    heart_jaw_hi: float    = 135.0,    # Heart: 턱각이 너무 크면(둥글면) 제외 (옵션)
    enable_heart_veto: bool = False,   # Heart 제외 규칙을 쓸지 여부(기본 끔)
) -> Tuple[bool, Optional[str]]:
    """
    반환: (통과여부, 제외사유문구)
    """
    # Square: 광대-턱 폭 차이가 너무 크면 Square 아님
    if label == 'Square' and cw is not None and jw is not None:
        gap = abs(cw - jw) / max(cw, 1e-8)
        if gap > square_gap_hard:
            return False, f'Square VETO: |Cw-Jw|/Cw={gap:.2f} > {square_gap_hard:.2f}'

    # Oblong: 세로/가로비가 낮으면 Oblong 아님
    if label == 'Oblong' and ar is not None:
        if ar < oblong_ar_min:
            return False, f'Oblong VETO: AR={ar:.2f} < {oblong_ar_min:.2f}'

    # Heart: 턱각이 너무 크면(뾰족X) Heart 아님 (옵션)
    if enable_heart_veto and label == 'Heart' and jaw_deg is not None:
        if jaw_deg > heart_jaw_hi:
            return False, f'Heart VETO: jaw_deg={jaw_deg:.1f}° > {heart_jaw_hi:.1f}°'

    return True, None


def pick_with_veto(
    probs: np.ndarray,
    class_names: List[str],
    *,
    ar: Optional[float]=None,
    jaw_deg: Optional[float]=None,
    cw: Optional[float]=None,
    jw: Optional[float]=None,
    # 임계값 파라미터
    square_gap_hard: float = 0.15,
    oblong_ar_min: float   = 1.35,
    heart_jaw_hi: float    = 135.0,
    enable_heart_veto: bool = False,
    k_view: int = 5,  # 몇 위까지 검사할지(보통 3~5)
) -> Dict[str, object]:
    """
    확률 내림차순으로 보며 VETO에 걸린 라벨은 제외, 통과하는 첫 라벨을 채택.
    모두 제외되면 원본 top-1로 폴백.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)

    order = np.argsort(-p)[:max(k_view, 1)]
    excluded = []

    for i in order:
        label = class_names[int(i)]
        ok, reason = _veto_check(
            label,
            ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw,
            square_gap_hard=square_gap_hard,
            oblong_ar_min=oblong_ar_min,
            heart_jaw_hi=heart_jaw_hi,
            enable_heart_veto=enable_heart_veto,
        )
        if ok:
            return {
                'final_idx': int(i),
                'final_label': label,
                'final_prob': float(p[int(i)]),
                'excluded': excluded,  # [(label, prob, reason), ...]
                'mode': 'model→hard-veto-scan'
            }
        else:
            excluded.append((label, float(p[int(i)]), reason))

    # 폴백: 다 제외되면 원본 1등 반환(로그 남김)
    top = int(np.argmax(p))
    return {
        'final_idx': top,
        'final_label': class_names[top],
        'final_prob': float(p[top]),
        'excluded': excluded,
        'mode': 'fallback_to_model_top1_all_vetoed'
    }


# =========================
# 4) 최종 의사결정 래퍼 (호환)
# =========================
def decide_final(
    probs: np.ndarray,
    class_names: List[str],
    *,
    ar: Optional[float]=None,
    jaw_deg: Optional[float]=None,
    cw: Optional[float]=None,
    jw: Optional[float]=None,
    square_gap_hard: float = 0.15,
    oblong_ar_min: float   = 1.35,
    heart_jaw_hi: float    = 135.0,
    enable_heart_veto: bool = False,
    k_view: int = 5,
) -> Dict[str, object]:
    """
    측정값이 하나라도 있으면 Hard VETO 스캔 사용,
    없으면 모델 Top-1을 그대로 반환.
    """
    use_rules = any(v is not None for v in (ar, jaw_deg, cw, jw))
    if not use_rules:
        p = np.asarray(probs, dtype=np.float64)
        p = p / np.clip(p.sum(), 1e-12, None)
        top = int(np.argmax(p))
        return {
            'final_idx': top,
            'final_label': class_names[top],
            'final_prob': float(p[top]),
            'excluded': [],
            'mode': 'model-top1 (no measures)'
        }
    return pick_with_veto(
        probs, class_names,
        ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw,
        square_gap_hard=square_gap_hard,
        oblong_ar_min=oblong_ar_min,
        heart_jaw_hi=heart_jaw_hi,
        enable_heart_veto=enable_heart_veto,
        k_view=k_view,
    )


# --- 과거 코드와의 호환용 별칭 (원래 decide_rule_vs_top2를 쓰던 호출부 보호) ---
def decide_rule_vs_top2(
    probs, class_names, ar=None, jaw_deg=None, cw=None, jw=None, **kwargs
):
    """
    과거 함수명 호환: 내부적으로 decide_final(Hard VETO)로 대체한다.
    반환은 (idx, label, reason_tag) 튜플로 간소화.
    """
    out = decide_final(
        probs, class_names,
        ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw,
        square_gap_hard=kwargs.get('square_gap_hard', 0.15),
        oblong_ar_min=kwargs.get('oblong_ar_min', 1.35),
        heart_jaw_hi=kwargs.get('heart_jaw_hi', 135.0),
        enable_heart_veto=kwargs.get('enable_heart_veto', False),
        k_view=kwargs.get('k_view', 5),
    )
    return out['final_idx'], out['final_label'], out['mode']


__all__ = [
    'FaceShapeModel',
    'topk_from_probs',
    'top2_strings',
    'pick_with_veto',
    'decide_final',
    'decide_rule_vs_top2',  # 호환용
]

