# faceshape.py — Keras3 로더 (보정 없음, 모델 결과 우선, Top-2 유틸 포함)

import os
from typing import Tuple, List, Optional
import numpy as np
import keras          # ✅ Keras 3
from PIL import Image


class FaceShapeModel:
    def __init__(self, model_path: str, classes_path: str, img_size: Tuple[int, int]=(224, 224)):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f"classes not found: {classes_path}")

        # Keras 3 저장 모델 로드 (Random*, Normalization 층 포함되어도 OK)
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


# ========== (호환용 더미: 규칙 보정 제거) ==========
def apply_rules(probs, class_names, **kwargs):
    """
    규칙 보정 없이 그대로 반환 (호환용 더미).
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    top = int(np.argmax(p))
    return {'rule_idx': top, 'rule_label': class_names[top], 'rule_probs': p}


def decide_rule_vs_top2(
    probs: np.ndarray,
    class_names: List[str],
    ar: Optional[float]=None, jaw_deg: Optional[float]=None,
    cw: Optional[float]=None, jw: Optional[float]=None
):
    """
    규칙/보정 없이 모델 top-1만 반환.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    top = int(np.argmax(p))
    return top, class_names[top], "model-top1 (no rules)"


# ========== Top-K 유틸 ==========
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
    "apply_rules",          # 호환용 (보정 없음)
    "decide_rule_vs_top2",  # 호환용 (모델 top1 고정)
    "topk_from_probs",
    "top2_strings",
]

