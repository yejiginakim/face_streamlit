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
# ========== 규칙 보정 + 재랭킹 ==========
import numpy as np

# 내부 유틸: 클래스명 → 인덱스 매핑
def _name_to_index_map(class_names):
    return {name: i for i, name in enumerate(class_names)}

def _safe_log_probs(probs, T=1.8):
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    return np.log(p + 1e-12) / float(T)

def _softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / np.clip(s, 1e-12, None)

def apply_rules(
    probs, class_names,
    ar=None, jaw_deg=None, cw=None, jw=None,
    # --- 하드/소프트 기준 기본값 (필요시 스트림릿에서 인자만 바꿔 호출) ---
    square_tol=0.10,      # 스퀘어 허용 오차(소프트 경고): |Cw-Jw|/Cw ≤ 10%
    square_hard=0.15,     # 스퀘어 하드 VETO: |Cw-Jw|/Cw > 15%면 후보 제외
    oblong_ar_cut=1.35,   # 오블롱 하드 VETO: AR < 1.35면 제외
    heart_jaw_hi=131.0,   # Heart 하향 기준: jaw_deg > 131° (턱이 뾰족하지 않음)
    round_up_gain=2.4,    # Round 상향 가중 (>1이면 상향)
    oval_up_gain=1.6,     # Oval  상향 가중
    heart_down_gain=0.20, # Heart 하향 가중 (0~1이면 하향)
    T=1.8,                # 온도 스케일링(과잉확신 완화)
    strict_veto=True,     # True면 하드 VETO는 확실히 제외(-1e9)
):
    """
    probs: 모델 원래 확률 벡터(또는 리스트)
    class_names: 모델의 클래스 이름 리스트(예: ['Oval','Round','Square','Oblong','Heart'])
    ar, jaw_deg, cw, jw: 선택 입력(없으면 규칙 완화/미적용)
    반환: dict(rule_idx, rule_label, rule_probs(np.ndarray))
    """
    names = list(class_names)
    idx = _name_to_index_map(names)

    # 클래스명 안전 접근(없는 경우 None)
    i_sq = idx.get('Square')
    i_ht = idx.get('Heart')
    i_ob = idx.get('Oblong')
    i_ov = idx.get('Oval')
    i_rd = idx.get('Round')

    # 0) 로그-확률(온도 스케일링)
    logits = _safe_log_probs(probs, T=T)

    # 1) 하드 VETO
    #    - 수치가 없으면 적용 생략 (None)
    if cw is not None and jw is not None and i_sq is not None:
        cw_jw_gap = abs(cw - jw) / max(cw, 1e-8)
        if cw_jw_gap > square_hard and strict_veto:
            logits[i_sq] = -1e9  # Square 제외
    if ar is not None and i_ob is not None:
        if float(ar) < float(oblong_ar_cut) and strict_veto:
            logits[i_ob] = -1e9  # Oblong 제외

    # 2) 소프트 가중(로그-도메인 더하기)
    add = np.zeros_like(logits)

    # Heart down: 턱각이 큰(둥근) 편이면 하향
    if (jaw_deg is not None) and (i_ht is not None):
        if float(jaw_deg) > float(heart_jaw_hi):
            add[i_ht] += np.log(max(heart_down_gain, 1e-6))  # 0~1 사이면 하향

    # Round/Oval up: AR 짧고 턱각 큰 편이면 상향
    if (ar is not None) and (jaw_deg is not None):
        if float(ar) < 1.25 and float(jaw_deg) > 132.0:
            if i_rd is not None:
                add[i_rd] += np.log(max(round_up_gain, 1e-6))
            if i_ov is not None:
                add[i_ov] += np.log(max(oval_up_gain, 1e-6))

    # Square soft down: 하드 VETO까지는 아니지만 살짝 큰 경우(옵션)
    if (cw is not None and jw is not None and i_sq is not None and not strict_veto):
        cw_jw_gap = abs(cw - jw) / max(cw, 1e-8)
        if square_tol < cw_jw_gap <= square_hard:
            add[i_sq] += np.log(0.3)  # 약한 하향

    # 3) 재정규화
    adj = logits + add
    adj_p = _softmax(adj)

    # 4) 결과
    rule_top = int(np.argmax(adj_p))
    return {
        'rule_idx': rule_top,
        'rule_label': names[rule_top],
        'rule_probs': adj_p
    }

def decide_rule_vs_top2(
    probs, class_names,
    ar=None, jaw_deg=None, cw=None, jw=None,
    **rule_kwargs
):
    """
    규칙 보정 적용 → 최종 top-1 레이블과 '이유 태그' 반환.
    rule_kwargs는 apply_rules의 파라미터 튜닝용으로 그대로 전달됨.
    """
    # 규칙 인자 중 하나라도 제공되면 보정 적용, 아니면 모델 top-1
    use_rules = any(v is not None for v in (ar, jaw_deg, cw, jw))

    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)

    if use_rules:
        out = apply_rules(
            p, class_names,
            ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw,
            **rule_kwargs
        )
        return out['rule_idx'], out['rule_label'], 'rules+model'
    else:
        top = int(np.argmax(p))
        return top, class_names[top], 'model-top1 (no rules)'


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

