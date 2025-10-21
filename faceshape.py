# faceshape.py
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from PIL import Image

# EfficientNet 계열 전처리가 있으면 사용, 없으면 [-1,1] 스케일 폴백
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    preprocess_input = None

# 커스텀/Lambda 활성화 대비(저장 시 swish/gelu 등)
_CUSTOM_OBJECTS = {
    "swish": tf.nn.swish,
    "gelu": getattr(tf.nn, "gelu", tf.keras.activations.gelu),
}

# ===== 하이퍼 =====
TH_MARGIN = 0.10
TEMP_UNCERT = 1.60

TH_AR_OBLONG, TH_JAW_SOFT = 1.50, 130.0
TH_AR_ROUND,  TH_JAW_ROUND = 1.25, 135.0
TH_CWJW_LOW,  TH_CWJW_HIGH = 0.95, 1.10

ROUND_BOOST, SQUARE_PENALTY = 1.40, 0.85
P_RND_DOWN, P_SQR_DOWN, P_OBL_DOWN, P_OVAL_UP = 0.60, 0.90, 0.70, 1.35
DELTA_LOCK, VETO_STRICT = 0.06, True


class FaceShapeModel:
    def __init__(self, model_path: str, classes_path: str, img_size=(224, 224)):
        # 클래스 로드 (훈련 당시 순서 유지)
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [x.strip() for x in f if x.strip()]
        self.num_classes = len(self.class_names)
        self.img_size = tuple(img_size)

        # 모델 로드: tf.keras 우선, 실패 시 safe_mode=False + custom_objects 폴백
        self.model = self._load_checkpoint(model_path)

    def _load_checkpoint(self, model_path: str):
        # 1차: 표준 로드 (tf.keras)
        try:
            return tfk.models.load_model(model_path, compile=False)
        except Exception as e1:
            # 2차: 직렬화 불일치(Keras3/커스텀 등) 대비 폴백
            try:
                return tfk.models.load_model(
                    model_path,
                    compile=False,
                    safe_mode=False,
                    custom_objects=_CUSTOM_OBJECTS,
                )
            except Exception as e2:
                raise RuntimeError(
                    "Failed to load model.\n"
                    f"Path: {model_path}\n"
                    f"1) {type(e1).__name__}: {e1}\n"
                    f"2) {type(e2).__name__}: {e2}\n"
                    "Tip: Ensure it was saved with tf.keras and you're loading with tf.keras."
                )

    def _preprocess_pil(self, pil_img: Image.Image):
        pil_resized = pil_img.resize(self.img_size)
        arr = tfk.utils.img_to_array(pil_resized)[None, ...]  # (1,H,W,3)
        if preprocess_input is not None:
            arr = preprocess_input(arr)
        else:
            # Fallback: 0~255 -> [-1,1]
            arr = (arr / 127.5) - 1.0
        return arr

    def predict_probs(self, pil_img: Image.Image):
        x = self._preprocess_pil(pil_img)
        p = self.model.predict(x, verbose=0)[0]
        p = np.asarray(p, dtype=np.float64)
        p = p / np.clip(p.sum(), 1e-12, None)
        return p


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

    # Oblong → Oval veto (조건 충족 시)
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

    # Round 보정 (근거 있을 때만)
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


def decide_rule_vs_top2(probs, class_names, ar=None, jaw_deg=None, cw=None, jw=None):
    """
    정책:
      - margin >= TH_MARGIN → top1 유지
      - margin <  TH_MARGIN →
          * rule 결과가 {top1, top2} 안이면 → top1 유지
          * rule 결과가 top2 밖이면 → rule 결과 채택
    """
    p = np.asarray(probs, dtype=np.float64); p /= np.clip(p.sum(), 1e-12, None)
    order = np.argsort(-p)
    top1, top2 = int(order[0]), int(order[1])
    margin = float(p[top1] - p[top2])

    if margin >= TH_MARGIN:
        return top1, class_names[top1], f"confident: margin={margin:.3f} ≥ {TH_MARGIN}"

    rr = apply_rules(p, class_names, ar=ar, jaw_deg=jaw_deg, cw=cw, jw=jw)
    rule_idx = rr['rule_idx']; rule_label = rr['rule_label']

    if rule_idx in (top1, top2):
        return top1, class_names[top1], f"uncertain: rule in top2 → keep top1 ({class_names[top1]})"
    return rule_idx, rule_label, f"uncertain: rule outside top2 → use rule ({rule_label})"

