# faceshape.py — PyTorch 로더 + Top-K 유틸 (룰/베토 완전 제거)
import os
from typing import Tuple, List, Optional, Callable, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


class FaceShapeModel:
    """
    PyTorch 분류 모델 로더.
    - TorchScript(.pt/.pth scripted/trace): torch.jit.load()
    - state_dict(.pth): model_builder 콜러블로 모델 생성 후 load_state_dict()
    - predict_probs() -> (C,) 확률 numpy.float64 반환
    """
    def __init__(
        self,
        model_path: str,
        classes_path: str,
        img_size: Tuple[int, int]=(224, 224),
        device: Optional[Union[str, torch.device]] = None,
        model_builder: Optional[Callable[[], nn.Module]] = None,
        normalize: Optional[str] = 'imagenet',  # 'imagenet' 또는 None
        half: bool = False,                     # GPU에서 float16 추론
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'model not found: {model_path}')
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f'classes not found: {classes_path}')

        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [ln.strip() for ln in f if ln.strip()]

        self.img_size = tuple(img_size)
        self.normalize = normalize
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) TorchScript 우선 시도
        self.model: nn.Module
        try:
            self.model = torch.jit.load(model_path, map_location='cpu')
        except Exception:
            # 2) state_dict 로드
            if model_builder is None:
                raise ValueError(
                    "state_dict로 저장된 가중치로 보입니다. model_builder를 넘겨주세요 "
                    "(예: lambda: torchvision.models.efficientnet_b4(num_classes=5))"
                )
            state = torch.load(model_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.model = model_builder()
            self.model.load_state_dict(state, strict=False)

        self.model.eval().to(self.device)

        self.use_half = bool(half and self.device.type == 'cuda')
        if self.use_half:
            try:
                self.model.half()
            except Exception:
                self.use_half = False

    def _preprocess(self, pil: Image.Image) -> torch.Tensor:
        pil = pil.convert('RGB').resize(self.img_size)
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # [0,1], HWC
        arr = np.transpose(arr, (2, 0, 1))               # CHW

        if self.normalize == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std

        t = torch.from_numpy(arr)  # float32
        if self.use_half:
            t = t.half()
        return t.unsqueeze(0)      # (1,3,H,W)

    def predict_probs(self, pil: Image.Image) -> np.ndarray:
        """
        모델 출력이 logits이든 probs든 softmax로 확률화.
        반환: (C,) np.float64, 합=1
        """
        x = self._preprocess(pil).to(self.device, non_blocking=True)
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            y = out.squeeze(0)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            if y.dim() == 1:
                probs = torch.softmax(y.float(), dim=0)
            else:
                probs = torch.softmax(y.float(), dim=1).squeeze(0)
            p = probs.detach().cpu().numpy().astype(np.float64)
            s = float(p.sum())
            return (p / s) if s > 0 else p

    def predict_topk(self, pil: Image.Image, k: int = 2):
        """편의 함수: 모델 확률 기준 Top-K 반환"""
        probs = self.predict_probs(pil)
        return topk_from_probs(probs, self.class_names, k=k)


# ---------- Top-K 유틸 ----------
def topk_from_probs(probs: np.ndarray, class_names: List[str], k: int = 2):
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    order = np.argsort(-p)[:k]
    return [(int(i), class_names[int(i)], float(p[int(i)])) for i in order]


def top2_strings(items):
    # items: [(idx, label, prob), ...]
    return [f'{label} ({prob*100:.1f}%)' for _, label, prob in items]


__all__ = [
    'FaceShapeModel',
    'topk_from_probs',
    'top2_strings',
]

