from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # noqa: BLE001
    ort = None  # type: ignore


_SESSION: Optional["ort.InferenceSession"] = None
MODEL_LOADED: bool = False


def _load_session() -> Optional[ort.InferenceSession]:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    if ort is None:
        return None
    model_path = Path("backend/models/presence_mobilenet.onnx")
    if model_path.exists():
        providers = ["CPUExecutionProvider"]
        try:
            _SESSION = ort.InferenceSession(str(model_path), providers=providers)
            globals()["MODEL_LOADED"] = True
        except Exception:
            _SESSION = None
            globals()["MODEL_LOADED"] = False
    return _SESSION


def _preprocess(img: np.ndarray) -> np.ndarray:
    # Resize to 224x224, normalize to 0..1, NCHW
    inp = cv2.resize(img, (224, 224))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    return np.expand_dims(inp, 0)


def predict_presence(path: str) -> float:
    """Return probability [0,1] that an image contains a billboard.

    If ONNX model exists, use it. Otherwise, fall back to heuristic edges-based estimator.
    """
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        return 0.0

    session = _load_session()
    if session is not None:
        inp = _preprocess(img)
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        # Assume output [N,2] softmax: [no_billboard, billboard]
        probs = outputs[0].astype(np.float32)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            prob_billboard = float(probs[0][1])
        else:
            # If the model outputs a single sigmoid value
            prob_billboard = float(probs.flatten()[0])
        return max(0.0, min(1.0, prob_billboard))

    # Heuristic fallback (same as before)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(w * h)

    best = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.02 * area_img:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh or 1)
        rectangularity = area / float(max(bw * bh, 1))
        if aspect >= 2.0 and rectangularity >= 0.6:
            area_norm = area / area_img
            score = min(1.0, 0.5 * area_norm + 0.3 * rectangularity + 0.2 * min(1.0, (aspect - 1.5) / 6.0))
            best = max(best, score)
    return float(best)


