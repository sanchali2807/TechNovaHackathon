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


def predict_presence(path: str, custom_model_path: str = None) -> dict:
    """Return probability scores for billboard classification using ONNX model.
    
    Returns dict with:
    - billboard: probability of billboard class
    - no_billboard: probability of no billboard class
    - accepted: boolean if image should be accepted (billboard >= 0.8)
    """
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        return {
            "billboard": 0.0,
            "no_billboard": 1.0,
            "accepted": False,
            "message": "Invalid image file"
        }

    # Try custom model first, then default model
    session = None
    if custom_model_path and Path(custom_model_path).exists():
        try:
            session = onnxruntime.InferenceSession(custom_model_path)
        except Exception as e:
            print(f"Failed to load custom model {custom_model_path}: {e}")
    
    if session is None:
        session = _load_session()
    
    if session is not None:
        inp = _preprocess(img)
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        probs = outputs[0].astype(np.float32)
        
        if probs.ndim == 2 and probs.shape[1] >= 2:
            # Two-class output: [no_billboard, billboard]
            no_billboard_prob = float(probs[0][0])
            billboard_prob = float(probs[0][1])
        else:
            # Single output - assume it's billboard probability
            billboard_prob = float(probs.flatten()[0])
            no_billboard_prob = 1.0 - billboard_prob
        
        # Apply threshold rule: billboard >= 0.3 to accept (lowered from 0.8)
        accepted = billboard_prob >= 0.3
        
        # Debug output
        print(f"ONNX Model Results: billboard={billboard_prob:.3f}, no_billboard={no_billboard_prob:.3f}, accepted={accepted}")
        
        message = "Billboard detected ✅" if accepted else "No billboard detected ❌ Please upload a billboard photo."
        
        return {
            "billboard": max(0.0, min(1.0, billboard_prob)),
            "no_billboard": max(0.0, min(1.0, no_billboard_prob)),
            "accepted": accepted,
            "message": message
        }

    # Heuristic fallback - return rejection for no ONNX model
    return {
        "billboard": 0.0,
        "no_billboard": 1.0,
        "accepted": False,
        "message": "No billboard detected Please upload a billboard photo."
    }
