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
        print("ERROR: onnxruntime not available - cannot load ONNX model")
        return None
    
    # Try both possible model locations
    model_paths = [
        Path("models/presence_mobilenet.onnx"),  # Relative to backend/
        Path("backend/models/presence_mobilenet.onnx"),  # From project root
        Path("app/models/presence_mobilenet.onnx")  # From backend/
    ]
    model_path = None
    for path in model_paths:
        print(f"Checking model path: {path.absolute()}")
        if path.exists():
            model_path = path
            print(f"Found ONNX model at: {model_path.absolute()}")
            break
    
    if model_path is None:
        print("ERROR: presence_mobilenet.onnx not found in any expected location")
        globals()["MODEL_LOADED"] = False
        return None
        
    providers = ["CPUExecutionProvider"]
    try:
        _SESSION = ort.InferenceSession(str(model_path), providers=providers)
        globals()["MODEL_LOADED"] = True
        print(f"SUCCESS: ONNX model loaded from {model_path}")
        print(f"Model inputs: {[inp.name for inp in _SESSION.get_inputs()]}")
        print(f"Model outputs: {[out.name for out in _SESSION.get_outputs()]}")
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model from {model_path}: {e}")
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
    - accepted: boolean if image should be accepted (billboard >= 0.6)
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
            session = ort.InferenceSession(custom_model_path)
            print(f"Using custom model: {custom_model_path}")
        except Exception as e:
            print(f"Failed to load custom model {custom_model_path}: {e}")
    
    if session is None:
        session = _load_session()
        if session is None:
            print("ERROR: No ONNX model available for inference")
    
    if session is not None:
        print(f"Running ONNX inference on image: {path}")
        inp = _preprocess(img)
        print(f"Preprocessed input shape: {inp.shape}, dtype: {inp.dtype}, min: {inp.min():.3f}, max: {inp.max():.3f}")
        
        try:
            outputs = session.run(None, {session.get_inputs()[0].name: inp})
            probs = outputs[0].astype(np.float32)
            print(f"Raw ONNX output shape: {probs.shape}, values: {probs}")
            
            if probs.ndim == 2 and probs.shape[1] >= 2:
                # Two-class output: [no_billboard, billboard]
                no_billboard_prob = float(probs[0][0])
                billboard_prob = float(probs[0][1])
                print(f"Two-class output detected: no_billboard={no_billboard_prob:.4f}, billboard={billboard_prob:.4f}")
            else:
                # Single output - assume it's billboard probability
                billboard_prob = float(probs.flatten()[0])
                no_billboard_prob = 1.0 - billboard_prob
                print(f"Single output detected: billboard={billboard_prob:.4f}, no_billboard={no_billboard_prob:.4f}")
            
            # Apply threshold rule: billboard >= 0.6 to accept (60% confidence - relaxed for debugging)
            accepted = billboard_prob >= 0.6
            
            # Enhanced debug output with both class scores
            print(f"\n=== ONNX PRESENCE DEBUG ===")
            print(f"Image: {path}")
            print(f"CLASS SCORES: billboard={billboard_prob:.4f} ({billboard_prob*100:.1f}%), no_billboard={no_billboard_prob:.4f} ({no_billboard_prob*100:.1f}%)")
            print(f"DECISION: billboard_confidence={billboard_prob:.4f}, threshold=0.6, accepted={accepted}")
            print(f"=== END ONNX DEBUG ===\n")
            
            confidence_pct = int(billboard_prob * 100)
            if accepted:
                message = f"Billboard detected with {confidence_pct}% confidence"
            else:
                message = "Low confidence. Please upload a clearer billboard photo."
            
            return {
                "billboard": max(0.0, min(1.0, billboard_prob)),
                "no_billboard": max(0.0, min(1.0, no_billboard_prob)),
                "accepted": accepted,
                "message": message,
                "confidence": confidence_pct
            }
        except Exception as e:
            print(f"ERROR during ONNX inference: {e}")
            return {
                "billboard": 0.0,
                "no_billboard": 1.0,
                "accepted": False,
                "message": f"Model inference failed: {str(e)}",
                "confidence": 0
            }

    # Model not available - return error instead of silent rejection
    print("ERROR: ONNX model not loaded - cannot perform billboard detection")
    return {
        "billboard": 0.0,
        "no_billboard": 1.0,
        "accepted": False,
        "message": "Billboard detection model not available. Please check server configuration.",
        "confidence": 0,
        "error": "model_not_loaded"
    }
