from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


def build_mobilenet_v2(num_classes: int = 2) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
    except Exception:
        weights = None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    model = build_mobilenet_v2(num_classes=2)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_dir = Path("backend/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "presence_mobilenet.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["prob"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
    )
    print(f"Exported ONNX to {out_path.resolve()}")


if __name__ == "__main__":
    main()


