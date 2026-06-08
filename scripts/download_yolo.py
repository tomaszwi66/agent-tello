"""Pre-download YOLOv8n model for offline drone flight.

Run ONCE before the drone session (with internet access):
    python scripts/download_yolo.py

Saves the model to models/yolov8n.pt (path set in config.YOLO_MODEL_PATH).
Subsequent flights are fully offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main() -> int:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    target = models_dir / "yolov8n.pt"

    if target.exists():
        print(f"Model already present: {target}  ({target.stat().st_size // 1024} KB)")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics==8.3.0")
        return 1

    print(f"Downloading YOLOv8n --- {target} ---")
    model = YOLO("yolov8n.pt")          # downloads to ultralytics cache
    # Export / copy to our models/ dir so it works offline.
    import shutil
    cached = Path(model.ckpt_path)
    shutil.copy2(cached, target)
    print(f"Saved: {target}  ({target.stat().st_size // 1024} KB)")

    # Quick smoke test.
    import numpy as np
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    out = model.predict(dummy, imgsz=320, verbose=False)
    print(f"Smoke test OK --- {len(out[0].boxes)} detections on blank frame (expected 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
