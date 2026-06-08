"""Stage 6 --- optional YOLO-based hazard surface detector.

Runs in a daemon thread parallel to the depth pipeline. Detects objects whose
surfaces fool the depth model (TV screens, laptop displays, refrigerators,
glass clocks) and marks the corresponding sectors as hazardous.

The navigation policy reads get_latest() and treats a hazardous sector the same
as a safety_stop --- i.e. don't fly into it, yaw away.

Why this matters:
  Flat reflective surfaces (TV glass, mirrors) give the depth model confused
  output --- sometimes they look infinitely far (mirror shows room behind),
  sometimes very close (specular reflection). Either artifact is dangerous.
  A 2D object detector (YOLO) is not fooled by reflections because it matches
  visual features, not stereo/monocular depth cues.

YOLO is OPTIONAL. If ultralytics is not installed, or the model file is
missing, YoloDetector raises on construction --- call build_yolo_detector()
which returns None gracefully so the flight script can skip it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from src import config

if TYPE_CHECKING:
    from src.video_stream import VideoStream

try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

# Fraction of a detection's width that must overlap with a sector to flag it.
_SECTOR_OVERLAP_FRACTION = 0.15
_YOLO_STALE_MS = 600.0


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    # Bounding box in normalised [0,1] coords relative to frame width/height.
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class YoloResult:
    detections: List[Detection]
    hazard_sectors: List[bool]   # [left, center, right]
    frame_ts_ns: int
    latency_ms: float



class YoloDetector:
    """Background YOLO hazard detector. One instance per flight."""

    def __init__(self, model_path: str, sector_count: int = config.SECTOR_COUNT) -> None:
        if not _HAS_ULTRALYTICS:
            raise RuntimeError(
                "ultralytics not installed --- run: pip install ultralytics==8.3.0"
            )
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {p}. "
                "Run: python scripts/download_yolo.py"
            )
        self._model_path = str(p)
        self._sector_count = sector_count
        self._model = None
        self._names = {}
        self._stream: Optional["VideoStream"] = None
        self._lock = threading.Lock()
        self._result: Optional[YoloResult] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.infer_count = 0

    def attach_stream(self, stream: "VideoStream") -> None:
        self._stream = stream

    def warmup(self) -> None:
        self._model = _YOLO(self._model_path, verbose=False)
        self._names = getattr(self._model, "names", {}) or {}
        dummy = np.zeros((config.YOLO_IMG_SIZE, config.YOLO_IMG_SIZE, 3), dtype=np.uint8)
        self._model.predict(dummy, imgsz=config.YOLO_IMG_SIZE, verbose=False)

    def start(self) -> None:
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        if self._model is None:
            self.warmup()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="yolo-detector"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[YoloResult]:
        with self._lock:
            return self._result

    # ---- background thread -----------------------------------------------

    def _loop(self) -> None:
        last_ts = -1
        while self._running:
            if self._stream is None:
                time.sleep(0.05)
                continue
            frame, ts_ns, _ = self._stream.get_latest()
            if frame is None or ts_ns == last_ts:
                time.sleep(0.015)
                continue
            last_ts = ts_ns
            try:
                t0 = time.perf_counter()
                kwargs = {}
                if not config.YOLO_DETECT_ALL_CLASSES:
                    kwargs["classes"] = list(config.YOLO_HAZARD_CLASSES)
                preds = self._model.predict(
                    frame,
                    imgsz=config.YOLO_IMG_SIZE,
                    verbose=False,
                    conf=config.YOLO_MIN_CONF,
                    **kwargs,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                dets = self._parse(preds)
                hazard = self._sector_hazards(dets)
                with self._lock:
                    self._result = YoloResult(
                        detections=dets,
                        hazard_sectors=hazard,
                        frame_ts_ns=ts_ns,
                        latency_ms=latency_ms,
                    )
                self.infer_count += 1
            except Exception:
                time.sleep(0.01)

    def _parse(self, results) -> List[Detection]:
        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xn = box.xyxyn[0].tolist()  # [x1,y1,x2,y2] normalised
                out.append(Detection(
                    cls_id=cls_id,
                    cls_name=str(self._names.get(cls_id, str(cls_id))),
                    conf=conf,
                    x1=xn[0], y1=xn[1], x2=xn[2], y2=xn[3],
                ))
        return out

    def _sector_hazards(self, dets: List[Detection]) -> List[bool]:
        n = self._sector_count
        hazard = [False] * n
        for det in dets:
            if det.cls_id not in config.YOLO_HAZARD_CLASSES:
                continue
            if det.conf < config.YOLO_HAZARD_MIN_CONF:
                continue
            for i in range(n):
                sec_x0 = i / n
                sec_x1 = (i + 1) / n
                overlap = max(0.0, min(det.x2, sec_x1) - max(det.x1, sec_x0))
                det_w = max(1e-6, det.x2 - det.x1)
                if overlap / det_w >= _SECTOR_OVERLAP_FRACTION:
                    hazard[i] = True
        return hazard


def build_yolo_detector(model_path: str) -> Optional[YoloDetector]:
    """Build + warm up a YoloDetector, or return None if unavailable."""
    if not config.YOLO_ENABLED:
        return None
    if not _HAS_ULTRALYTICS:
        return None
    try:
        det = YoloDetector(model_path)
        det.warmup()
        return det
    except Exception as e:
        # Non-fatal: fly without YOLO rather than aborting.
        return None

