"""Stage 4 --- background depth inference pipeline.

Runs in a daemon thread: pulls latest frame from VideoStream, runs
DepthProvider.infer(), smooths with TemporalSmoother, splits into sectors.
Exposes latest result via get_latest() --- single-slot, non-blocking.

Control loop reads get_latest() every tick without blocking on inference.
Inference (~30 ms) runs independently from the 50 ms control period.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src import config
from src.depth_provider import DepthProvider
from src.perception.depth_smoother import TemporalSmoother
from src.perception.sectors import SectorResult, split_sectors
from src.video_stream import VideoStream


@dataclass
class PipelineResult:
    sectors: list[SectorResult]
    depth_map: np.ndarray       # smoothed, float32
    frame_ts_ns: int
    latency_ms: float
    safety_stop: bool           # True if any sector requests stop


class DepthPipeline:
    """Background depth inference + sector analysis pipeline."""

    def __init__(
        self,
        stream: VideoStream,
        provider: DepthProvider,
        smoother: TemporalSmoother | None = None,
    ) -> None:
        self._stream = stream
        self._provider = provider
        self._smoother = smoother or TemporalSmoother(
            asymmetric=True,
            alpha_approach=config.SMOOTHING_ALPHA_APPROACH,
            alpha_recede=config.SMOOTHING_ALPHA_RECEDE,
            larger_is_closer=not config.DEPTH_MODEL_METRIC,
        )
        self._lock = threading.Lock()
        self._result: PipelineResult | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._infer_count = 0
        self._stop_count = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="depth-pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> PipelineResult | None:
        with self._lock:
            return self._result

    @property
    def infer_count(self) -> int:
        return self._infer_count

    @property
    def stop_count(self) -> int:
        return self._stop_count

    def _loop(self) -> None:
        last_ts = -1
        while self._running:
            frame, ts_ns, age_ms = self._stream.get_latest()
            if frame is None or ts_ns == last_ts:
                time.sleep(0.005)
                continue
            last_ts = ts_ns

            try:
                result = self._provider.infer(frame, ts_ns)
                depth = self._smoother.update(result.depth_map)
                sectors = split_sectors(depth)
                stop = any(s.safety_stop for s in sectors)

                pipeline_result = PipelineResult(
                    sectors=sectors,
                    depth_map=depth,
                    frame_ts_ns=ts_ns,
                    latency_ms=result.latency_ms,
                    safety_stop=stop,
                )
                with self._lock:
                    self._result = pipeline_result
                self._infer_count += 1
                if stop:
                    self._stop_count += 1
            except Exception:
                # Never crash the background thread --- control loop must keep running.
                time.sleep(0.01)
