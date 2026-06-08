"""MockDepthProvider --- synthetic depth for offline tests.

Stable, deterministic, near-zero latency. Use this to validate the control
pipeline without a model or a drone in the loop.
"""

from __future__ import annotations

import time

import numpy as np

from src.depth_provider import DepthProvider, DepthResult


class MockDepthProvider(DepthProvider):
    """Returns a vertical gradient depth map.

    Convention used here: larger value = farther. The bottom of the frame
    (foreground) is "near", the top is "far". Useful to sanity-check
    sectors/safety-stop logic in later stages.
    """

    def __init__(self, height: int = 252, width: int = 336, near_m: float = 0.5, far_m: float = 10.0) -> None:
        self._h = int(height)
        self._w = int(width)
        col = np.linspace(far_m, near_m, self._h, dtype=np.float32)
        self._base = np.tile(col[:, None], (1, self._w))

    @property
    def name(self) -> str:
        return f"mock-gradient-{self._h}x{self._w}"

    def warmup(self) -> None:
        return

    def infer(self, frame: np.ndarray, frame_ts_ns: int) -> DepthResult:
        t0 = time.perf_counter()
        # Tiny temporal jitter so smoothing/CI tests can see motion.
        jitter = 0.02 * float(np.sin(frame_ts_ns / 1e9))
        depth = self._base + jitter
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DepthResult(
            depth_map=depth,
            frame_ts_ns=frame_ts_ns,
            latency_ms=latency_ms,
            model_name=self.name,
        )
