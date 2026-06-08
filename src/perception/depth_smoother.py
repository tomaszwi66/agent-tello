"""Temporal smoothing for depth maps.

Stage 2: symmetric EMA (alpha scalar).
Stage 4: asymmetric EMA --- per spec sec 4.1.
    Approach (obstacle closing): alpha_approach  (high -> fast response)
    Recede  (obstacle moving away): alpha_recede (low  -> slow decay)

Depth models differ:
    relative DA V2: larger value = closer
    metric DA V2:   smaller value = closer
"""

from __future__ import annotations

import numpy as np

from src import config


class TemporalSmoother:
    def __init__(
        self,
        alpha: float = config.SMOOTHING_ALPHA,
        asymmetric: bool = False,
        alpha_approach: float = config.SMOOTHING_ALPHA_APPROACH,
        alpha_recede: float = config.SMOOTHING_ALPHA_RECEDE,
        larger_is_closer: bool = not config.DEPTH_MODEL_METRIC,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = float(alpha)
        self._asymmetric = asymmetric
        self._alpha_approach = float(alpha_approach)
        self._alpha_recede = float(alpha_recede)
        self._larger_is_closer = bool(larger_is_closer)
        self._prev: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def asymmetric(self) -> bool:
        return self._asymmetric

    def reset(self) -> None:
        self._prev = None

    def update(self, depth: np.ndarray) -> np.ndarray:
        if self._prev is None or self._prev.shape != depth.shape:
            self._prev = depth.copy()
            return self._prev

        if self._asymmetric:
            # Per-pixel alpha: approach pixels get higher alpha (react fast).
            if self._larger_is_closer:
                approaching = depth > self._prev
            else:
                approaching = depth < self._prev
            a = np.where(approaching, self._alpha_approach, self._alpha_recede)
            self._prev = (1.0 - a) * self._prev + a * depth
        else:
            np.multiply(self._prev, 1.0 - self._alpha, out=self._prev)
            self._prev += self._alpha * depth

        return self._prev
