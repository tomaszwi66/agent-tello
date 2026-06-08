"""DepthProvider abstraction --- RESERVED SLOT for Stage 2.

The control system imports only this interface. Concrete depth models
(Depth Anything 3 Small/Base, MockDepth) implement it independently.
This decoupling is a project rule: swap depth without touching control.

No implementations live here yet. Stage 2 will add them under
src/perception/ when that scope opens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthResult:
    depth_map: np.ndarray   # float32, HxW. Convention to be fixed in Stage 2.
    frame_ts_ns: int        # timestamp of the source frame
    latency_ms: float       # wall-clock inference time
    model_name: str         # for telemetry / debug


class DepthProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def warmup(self) -> None:
        """Allocate buffers, run a dummy inference. Call before flight."""

    @abstractmethod
    def infer(self, frame: np.ndarray, frame_ts_ns: int) -> DepthResult: ...
