"""Stage 3/4 --- sector-based depth analysis.

Supports two depth conventions:

  metric=False (DA V2 relative, default):
    Larger raw value = CLOSER (inverse/relative depth).
    Normalized per-frame to [0,1]. Stop if norm_p90 > SAFETY_STOP_NORM_THRESHOLD.

  metric=True (metric model, e.g. DA V2 Metric / DA3 Metric):
    Larger raw value = FARTHER (metres).
    No normalization. Stop if raw_p10 < SAFETY_STOP_DISTANCE_M.
    p10 = 10th percentile --- represents the closer objects in the sector.

Both modes apply floor cropping (bottom SECTOR_FLOOR_CROP rows ignored) to
avoid false positives from the Tello's slightly downward camera angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src import config


class SectorID(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


@dataclass(frozen=True)
class SectorResult:
    sector_id: SectorID
    # Normalized stats (relative mode only, else 0.0)
    norm_p90: float
    norm_max: float
    # Raw stats (always populated)
    raw_p10: float    # 10th percentile --- closest objects in sector
    raw_p90: float
    raw_max: float
    raw_min: float
    pixel_count: int
    safety_stop: bool
    metric: bool      # True = values are metres


def _normalize(depth: np.ndarray) -> np.ndarray:
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - lo) / (hi - lo)).astype(np.float32)


def split_sectors(
    depth_map: np.ndarray,
    n: int = config.SECTOR_COUNT,
    floor_crop: float = config.SECTOR_FLOOR_CROP,
    metric: bool = config.DEPTH_MODEL_METRIC,
    norm_threshold: float = config.SAFETY_STOP_NORM_THRESHOLD,
    distance_m: float = config.SAFETY_STOP_DISTANCE_M,
    pixel_fraction: float = config.SAFETY_STOP_PIXEL_FRACTION,
) -> list[SectorResult]:
    """Analyze depth map and return per-sector safety stats.

    Args:
        depth_map: float32 (H, W).
        n: number of sectors (1---5).
        floor_crop: bottom fraction to exclude.
        metric: True = depth in metres (larger=farther), False = relative (larger=closer).
        norm_threshold: relative mode threshold (0---1).
        distance_m: metric mode threshold in metres.
        pixel_fraction: fraction of sector pixels that must trigger to set safety_stop.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2-D, got shape {depth_map.shape}")
    if not 1 <= n <= 5:
        raise ValueError(f"n must be 1-5, got {n}")

    h, w = depth_map.shape
    crop_rows = int(h * floor_crop)
    active = depth_map[: h - crop_rows, :] if crop_rows > 0 else depth_map

    norm = None if metric else _normalize(active)
    results: list[SectorResult] = []

    for i in range(n):
        x0 = i * w // n
        x1 = (i + 1) * w // n if i < n - 1 else w

        raw_patch = active[:, x0:x1].ravel()

        if metric:
            # Metric: smaller value = closer = dangerous.
            hot = np.count_nonzero(raw_patch < distance_m)
            stop = (hot / raw_patch.size) >= pixel_fraction
            n_p90, n_max = 0.0, 0.0
        else:
            norm_patch = norm[:, x0:x1].ravel()
            hot = np.count_nonzero(norm_patch > norm_threshold)
            stop = (hot / norm_patch.size) >= pixel_fraction
            n_p90 = float(np.percentile(norm_patch, 90))
            n_max = float(norm_patch.max())

        results.append(SectorResult(
            sector_id=SectorID(i) if n == 3 else SectorID(min(i, 2)),
            norm_p90=n_p90,
            norm_max=n_max,
            raw_p10=float(np.percentile(raw_patch, 10)),
            raw_p90=float(np.percentile(raw_patch, 90)),
            raw_max=float(raw_patch.max()),
            raw_min=float(raw_patch.min()),
            pixel_count=int(raw_patch.size),
            safety_stop=stop,
            metric=metric,
        ))

    return results


def any_stop(sectors: list[SectorResult]) -> bool:
    return any(s.safety_stop for s in sectors)
