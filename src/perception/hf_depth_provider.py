"""HuggingFace depth provider (default: Depth Anything V2 Small).

Single class --- switchable model via model_id. DA V2 Small is the validated
default for Stage 2; DA3 Small is supported as an optional benchmark backend
by passing config.DEPTH_MODEL_ID_DA3 (or any HF id).

Conventions:
- Input frame is RGB uint8 (H, W, 3) --- djitellopy already produces RGB.
- Output depth_map is float32 (H_out, W_out). The HF processor decides
  H_out, W_out; we do NOT post-resize to input HxW here. Downstream stages
  resize when they need pixel correspondence (e.g. heatmap overlay).
- Larger predicted value = closer (HF DA V2 convention: it outputs inverse
  depth / relative depth). We expose raw output; pseudo-metric calibration
  is a separate stage (spec sec. 6).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

# Force fully offline mode for HuggingFace --- must be set BEFORE any transformers
# imports. When connected to Tello AP there is no internet; without this flag
# huggingface_hub makes DNS requests that time out (~5s each) and warmup takes 30s.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src import config
from src.depth_provider import DepthProvider, DepthResult


class HFDepthProvider(DepthProvider):
    def __init__(
        self,
        model_id: str = config.DEPTH_MODEL_ID,
        device: str = config.DEPTH_DEVICE,
        dtype: str = config.DEPTH_DTYPE,
        display_name: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype_str = dtype
        self._display_name = display_name or model_id.split("/")[-1]
        # Lazy: heavy imports happen only at warmup so MockProvider use
        # doesn't pay for torch import time.
        self._torch = None
        self._processor = None
        self._model = None
        self._torch_dtype = None

    @property
    def name(self) -> str:
        return f"{self._display_name}@{self._device}/{self._dtype_str}"

    def warmup(self) -> None:
        import torch  # noqa: WPS433 --- intentional lazy import
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if self._dtype_str not in dtype_map:
            raise ValueError(f"unsupported dtype: {self._dtype_str}")
        self._torch = torch
        self._torch_dtype = dtype_map[self._dtype_str]

        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        # local_files_only=True: model must be pre-cached; avoids HF DNS checks
        # when connected to Tello AP (no internet during flight sessions).
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id, local_files_only=True
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self._model_id, torch_dtype=self._torch_dtype, local_files_only=True
        ).to(self._device)
        self._model.eval()

        # Two dummy passes --- first triggers cuDNN autotune, second gives a stable timing.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy, 0)
        if self._device == "cuda":
            torch.cuda.synchronize()

    def infer(self, frame: np.ndarray, frame_ts_ns: int) -> DepthResult:
        if self._model is None:
            raise RuntimeError("HFDepthProvider.infer called before warmup()")
        torch = self._torch  # local alias
        t0 = time.perf_counter()

        # Frame is RGB uint8 H x W x 3 from djitellopy/PyAV.
        # HF processor handles resize, normalize, batchify.
        inputs = self._processor(images=frame, return_tensors="pt")
        # Move to device. Float tensors get the requested dtype; ints stay int.
        on_device = {}
        for k, v in inputs.items():
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                on_device[k] = v.to(self._device, dtype=self._torch_dtype, non_blocking=True)
            else:
                on_device[k] = v.to(self._device, non_blocking=True)

        with torch.inference_mode():
            outputs = self._model(**on_device)
            depth = outputs.predicted_depth  # (B, H, W) --- typically B=1

        if self._device == "cuda":
            torch.cuda.synchronize()
        depth_np = depth.squeeze(0).to(torch.float32).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DepthResult(
            depth_map=depth_np,
            frame_ts_ns=frame_ts_ns,
            latency_ms=latency_ms,
            model_name=self.name,
        )


def build_default() -> HFDepthProvider:
    """Relative DA V2 Small (larger=closer)."""
    return HFDepthProvider(model_id=config.DEPTH_MODEL_ID, display_name="DA-V2-Small")


def build_active() -> HFDepthProvider:
    """Returns the active model based on config.DEPTH_MODEL_METRIC."""
    if config.DEPTH_MODEL_METRIC:
        return build_metric()
    return build_default()


def build_metric() -> HFDepthProvider:
    """Metric model (larger=farther, metres). Uses DEPTH_MODEL_ID_METRIC from config.
    Verify model ID and download before first use (needs internet, not available on Tello AP).
    """
    return HFDepthProvider(model_id=config.DEPTH_MODEL_ID_METRIC, display_name="DA-Metric")


def build_da3_small() -> HFDepthProvider:
    """DA3 Small --- benchmark backend (relative depth)."""
    return HFDepthProvider(model_id=config.DEPTH_MODEL_ID_DA3, display_name="DA3-Small")
