"""Video stream ingestion.

Single-slot latest-frame buffer (no queue --- reactive system wants the
newest frame, never a stale one). djitellopy already maintains its own
background thread for h264 decoding; we only poll its latest frame.

Public API exposes state (fps, age, freeze flag). Reactions are decided
elsewhere --- this module observes, it does not act.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

from src import config
from src.tello_client import TelloClient


class VideoStream:
    def __init__(self, client: TelloClient, telemetry=None, fps_ema_alpha: float = 0.1) -> None:
        self._client = client
        self._telemetry = telemetry
        self._fps_alpha = fps_ema_alpha
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts_ns: int = 0
        self._last_seen_fingerprint: int = -1
        self._frames_seen: int = 0
        self._fps_ema: float = 0.0
        self._last_frame_perf: float = 0.0
        self._last_restart_perf: float = 0.0
        self._suppress_restarts: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_reader = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._client.stream_on()
        self._frame_reader = self._client.get_frame_read()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="VideoStream", daemon=True)
        self._thread.start()
        if self._telemetry is not None:
            self._telemetry.event("video_stream_start")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._client.stream_off()
        except Exception:
            pass
        if self._telemetry is not None:
            self._telemetry.event("video_stream_stop", frames=self._frames_seen)

    def _run(self) -> None:
        # djitellopy's frame_reader exposes .frame (latest decoded ndarray).
        # We can't get a true frame-id. Some versions reuse the same ndarray
        # and update it in-place, so object identity is not reliable.
        # The Tello briefly drops its video stream during takeoff --- djitellopy's
        # internal PyAV thread crashes with an I/O error. We detect this via
        # frame_reader.stopped and restart streamon to recover automatically.
        while not self._stop.is_set():
            # Check if djitellopy's internal thread died --- restart if so.
            if (
                not self._suppress_restarts
                and self._frame_reader is not None
                and getattr(self._frame_reader, "stopped", False)
            ):
                if self._telemetry is not None:
                    self._telemetry.event("video_stream_restart", reason="frame_reader_stopped")
                self._restart_reader()
                time.sleep(0.5)
                continue

            frame = None
            try:
                frame = self._frame_reader.frame if self._frame_reader is not None else None
            except Exception as e:
                if self._telemetry is not None:
                    self._telemetry.event("frame_read_error", err=str(e))
            if frame is not None:
                fingerprint = self._fingerprint(frame)
                if fingerprint != self._last_seen_fingerprint:
                    self._on_new_frame(frame, fingerprint)
            now = time.perf_counter()
            if self._last_frame_perf > 0:
                stale_s = now - self._last_frame_perf
            else:
                stale_s = now - self._last_restart_perf if self._last_restart_perf > 0 else 0.0
            if (
                not self._suppress_restarts
                and stale_s > 2.5
                and now - self._last_restart_perf > 2.5
            ):
                if self._telemetry is not None:
                    self._telemetry.event(
                        "video_stream_restart",
                        reason="stale_frames",
                        stale_s=round(stale_s, 2),
                    )
                self._restart_reader()
            # Poll at ~60Hz; sleep to avoid CPU spin.
            time.sleep(0.005)

    def _restart_reader(self) -> None:
        """Restart the frame reader after a crash (Tello drops UDP stream during takeoff).
        We do NOT call streamoff/streamon --- the Tello resumes streaming automatically
        once airborne. Just open a new PyAV container on the same port.
        """
        self._last_restart_perf = time.perf_counter()
        try:
            restart = getattr(self._client, "restart_frame_read", None)
            self._frame_reader = restart() if restart is not None else self._client.get_frame_read()
            if self._telemetry is not None:
                self._telemetry.event("video_stream_restarted")
        except Exception as e:
            if self._telemetry is not None:
                self._telemetry.event("video_stream_restart_failed", err=str(e))

    def restart_reader(self) -> None:
        self._restart_reader()

    def suppress_restarts(self, value: bool = True) -> None:
        self._suppress_restarts = bool(value)

    @staticmethod
    def _fingerprint(frame: np.ndarray) -> int:
        try:
            h, w = frame.shape[:2]
            sample = frame[:: max(1, h // 32), :: max(1, w // 32), :]
            return hash(sample.tobytes())
        except Exception:
            return id(frame)

    def _on_new_frame(self, frame: np.ndarray, fingerprint: int) -> None:
        now_perf = time.perf_counter()
        now_ns = time.time_ns()
        dt = now_perf - self._last_frame_perf if self._last_frame_perf > 0 else 0.0
        with self._lock:
            self._latest = frame.copy()
            self._latest_ts_ns = now_ns
            self._last_seen_fingerprint = fingerprint
            self._frames_seen += 1
            self._last_frame_perf = now_perf
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_ema = (
                    inst_fps if self._fps_ema == 0.0
                    else (1 - self._fps_alpha) * self._fps_ema + self._fps_alpha * inst_fps
                )

    def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            frame = self._latest
            ts_ns = self._latest_ts_ns
            last_perf = self._last_frame_perf
        if frame is None:
            return None, 0, float("inf")
        age_ms = (time.perf_counter() - last_perf) * 1000.0 if last_perf > 0 else float("inf")
        return frame, ts_ns, age_ms

    def fps(self) -> float:
        with self._lock:
            return self._fps_ema

    def is_frozen(self, threshold_ms: int = config.FRAME_STALENESS_WARN_MS) -> bool:
        _, _, age = self.get_latest()
        return age > threshold_ms

    def frames_seen(self) -> int:
        with self._lock:
            return self._frames_seen
