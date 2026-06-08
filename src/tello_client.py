"""Thin wrapper around djitellopy.Tello.

Responsibilities:
- Connection + preflight (battery gate).
- Send-rate-controlled RC dispatch with spec-limit clamping.
- Liveness timestamp from state UDP packets (for watchdog).
- Clean shutdown (auto-land on close if airborne).

Spec rule: send_rc clamps EVERY channel to spec limits.
There must be no path that bypasses these clamps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from djitellopy import Tello as _RawTello
import djitellopy.tello as _dj_tello

from src import config


# --- liveness instrumentation -------------------------------------------
# djitellopy's udp_state_receiver is a @staticmethod that calls
# `Tello.parse_state(data)` directly, so a subclass override is bypassed.
# We instead wrap the static method at import time. Single-drone deployment,
# single shared timestamp is fine.
_state_last_perf: float = 0.0
_state_packets: int = 0


def _install_parse_state_hook() -> None:
    global _state_last_perf, _state_packets
    if getattr(_dj_tello.Tello.parse_state, "_telexp_wrapped", False):
        return
    original = _dj_tello.Tello.parse_state

    def wrapped(state):
        global _state_last_perf, _state_packets
        result = original(state)
        _state_last_perf = time.perf_counter()
        _state_packets += 1
        return result

    wrapped._telexp_wrapped = True  # type: ignore[attr-defined]
    _dj_tello.Tello.parse_state = staticmethod(wrapped)


_install_parse_state_hook()


class TelloError(RuntimeError):
    pass


@dataclass
class DroneState:
    battery_pct: int
    height_cm: int
    baro_cm: float
    temp_low_c: int
    temp_high_c: int
    pitch_deg: int
    roll_deg: int
    yaw_deg: int
    vgx: int
    vgy: int
    vgz: int
    flight_time_s: int


def _clamp(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


class TelloClient:
    def __init__(self, telemetry=None) -> None:
        self._tello = _RawTello()
        self._telemetry = telemetry
        self._airborne = False
        self._stream_on = False
        self._t0 = time.perf_counter()

    # ---- lifecycle -----------------------------------------------------

    def connect(self) -> DroneState:
        self._tello.connect()
        # Wait for first state UDP packet so the watchdog has a valid signal.
        deadline = time.perf_counter() + 3.0
        while _state_last_perf <= 0.0:
            if time.perf_counter() > deadline:
                raise TelloError("no state UDP packets within 3s of connect")
            time.sleep(0.05)
        state = self.read_state()
        if state.battery_pct < config.MIN_BATTERY_TAKEOFF_PCT:
            raise TelloError(
                f"battery too low for takeoff: {state.battery_pct}% < {config.MIN_BATTERY_TAKEOFF_PCT}%"
            )
        # Pin internal speed --- Tello scales rc_control channels by this value.
        # Without it, RC=30 might be 30% of an unknown default (---10cm/s).
        try:
            self._tello.set_speed(config.TELLO_SPEED_CMS)
            self._event("set_speed", cms=config.TELLO_SPEED_CMS)
        except Exception as e:
            self._event("set_speed_failed", err=str(e))
        self._event("connected", battery=state.battery_pct, height=state.height_cm)
        return state

    def takeoff(self) -> None:
        self.sync_airborne_from_state()
        if self._airborne:
            self._event("takeoff_skipped_already_airborne")
            return
        self._event("takeoff_begin")
        self._tello.takeoff()
        self._airborne = True
        self._event("takeoff_complete")

    def land(self) -> None:
        self.sync_airborne_from_state()
        if not self._airborne:
            return
        self._event("land_begin")
        self._tello.land()
        self._airborne = False
        self._event("land_complete")

    def emergency(self) -> None:
        self._event("emergency")
        try:
            self._tello.emergency()
        finally:
            self._airborne = False

    def stream_on(self) -> None:
        if self._stream_on:
            return
        self._tello.streamon()
        self._stream_on = True
        self._event("stream_on")

    def stream_off(self) -> None:
        if not self._stream_on:
            return
        try:
            self._tello.streamoff()
        finally:
            self._stream_on = False
            self._event("stream_off")

    def get_frame_read(self):
        return self._tello.get_frame_read()

    def restart_frame_read(self):
        """Force djitellopy to create a fresh video reader after its thread dies."""
        old = getattr(self._tello, "background_frame_read", None)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        try:
            self._tello.background_frame_read = None
        except Exception:
            pass
        return self._tello.get_frame_read()

    def move_up_cm(self, cm: int) -> None:
        cm = int(max(20, min(100, cm)))
        self._event("move_up_begin", cm=cm)
        self._tello.move_up(cm)
        self._event("move_up_complete", cm=cm)

    def move_down_cm(self, cm: int) -> None:
        cm = int(max(20, min(100, cm)))
        self._event("move_down_begin", cm=cm)
        self._tello.move_down(cm)
        self._event("move_down_complete", cm=cm)

    def close(self) -> None:
        # Always try to land first if airborne.
        if self._airborne:
            try:
                self.land()
            except Exception as e:
                self._event("land_on_close_failed", err=str(e))
                try:
                    self.emergency()
                except Exception:
                    pass
        if self._stream_on:
            try:
                self.stream_off()
            except Exception:
                pass
        try:
            self._tello.end()
        except Exception:
            pass

    # ---- control -------------------------------------------------------

    def send_rc(self, lr: float = 0, fb: float = 0, ud: float = 0, yaw: float = 0) -> None:
        """Send RC command. Channels are PERCENT [-100..100] before clamping.

        Spec hard caps applied here. No path bypasses this clamp.
        """
        lr_c = _clamp(lr, -config.RC_LATERAL_CAP, config.RC_LATERAL_CAP)
        fb_c = _clamp(fb, -config.RC_FORWARD_CAP, config.RC_FORWARD_CAP)
        ud_c = _clamp(ud, -config.RC_VERTICAL_CAP, config.RC_VERTICAL_CAP)
        yaw_c = _clamp(yaw, -config.RC_YAW_CAP, config.RC_YAW_CAP)
        self._tello.send_rc_control(lr_c, fb_c, ud_c, yaw_c)

    # ---- liveness ------------------------------------------------------

    def last_state_age_ms(self) -> float:
        """Milliseconds since the last state UDP packet from the drone."""
        if _state_last_perf <= 0.0:
            return float("inf")
        return (time.perf_counter() - _state_last_perf) * 1000.0

    def state_packets(self) -> int:
        return _state_packets

    def sync_airborne_from_state(self) -> bool:
        """Best-effort correction for cases where Tello lands outside our path."""
        try:
            state = self.read_state()
        except Exception as e:
            self._event("airborne_sync_failed", err=str(e))
            return self._airborne
        was = self._airborne
        self._airborne = state.height_cm >= 15
        if self._airborne != was:
            self._event(
                "airborne_sync",
                was=was,
                now=self._airborne,
                height=state.height_cm,
                flight_time=state.flight_time_s,
            )
        return self._airborne

    # ---- telemetry-friendly state read ---------------------------------

    def read_state(self) -> DroneState:
        t = self._tello
        return DroneState(
            battery_pct=int(t.get_battery()),
            height_cm=int(t.get_height()),
            baro_cm=float(t.get_barometer()),
            temp_low_c=int(t.get_lowest_temperature()),
            temp_high_c=int(t.get_highest_temperature()),
            pitch_deg=int(t.get_pitch()),
            roll_deg=int(t.get_roll()),
            yaw_deg=int(t.get_yaw()),
            vgx=int(t.get_speed_x()),
            vgy=int(t.get_speed_y()),
            vgz=int(t.get_speed_z()),
            flight_time_s=int(t.get_flight_time()),
        )

    @property
    def airborne(self) -> bool:
        return self._airborne

    # ---- internals -----------------------------------------------------

    def _event(self, kind: str, **fields) -> None:
        if self._telemetry is not None:
            self._telemetry.event(kind, **fields)
