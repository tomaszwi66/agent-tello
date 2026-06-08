"""Stable 3-sector reactive navigation policy.

This is the known-good practical version: simple LEFT / CENTER / RIGHT
reasoning, forward + yaw when the centre is clear, and hard stop/yaw only when
the centre is genuinely blocked. It is less elegant than the 7-sector
experiment, but it proved reliable in flight.

Convention: Tello yaw + = clockwise = RIGHT, yaw - = LEFT.
Metric depth: larger raw_p10 = farther = freer.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from src import config
from src.control_loop import HOVER, RcCommand
from src.perception.depth_pipeline import DepthPipeline, PipelineResult
from src.tello_client import TelloClient

if TYPE_CHECKING:
    from src.perception.yolo_detector import YoloDetector

_DEPTH_STALE_MS: float = 400.0
_YOLO_STALE_MS: float = 600.0

_STOP_M = config.NAV_FORWARD_STOP_M
_RESUME_M = config.NAV_FORWARD_RESUME_M
_RESUME_FRAMES = config.NAV_FORWARD_RESUME_FRAMES
_SIDE_RATIO = config.NAV_SIDE_FREER_RATIO
_SIDE_MIN = config.NAV_SIDE_MIN_M
_EXPLORE_TRIGGER = config.NAV_EXPLORE_TRIGGER_S
_EXPLORE_SPIN = config.NAV_EXPLORE_SPIN_S
_EXPLORE_COOL = config.NAV_EXPLORE_COOLDOWN_S
_INFLATION_M = config.NAV_INFLATION_M
_MOTION_LATENCY = config.NAV_MOTION_LATENCY_S

_BRAKE_DUR = 0.35
_BRAKE_FB = -float(config.RC_FORWARD_CAP)
_STEER_DEADBAND_M = 0.35
_STEER_GAIN = 0.55
_CRUISE_YAW_CAP = 22.0
_AVOID_YAW_CAP = 36.0
_APERTURE_YAW_CAP = 18.0
_FIELD_YAW_CAP = 30.0
_FIELD_SIDE_SAFE_M = 0.95
_FIELD_SIDE_CRITICAL_M = 0.32
_FIELD_CENTER_GAIN = 0.40
_FIELD_FAR_GAIN = 2.5
_FIELD_FAR_CAP = 8.0
_SIDE_EDGE_SOFT_M = 0.55
_SIDE_BLOCK_PASS_CENTER_M = 1.10
_APERTURE_ADVANTAGE_M = 0.35
_RECOVERY_YAW_CAP = config.NAV_RECOVERY_YAW_CAP
_RECOVERY_YAW_STEP = config.NAV_RECOVERY_YAW_STEP
_RECOVERY_MIN_SCAN_S = config.NAV_RECOVERY_MIN_SCAN_S
_RECOVERY_EXIT_CENTER_M = config.NAV_RECOVERY_EXIT_CENTER_M
_RECOVERY_ALTERNATE_BAND_M = config.NAV_RECOVERY_ALTERNATE_BAND_M
_RECOVERY_FAR_BAND_M = config.NAV_RECOVERY_FAR_BAND_M
_RECOVERY_EXIT_FRAMES = config.NAV_RECOVERY_EXIT_FRAMES
_RECOVERY_CAPTURE_CENTER_M = config.NAV_RECOVERY_CAPTURE_CENTER_M
_RECOVERY_CAPTURE_FAR_M = config.NAV_RECOVERY_CAPTURE_FAR_M
_RECOVERY_CAPTURE_SIDE_M = config.NAV_RECOVERY_CAPTURE_SIDE_M
_RECOVERY_FAST_CAPTURE_MIN_SCAN_S = config.NAV_RECOVERY_FAST_CAPTURE_MIN_SCAN_S
_RECOVERY_FAST_CAPTURE_FRAMES = config.NAV_RECOVERY_FAST_CAPTURE_FRAMES
_RECOVERY_FAST_CAPTURE_CENTER_M = config.NAV_RECOVERY_FAST_CAPTURE_CENTER_M
_RECOVERY_FAST_CAPTURE_FAR_M = config.NAV_RECOVERY_FAST_CAPTURE_FAR_M
_RECOVERY_FAST_CAPTURE_SIDE_M = config.NAV_RECOVERY_FAST_CAPTURE_SIDE_M
_YAW_SMOOTH_ALPHA = 0.35
_YAW_MAX_STEP = 8.0
_SLOW_ZONE_RAW_M = 1.05
_RECOVERY_CLEAR_M = config.NAV_FORWARD_RESUME_M
_SPEED_OPEN_M = config.NAV_SPEED_OPEN_M
_SPEED_CRUISE_M = config.NAV_SPEED_CRUISE_M
_SPEED_CAUTION_M = config.NAV_SPEED_CAUTION_M
_SPEED_SLOW_M = config.NAV_SPEED_SLOW_M
_SPEED_MIN_FORWARD_M = config.NAV_SPEED_MIN_FORWARD_M
_SIDE_TIGHT_M = config.NAV_SIDE_TIGHT_M
_SIDE_CAUTION_M = config.NAV_SIDE_CAUTION_M
_NARROW_NO_FORWARD_SIDE_M = config.NAV_NARROW_NO_FORWARD_SIDE_M
_NARROW_OPEN_CENTER_M = config.NAV_NARROW_OPEN_CENTER_M
_NARROW_SLOW_SIDE_M = config.NAV_NARROW_SLOW_SIDE_M
_DOORWAY_CENTER_M = config.NAV_DOORWAY_CENTER_M
_DOORWAY_MIN_SIDE_M = config.NAV_DOORWAY_MIN_SIDE_M
_DOORWAY_SPEED_CAP = config.NAV_DOORWAY_SPEED_CAP
_FLAT_FRONT_CENTER_M = config.NAV_FLAT_FRONT_CENTER_M
_FLAT_FRONT_SPREAD_M = config.NAV_FLAT_FRONT_SPREAD_M
_FLAT_FRONT_FAR_M = config.NAV_FLAT_FRONT_FAR_M
_YAW_DIRECTION = 1.0

_FORWARD_VELOCITY_MPS = (
    config.RC_FORWARD_CAP / 100.0 * config.TELLO_SPEED_CMS / 100.0
)


def _dist(sec) -> float:
    return sec.raw_p10 if sec.metric else (1.0 - sec.norm_p90)


def _far(sec) -> float:
    return sec.raw_p90 if sec.metric else _dist(sec)


def _yaw_cmd(value: float) -> float:
    return _YAW_DIRECTION * value


@dataclass
class _State:
    forward_engaged: bool = False
    clear_streak: int = 0
    brake_until: float = 0.0
    last_forward_ts: float = field(default_factory=time.perf_counter)
    explore_dir: float = 0.0
    explore_started_ts: float = 0.0
    explore_until: float = 0.0
    explore_cooldown_until: float = 0.0
    recovery_clear_streak: int = 0
    recovery_capture_streak: int = 0
    last_explore_choice: float = -1.0
    last_yaw_cmd: float = 0.0


class YawPolicy:
    """Stateful reactive policy. One instance per flight session."""

    handles_safety_stop = True

    def __init__(
        self,
        pipeline: DepthPipeline,
        yolo_detector: Optional["YoloDetector"] = None,
    ) -> None:
        self._pipeline = pipeline
        self._yolo = yolo_detector
        self._st = _State()
        self.debug = {}

    def __call__(self, _client: TelloClient, _t: float) -> RcCommand:
        return self._decide(self._pipeline.get_latest())

    def _decide(self, pr: PipelineResult | None) -> RcCommand:
        now = time.perf_counter()

        if now < self._st.brake_until:
            self._set_debug("brake", None)
            return RcCommand(fb=_BRAKE_FB)

        if pr is None:
            self._set_debug("no_depth", None)
            return self._stop(now)
        if (time.time_ns() - pr.frame_ts_ns) / 1e6 > _DEPTH_STALE_MS:
            self._set_debug("stale_depth", pr)
            return self._stop(now)
        if len(pr.sectors) < 3:
            self._set_debug("no_sectors", pr)
            return self._stop(now)

        left, center, right = pr.sectors[0], pr.sectors[1], pr.sectors[2]
        yolo = self._yolo_hazards(now)

        l_raw = _dist(left)
        c_raw = _dist(center)
        r_raw = _dist(right)
        l_far = _far(left)
        c_far = _far(center)
        r_far = _far(right)

        raw_dist = [l_raw, c_raw, r_raw]
        yolo_block = [
            bool(i < len(yolo) and yolo[i] and raw_dist[i] < config.NAV_YOLO_DISTANCE_GATE_M)
            for i in range(3)
        ]
        velocity = _FORWARD_VELOCITY_MPS if self._st.forward_engaged else 0.0
        margin = _INFLATION_M + velocity * _MOTION_LATENCY

        def eff(raw: float) -> float:
            return max(0.0, raw - margin)

        l_eff = eff(l_raw)
        c_eff = eff(c_raw)
        center_current_clear = (
            not center.safety_stop
            and not yolo_block[1]
            and c_eff >= _RESUME_M
        )
        r_eff = eff(r_raw)

        l_block = left.safety_stop or yolo_block[0]
        c_block = (center.safety_stop or yolo_block[1]) and not center_current_clear
        r_block = right.safety_stop or yolo_block[2]
        front_spread = max(l_eff, c_eff, r_eff) - min(l_eff, c_eff, r_eff)
        flat_front = (
            c_eff < _FLAT_FRONT_CENTER_M
            and c_far < _FLAT_FRONT_FAR_M
            and front_spread < _FLAT_FRONT_SPREAD_M
        )
        debug_fields = dict(
            l_raw=l_raw, c_raw=c_raw, r_raw=r_raw,
            l_far=l_far, c_far=c_far, r_far=r_far,
            l_eff=l_eff, c_eff=c_eff, r_eff=r_eff,
            l_block=l_block, c_block=c_block, r_block=r_block,
            flat_front=flat_front,
            front_spread=front_spread,
            yolo=yolo,
            yolo_block=yolo_block,
        )
        self._set_debug("deciding", pr, **debug_fields)

        if self._should_recover(now, l_eff, c_eff, r_eff, c_far):
            self._set_debug("stuck_recovery_scan", pr, **debug_fields)
            return self._explore(now, l_eff, c_eff, r_eff, l_far, c_far, r_far)

        if yolo_block[1]:
            if self._st.forward_engaged:
                self._set_debug("yolo_center_stop", pr, **debug_fields)
                return self._stop(now)
            self._set_debug("yolo_center_yaw", pr, **debug_fields)
            return self._yaw_freer(l_block, r_block, l_eff, r_eff)

        # Long corridors and doors often look like "very far centre + closer
        # sides". That is the target, not a reason to scan. Only stop for a
        # genuinely close centre obstacle.
        center_too_close = c_block or c_eff < _STOP_M or flat_front
        if center_too_close:
            if self._st.forward_engaged:
                self._set_debug("center_hard_stop", pr, **debug_fields)
                return self._stop(now)
            self._set_debug("center_hard_yaw", pr, **debug_fields)
            return self._yaw_freer_or_explore(
                l_block, r_block, l_eff, c_eff, r_eff, l_far, c_far, r_far, now
            )

        steer_yaw = self._steer_yaw(
            l_block, r_block, l_eff, c_eff, r_eff, l_far, c_far, r_far
        )

        if self._st.forward_engaged:
            self._st.explore_dir = 0.0
            self._st.recovery_clear_streak = 0
            speed_cap = self._speed_cap(c_eff, min(l_eff, r_eff))
            self._set_debug(
                "forward", pr, **debug_fields, steer_yaw=steer_yaw, speed_cap=speed_cap
            )
            cmd = self._forward_cmd(
                steer_yaw, center_dist=c_eff, side_clearance=min(l_eff, r_eff)
            )
            if cmd.fb > 0:
                self._st.last_forward_ts = now
            else:
                self._st.forward_engaged = False
                self._st.clear_streak = 0
            return cmd

        if c_eff >= _RESUME_M and not c_block:
            self._st.clear_streak += 1
            if self._st.clear_streak >= _RESUME_FRAMES:
                speed_cap = self._speed_cap(c_eff, min(l_eff, r_eff))
                self._set_debug(
                    "resume_forward", pr, **debug_fields,
                    steer_yaw=steer_yaw, speed_cap=speed_cap,
                )
                cmd = self._forward_cmd(
                    steer_yaw, center_dist=c_eff, side_clearance=min(l_eff, r_eff)
                )
                if cmd.fb > 0:
                    self._st.forward_engaged = True
                    self._st.clear_streak = 0
                    self._st.last_forward_ts = now
                return cmd
        else:
            self._st.clear_streak = 0

        if abs(steer_yaw) > 0:
            self._set_debug("align_before_forward", pr, **debug_fields, steer_yaw=steer_yaw)
            return self._yaw_only(steer_yaw)
        self._set_debug("hover_wait_clear", pr, **debug_fields, steer_yaw=steer_yaw)
        return HOVER

    def _set_debug(self, reason: str, pr: PipelineResult | None, **fields) -> None:
        dbg = {
            "reason": reason,
            "recovery_dir": self._st.explore_dir,
            "recovery_clear_streak": self._st.recovery_clear_streak,
            "recovery_capture_streak": self._st.recovery_capture_streak,
            **fields,
        }
        if pr is not None and len(pr.sectors) >= 3:
            l, c, r = pr.sectors[0], pr.sectors[1], pr.sectors[2]
            dbg.update(
                nav_L_raw=round(_dist(l), 3),
                nav_C_raw=round(_dist(c), 3),
                nav_R_raw=round(_dist(r), 3),
                nav_L_far=round(_far(l), 3),
                nav_C_far=round(_far(c), 3),
                nav_R_far=round(_far(r), 3),
                nav_L_stop=int(l.safety_stop),
                nav_C_stop=int(c.safety_stop),
                nav_R_stop=int(r.safety_stop),
            )
        self.debug = dbg

    def _stop(self, now: float) -> RcCommand:
        if self._st.forward_engaged:
            self._st.brake_until = now + _BRAKE_DUR
            self._st.forward_engaged = False
            self._st.clear_streak = 0
            self._st.last_yaw_cmd = 0.0
            return RcCommand(fb=_BRAKE_FB)
        self._st.clear_streak = 0
        self._st.last_yaw_cmd = 0.0
        return HOVER

    def _steer_yaw(
        self,
        l_block: bool,
        r_block: bool,
        l_eff: float,
        c_eff: float,
        r_eff: float,
        l_far: float,
        c_far: float,
        r_far: float,
    ) -> float:
        # Continuous potential-field steering:
        # - side walls repel the drone smoothly,
        # - wider side clearance recentres it,
        # - far openings attract only gently so the centre path remains primary.
        # Positive yaw means RIGHT, negative yaw means LEFT.
        l_rep = self._side_repulsion(l_eff)
        r_rep = self._side_repulsion(r_eff)
        wall_yaw = (l_rep - r_rep) * _AVOID_YAW_CAP

        side_diff = r_eff - l_eff
        if abs(side_diff) < _STEER_DEADBAND_M:
            center_yaw = 0.0
        else:
            scale = min(1.0, abs(side_diff) / max(max(l_eff, r_eff), 0.1) * _FIELD_CENTER_GAIN)
            center_yaw = math.copysign(_CRUISE_YAW_CAP * scale, side_diff)

        far_yaw = 0.0
        if c_eff < _SPEED_OPEN_M:
            far_diff = r_far - l_far
            if abs(far_diff) > _RECOVERY_FAR_BAND_M:
                far_yaw = max(
                    -_FIELD_FAR_CAP,
                    min(_FIELD_FAR_CAP, far_diff * _FIELD_FAR_GAIN),
                )

        yaw = wall_yaw + center_yaw + far_yaw
        return max(-_FIELD_YAW_CAP, min(_FIELD_YAW_CAP, yaw))

    @staticmethod
    def _side_repulsion(side_eff: float) -> float:
        if side_eff >= _FIELD_SIDE_SAFE_M:
            return 0.0
        x = (_FIELD_SIDE_SAFE_M - max(side_eff, _FIELD_SIDE_CRITICAL_M)) / _FIELD_SIDE_SAFE_M
        return x * x

    def _side_aperture_is_promising(
        self,
        side_eff: float,
        side_far: float,
        center_eff: float,
        center_far: float,
    ) -> bool:
        if center_eff < _RESUME_M:
            return False
        if side_eff < _SIDE_EDGE_SOFT_M:
            return False
        return side_far >= max(_SIDE_MIN, center_far + _APERTURE_ADVANTAGE_M)

    def _side_avoid_yaw(self, direction: float, side_eff: float) -> float:
        closeness = max(0.0, min(1.0, (_SIDE_MIN - side_eff) / max(_SIDE_MIN, 0.1)))
        yaw = _CRUISE_YAW_CAP + (_AVOID_YAW_CAP - _CRUISE_YAW_CAP) * closeness
        return math.copysign(yaw, direction)

    def _smooth_yaw(self, target: float, max_step: float = _YAW_MAX_STEP) -> float:
        blended = self._st.last_yaw_cmd + (target - self._st.last_yaw_cmd) * _YAW_SMOOTH_ALPHA
        delta = max(-max_step, min(max_step, blended - self._st.last_yaw_cmd))
        yaw = self._st.last_yaw_cmd + delta
        if abs(yaw) < 3.0:
            yaw = 0.0
        self._st.last_yaw_cmd = yaw
        return yaw

    def _forward_cmd(self, yaw: float, center_dist: float, side_clearance: float) -> RcCommand:
        yaw = self._smooth_yaw(yaw)
        yaw_ratio = min(1.0, abs(yaw) / max(_AVOID_YAW_CAP, 1.0))
        cap = self._speed_cap(center_dist, side_clearance)
        if cap <= 0.0:
            return RcCommand(yaw=_yaw_cmd(yaw))
        fb = cap * (1.0 - 0.55 * yaw_ratio)
        min_forward = min(cap, 14.0 if yaw_ratio > 0.55 else 18.0)
        return RcCommand(fb=max(min_forward, fb), yaw=_yaw_cmd(yaw))

    def _speed_cap(self, center_dist: float, side_clearance: float) -> float:
        doorway_open = (
            center_dist >= _DOORWAY_CENTER_M
            and side_clearance >= _DOORWAY_MIN_SIDE_M
        )
        if side_clearance < 0.25 and center_dist < _SPEED_CAUTION_M:
            return 0.0

        if center_dist < _SPEED_MIN_FORWARD_M:
            return 0.0
        if center_dist >= _SPEED_OPEN_M:
            cap = float(config.RC_FORWARD_CAP)
        elif center_dist >= _SPEED_CRUISE_M:
            cap = float(config.RC_FORWARD_CAP)
        elif center_dist >= _SPEED_CAUTION_M:
            cap = 26.0
        elif center_dist >= _SPEED_SLOW_M:
            cap = 22.0
        else:
            cap = 18.0

        if doorway_open:
            cap = min(cap, _DOORWAY_SPEED_CAP)
        if side_clearance < _NARROW_NO_FORWARD_SIDE_M:
            cap = min(cap, 12.0)
        elif side_clearance < _NARROW_SLOW_SIDE_M:
            cap = min(cap, 16.0)
        elif side_clearance < _SIDE_TIGHT_M and center_dist < _SPEED_CRUISE_M:
            cap = min(cap, 20.0)
        elif side_clearance < _SIDE_CAUTION_M and center_dist < _SPEED_CAUTION_M:
            cap = min(cap, 26.0)
        return cap

    def _yaw_freer(self, l_block: bool, r_block: bool, l_eff: float, r_eff: float) -> RcCommand:
        l_ok = not l_block
        r_ok = not r_block
        if l_ok and not r_ok:
            return self._yaw_only(-_AVOID_YAW_CAP)
        if r_ok and not l_ok:
            return self._yaw_only(_AVOID_YAW_CAP)
        if l_ok and r_ok:
            return (
                self._yaw_only(-_AVOID_YAW_CAP)
                if l_eff >= r_eff
                else self._yaw_only(_AVOID_YAW_CAP)
            )
        return HOVER

    def _yaw_freer_or_explore(
        self,
        l_block,
        r_block,
        l_eff,
        c_eff,
        r_eff,
        l_far,
        c_far,
        r_far,
        now,
    ) -> RcCommand:
        cmd = self._yaw_freer(l_block, r_block, l_eff, r_eff)
        if cmd is not HOVER:
            return cmd
        stuck_s = now - self._st.last_forward_ts
        if stuck_s >= _EXPLORE_TRIGGER and now >= self._st.explore_cooldown_until:
            return self._explore(now, l_eff, c_eff, r_eff, l_far, c_far, r_far)
        if now < self._st.explore_until and self._st.explore_dir != 0.0:
            return self._yaw_only(self._st.explore_dir, max_step=_RECOVERY_YAW_STEP)
        return HOVER

    def _should_recover(
        self,
        now: float,
        l_eff: float,
        c_eff: float,
        r_eff: float,
        c_far: float | None = None,
    ) -> bool:
        if self._st.forward_engaged:
            return False
        if self._st.explore_dir != 0.0 and now < self._st.explore_cooldown_until:
            if now >= self._st.explore_until:
                self._st.explore_dir = -self._st.explore_dir
                self._st.explore_started_ts = now
                self._st.explore_until = now + _EXPLORE_SPIN
                self._st.explore_cooldown_until = self._st.explore_until + _EXPLORE_COOL
                self._st.recovery_clear_streak = 0
                self._st.recovery_capture_streak = 0

            in_min_scan = now - self._st.explore_started_ts < _RECOVERY_MIN_SCAN_S
            in_fast_min_scan = (
                now - self._st.explore_started_ts < _RECOVERY_FAST_CAPTURE_MIN_SCAN_S
            )
            strong_route_capture = (
                not in_fast_min_scan
                and c_eff >= _RECOVERY_FAST_CAPTURE_CENTER_M
                and (c_far is None or c_far >= _RECOVERY_FAST_CAPTURE_FAR_M)
                and max(l_eff, r_eff) >= _RECOVERY_FAST_CAPTURE_SIDE_M
                and min(l_eff, r_eff) >= _NARROW_NO_FORWARD_SIDE_M
            )
            doorway_capture = (
                not in_fast_min_scan
                and c_eff >= _DOORWAY_CENTER_M
                and (c_far is None or c_far >= _RECOVERY_FAST_CAPTURE_FAR_M)
                and min(l_eff, r_eff) >= _DOORWAY_MIN_SIDE_M
            )
            if strong_route_capture or doorway_capture:
                self._st.recovery_capture_streak += 1
            else:
                self._st.recovery_capture_streak = 0
            if self._st.recovery_capture_streak >= _RECOVERY_FAST_CAPTURE_FRAMES:
                self._st.explore_dir = 0.0
                self._st.explore_until = 0.0
                self._st.explore_cooldown_until = 0.0
                self._st.recovery_clear_streak = 0
                self._st.recovery_capture_streak = 0
                self._st.clear_streak = _RESUME_FRAMES
                return False

            route_capture = (
                not in_min_scan
                and c_eff >= _RECOVERY_CAPTURE_CENTER_M
                and (c_far is None or c_far >= _RECOVERY_CAPTURE_FAR_M)
                and max(l_eff, r_eff) >= _RECOVERY_CAPTURE_SIDE_M
            )
            if route_capture:
                self._st.explore_dir = 0.0
                self._st.explore_until = 0.0
                self._st.explore_cooldown_until = 0.0
                self._st.recovery_clear_streak = 0
                self._st.recovery_capture_streak = 0
                self._st.clear_streak = _RESUME_FRAMES
                return False

            clear_enough = (
                c_eff >= _RECOVERY_EXIT_CENTER_M
                and min(l_eff, r_eff) >= _SIDE_EDGE_SOFT_M
            )
            if clear_enough:
                self._st.recovery_clear_streak += 1
            else:
                self._st.recovery_clear_streak = 0

            if not in_min_scan and self._st.recovery_clear_streak >= _RECOVERY_EXIT_FRAMES:
                self._st.explore_dir = 0.0
                self._st.explore_until = 0.0
                self._st.explore_cooldown_until = 0.0
                self._st.recovery_clear_streak = 0
                self._st.recovery_capture_streak = 0
                return False
            return True
        stuck_s = now - self._st.last_forward_ts
        if stuck_s < _EXPLORE_TRIGGER:
            return False
        # Dead-end/corner band: the centre is not "crash-close", but it is too
        # tight to resume. In that band plain hover can get stuck forever.
        return c_eff < _RECOVERY_EXIT_CENTER_M

    def _explore(
        self,
        now: float,
        l_eff: float,
        c_eff: float,
        r_eff: float,
        l_far: float | None = None,
        c_far: float | None = None,
        r_far: float | None = None,
    ) -> RcCommand:
        if self._st.forward_engaged:
            return self._stop(now)
        if self._st.explore_dir != 0.0 and now < self._st.explore_cooldown_until:
            return self._yaw_only(self._st.explore_dir, max_step=_RECOVERY_YAW_STEP)
        if now < self._st.explore_cooldown_until:
            return HOVER
        if self._st.explore_dir != 0.0:
            prefer_left = self._st.explore_dir < 0
        elif (
            l_far is not None
            and r_far is not None
            and math.isfinite(l_far)
            and math.isfinite(r_far)
            and abs(l_far - r_far) >= _RECOVERY_FAR_BAND_M
        ):
            prefer_left = l_far > r_far
        elif abs(l_eff - r_eff) < _RECOVERY_ALTERNATE_BAND_M:
            prefer_left = random.choice((True, False))
        else:
            prefer_left = l_eff >= r_eff
        self._st.last_explore_choice = -1.0 if prefer_left else 1.0
        self._st.explore_dir = -_RECOVERY_YAW_CAP if prefer_left else _RECOVERY_YAW_CAP
        self._st.explore_started_ts = now
        self._st.recovery_clear_streak = 0
        self._st.recovery_capture_streak = 0
        self._st.explore_until = now + _EXPLORE_SPIN
        self._st.explore_cooldown_until = now + _EXPLORE_SPIN + _EXPLORE_COOL
        return self._yaw_only(self._st.explore_dir, max_step=_RECOVERY_YAW_STEP)

    def _yaw_only(self, target_yaw: float, max_step: float = _YAW_MAX_STEP) -> RcCommand:
        return RcCommand(yaw=_yaw_cmd(self._smooth_yaw(target_yaw, max_step=max_step)))

    def _yolo_hazards(self, now: float) -> List[bool]:
        if self._yolo is None:
            return [False, False, False]
        r = self._yolo.get_latest()
        if r is None:
            return [False, False, False]
        if (time.time_ns() - r.frame_ts_ns) / 1e6 > _YOLO_STALE_MS:
            return [False, False, False]
        return r.hazard_sectors


def make_yaw_policy(pipeline: DepthPipeline, yolo_detector=None):
    """ControlLoop-compatible policy factory."""
    return YawPolicy(pipeline, yolo_detector=yolo_detector)
