"""Fixed-rate reactive control loop.

Spec philosophy: hover on uncertainty. Hover on degraded comms.
The loop reads watchdog state and either runs the supplied policy
or overrides with a safe action.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from src import config
from src.tello_client import TelloClient
from src.watchdog import CommState, LoopDeadlineWatchdog, comm_state_from_age_ms

if TYPE_CHECKING:
    from src.perception.depth_pipeline import DepthPipeline


@dataclass
class RcCommand:
    lr: float = 0.0
    fb: float = 0.0
    ud: float = 0.0
    yaw: float = 0.0


HOVER = RcCommand()

# Policy: receives (client, t_loop_s) -> RcCommand.
Policy = Callable[[TelloClient, float], RcCommand]


def hover_policy(_client: TelloClient, _t: float) -> RcCommand:
    return HOVER


class ControlLoop:
    def __init__(
        self,
        client: TelloClient,
        policy: Policy,
        duration_s: float,
        telemetry=None,
        loop_hz: int = config.LOOP_HZ,
        sample_every_n_ticks: Optional[int] = None,
        depth_pipeline: Optional["DepthPipeline"] = None,
        install_signal_handler: bool = True,
    ) -> None:
        self._client = client
        self._policy = policy
        self._duration_s = duration_s
        self._telemetry = telemetry
        self._period_s = 1.0 / loop_hz
        self._deadline = LoopDeadlineWatchdog()
        self._sample_every = sample_every_n_ticks or max(1, loop_hz // config.TELEMETRY_SAMPLE_HZ)
        self._depth_pipeline = depth_pipeline
        self._policy_handles_safety_stop = bool(getattr(policy, "handles_safety_stop", False))
        self._stop = False
        self._safety_stop_active = False  # edge-detection for logging
        self._previous_sigint_handler = None
        if install_signal_handler:
            self._install_signal_handler()

    def _install_signal_handler(self) -> None:
        def _handler(signum, _frame):
            self._stop = True
            if self._telemetry is not None:
                self._telemetry.event("signal_received", signum=signum)
        try:
            self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            # Not in main thread --- caller is responsible.
            pass

    def _restore_signal_handler(self) -> None:
        if self._previous_sigint_handler is None:
            return
        try:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
        except ValueError:
            pass
        self._previous_sigint_handler = None

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        t_start = time.perf_counter()
        tick = 0

        if self._telemetry is not None:
            self._telemetry.event(
                "loop_start",
                loop_hz=int(round(1.0 / self._period_s)),
                duration_s=self._duration_s,
            )

        while not self._stop:
            self._deadline.tick_start()
            now = time.perf_counter()
            t_loop = now - t_start
            if t_loop >= self._duration_s:
                break

            # Derive comm state directly from state-packet age.
            state_age_ms = self._client.last_state_age_ms()
            comm_state = comm_state_from_age_ms(state_age_ms)

            # Decide action.
            if comm_state == CommState.LOST:
                self._on_comm_lost()
                break
            if comm_state == CommState.DEGRADED:
                cmd = HOVER
                degraded = True
            else:
                cmd = self._policy(self._client, t_loop)
                degraded = False

            # Older hover-only policies need a hard override. Navigation
            # policies handle depth stops themselves so they can still yaw.
            safety_stop = self._check_safety_stop()
            if safety_stop and not self._policy_handles_safety_stop:
                cmd = HOVER

            # Dispatch.
            try:
                self._client.send_rc(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)
                send_ok = True
                send_err = ""
            except Exception as e:
                send_ok = False
                send_err = str(e)
                if self._telemetry is not None:
                    self._telemetry.event("send_rc_failed", err=send_err)

            # Telemetry sample.
            if tick % self._sample_every == 0:
                self._sample(t_loop, state_age_ms, comm_state, cmd, degraded, send_ok, safety_stop)

            # Sleep to deadline.
            overrun = self._deadline.overrun_ms(self._period_s)
            if overrun > 0:
                if self._telemetry is not None:
                    self._telemetry.event("loop_overrun_ms", value=overrun, tick=tick)
            else:
                remain = self._period_s - (time.perf_counter() - now)
                if remain > 0:
                    time.sleep(remain)

            tick += 1

        # Loop end --- send one final hover for safety.
        try:
            self._client.send_rc(0, 0, 0, 0)
        except Exception:
            pass

        if self._telemetry is not None:
            self._telemetry.event("loop_end", ticks=tick, elapsed_s=time.perf_counter() - t_start)
        self._restore_signal_handler()

    _DEPTH_STALE_MS: float = 400.0  # ignore depth older than this

    def _check_safety_stop(self) -> bool:
        if self._depth_pipeline is None:
            return False
        pr = self._depth_pipeline.get_latest()
        if pr is None:
            return False
        age_ms = (time.time_ns() - pr.frame_ts_ns) / 1e6
        if age_ms > self._DEPTH_STALE_MS:
            return False  # stale data --- don't block flight
        stop = pr.safety_stop
        # Log edge transitions only (not every tick) to keep events.jsonl clean.
        if stop != self._safety_stop_active:
            self._safety_stop_active = stop
            if self._telemetry is not None:
                self._telemetry.event(
                    "safety_stop_changed",
                    active=stop,
                    sectors=[
                        {"id": str(s.sector_id), "stop": s.safety_stop,
                         "p10_m": round(s.raw_p10, 3) if s.metric else None,
                         "norm_p90": round(s.norm_p90, 3) if not s.metric else None}
                        for s in pr.sectors
                    ],
                )
        return stop

    def _on_comm_lost(self) -> None:
        if self._telemetry is not None:
            self._telemetry.event("comm_lost", state_age_ms=self._client.last_state_age_ms())
        # Send a final hover (best effort; UDP may be dropped). Do NOT attempt
        # blocking land() here --- that will stall on a dead link. The drone's
        # own 5s no-command watchdog will auto-hover then auto-land. The
        # calling script's cleanup will retry land() if comms recover.
        try:
            self._client.send_rc(0, 0, 0, 0)
        except Exception:
            pass

    def _sample(
        self,
        t_loop: float,
        state_age_ms: float,
        comm_state: CommState,
        cmd: RcCommand,
        degraded: bool,
        send_ok: bool,
        safety_stop: bool = False,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            st = self._client.read_state()
            nav = getattr(self._policy, "debug", {}) or {}
            self._telemetry.sample(
                t_loop_s=round(t_loop, 4),
                state_age_ms=round(state_age_ms, 2),
                comm=comm_state.value,
                degraded=int(degraded),
                send_ok=int(send_ok),
                rc_lr=cmd.lr, rc_fb=cmd.fb, rc_ud=cmd.ud, rc_yaw=cmd.yaw,
                battery=st.battery_pct, height=st.height_cm, baro=st.baro_cm,
                pitch=st.pitch_deg, roll=st.roll_deg, yaw=st.yaw_deg,
                vgx=st.vgx, vgy=st.vgy, vgz=st.vgz,
                temp_hi=st.temp_high_c, temp_lo=st.temp_low_c,
                flight_time=st.flight_time_s,
                safety_stop=int(safety_stop),
                nav_reason=nav.get("reason", ""),
                nav_L_raw=nav.get("nav_L_raw", ""),
                nav_C_raw=nav.get("nav_C_raw", ""),
                nav_R_raw=nav.get("nav_R_raw", ""),
                nav_L_far=nav.get("nav_L_far", ""),
                nav_C_far=nav.get("nav_C_far", ""),
                nav_R_far=nav.get("nav_R_far", ""),
                nav_L_eff=round(nav["l_eff"], 3) if "l_eff" in nav else "",
                nav_C_eff=round(nav["c_eff"], 3) if "c_eff" in nav else "",
                nav_R_eff=round(nav["r_eff"], 3) if "r_eff" in nav else "",
                nav_L_stop=nav.get("nav_L_stop", ""),
                nav_C_stop=nav.get("nav_C_stop", ""),
                nav_R_stop=nav.get("nav_R_stop", ""),
                nav_L_block=int(nav["l_block"]) if "l_block" in nav else "",
                nav_C_block=int(nav["c_block"]) if "c_block" in nav else "",
                nav_R_block=int(nav["r_block"]) if "r_block" in nav else "",
                nav_steer_yaw=round(nav["steer_yaw"], 2) if "steer_yaw" in nav else "",
                nav_speed_cap=round(nav["speed_cap"], 2) if "speed_cap" in nav else "",
                nav_recovery_dir=round(nav["recovery_dir"], 2) if "recovery_dir" in nav else "",
                nav_recovery_clear=nav.get("recovery_clear_streak", ""),
                nav_recovery_capture=nav.get("recovery_capture_streak", ""),
                nav_yolo="".join("1" if x else "0" for x in nav.get("yolo", [])),
                nav_yolo_block="".join("1" if x else "0" for x in nav.get("yolo_block", [])),
            )
        except Exception as e:
            self._telemetry.event("sample_failed", err=str(e))
