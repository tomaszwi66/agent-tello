"""Stage 5/6 autonomous flight: forward + reactive yaw + YOLO hazard avoidance.

Behaviour:
  - Steers toward the farthest sector (not just "forward if clear")
  - Detects corridor openings proactively and yaws into them
  - Corner artifact detection (prevents false "infinity" between perpendicular walls)
  - Window/mirror anomaly detection (physically impossible depth values)
  - YOLO detects TV/screens/reflective objects and treats them as blocked
  - Active braking on FORWARD -> stop transition (Tello has momentum)
  - Hysteresis: won't resume forward from a single clear frame
  - Exploration spin when stuck too long
  - Depth stale -> hover until stream recovers
  - Any exception -> guaranteed land() then emergency() fallback
  - Ctrl+C -> safe land (3s timeout, then forced exit)

Window (default on, --no-window to disable):
  Left  : RGB camera + YOLO boxes + policy decision
  Right : depth heatmap with sector overlay (close = hot, far = cool)
  q     : quit and land
  p     : screenshot to logs/

YOLO is optional: if models/yolov8n.pt not found, runs without it.
Pre-download: python scripts/download_yolo.py
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.control_loop import ControlLoop
from src.navigation.yaw_policy import _DEPTH_STALE_MS, make_yaw_policy
from src.perception.depth_pipeline import DepthPipeline
from src.perception.depth_smoother import TemporalSmoother
from src.perception.sectors import SectorID
from src.telemetry import RunLogger, force_exit, stderr
from src.tello_client import TelloClient
from src.video_stream import VideoStream

_SECTOR_COLORS = [(255, 100, 50), (50, 200, 50), (50, 100, 255)]
_STOP_COLOR = (0, 0, 220)
_YOLO_COLOR = (0, 200, 255)
_SCALE = 0.5
_LOG_INTERVAL_S = 3.0


def _sector_name(i: int, n: int) -> str:
    if n == 3:
        return SectorID(i).name
    if i == n // 2:
        return "C"
    return f"S{i}"


def _center_indices(n: int) -> list[int]:
    center = n // 2
    return [center]


# Visualization helpers

def _colorize(depth: np.ndarray, hw: tuple) -> np.ndarray:
    d = depth.astype(np.float32)
    lo, hi = d.min(), d.max()
    if hi - lo < 1e-6:
        n = np.zeros_like(d, dtype=np.uint8)
    else:
        n = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
        if config.DEPTH_MODEL_METRIC:
            n = 255 - n
    r = cv2.resize(n, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(r, cv2.COLORMAP_INFERNO)


def _put(img, text, xy, scale=0.55, fg=(255, 255, 255), bold=False):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, fg,
                2 if bold else 1, cv2.LINE_AA)


def _policy_label(pr, policy_obj=None) -> tuple:
    """Read actual policy state from policy object if available."""
    if pr is None:
        return "HOVER [no data]", (100, 100, 100)
    age_ms = (time.time_ns() - pr.frame_ts_ns) / 1e6
    if age_ms > _DEPTH_STALE_MS:
        return f"HOVER [stale {age_ms:.0f}ms]", (100, 100, 200)
    if len(pr.sectors) < 3:
        return "HOVER [no sectors]", (100, 100, 100)

    # Try to read actual policy state.
    if policy_obj is not None and hasattr(policy_obj, "_st"):
        st = policy_obj._st
        if st.forward_engaged:
            return "FORWARD >>", (50, 220, 50)
        if time.perf_counter() < st.brake_until:
            return "BRAKING |||", (200, 200, 0)
        if st.explore_dir != 0.0 and time.perf_counter() < st.explore_until:
            d = "LEFT" if st.explore_dir < 0 else "RIGHT"
            return f"EXPLORE {d}", (200, 100, 200)

    # Fall back to sector-based label.
    sectors = pr.sectors
    n = len(sectors)
    center_band = _center_indices(n)
    if not any(sectors[i].safety_stop for i in center_band):
        return "FORWARD >>", (50, 220, 50)
    mid = n // 2
    lf = any(not s.safety_stop for s in sectors[:mid])
    rf = any(not s.safety_stop for s in sectors[mid + 1:])
    if lf and not rf:
        return "YAW LEFT <<", (255, 180, 50)
    if rf and not lf:
        return "YAW RIGHT >>", (50, 180, 255)
    if lf and rf:
        return "YAW (compare)", (200, 200, 50)
    return "HOVER [blocked]", (0, 0, 200)


def _draw_depth(depth_vis, sectors, yolo_hazards=None):
    h, w = depth_vis.shape[:2]
    n = len(sectors)
    crop_y = int(h * config.SECTOR_FLOOR_CROP)
    cv2.line(depth_vis, (0, h - crop_y), (w, h - crop_y), (60, 60, 60), 1)
    if yolo_hazards is None:
        yolo_hazards = [False] * n
    for i, sec in enumerate(sectors):
        x0 = i * w // n
        x1 = (i + 1) * w // n if i < n - 1 else w
        is_yolo = yolo_hazards[i] if i < len(yolo_hazards) else False
        color = _YOLO_COLOR if is_yolo else (_STOP_COLOR if sec.safety_stop else _SECTOR_COLORS[i % 3])
        thick = 3 if (sec.safety_stop or is_yolo) else 2
        cv2.rectangle(depth_vis, (x0, 0), (x1 - 1, h - 1), color, thick)
        label = _sector_name(i, n)
        suffix = " YOLO" if is_yolo else (" STOP" if sec.safety_stop else "")
        _put(depth_vis, f"{label}{suffix}", (x0 + 6, 22), fg=color, bold=sec.safety_stop or is_yolo)
        if sec.metric:
            _put(depth_vis, f"p10={sec.raw_p10:.2f}m", (x0 + 6, 42), scale=0.45, fg=color)
        else:
            _put(depth_vis, f"n={sec.norm_p90:.2f}", (x0 + 6, 42), scale=0.45, fg=color)


def _draw_yolo_boxes(rgb, yolo_result, frame_hw):
    """Draw YOLO detection bounding boxes on the RGB frame."""
    if yolo_result is None:
        return
    h, w = frame_hw
    for det in yolo_result.detections:
        x1 = int(det.x1 * w); y1 = int(det.y1 * h)
        x2 = int(det.x2 * w); y2 = int(det.y2 * h)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), _YOLO_COLOR, 2)
        _put(rgb, f"{det.cls_name} {det.conf:.0%}", (x1, max(12, y1 - 5)),
             scale=0.45, fg=_YOLO_COLOR)


# Emergency landing

def _emergency_land(client, loop, stream, pipeline, yolo_det, loop_done, log, no_window):
    """Guaranteed land sequence. Never raises."""
    stderr("=== EMERGENCY LANDING SEQUENCE ===")
    if loop is not None:
        try:
            loop.request_stop()
        except Exception:
            pass
    if stream is not None:
        try:
            suppress = getattr(stream, "suppress_restarts", None)
            if suppress is not None:
                suppress(True)
        except Exception:
            pass
    for _ in range(5):
        try:
            client.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        time.sleep(0.05)
    if loop_done is not None:
        loop_done.wait(timeout=2.0)

    # land() blocks on UDP ACK; use thread + timeout so Ctrl+C is responsive.
    landed = False
    land_done = threading.Event()
    land_exc = [None]

    def _do_land():
        try:
            client.land()
        except Exception as e:
            land_exc[0] = e
        finally:
            land_done.set()

    threading.Thread(target=_do_land, daemon=True).start()
    if land_done.wait(timeout=8.0):
        if land_exc[0] is None:
            stderr("landed cleanly")
            landed = True
        else:
            try:
                client.sync_airborne_from_state()
                state = client.read_state()
                if not client.airborne or state.height_cm < 15:
                    stderr(f"land() returned error but Tello is grounded: {land_exc[0]}")
                    landed = True
                else:
                    raise RuntimeError("still airborne")
            except Exception:
                stderr(f"land() failed: {land_exc[0]} -> emergency()")
                try:
                    client.emergency()
                    stderr("motors cut")
                except Exception as e2:
                    stderr(f"emergency() failed too: {e2}")
    else:
        stderr("land() timeout 8s - checking state before emergency fallback")
        try:
            client.sync_airborne_from_state()
            state = client.read_state()
            if not client.airborne or state.height_cm < 15:
                stderr("Tello appears grounded after land timeout")
                landed = True
            else:
                try:
                    client.emergency()
                    stderr("motors cut after land timeout")
                except Exception as e2:
                    stderr(f"emergency() failed after land timeout: {e2}")
        except Exception as e:
            stderr(f"state unavailable after land timeout: {e}")
            try:
                client.emergency()
                stderr("motors cut after unknown land state")
            except Exception as e2:
                stderr(f"emergency() failed after unknown land state: {e2}")

    try:
        log.event("emergency_land_sequence", landed=landed)
    except Exception:
        pass
    for obj in (yolo_det, pipeline, stream):
        if obj is not None:
            try:
                obj.stop()
            except Exception:
                pass
    try:
        client.close()
    except Exception:
        pass
    if not no_window:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# Main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--mock-depth", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-yolo", action="store_true", help="disable YOLO even if model present")
    ap.add_argument("--scale", type=float, default=_SCALE)
    args = ap.parse_args()

    shot_idx = [0]

    with RunLogger("s5_forward_yaw") as log:
        client = TelloClient(telemetry=log)
        try:
            client.connect()
        except Exception as e:
            stderr(f"connect failed: {e}")
            return 2

                # Depth model warmup
        if args.mock_depth:
            from src.perception.mock_provider import MockDepthProvider
            provider = MockDepthProvider()
            provider.warmup()
        else:
            from src.perception.hf_depth_provider import build_active
            provider = build_active()
            stderr(f"warming up depth: {provider.name}")
            t0 = time.perf_counter()
            provider.warmup()
            stderr(f"depth warmup done in {time.perf_counter() - t0:.2f}s")

        # YOLO warmup (optional)
        yolo_det = None
        if not args.no_yolo:
            from src.perception.yolo_detector import build_yolo_detector
            model_path = str(
                Path(__file__).resolve().parents[1] / config.YOLO_MODEL_PATH
            )
            stderr(f"loading YOLO from {model_path} ...")
            t0 = time.perf_counter()
            yolo_det = build_yolo_detector(model_path)
            if yolo_det is not None:
                stderr(f"YOLO ready in {time.perf_counter() - t0:.2f}s")
                log.event("yolo_ready", model=model_path)
            else:
                stderr("YOLO unavailable - depth-only navigation")
                log.event("yolo_skipped")

        # Takeoff
        stderr("taking off...")
        try:
            client.takeoff()
        except Exception as e:
            stderr(f"takeoff failed: {e}")
            client.close()
            return 2

        # DRONE IS AIRBORNE. Everything below in try/finally.
        stream = None
        pipeline = None
        policy_obj = None
        loop = None
        loop_done = None
        try:
            stderr("stabilizing...")
            time.sleep(2.5)

            stream = VideoStream(client, telemetry=log)
            stream.start()

            if yolo_det is not None:
                yolo_det.attach_stream(stream)
                yolo_det.start()

            smoother = TemporalSmoother(
                asymmetric=True,
                alpha_approach=config.SMOOTHING_ALPHA_APPROACH,
                alpha_recede=config.SMOOTHING_ALPHA_RECEDE,
            )
            pipeline = DepthPipeline(stream, provider, smoother)
            pipeline.start()

            stderr("waiting for first depth frame...")
            t_wait = time.perf_counter()
            while pipeline.get_latest() is None:
                if time.perf_counter() - t_wait > 5.0:
                    raise RuntimeError("pipeline timeout - no depth in 5s")
                time.sleep(0.05)
            stderr("depth ready - reactive navigation active"
                   + (" + YOLO" if yolo_det else ""))

            stderr("checking fresh Tello state telemetry...")
            fresh_deadline = time.perf_counter() + 4.0
            fresh_state = False
            while time.perf_counter() < fresh_deadline:
                state_age_ms = client.last_state_age_ms()
                if state_age_ms < config.COMMAND_TIMEOUT_MS:
                    fresh_state = True
                    break
                try:
                    client.send_rc(0, 0, 0, 0)
                except Exception as e:
                    log.event("preloop_hover_failed", err=str(e))
                time.sleep(0.1)
            if not fresh_state:
                state_age_ms = client.last_state_age_ms()
                log.event("preloop_comm_stale", state_age_ms=round(state_age_ms, 1))
                raise RuntimeError(
                    f"state telemetry stale before control loop: {state_age_ms:.0f}ms"
                )

            policy_obj = make_yaw_policy(pipeline, yolo_detector=yolo_det)
            loop = ControlLoop(
                client=client, policy=policy_obj,
                duration_s=args.duration, telemetry=log,
                depth_pipeline=pipeline,
            )

            loop_done = threading.Event()
            loop_exc = [None]

            def _run_loop():
                try:
                    loop.run()
                except Exception as e:
                    loop_exc[0] = e
                    stderr(f"flight error: {e}")
                finally:
                    loop_done.set()

            flight_thread = threading.Thread(target=_run_loop, daemon=True)
            flight_thread.start()

            t_start = time.perf_counter()
            last_status = t_start
            last_log = t_start

            try:
                while not loop_done.is_set():
                    pr = pipeline.get_latest()
                    label, label_color = _policy_label(pr, policy_obj)
                    yolo_result = yolo_det.get_latest() if yolo_det else None
                    yolo_hazards = yolo_result.hazard_sectors if yolo_result else None

                    now = time.perf_counter()
                    if now - last_status >= 1.0:
                        elapsed = now - t_start
                        if pr is not None:
                            age = f"{(time.time_ns() - pr.frame_ts_ns) / 1e6:.0f}ms"
                            s = pr.sectors
                            vals = [
                                f"{_sector_name(i, len(s))}={sec.raw_p10:.2f}m"
                                if sec.metric else f"{_sector_name(i, len(s))}={sec.norm_p90:.2f}"
                                for i, sec in enumerate(s)
                            ]
                            sect = " ".join(vals)
                        else:
                            age, sect = "N/A", "no-data"
                        yolo_str = ""
                        if yolo_result:
                            names = [d.cls_name for d in yolo_result.detections]
                            yolo_str = f"  yolo=[{','.join(names) or 'clear'}]"
                        stderr(f"t={elapsed:.1f}s  {label}  age={age}  {sect}"
                               f"  stops={pipeline.stop_count}  fps={stream.fps():.1f}"
                               f"{yolo_str}")
                        last_status = now

                    if now - last_log >= _LOG_INTERVAL_S and pr is not None:
                        s = pr.sectors
                        yolo_names = (
                            [d.cls_name for d in yolo_result.detections]
                            if yolo_result else []
                        )
                        try:
                            mid = len(s) // 2
                            log.event(
                                "nav_state",
                                t_s=round(now - t_start, 2),
                                decision=label,
                                depth_age_ms=round(
                                    (time.time_ns() - pr.frame_ts_ns) / 1e6, 1
                                ),
                                L_p10=round(s[0].raw_p10, 3),
                                C_p10=round(s[mid].raw_p10, 3),
                                R_p10=round(s[-1].raw_p10, 3),
                                L_stop=int(s[0].safety_stop),
                                C_stop=int(s[mid].safety_stop),
                                R_stop=int(s[-1].safety_stop),
                                sector_p10=[round(sec.raw_p10, 3) for sec in s],
                                sector_stop=[int(sec.safety_stop) for sec in s],
                                stream_fps=round(stream.fps(), 1),
                                total_stops=pipeline.stop_count,
                                yolo_hazards=yolo_hazards or [False, False, False],
                                yolo_objects=yolo_names,
                            )
                        except Exception as log_e:
                            stderr(f"log.event failed (non-fatal): {log_e}")
                        last_log = now

                    if not args.no_window:
                        frame, _, _ = stream.get_latest()
                        side = None
                        if frame is not None:
                            h, w = frame.shape[:2]
                            rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            _draw_yolo_boxes(rgb, yolo_result, (h, w))
                            if pr is not None:
                                depth_vis = _colorize(pr.depth_map, (h, w))
                                _draw_depth(depth_vis, pr.sectors, yolo_hazards)
                            else:
                                depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
                                _put(depth_vis, "NO DEPTH",
                                     (w // 2 - 60, h // 2), scale=0.7)
                            _put(rgb, label, (8, h - 16),
                                 scale=0.65, fg=label_color, bold=True)
                            _put(rgb, f"t={now - t_start:.1f}s  fps={stream.fps():.1f}",
                                 (8, 22))
                            if yolo_det is None:
                                _put(rgb, "YOLO: off", (8, 42), scale=0.4,
                                     fg=(100, 100, 100))
                            side = np.hstack([rgb, depth_vis])
                            if args.scale != 1.0:
                                side = cv2.resize(
                                    side,
                                    (int(side.shape[1] * args.scale),
                                     int(side.shape[0] * args.scale)),
                                )
                            cv2.imshow("s5-forward-yaw", side)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            stderr("q pressed - landing")
                            break
                        if key == ord("p") and side is not None:
                            shot_dir = Path(__file__).resolve().parents[1] / "logs"
                            shot_dir.mkdir(exist_ok=True)
                            shot_idx[0] += 1
                            p = shot_dir / f"s5_shot_{shot_idx[0]}.png"
                            cv2.imwrite(str(p), side)
                            stderr(f"screenshot: {p}")
                    else:
                        time.sleep(0.05)
            except KeyboardInterrupt:
                stderr("Ctrl+C requested")

        except BaseException as e:
            stderr(f"FATAL airborne error: {e}")
            traceback.print_exc()
            try:
                log.event("airborne_fatal", err=str(e), tb=traceback.format_exc())
            except Exception:
                pass
        finally:
            _emergency_land(
                client, loop, stream, pipeline, yolo_det, loop_done, log, args.no_window
            )

        stderr(f"summary - depth_infer={pipeline.infer_count if pipeline else 0}  "
               f"stops={pipeline.stop_count if pipeline else 0}"
               + (f"  yolo_infer={yolo_det.infer_count}" if yolo_det else ""))
        try:
            log.event(
                "flight_summary",
                infer_count=pipeline.infer_count if pipeline else 0,
                stop_count=pipeline.stop_count if pipeline else 0,
                model=provider.name,
                yolo_infer_count=yolo_det.infer_count if yolo_det else 0,
            )
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    force_exit(main())


