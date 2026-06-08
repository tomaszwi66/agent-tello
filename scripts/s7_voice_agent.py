"""Voice-operated local physical agent for Tello.

The drone starts on the ground and waits for English voice commands. Gemma can
interpret fuzzy commands, but obvious safety commands are handled by a local
deterministic parser first.
"""

from __future__ import annotations

import argparse
import logging
import queue
import re
import sys
import threading
import time
import traceback
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.s5_forward_yaw import (
    _SCALE,
    _colorize,
    _draw_depth,
    _draw_yolo_boxes,
    _policy_label,
    _put,
)
from src import config
from src.agent.mission_agent import MissionAgent
from src.agent.ollama_vl import OllamaVisionClient, ToolCommand
from src.agent.speech_io import EnglishListener, EnglishSpeaker
from src.control_loop import ControlLoop
from src.navigation.yaw_policy import make_yaw_policy
from src.perception.depth_pipeline import DepthPipeline
from src.perception.depth_smoother import TemporalSmoother
from src.telemetry import RunLogger, force_exit, stderr
from src.tello_client import TelloClient
from src.video_stream import VideoStream

_SEARCH_GOAL_ASR_CORRECTIONS = {
    "chairs": "chair",
    "share": "chair",
    "stare": "chair",
    "cheer": "chair",
}


class VoiceDroneAgent:
    def __init__(self, args, log: RunLogger) -> None:
        self.args = args
        self.log = log
        self.speaker = EnglishSpeaker(
            enabled=args.speak,
            output_device=args.speaker_device,
            duplex_input_device=args.mic_device,
            start_silence_s=args.speaker_start_silence,
        )
        self.listener = EnglishListener(
            model_path=args.whisper_model,
            device=args.whisper_device,
            input_device=args.mic_device,
            min_rms=args.mic_min_rms,
        )
        self.vlm = OllamaVisionClient(
            model=args.model,
            vision_model=args.vision_model,
            host=args.ollama_host,
            timeout_s=args.vlm_timeout,
            image_width=args.vlm_image_width,
        )
        self.client = TelloClient(telemetry=log)
        self.provider = None
        self.stream = None
        self.pipeline = None
        self.yolo_det = None
        self.policy_obj = None
        self.loop = None
        self.loop_done = None
        self.flight_thread = None
        self.mission_agent = None
        self.preview_thread = None
        self.preview_stop = threading.Event()
        self.heartbeat_thread = None
        self.heartbeat_stop = threading.Event()
        self._autonomous = False
        self._manual_rc_active = False
        self._landing_active = False
        self._shot_idx = 0
        self._last_heard = ""
        self._silent_listens = 0
        self._vision_memory = deque(maxlen=6)

    def run(self) -> int:
        self._preflight()
        self.client.connect()
        self._start_hover_heartbeat()
        self._warm_vision()
        self._ensure_stream()
        self._start_preview()
        self.speaker.say("I am ready. Listening.", wait=True)
        stderr("Ready. Speak commands. Ctrl+C to land/exit.")

        try:
            while True:
                self._pump_window(0.05)
                cmd_text = self._listen_command()
                if not cmd_text:
                    self._silent_listens += 1
                    if self.args.prompt_after_silence > 0 and self._silent_listens >= self.args.prompt_after_silence:
                        self.speaker.say("Listening.", wait=False)
                        self._silent_listens = 0
                    continue
                self._silent_listens = 0
                stderr(f"heard: {cmd_text}")
                self._last_heard = cmd_text
                self.log.event("voice_heard", text=cmd_text)

                cmd = self._parse_command(cmd_text)
                if cmd is None:
                    try:
                        cmd = self.vlm.command(
                            cmd_text,
                            airborne=self.client.airborne,
                            autonomous=self._autonomous,
                        )
                    except Exception as e:
                        stderr(f"command parser failed: {e}")
                        self.log.event("agent_command_failed", heard=cmd_text, err=str(e))
                        self.speaker.say("I did not understand that command. Please repeat it.")
                        continue

                stderr(
                    f"tool={cmd.tool} arg={cmd.argument!r} conf={cmd.confidence:.2f}"
                )
                self.log.event(
                    "agent_tool",
                    heard=cmd_text,
                    tool=cmd.tool,
                    argument=cmd.argument,
                    confidence=round(cmd.confidence, 3),
                    answer=cmd.answer_text,
                )
                self._execute(cmd)
                self._log_drone_state(f"after_{cmd.tool}")
                self._check_battery_safety()
        except KeyboardInterrupt:
            stderr("Ctrl+C requested")
        finally:
            self.shutdown()
        return 0

    def _preflight(self) -> None:
        logging.getLogger("djitellopy").setLevel(logging.WARNING)
        logging.getLogger("djitellopy.tello").setLevel(logging.WARNING)
        if not self.listener.available():
            raise RuntimeError(
                f"no local Whisper model: {self.args.whisper_model}. "
                "Run scripts/download_whisper.py first."
            )
        stderr(f"using Whisper model: {self.listener.model_description()}")
        stderr(f"using microphone: {self.listener.input_description()}")
        stderr(f"using speaker: {self.speaker.output_description()}")
        stderr("warming up Whisper and microphone...")
        t0 = time.perf_counter()
        self.listener.warmup(audio_seconds=0.15)
        stderr(f"audio warmup done in {time.perf_counter() - t0:.2f}s")
        stderr(f"checking Ollama brain={self.vlm.model} vision={self.vlm.vision_model}...")
        self.vlm.ping()
        stderr("agent command parser ready")

    def _warm_vision(self) -> None:
        stderr("warming up vision model...")
        t0 = time.perf_counter()
        warm_frame = np.zeros((64, 96, 3), dtype=np.uint8)
        self.vlm.describe(warm_frame, "test")
        stderr(f"vision warmup done in {time.perf_counter() - t0:.2f}s")

    def _warm_perception(self) -> None:
        if self.provider is None and self.args.mock_depth:
            from src.perception.mock_provider import MockDepthProvider

            self.provider = MockDepthProvider()
            self.provider.warmup()
        elif self.provider is None:
            from src.perception.hf_depth_provider import build_active

            self.provider = build_active()
            stderr(f"warming up depth: {self.provider.name}")
            t0 = time.perf_counter()
            self.provider.warmup()
            stderr(f"depth warmup done in {time.perf_counter() - t0:.2f}s")

        if self.yolo_det is None and not self.args.no_yolo:
            from src.perception.yolo_detector import build_yolo_detector

            model_path = str(Path(__file__).resolve().parents[1] / config.YOLO_MODEL_PATH)
            stderr(f"loading YOLO from {model_path} ...")
            self.yolo_det = build_yolo_detector(model_path)
            if self.yolo_det is None:
                stderr("YOLO unavailable - depth-only navigation")

    def _ensure_stream(self) -> None:
        if self.stream is None:
            self.stream = VideoStream(self.client, telemetry=self.log)
            self.stream.start()
        if self.yolo_det is not None:
            self.yolo_det.attach_stream(self.stream)
            self.yolo_det.start()

    def _ensure_pipeline(self) -> None:
        self._warm_perception()
        self._ensure_stream()
        if self.pipeline is not None:
            return
        smoother = TemporalSmoother(
            asymmetric=True,
            alpha_approach=config.SMOOTHING_ALPHA_APPROACH,
            alpha_recede=config.SMOOTHING_ALPHA_RECEDE,
        )
        self.pipeline = DepthPipeline(self.stream, self.provider, smoother)
        self.pipeline.start()
        stderr("waiting for first depth frame...")
        deadline = time.perf_counter() + 5.0
        while self.pipeline.get_latest() is None:
            self._pump_window(0.02)
            if time.perf_counter() > deadline:
                raise RuntimeError("pipeline timeout - no depth in 5s")
            time.sleep(0.05)

    def _listen_command(self) -> str:
        if self.args.prompt_each_listen:
            self.speaker.say("Tell me what to do.", wait=True)
        stderr(
            f"listening (start_timeout={self.args.listen_start_timeout:.1f}s "
            f"max={self.args.listen_seconds:.1f}s silence={self.args.listen_silence:.1f}s)..."
        )
        if self.args.fixed_listen_window:
            if self.args.listen_beep:
                self.speaker.beep()
            text = self.listener.listen_once(
                seconds=self.args.listen_seconds,
                whisper_vad=self.args.whisper_vad,
            ).strip()
        else:
            text = self.listener.listen_utterance(
                start_timeout_s=self.args.listen_start_timeout,
                max_s=self.args.listen_seconds,
                silence_s=self.args.listen_silence,
                pre_roll_s=self.args.listen_pre_roll,
                whisper_vad=self.args.whisper_vad,
                cue=self.speaker.beep if self.args.listen_beep else None,
                input_warmup_s=self.args.listen_input_warmup,
                cue_tone_device=(
                    self.args.speaker_device
                    if self.args.listen_beep and self.args.speak
                    else None
                ),
            ).strip()
        stderr(
            f"audio rms={self.listener.last_rms:.5f} "
            f"dur={self.listener.last_duration_s:.2f}s "
            f"mode={self.listener.last_record_mode}"
        )
        return self._dedupe_repeated_text(text)

    def _parse_command(self, text: str) -> ToolCommand | None:
        t = self._norm(text)
        if not t:
            return None
        if self._is_whisper_boilerplate(t):
            return ToolCommand("ignore", "", "", 1.0, text)
        if t in ("thanks", "thank you", "ok", "okay", "alright", "good"):
            return ToolCommand("ignore", "", "", 1.0, text)
        if any(x in t for x in ("emergency", "cut motors", "kill motors", "motor stop")):
            return ToolCommand("emergency", "", "Emergency motor stop.", 1.0, text)
        if any(x in t for x in ("land", "landing", "end flight", "finish flight")):
            return ToolCommand("land", "", "Landing.", 1.0, text)
        if any(x in t for x in ("stop", "wait", "hold", "hover")):
            return ToolCommand("hover", "", "Holding position.", 1.0, text)
        m = re.search(r"(find|search for|look for|where is)\s+(.+)", t)
        if m:
            goal = self._canonical_search_goal(m.group(2).strip(" .?!"))
            return ToolCommand("search", goal, f"Searching for {goal}.", 0.95, text)
        if any(x in t for x in ("take off", "takeoff", "start")):
            return ToolCommand("takeoff", "", "Taking off.", 1.0, text)
        if any(x in t for x in ("what do you see", "describe", "tell me what you see")):
            return ToolCommand("describe", text, "Looking.", 1.0, text)
        m = re.search(r"(do you\s+)?see\s+(.+)", t)
        if m:
            goal = self._canonical_search_goal(m.group(2).strip(" .?!"))
            return ToolCommand("search", goal, f"Searching for {goal}.", 0.88, text)
        if "right" in t and any(x in t for x in ("turn", "rotate", "look")):
            return ToolCommand("turn_right", "", "Turning right ninety degrees.", 0.95, text)
        if "left" in t and any(x in t for x in ("turn", "rotate", "look")):
            return ToolCommand("turn_left", "", "Turning left ninety degrees.", 0.95, text)
        if "forward" in t or "move ahead" in t or "go ahead" in t:
            return ToolCommand("forward", "", "Moving forward.", 0.9, text)
        if "back" in t or "backward" in t:
            return ToolCommand("back", "", "Moving back.", 0.9, text)
        if "go up" in t or "move up" in t or "climb" in t or "higher" in t:
            return ToolCommand("up", "", "Climbing half a meter.", 0.95, text)
        if "go down" in t or "move down" in t or "descend" in t or "lower" in t:
            return ToolCommand("down", "", "Descending half a meter.", 0.95, text)
        if (
            re.search(r"\b(fly|explore|autonomous)\b", t)
            or any(x in t for x in ("fly around", "play around", "play round"))
        ):
            return ToolCommand("start_autonomy", "", "Flying autonomously.", 0.9, text)
        return None

    @staticmethod
    def _canonical_search_goal(goal: str) -> str:
        tokens = goal.split()
        corrected = [_SEARCH_GOAL_ASR_CORRECTIONS.get(tok, tok) for tok in tokens]
        return " ".join(corrected).strip()

    @staticmethod
    def _norm(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower().strip()

    @staticmethod
    def _is_whisper_boilerplate(norm_text: str) -> bool:
        junk = (
            "thank you for watching",
            "thanks for watching",
            "all rights reserved",
            "subtitles by",
            "subscribe to the channel",
            "this is a drone command",
        )
        return any(x in norm_text for x in junk)

    @staticmethod
    def _dedupe_repeated_text(text: str) -> str:
        text = " ".join(text.split())
        if not text:
            return ""
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if len(parts) >= 2 and VoiceDroneAgent._norm(parts[0]) == VoiceDroneAgent._norm(parts[1]):
            return parts[0]
        half = len(text) // 2
        if half > 6 and VoiceDroneAgent._norm(text[:half]) == VoiceDroneAgent._norm(text[half:]):
            return text[:half].strip()
        return text

    def _execute(self, cmd: ToolCommand) -> None:
        say = cmd.answer_text
        try:
            if cmd.tool == "takeoff":
                self.speaker.say(say or "Taking off.", wait=True)
                self._takeoff()
            elif cmd.tool == "land":
                self.speaker.say(say or "Landing.", wait=True)
                self._stop_autonomy()
                self._land()
            elif cmd.tool == "hover":
                self._stop_autonomy()
                self.client.send_rc(0, 0, 0, 0)
                self.speaker.say(say or "Holding position.", wait=True)
            elif cmd.tool == "emergency":
                self._stop_autonomy()
                self.client.emergency()
                self.speaker.say(say or "Emergency motor stop.", wait=True)
            elif cmd.tool == "start_autonomy":
                if not self._battery_ok_for_mission():
                    return
                self.speaker.say(say or "Flying autonomously.", wait=True)
                self._ensure_pipeline()
                self._takeoff()
                self._start_autonomy()
            elif cmd.tool == "stop_autonomy":
                self._stop_autonomy()
                self.speaker.say(say or "Stopping autonomous flight.", wait=True)
            elif cmd.tool == "search":
                goal = cmd.argument or "the requested target"
                if not self._battery_ok_for_mission():
                    return
                self.speaker.say(say or f"Searching for {goal}.", wait=True)
                self._ensure_pipeline()
                self._takeoff()
                self._start_autonomy()
                self._start_search(goal)
            elif cmd.tool == "describe":
                self._describe(cmd.argument or "Describe what you see.")
            elif cmd.tool == "turn_left":
                self._manual_rc(yaw=-45, seconds=2.0, phrase=say or "Turning left ninety degrees.")
            elif cmd.tool == "turn_right":
                self._manual_rc(yaw=45, seconds=2.0, phrase=say or "Turning right ninety degrees.")
            elif cmd.tool == "forward":
                self._manual_rc(fb=30, seconds=2.0, phrase=say or "Moving forward.")
            elif cmd.tool == "back":
                self._manual_rc(fb=-20, seconds=0.7, phrase=say or "Moving back.")
            elif cmd.tool == "up":
                self._vertical_move(up=True, cm=50, phrase=say or "Climbing half a meter.")
            elif cmd.tool == "down":
                self._vertical_move(up=False, cm=50, phrase=say or "Descending half a meter.")
            elif cmd.tool == "noop":
                self._reply_short(self._last_heard or cmd.raw_text or "Hello.")
            elif cmd.tool == "clarify":
                self._reply_short(self._last_heard or cmd.raw_text or "Please clarify.")
            elif cmd.tool == "ignore":
                self.log.event("voice_ignored", text=cmd.raw_text)
            else:
                self.speaker.say(say or "Please clarify the command.", wait=True)
        except Exception as e:
            stderr(f"tool failed: {e}")
            self.log.event("tool_failed", tool=cmd.tool, err=str(e), tb=traceback.format_exc())
            self.speaker.say(self._friendly_tool_error(e), wait=True)

    @staticmethod
    def _friendly_tool_error(err: Exception) -> str:
        text = str(err)
        if "10051" in text or "unreachable network" in text.lower():
            return (
                "I lost the Wi-Fi connection to Tello. "
                "Connect the computer to the drone network and try again."
            )
        if "timeout" in text.lower():
            return "Tello did not respond in time. Check the drone Wi-Fi and try again."
        return "I could not execute that command."

    def _takeoff(self) -> None:
        self.client.sync_airborne_from_state()
        if self.client.airborne:
            return
        self.client.takeoff()
        time.sleep(2.5)

    def _battery_ok_for_mission(self) -> bool:
        try:
            state = self.client.read_state()
        except Exception:
            return True
        if state.battery_pct < config.MIN_BATTERY_TAKEOFF_PCT:
            self.log.event(
                "mission_blocked_low_battery",
                battery=state.battery_pct,
                threshold=config.MIN_BATTERY_TAKEOFF_PCT,
            )
            self.speaker.say(
                f"Battery is {state.battery_pct} percent. I will not start a new mission.",
                wait=True,
            )
            return False
        return True

    def _land(self) -> None:
        self._landing_active = True
        try:
            self.client.sync_airborne_from_state()
            if not self.client.airborne:
                return
            try:
                self.client.land()
            except Exception as e:
                time.sleep(0.8)
                try:
                    self.client.sync_airborne_from_state()
                    state = self.client.read_state()
                    if not self.client.airborne or state.height_cm < 15:
                        self.log.event(
                            "land_response_error_but_grounded",
                            err=str(e),
                            height=state.height_cm,
                        )
                        return
                except Exception:
                    pass
                raise
        finally:
            self._landing_active = False

    def _start_hover_heartbeat(self) -> None:
        if self.heartbeat_thread is not None:
            return
        self.heartbeat_stop.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._hover_heartbeat_loop,
            daemon=True,
            name="s7-hover-heartbeat",
        )
        self.heartbeat_thread.start()

    def _hover_heartbeat_loop(self) -> None:
        active_prev = False
        last_err_log = 0.0
        while not self.heartbeat_stop.is_set():
            active = (
                self.client.airborne
                and not self._autonomous
                and not self._manual_rc_active
                and not self._landing_active
            )
            if active and not active_prev:
                self.log.event("hover_heartbeat_active", active=True)
                stderr("hover heartbeat active")
            elif not active and active_prev:
                self.log.event("hover_heartbeat_active", active=False)
            active_prev = active
            if active:
                try:
                    self.client.send_rc(0, 0, 0, 0)
                except Exception as e:
                    now = time.perf_counter()
                    if now - last_err_log > 2.0:
                        last_err_log = now
                        self.log.event("hover_heartbeat_failed", err=str(e))
                        stderr(f"hover heartbeat failed: {e}")
            time.sleep(0.30)

    def _start_autonomy(self) -> None:
        self._ensure_pipeline()
        if self._autonomous and self.loop_done is not None and not self.loop_done.is_set():
            return
        self.policy_obj = make_yaw_policy(self.pipeline, yolo_detector=self.yolo_det)
        self.loop = ControlLoop(
            client=self.client,
            policy=self.policy_obj,
            duration_s=self.args.max_autonomy_s,
            telemetry=self.log,
            depth_pipeline=self.pipeline,
            install_signal_handler=False,
        )
        self.loop_done = threading.Event()

        def run_loop():
            try:
                self.loop.run()
            finally:
                self.loop_done.set()
                self._autonomous = False

        self._autonomous = True
        self.flight_thread = threading.Thread(target=run_loop, daemon=True)
        self.flight_thread.start()

    def _stop_autonomy(self) -> None:
        if self.loop is not None:
            self.loop.request_stop()
        if self.loop_done is not None:
            self.loop_done.wait(timeout=2.0)
        self._autonomous = False
        try:
            self.client.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        if self.mission_agent is not None:
            self.mission_agent.stop()
            self.mission_agent = None

    def _start_search(self, goal: str) -> None:
        if self.mission_agent is not None:
            self.mission_agent.stop()
        self.mission_agent = MissionAgent(
            stream=self.stream,
            vlm=self.vlm,
            goal=goal,
            telemetry=self.log,
            speaker=self.speaker,
            yolo_detector=self.yolo_det,
            interval_s=self.args.agent_interval,
            found_confidence=self.args.found_confidence,
        )
        self.mission_agent.start()
        threading.Thread(target=self._watch_search_found, daemon=True).start()

    def _watch_search_found(self) -> None:
        agent = self.mission_agent
        while agent is not None and not agent.found():
            self._pump_window(0.1)
            time.sleep(0.1)
        if agent is not None and agent.found():
            self._remember_vision(f"I found the target: {agent.latest().goal}.")
            self._stop_autonomy()

    def _start_preview(self) -> None:
        if self.args.no_window or self.preview_thread is not None:
            return
        self.preview_stop.clear()
        self.preview_thread = threading.Thread(
            target=self._preview_loop,
            daemon=True,
            name="s7-preview",
        )
        self.preview_thread.start()

    def _preview_loop(self) -> None:
        while not self.preview_stop.is_set():
            self._draw_window_once()
            time.sleep(0.03)

    def _describe(self, question: str) -> None:
        self._ensure_stream()
        if self.mission_agent is not None and self.mission_agent.uses_vlm():
            self.mission_agent.stop()
            self.mission_agent = None
            self.log.event("agent_vlm_mission_stopped_for_describe")
        frame = self._wait_for_frame(timeout_s=4.0)
        if frame is None:
            self.log.event("agent_camera_frame_missing_retry")
            try:
                restart = getattr(self.stream, "restart_reader", None)
                if restart is not None:
                    restart()
            except Exception as e:
                self.log.event("agent_camera_restart_failed", err=str(e))
            frame = self._wait_for_frame(timeout_s=4.0)
            if frame is None:
                self.speaker.say("I do not have a fresh camera frame yet.", wait=True)
                return
        try:
            debug_path = Path(__file__).resolve().parents[1] / "logs" / "s7_vlm_last.jpg"
            debug_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            stderr(f"vlm frame saved: {debug_path}")
        except Exception as e:
            stderr(f"vlm frame save failed: {e}")
        self.speaker.say("Looking.", wait=True)
        result_q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def run_describe() -> None:
            try:
                result_q.put(("ok", self.vlm.describe(frame, question)))
            except Exception as e:
                result_q.put(("err", e))

        worker = threading.Thread(target=run_describe, daemon=True, name="vlm-describe")
        worker.start()
        while worker.is_alive():
            self._pump_window(0.05)
            time.sleep(0.02)
        kind, payload = result_q.get()
        if kind == "err":
            stderr(f"describe failed: {payload}")
            self.log.event("agent_describe_failed", question=question, err=str(payload))
            self.speaker.say(
                "The vision model did not answer in time. Try a faster model.",
                wait=True,
            )
            return
        answer = str(payload).strip()
        stderr(f"description: {answer}")
        self.log.event("agent_describe", question=question, answer=answer)
        self._remember_vision(answer)
        self.speaker.say(answer, wait=True)

    def _wait_for_frame(self, timeout_s: float):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            self._pump_window(0.03)
            frame, _ts, age_ms = self.stream.get_latest()
            if frame is not None and age_ms < 1000:
                return frame.copy()
            time.sleep(0.05)
        return None

    def _manual_rc(self, lr=0, fb=0, ud=0, yaw=0, seconds=0.5, phrase="Executing.") -> None:
        self.client.sync_airborne_from_state()
        if not self.client.airborne:
            self._takeoff()
        self._stop_autonomy()
        self.speaker.say(phrase, wait=True)
        self._manual_rc_active = True
        try:
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                self.client.send_rc(lr, fb, ud, yaw)
                self._pump_window(0.02)
                time.sleep(0.05)
            self.client.send_rc(0, 0, 0, 0)
        finally:
            self._manual_rc_active = False

    def _vertical_move(self, *, up: bool, cm: int = 50, phrase: str = "Changing altitude.") -> None:
        self.client.sync_airborne_from_state()
        if not self.client.airborne:
            self._takeoff()
        self._stop_autonomy()
        if not up:
            try:
                state = self.client.read_state()
                if state.height_cm < cm + 35:
                    self.speaker.say(
                        f"I am at {state.height_cm} centimeters. I will not descend because that is too close to the floor.",
                        wait=True,
                    )
                    return
            except Exception:
                pass
        self.speaker.say(phrase, wait=True)
        self._manual_rc_active = True
        try:
            if up:
                self.client.move_up_cm(cm)
            else:
                self.client.move_down_cm(cm)
            self.client.send_rc(0, 0, 0, 0)
        finally:
            self._manual_rc_active = False

    def _reply_short(self, text: str) -> None:
        try:
            answer = self.vlm.reply(text, memory=self._vision_memory_text())
        except Exception as e:
            stderr(f"chat reply failed: {e}")
            self.log.event("agent_reply_failed", text=text, err=str(e))
            answer = "I heard you, but I could not answer in time."
        self.log.event("agent_reply", text=text, answer=answer)
        self.speaker.say(answer or "I am here.", wait=True)

    def _remember_vision(self, answer: str) -> None:
        answer = " ".join(str(answer).split())
        if answer:
            self._vision_memory.append(answer)

    def _vision_memory_text(self) -> str:
        if not self._vision_memory:
            return ""
        return " | ".join(self._vision_memory)

    def _log_drone_state(self, label: str) -> None:
        try:
            state = self.client.read_state()
        except Exception as e:
            stderr(f"state {label}: unavailable ({e})")
            self.log.event("agent_state_failed", label=label, err=str(e))
            return
        airborne = state.height_cm >= 15
        stderr(
            f"state {label}: battery={state.battery_pct}% "
            f"height={state.height_cm}cm flight={state.flight_time_s}s "
            f"airborne={airborne}"
        )
        self.log.event(
            "agent_state",
            label=label,
            battery=state.battery_pct,
            height=state.height_cm,
            flight_time=state.flight_time_s,
            airborne=airborne,
        )

    def _check_battery_safety(self) -> None:
        try:
            state = self.client.read_state()
        except Exception:
            return
        if self.client.airborne and state.battery_pct <= config.MIN_BATTERY_ABORT_PCT:
            stderr(
                f"battery abort: {state.battery_pct}% <= "
                f"{config.MIN_BATTERY_ABORT_PCT}%"
            )
            self.log.event(
                "battery_abort_land",
                battery=state.battery_pct,
                threshold=config.MIN_BATTERY_ABORT_PCT,
                height=state.height_cm,
            )
            self.speaker.say(
                f"Battery {state.battery_pct} percent. Emergency landing.",
                wait=True,
            )
            self._stop_autonomy()
            self._land()

    def _pump_window(self, duration_s: float) -> None:
        if self.preview_thread is not None and self.preview_thread.is_alive():
            time.sleep(max(0.0, duration_s))
            return
        end = time.perf_counter() + duration_s
        while time.perf_counter() < end:
            self._draw_window_once()
            time.sleep(0.015)

    def _draw_window_once(self) -> None:
        if self.args.no_window or self.stream is None:
            return
        frame, _ts, _age = self.stream.get_latest()
        if frame is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            _put(blank, "WAITING FOR TELLO CAMERA", (80, 180), scale=0.8)
            cv2.imshow("s7-voice-agent", blank)
            cv2.waitKey(1)
            return
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pr = self.pipeline.get_latest() if self.pipeline else None
        yolo_result = self.yolo_det.get_latest() if self.yolo_det else None
        _draw_yolo_boxes(rgb, yolo_result, (h, w))
        if pr is not None:
            depth_vis = _colorize(pr.depth_map, (h, w))
            _draw_depth(depth_vis, pr.sectors, yolo_result.hazard_sectors if yolo_result else None)
            label, color = _policy_label(pr, self.policy_obj)
        else:
            depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
            _put(depth_vis, "NO DEPTH YET", (w // 2 - 90, h // 2), scale=0.7)
            label, color = "CAMERA ONLY", (200, 200, 200)
        status = "AIR" if self.client.airborne else "GROUND"
        _put(rgb, f"{status}  {label}", (8, h - 16), scale=0.65, fg=color, bold=True)
        _put(
            rgb,
            f"video {self.stream.frames_seen()}  {self.stream.fps():.1f} fps",
            (8, 22),
            scale=0.55,
            fg=(230, 230, 230),
        )
        side = np.hstack([rgb, depth_vis])
        if self.args.scale != 1.0:
            side = cv2.resize(
                side,
                (int(side.shape[1] * self.args.scale), int(side.shape[0] * self.args.scale)),
            )
        cv2.imshow("s7-voice-agent", side)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            shot_dir = Path(__file__).resolve().parents[1] / "logs"
            shot_dir.mkdir(exist_ok=True)
            self._shot_idx += 1
            p = shot_dir / f"s7_shot_{self._shot_idx}.png"
            cv2.imwrite(str(p), side)
            stderr(f"screenshot: {p}")

    def shutdown(self) -> None:
        self._stop_autonomy()
        if self.heartbeat_thread is not None:
            self.heartbeat_stop.set()
            self.heartbeat_thread.join(timeout=1.0)
            self.heartbeat_thread = None
        if self.preview_thread is not None:
            self.preview_stop.set()
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
        if self.client.airborne:
            try:
                self.client.land()
            except Exception as e:
                stderr(f"land failed: {e}; emergency fallback")
                try:
                    self.client.emergency()
                except Exception:
                    pass
        for obj in (self.yolo_det, self.pipeline, self.stream):
            if obj is not None:
                try:
                    obj.stop()
                except Exception:
                    pass
        try:
            self.client.close()
        except Exception:
            pass
        if not self.args.no_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            self.speaker.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--vision-model", default="qwen2.5vl:3b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--vlm-timeout", type=float, default=45.0)
    ap.add_argument("--vlm-image-width", type=int, default=224)
    ap.add_argument("--whisper-model", default="models/whisper-large-v3-turbo")
    ap.add_argument("--whisper-device", default="cuda")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--mic-min-rms", type=float, default=0.0035)
    ap.add_argument("--speaker-device", default=None)
    ap.add_argument("--speaker-start-silence", type=float, default=0.25)
    ap.add_argument("--list-audio-devices", action="store_true")
    ap.add_argument("--listen-seconds", type=float, default=6.0)
    ap.add_argument("--listen-start-timeout", type=float, default=12.0)
    ap.add_argument("--listen-silence", type=float, default=0.9)
    ap.add_argument("--listen-pre-roll", type=float, default=1.5)
    ap.add_argument("--listen-input-warmup", type=float, default=1.0)
    ap.add_argument("--whisper-vad", action="store_true")
    ap.add_argument("--fixed-listen-window", action="store_true")
    ap.add_argument("--listen-beep", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prompt-each-listen", action="store_true")
    ap.add_argument("--prompt-after-silence", type=int, default=0)
    ap.add_argument("--speak", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--agent-interval", type=float, default=2.0)
    ap.add_argument("--found-confidence", type=float, default=0.62)
    ap.add_argument("--max-autonomy-s", type=float, default=600.0)
    ap.add_argument("--mock-depth", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--scale", type=float, default=_SCALE)
    args = ap.parse_args()

    if args.list_audio_devices:
        print("Input audio devices:")
        for idx, name, channels, rate, hostapi in EnglishListener.list_input_devices():
            print(f"{idx:3d}  {name}  inputs={channels}  default_rate={rate:.0f}  api={hostapi}")
        print("\nOutput audio devices:")
        for idx, name, channels, rate, hostapi in EnglishSpeaker.list_output_devices():
            print(f"{idx:3d}  {name}  outputs={channels}  default_rate={rate:.0f}  api={hostapi}")
        return 0

    if args.mic_device is not None:
        try:
            args.mic_device = int(args.mic_device)
        except ValueError:
            pass
    if args.speaker_device is not None:
        try:
            args.speaker_device = int(args.speaker_device)
        except ValueError:
            pass

    with RunLogger("s7_voice_agent") as log:
        agent = VoiceDroneAgent(args, log)
        try:
            return agent.run()
        except Exception as e:
            stderr(f"fatal: {e}")
            log.event("fatal", err=str(e), tb=traceback.format_exc())
            try:
                agent.shutdown()
            except Exception:
                pass
            return 2


if __name__ == "__main__":
    force_exit(main())

