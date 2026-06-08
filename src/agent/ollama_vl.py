"""Local Ollama vision-language adapter.

The flight controller must stay reactive and deterministic. This module is the
slow "cortex" layer: it asks a local VLM what is visible in the current camera
frame and returns a small structured verdict for the mission supervisor.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class VisionVerdict:
    found: bool
    confidence: float
    answer_text: str
    evidence_text: str
    target_visible: str
    latency_ms: float
    model: str


@dataclass(frozen=True)
class ToolCommand:
    tool: str
    argument: str
    answer_text: str
    confidence: float
    raw_text: str


class OllamaVisionClient:
    def __init__(
        self,
        model: str = "gemma4:e4b",
        vision_model: str | None = None,
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 30.0,
        image_width: int = 256,
    ) -> None:
        self.model = model
        self.vision_model = vision_model or model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.image_width = image_width

    def ping(self) -> None:
        r = requests.get(f"{self.host}/api/tags", timeout=5.0)
        r.raise_for_status()

    def analyze(self, frame_rgb: np.ndarray, goal_text: str) -> VisionVerdict:
        t0 = time.perf_counter()
        image_b64 = self._encode_frame(frame_rgb)
        prompt = self._prompt(goal_text)
        payload: dict[str, Any] = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 64,
            },
        }
        r = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        text = str(r.json().get("response", "")).strip()
        data = self._parse_json(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return VisionVerdict(
            found=bool(data.get("found", False)),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
            answer_text=str(data.get("answer_en", data.get("answer_text", ""))).strip(),
            evidence_text=str(data.get("evidence_en", data.get("evidence_text", ""))).strip(),
            target_visible=str(data.get("target_visible", "")).strip(),
            latency_ms=latency_ms,
            model=self.vision_model,
        )

    def describe(self, frame_rgb: np.ndarray, question_text: str = "Describe what you see.") -> str:
        image_b64 = self._encode_frame(frame_rgb)
        prompt = (
            "You are Tello, a local physical drone agent, looking through your own camera. "
            "This is your vision from the drone body. "
            "Answer in English with one short first-person sentence as Tello. "
            "Do not write 'in the image', 'the camera shows', or 'the drone sees'. "
            "Speak as if you are seeing and flying yourself. "
            f"User question: {question_text!r}"
        )
        payload: dict[str, Any] = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 48,
            },
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        answer = str(r.json().get("message", {}).get("content", "")).strip()
        if self._looks_like_vision_refusal(answer):
            payload["messages"] = [
                {
                    "role": "user",
                    "content": (
                        "You are Tello. What do you see through your camera? "
                        "Answer in English with one short first-person sentence."
                    ),
                    "images": [image_b64],
                }
            ]
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            answer = str(r.json().get("message", {}).get("content", "")).strip()
        return self._english_drone_voice(answer)

    @staticmethod
    def _english_drone_voice(text: str) -> str:
        out = text.strip()
        replacements = (
            ("In the image I see ", "I see "),
            ("In this image I see ", "I see "),
            ("The image shows ", "I see "),
            ("The camera shows ", "I see "),
            ("The drone sees ", "I see "),
            ("Tello sees ", "I see "),
            ("in the image I see ", "I see "),
            ("in this image I see ", "I see "),
            ("the image shows ", "I see "),
            ("the camera shows ", "I see "),
            ("the drone sees ", "I see "),
            ("tello sees ", "I see "),
        )
        for old, new in replacements:
            if out.startswith(old):
                out = new + out[len(old):]
                break
        return out

    @staticmethod
    def _looks_like_vision_refusal(text: str) -> bool:
        t = text.lower()
        return any(
            s in t
            for s in (
                "i cannot see",
                "i can't see",
                "cannot describe",
                "can't describe",
                "i do not have access",
                "i don't have access",
                "as an ai",
            )
        )

    def command(self, text_value: str, *, airborne: bool, autonomous: bool) -> ToolCommand:
        prompt = self._command_prompt(text_value, airborne=airborne, autonomous=autonomous)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30s",
            "options": {"temperature": 0.0, "num_ctx": 2048, "num_predict": 96},
        }
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        raw = str(r.json().get("response", "")).strip()
        data = self._parse_json(raw)
        tool = str(data.get("tool", "clarify")).strip()
        allowed = {
            "takeoff",
            "land",
            "hover",
            "emergency",
            "start_autonomy",
            "stop_autonomy",
            "search",
            "describe",
            "turn_left",
            "turn_right",
            "forward",
            "back",
            "up",
            "down",
            "clarify",
            "noop",
        }
        if tool not in allowed:
            tool = "clarify"
        return ToolCommand(
            tool=tool,
            argument=str(data.get("argument", "")).strip(),
            answer_text=str(data.get("answer_text", "")).strip(),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
            raw_text=raw,
        )

    def reply(self, text_value: str, memory: str = "") -> str:
        memory_text = (
            f"Recent things I remember seeing: {memory!r}. "
            if memory
            else "I do not have remembered camera descriptions yet. "
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": (
                "Answer in English, very briefly, at most two sentences. "
                "You are Tello, a local physical drone agent. "
                "Speak in first person: I see, I fly, I remember, I am. "
                "Do not describe yourself as software, a model, a camera, or a drone in third person. "
                "If this is casual conversation, answer naturally as Tello and do not pretend to perform an action. "
                f"{memory_text}"
                f"User: {text_value!r}"
            ),
            "stream": False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0.3,
                "num_ctx": 2048,
                "num_predict": 70,
            },
        }
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return str(r.json().get("response", "")).strip()

    def _encode_frame(self, frame_rgb: np.ndarray) -> str:
        if frame_rgb.ndim != 3:
            raise ValueError(f"expected RGB frame HxWx3, got {frame_rgb.shape}")
        h, w = frame_rgb.shape[:2]
        if w > self.image_width:
            scale = self.image_width / float(w)
            frame_rgb = cv2.resize(
                frame_rgb,
                (self.image_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for Ollama")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _prompt(goal_text: str) -> str:
        return (
            "You are Tello, a local physical drone agent, looking through your own camera. "
            f"Target: {goal_text!r}. "
            "Answer only JSON. "
            "If the target is visible, found=true. If not, found=false. "
            "{"
            "\"found\":false,"
            "\"confidence\":0.0,"
            "\"target_visible\":\"\","
            "\"answer_en\":\"short English first-person answer\","
            "\"evidence_en\":\"short English evidence\""
            "}"
        )

    @staticmethod
    def _command_prompt(text_value: str, *, airborne: bool, autonomous: bool) -> str:
        return (
            "You are the local control brain of a DJI/Ryze Tello drone. "
            "User-facing answers must sound as if Tello is speaking in first person. "
            "The user speaks English, sometimes casually. "
            "Choose exactly one safe tool. "
            "You do not directly control RC channels; you only choose one tool from the list.\n\n"
            f"State: airborne={airborne}, autonomous={autonomous}.\n"
            f"User utterance: {text_value!r}\n\n"
            "Available tools:\n"
            "- takeoff: take off and hover.\n"
            "- land: land safely.\n"
            "- hover: stop motion and hover.\n"
            "- emergency: cut motors only when the user clearly asks for emergency motor stop.\n"
            "- start_autonomy: fly autonomously and explore without a specific target.\n"
            "- stop_autonomy: stop autonomous flight but do not land.\n"
            "- search: find an object or place; argument is the target, for example 'red chair'.\n"
            "- describe: describe the current camera view; argument may contain the question.\n"
            "- turn_left: turn left about 90 degrees.\n"
            "- turn_right: turn right about 90 degrees.\n"
            "- forward: move forward about one meter when safe.\n"
            "- back: move back a little, about 20 cm.\n"
            "- up: climb about 50 cm.\n"
            "- down: descend about 50 cm when safe.\n"
            "- clarify: use when the command is unclear or risky.\n"
            "- noop: use when it is not a drone command.\n\n"
            "Rules:\n"
            "- 'find X', 'search for X', 'where is X' -> search with X as argument.\n"
            "- 'describe what you see', 'what do you see' -> describe.\n"
            "- 'fly', 'explore', 'fly around' without a target -> start_autonomy.\n"
            "- 'take off' -> takeoff.\n"
            "- 'land', 'end the flight' -> land.\n"
            "- 'stop', 'wait', 'hold position' -> hover.\n"
            "- 'go up', 'climb', 'higher' -> up.\n"
            "- 'go down', 'descend', 'lower' -> down.\n"
            "- If a tool requires flight and the drone is not airborne, still choose that tool; the executor will take off safely if needed.\n\n"
            "Answer only JSON without markdown:\n"
            "{"
            "\"tool\":\"tool_name\", "
            "\"argument\":\"text or empty\", "
            "\"answer_text\":\"short spoken English answer for the user\", "
            "\"confidence\":0.0"
            "}"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                raise
            return json.loads(m.group(0))

