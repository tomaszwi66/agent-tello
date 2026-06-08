"""Mission supervisor that watches camera frames with fast YOLO and local VLM."""

from __future__ import annotations

import threading
import time
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Optional

from src.agent.ollama_vl import OllamaVisionClient, VisionVerdict


@dataclass(frozen=True)
class MissionState:
    goal: str
    active: bool
    found: bool
    checks: int
    last_verdict: Optional[VisionVerdict]


# COCO / YOLOv8 class ids with English aliases. Anything outside this list uses
# the slower VLM fallback because ordinary YOLOv8n is not open-vocabulary.
_YOLO_GOAL_ALIASES: dict[int, tuple[str, ...]] = {
    0: ("czlowiek", "czlowieka", "osoba", "osobe", "ludzi", "person", "human"),
    1: ("rower", "roweru", "bicycle"),
    2: ("samochod", "samochodu", "auto", "car"),
    13: ("lawka", "lawke", "bench"),
    14: ("ptak", "bird"),
    15: ("kot", "kota", "cat"),
    16: ("pies", "psa", "dog"),
    24: ("plecak", "plecaka", "backpack"),
    26: ("torba", "torbe", "torebka", "handbag"),
    28: ("walizka", "walizke", "suitcase"),
    32: ("pilka", "pilke", "ball", "sports ball"),
    39: ("butelka", "butelke", "bottle"),
    40: ("kieliszek", "szklanka", "wine glass", "glass"),
    41: ("kubek", "kubka", "filizanka", "cup"),
    45: ("miska", "miske", "bowl"),
    46: ("banan", "banana"),
    47: ("jablko", "apple"),
    56: ("chair", "seat"),
    57: ("kanapa", "sofa", "couch"),
    58: ("roslina", "doniczka", "kwiat", "potted plant", "plant"),
    59: ("lozko", "bed"),
    60: ("stol", "stolu", "stolik", "table", "dining table"),
    61: ("toaleta", "sedes", "toilet"),
    62: ("telewizor", "tv", "ekran"),
    63: ("laptop", "komputer", "notebook"),
    64: ("mysz", "myszka", "mouse"),
    65: ("pilot", "remote"),
    66: ("klawiatura", "keyboard"),
    67: ("telefon", "komorka", "smartfon", "cell phone"),
    68: ("mikrofalowka", "microwave"),
    69: ("piekarnik", "oven"),
    71: ("zlew", "umywalka", "sink"),
    72: ("lodowka", "refrigerator", "fridge"),
    73: ("ksiazka", "ksiazke", "book"),
    74: ("zegar", "zegarek", "clock"),
    75: ("wazon", "vase"),
    76: ("nozyczki", "scissors"),
    77: ("mis", "misiek", "teddy bear"),
    79: ("szczoteczka", "toothbrush"),
}

_FUZZY_ALIAS_MIN_RATIO = 0.84
_ASR_GOAL_CORRECTIONS: dict[str, int] = {
    "chairs": 56,
    "share": 56,
    "stare": 56,
    "cheer": 56,
}


class MissionAgent:
    def __init__(
        self,
        stream,
        vlm: OllamaVisionClient,
        goal: str,
        telemetry=None,
        speaker=None,
        yolo_detector=None,
        interval_s: float = 0.85,
        found_confidence: float = 0.62,
    ) -> None:
        self._stream = stream
        self._vlm = vlm
        self._goal = goal
        self._telemetry = telemetry
        self._speaker = speaker
        self._yolo = yolo_detector
        self._interval_s = interval_s
        self._found_confidence = found_confidence
        self._yolo_targets = self._resolve_yolo_targets_v2(goal)
        self._lock = threading.Lock()
        self._last_verdict: Optional[VisionVerdict] = None
        self._checks = 0
        self._found = False
        self._running = False
        self._last_yolo_seen_log = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        mode = "yolo" if self._yolo_targets else "vlm"
        self._event("agent_start", goal=self._goal, mode=mode, model=self._vlm.model)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mission-agent")
        self._thread.start()
    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def found(self) -> bool:
        with self._lock:
            return self._found

    def uses_vlm(self) -> bool:
        return not self._yolo_targets

    def latest(self) -> MissionState:
        with self._lock:
            return MissionState(
                goal=self._goal,
                active=self._running,
                found=self._found,
                checks=self._checks,
                last_verdict=self._last_verdict,
            )

    def _loop(self) -> None:
        if self._yolo_targets and self._yolo is not None:
            self._loop_yolo()
        else:
            self._loop_vlm()

    def _loop_yolo(self) -> None:
        while self._running and not self.found():
            if self._check_yolo_targets():
                return
            time.sleep(0.05)

    def _loop_vlm(self) -> None:
        last_ts = -1
        while self._running and not self.found():
            frame, ts_ns, age_ms = self._stream.get_latest()
            if frame is None or ts_ns == last_ts or age_ms > 500:
                time.sleep(0.05)
                continue
            last_ts = ts_ns
            try:
                verdict = self._vlm.analyze(frame, self._goal)
                is_found = verdict.found and verdict.confidence >= self._found_confidence
                with self._lock:
                    self._last_verdict = verdict
                    self._checks += 1
                    if is_found:
                        self._found = True
                self._event(
                    "agent_vision",
                    goal=self._goal,
                    found=verdict.found,
                    confidence=round(verdict.confidence, 3),
                    answer=verdict.answer_text,
                    evidence=verdict.evidence_text,
                    latency_ms=round(verdict.latency_ms, 1),
                    checks=self._checks,
                )
                if is_found:
                    msg = verdict.answer_text or f"I found {self._goal}."
                    self._event("agent_found", goal=self._goal, answer=msg)
                    self._say(msg)
                    return
            except Exception as e:
                self._event("agent_error", err=str(e))
                time.sleep(0.5)
            time.sleep(self._interval_s)

    def _check_yolo_targets(self) -> bool:
        try:
            result = self._yolo.get_latest() if self._yolo is not None else None
        except Exception:
            return False
        if result is None:
            return False
        for det in result.detections:
            if det.cls_id not in self._yolo_targets:
                continue
            now = time.perf_counter()
            if now - self._last_yolo_seen_log > 0.5:
                self._last_yolo_seen_log = now
                self._event(
                    "agent_yolo_seen",
                    goal=self._goal,
                    cls=det.cls_name,
                    cls_id=det.cls_id,
                    confidence=round(det.conf, 3),
                    x1=round(det.x1, 3),
                    y1=round(det.y1, 3),
                    x2=round(det.x2, 3),
                    y2=round(det.y2, 3),
                )
            if det.conf < 0.28:
                continue
            msg = f"I see {self._goal}. Stopping Tello."
            with self._lock:
                self._found = True
                self._checks += 1
            self._event(
                "agent_found_yolo",
                goal=self._goal,
                cls=det.cls_name,
                cls_id=det.cls_id,
                confidence=round(det.conf, 3),
                x1=round(det.x1, 3),
                y1=round(det.y1, 3),
                x2=round(det.x2, 3),
                y2=round(det.y2, 3),
            )
            self._say(msg)
            return True
        return False

    @classmethod
    def _resolve_yolo_targets(cls, goal: str) -> set[int]:
        text = cls._norm(goal)
        out: set[int] = set()
        for cls_id, aliases in _YOLO_GOAL_ALIASES.items():
            if any(alias in text for alias in aliases):
                out.add(cls_id)
        return out

    @staticmethod
    def _norm(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    @classmethod
    def _resolve_yolo_targets_v2(cls, goal: str) -> set[int]:
        text = cls._norm_v3(goal)
        tokens = [tok for tok in text.replace("-", " ").split() if len(tok) >= 4]
        out: set[int] = set()
        for token in tokens:
            if token in _ASR_GOAL_CORRECTIONS:
                out.add(_ASR_GOAL_CORRECTIONS[token])
        for cls_id, aliases in _YOLO_GOAL_ALIASES.items():
            if any(alias in text for alias in aliases):
                out.add(cls_id)
                continue
            for token in tokens:
                if any(
                    SequenceMatcher(None, token, alias).ratio() >= _FUZZY_ALIAS_MIN_RATIO
                    for alias in aliases
                    if len(alias) >= 4 and " " not in alias
                ):
                    out.add(cls_id)
                    break
        return out

    @staticmethod
    def _norm_v2(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    @staticmethod
    def _norm_v3(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    def _event(self, kind: str, **fields) -> None:
        if self._telemetry is not None:
            try:
                self._telemetry.event(kind, **fields)
            except Exception:
                pass

    def _say(self, text: str) -> None:
        if self._speaker is not None:
            try:
                self._speaker.say(text, wait=True)
            except Exception:
                pass

