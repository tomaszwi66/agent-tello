"""Structured run logging.

One run = one timestamped directory under LOG_DIR.
- events.jsonl: discrete events (state transitions, commands, warnings).
- samples.csv:  periodic samples (battery, height, jitter, ...).

Append-only, flushed every FLUSH_EVERY writes. Stdlib only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from src import config

FLUSH_EVERY = 16


class RunLogger:
    def __init__(self, run_name: str, root: str | os.PathLike = config.LOG_DIR) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{ts}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._samples_path = self.run_dir / "samples.csv"
        self._events_file = self._events_path.open("a", encoding="utf-8")
        self._samples_file: Any = None
        self._samples_writer: csv.DictWriter | None = None
        self._samples_fields: list[str] | None = None
        self._write_count = 0
        self._lock = Lock()
        self.event("run_start", run=run_name, run_dir=str(self.run_dir))

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"ts_ns": time.time_ns(), "wall": time.time(), "kind": kind, **fields}
        with self._lock:
            self._events_file.write(json.dumps(rec, default=_json_default) + "\n")
            self._bump_flush()

    def sample(self, **fields: Any) -> None:
        with self._lock:
            if self._samples_writer is None:
                self._samples_fields = ["ts_ns", "wall"] + list(fields.keys())
                self._samples_file = self._samples_path.open("a", newline="", encoding="utf-8")
                self._samples_writer = csv.DictWriter(self._samples_file, fieldnames=self._samples_fields)
                self._samples_writer.writeheader()
            row = {"ts_ns": time.time_ns(), "wall": time.time(), **fields}
            # csv.DictWriter requires keys to be a subset of fieldnames; pad missing.
            assert self._samples_fields is not None
            for k in self._samples_fields:
                row.setdefault(k, "")
            self._samples_writer.writerow(row)
            self._bump_flush()

    def _bump_flush(self) -> None:
        self._write_count += 1
        if self._write_count % FLUSH_EVERY == 0:
            self._events_file.flush()
            if self._samples_file is not None:
                self._samples_file.flush()

    def close(self) -> None:
        # Write run_end inline --- don't call self.event() because that would
        # try to re-acquire self._lock from the same thread (Lock is not
        # reentrant -> deadlock that blocks process exit).
        with self._lock:
            rec = {"ts_ns": time.time_ns(), "wall": time.time(), "kind": "run_end"}
            try:
                self._events_file.write(json.dumps(rec, default=_json_default) + "\n")
                self._events_file.flush()
                self._events_file.close()
            except Exception:
                pass
            if self._samples_file is not None:
                try:
                    self._samples_file.flush()
                    self._samples_file.close()
                except Exception:
                    pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.event("run_exception", exc_type=str(exc_type), exc=str(exc))
        self.close()


def _json_default(o: Any) -> Any:
    try:
        return float(o)
    except Exception:
        return str(o)


def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def force_exit(code: int) -> None:
    """Force-exit the process.

    Works around djitellopy/cv2 cleanup hangs on Windows where daemon UDP
    receiver threads and PyAV/OpenCV native resources can stall Python's
    normal interpreter shutdown. By this point our own state is flushed
    and the drone has been commanded to land, so a hard exit is safe.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
