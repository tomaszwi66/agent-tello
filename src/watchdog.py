"""Comm + loop deadline watchdogs.

Watchdogs OBSERVE; they do not act. They return state.
The control loop decides what to do with that state.
"""

from __future__ import annotations

import time
from enum import Enum

from src import config


class CommState(Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"  # one command timeout, recoverable -> hover
    LOST = "LOST"          # sustained loss -> land/emergency


def comm_state_from_age_ms(
    age_ms: float,
    degraded_ms: int = config.COMMAND_TIMEOUT_MS,
    lost_ms: int = config.COMMAND_LAND_TIMEOUT_MS,
) -> CommState:
    """Derive comm state directly from state-packet age.

    Single source of truth: the time since the last UDP state packet.
    OK <degraded_ms <= DEGRADED <lost_ms <= LOST.
    """
    if age_ms >= lost_ms:
        return CommState.LOST
    if age_ms >= degraded_ms:
        return CommState.DEGRADED
    return CommState.OK


class LoopDeadlineWatchdog:
    """Observes whether the control loop met its period budget.

    Use:
        ldw.tick_start()
        ... do work ...
        overrun_ms = ldw.tick_end(period_s)
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def tick_start(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def overrun_ms(self, period_s: float) -> float:
        return max(0.0, (time.perf_counter() - self._start - period_s) * 1000.0)
