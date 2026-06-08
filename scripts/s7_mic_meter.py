"""Measure microphone RMS levels for selected input devices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.speech_io import EnglishListener


def _parse_devices(value: str | None):
    if not value:
        return [idx for idx, *_rest in EnglishListener.list_input_devices()]
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            out.append(part)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="Comma-separated device ids or name fragments.")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    print("Speak normally while each device is measured.")
    for dev in _parse_devices(args.devices):
        listener = EnglishListener(
            device="cpu",
            input_device=dev,
            sample_rate=args.sample_rate,
            min_rms=0.0,
        )
        try:
            desc = listener.input_description()
            rms = listener.record_level(seconds=args.seconds)
            print(f"{str(dev):>8}  rms={rms:.5f}  {desc}")
        except Exception as e:
            print(f"{str(dev):>8}  ERROR  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
