"""List input audio devices for the voice drone agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.speech_io import EnglishListener, EnglishSpeaker


def main() -> int:
    print("Input audio devices:")
    for idx, name, channels, rate, hostapi in EnglishListener.list_input_devices():
        print(f"{idx:3d}  {name}  inputs={channels}  default_rate={rate:.0f}  api={hostapi}")
    print("\nOutput audio devices:")
    for idx, name, channels, rate, hostapi in EnglishSpeaker.list_output_devices():
        print(f"{idx:3d}  {name}  outputs={channels}  default_rate={rate:.0f}  api={hostapi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
