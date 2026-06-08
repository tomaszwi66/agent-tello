"""Safe local microphone and speaker test for English mission commands."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.speech_io import EnglishListener, EnglishSpeaker


def _device_value(value: str | None):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--model", default="models/whisper-large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--mic-min-rms", type=float, default=0.0035)
    ap.add_argument("--speaker-device", default=None)
    ap.add_argument("--speaker-start-silence", type=float, default=0.25)
    ap.add_argument("--start-timeout", type=float, default=10.0)
    ap.add_argument("--silence-seconds", type=float, default=0.9)
    ap.add_argument("--pre-roll", type=float, default=1.5)
    ap.add_argument("--input-warmup", type=float, default=1.0)
    ap.add_argument("--whisper-vad", action="store_true")
    ap.add_argument("--save-wav", default=None)
    ap.add_argument("--fixed-window", action="store_true")
    ap.add_argument("--list-audio-devices", action="store_true")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--say-only", default=None)
    args = ap.parse_args()

    if args.list_audio_devices:
        print("Input audio devices:")
        for idx, name, channels, rate, hostapi in EnglishListener.list_input_devices():
            print(f"{idx:3d}  {name}  inputs={channels}  default_rate={rate:.0f}  api={hostapi}")
        print("\nOutput audio devices:")
        for idx, name, channels, rate, hostapi in EnglishSpeaker.list_output_devices():
            print(f"{idx:3d}  {name}  outputs={channels}  default_rate={rate:.0f}  api={hostapi}")
        return 0

    speaker = EnglishSpeaker(
        enabled=not args.no_speak,
        output_device=_device_value(args.speaker_device),
        duplex_input_device=_device_value(args.mic_device),
        start_silence_s=args.speaker_start_silence,
    )
    if args.say_only:
        print(f"Using output: {speaker.output_description()}")
        speaker.say(args.say_only, wait=True)
        print(f"TTS BACKEND: {speaker.last_tts_backend}")
        if speaker.last_tts_error:
            print(f"TTS ERROR: {speaker.last_tts_error}")
        speaker.close()
        return 0

    listener = EnglishListener(
        model_path=args.model,
        device=args.device,
        input_device=_device_value(args.mic_device),
        min_rms=args.mic_min_rms,
    )
    if not listener.available():
        print("No local Whisper model found.")
        print("Run: python scripts/download_whisper.py")
        return 2

    print(f"Using Whisper model: {listener.model_description()}")
    print(f"Using input: {listener.input_description()}")
    print(f"Using output: {speaker.output_description()}")
    print("Speak after the beep.")
    if args.fixed_window:
        speaker.beep()
        text = listener.listen_once(seconds=args.seconds, whisper_vad=args.whisper_vad)
    else:
        text = listener.listen_utterance(
            start_timeout_s=args.start_timeout,
            max_s=args.seconds,
            silence_s=args.silence_seconds,
            pre_roll_s=args.pre_roll,
            whisper_vad=args.whisper_vad,
            cue=speaker.beep,
            input_warmup_s=args.input_warmup,
            cue_tone_device=_device_value(args.speaker_device) if not args.no_speak else None,
        )
    if args.save_wav:
        _save_wav(Path(args.save_wav), listener.last_audio, listener.sample_rate)
    print(f"AUDIO MODE: {listener.last_record_mode}")
    print(f"RMS: {listener.last_rms:.5f}  threshold={args.mic_min_rms:.5f}")
    print(f"DURATION: {listener.last_duration_s:.2f}s")
    print(f"HEARD: {text}")
    if text:
        speaker.say(f"I heard: {text}", wait=True)
    else:
        speaker.say("I did not hear a command.", wait=True)
    speaker.close()
    return 0


def _save_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np_clip_int16(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    print(f"WAV: {path}")


def np_clip_int16(audio):
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2")


if __name__ == "__main__":
    raise SystemExit(main())
