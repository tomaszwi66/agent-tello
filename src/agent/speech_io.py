"""Local English speech helpers.

This stays optional: missions can run fully offline with typed text. If a local
Whisper/faster-whisper model is available, the same script can capture a short
spoken command from the microphone.
"""

from __future__ import annotations

import queue
import locale
import subprocess
import sys
import threading
import time
import tempfile
import wave
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import unicodedata

import numpy as np

_AUDIO_IO_LOCK = threading.RLock()


@dataclass
class _SpeechItem:
    text: str
    done: threading.Event | None = None


class EnglishSpeaker:
    def __init__(
        self,
        enabled: bool = True,
        output_device: int | str | None = None,
        duplex_input_device: int | str | None = None,
        piper_model: str | None = "models/tts/piper/en_US-lessac-medium.onnx",
        start_silence_s: float = 0.25,
    ) -> None:
        self.enabled = enabled
        self.output_device = output_device
        self.duplex_input_device = duplex_input_device
        self.piper_model = piper_model
        self.start_silence_s = start_silence_s
        self._resolved_output_device = None
        self._resolved_duplex_input_device = None
        self.last_tts_backend = ""
        self.last_tts_error = ""
        self._q: queue.Queue[_SpeechItem | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        if enabled:
            self._worker = threading.Thread(target=self._loop, daemon=True, name="english-speaker")
            self._worker.start()

    def output_description(self) -> str:
        if self.output_device is None:
            return "default"
        try:
            import sounddevice as sd

            device = self._resolve_output_device()
            info = sd.query_devices(device, "output")
            return (
                f"{device}: {info.get('name', '')} "
                f"outputs={info.get('max_output_channels', '?')} "
                f"default_rate={float(info.get('default_samplerate', 0.0) or 0.0):.0f}"
            )
        except Exception as e:
            return f"{self.output_device!r} ({e})"

    def say(self, text: str, wait: bool = False) -> None:
        if not self.enabled or not text:
            return
        done = threading.Event() if wait else None
        self._q.put(_SpeechItem(text=text, done=done))
        if done is not None:
            if not done.wait(timeout=30.0):
                self.last_tts_error = "timeout"
                print("tts failed: timeout", flush=True)

    def beep(self, freq_hz: int = 880, duration_ms: int = 90) -> None:
        if not self.enabled:
            return
        if self.output_device is not None:
            try:
                import sounddevice as sd

                sample_rate = 48000
                duration_s = max(0.02, duration_ms / 1000.0)
                t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
                audio = (0.18 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
                sd.play(audio, samplerate=sample_rate, device=self._resolve_output_device())
                sd.wait()
                return
            except Exception:
                pass
        try:
            import winsound

            winsound.Beep(freq_hz, duration_ms)
        except Exception:
            print("\a", end="", flush=True)

    def close(self) -> None:
        if self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=2.0)
            self._worker = None

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self._say_blocking(item.text)
            finally:
                if item.done is not None:
                    item.done.set()

    def _say_blocking(self, text: str) -> None:
        if self.output_device is not None and self._say_via_wav_device(text):
            if self.last_tts_backend == "piper":
                self.last_tts_backend = "piper_wav_device"
            else:
                self.last_tts_backend = "wav_device"
            self.last_tts_error = ""
            return
        try:
            import pyttsx3

            engine = pyttsx3.init()
            self._select_english_voice(engine)
            engine.setProperty("rate", 175)
            engine.say(text)
            engine.runAndWait()
            self.last_tts_backend = "pyttsx3"
            self.last_tts_error = ""
            return
        except Exception as e:
            self.last_tts_backend = "failed"
            self.last_tts_error = str(e)
            print(f"tts failed: {e}", flush=True)
            pass

    def _say_via_wav_device(self, text: str) -> bool:
        try:
            wav_path = self._synthesize_to_wav(text)
            if wav_path is None:
                self.last_tts_error = "synthesis produced no wav"
                print("tts wav failed: synthesis produced no wav", flush=True)
                return False
            try:
                self._play_wav(wav_path)
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return True
        except Exception as e:
            self.last_tts_error = str(e)
            print(f"tts wav failed: {e}", flush=True)
            return False

    def _synthesize_to_wav(self, text: str) -> Path | None:
        piper_path = self._synthesize_with_piper(text)
        if piper_path is not None:
            return piper_path

        path = Path(tempfile.gettempdir()) / f"agent_tello_tts_{time.time_ns()}.wav"
        text_path = Path(tempfile.gettempdir()) / f"agent_tello_tts_{time.time_ns()}.txt"
        ps_path = Path(tempfile.gettempdir()) / f"agent_tello_tts_{time.time_ns()}.ps1"
        ps = "\n".join(
            [
                "param([string]$outPath, [string]$textPath)",
                "Add-Type -AssemblyName System.Speech",
                "$text=[System.IO.File]::ReadAllText($textPath, [System.Text.Encoding]::UTF8)",
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer",
                "$s.Rate=0",
                "try {",
                "  $c=[System.Globalization.CultureInfo]::GetCultureInfo('en-US')",
                "  $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet,[System.Speech.Synthesis.VoiceAge]::NotSet,0,$c)",
                "} catch {}",
                "$s.SetOutputToWaveFile($outPath)",
                "$s.Speak($text)",
                "$s.Dispose()",
            ]
        )
        try:
            text_path.write_text(text, encoding="utf-8")
            ps_path.write_text(ps, encoding="utf-8")
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ps_path),
                    str(path),
                    str(text_path),
                ],
                timeout=20,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.last_tts_error = (proc.stderr or proc.stdout or "").strip()
        except Exception as e:
            self.last_tts_error = str(e)
        finally:
            try:
                text_path.unlink(missing_ok=True)
                ps_path.unlink(missing_ok=True)
            except Exception:
                pass
        return path if path.exists() and path.stat().st_size > 44 else None

    def _synthesize_with_piper(self, text: str) -> Path | None:
        if not self.piper_model:
            return None
        model = Path(self.piper_model)
        if not model.exists():
            return None
        piper_exe = Path(sys.executable).with_name("piper.exe")
        if not piper_exe.exists():
            piper_exe = Path("piper.exe")
        path = Path(tempfile.gettempdir()) / f"agent_tello_piper_{time.time_ns()}.wav"
        try:
            stdin_encoding = locale.getpreferredencoding(False) or "utf-8"
            proc = subprocess.run(
                [str(piper_exe), "-m", str(model), "-f", str(path)],
                input=self._clean_tts_text(text),
                timeout=20,
                check=False,
                capture_output=True,
                text=True,
                encoding=stdin_encoding,
                errors="replace",
            )
            if proc.returncode != 0:
                self.last_tts_error = (proc.stderr or proc.stdout or "").strip()
                return None
            if path.exists() and path.stat().st_size > 44:
                self.last_tts_backend = "piper"
                return path
        except Exception as e:
            self.last_tts_error = str(e)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _clean_tts_text(text: str) -> str:
        return (
            text.replace("\u2122", "")
            .replace("\u00ae", "")
            .replace("\u00a9", "")
            .replace("\u00c4\u2122", "\u0119")
            .replace("\u00c4\u2026", "\u0105")
            .replace("\u0139\u201a", "\u0142")
            .replace("\u0139\u201e", "\u0144")
            .replace("\u0139\u203a", "\u015b")
            .replace("\u0139\u015f", "\u017a")
            .replace("\u0139\u013d", "\u017c")
            .replace("\u00c3\u00b3", "\u00f3")
            .replace("\u00c4\u2021", "\u0107")
        )

    def _play_wav(self, path: Path) -> None:
        import sounddevice as sd

        with _AUDIO_IO_LOCK:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                if self.start_silence_s > 0:
                    pad_shape = (int(sample_rate * self.start_silence_s),)
                    if channels > 1:
                        pad_shape = (pad_shape[0], channels)
                    pad = np.zeros(pad_shape, dtype=np.float32)
                    audio = np.concatenate([pad, audio], axis=0)
                if self.duplex_input_device is not None:
                    try:
                        self._play_audio_duplex(audio, sample_rate)
                        return
                    except Exception as e:
                        self.last_tts_error = str(e)
                        self._resolved_output_device = None
                        self._resolved_duplex_input_device = None
                        print(f"tts duplex failed: {e}", flush=True)
                sd.play(audio, samplerate=sample_rate, device=self._resolve_output_device())
                sd.wait()

    def _play_audio_duplex(self, audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        if audio.ndim == 1:
            audio_2d = audio.reshape(-1, 1)
        else:
            audio_2d = audio
        target_rate = 16000
        if sample_rate != target_rate and len(audio_2d) > 1:
            old_x = np.linspace(0.0, 1.0, len(audio_2d), endpoint=False)
            new_len = max(1, int(len(audio_2d) * target_rate / float(sample_rate)))
            new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
            chans = [
                np.interp(new_x, old_x, audio_2d[:, ch]).astype(np.float32)
                for ch in range(audio_2d.shape[1])
            ]
            audio_2d = np.stack(chans, axis=1)
            sample_rate = target_rate
        out_channels = int(audio_2d.shape[1])
        pos = 0

        def cb(indata, outdata, frames, time_info, status):
            nonlocal pos
            outdata.fill(0.0)
            end = min(pos + frames, len(audio_2d))
            n = end - pos
            if n > 0:
                outdata[:n, :out_channels] = audio_2d[pos:end, :out_channels]
            pos = end

        with sd.Stream(
            device=(self._resolve_duplex_input_device(), self._resolve_output_device()),
            channels=(1, out_channels),
            samplerate=sample_rate,
            dtype="float32",
            callback=cb,
        ):
            deadline = time.perf_counter() + (len(audio_2d) / float(sample_rate)) + 2.0
            while pos < len(audio_2d) and time.perf_counter() < deadline:
                time.sleep(0.02)

    @staticmethod
    def _select_english_voice(engine) -> None:
        try:
            voices = engine.getProperty("voices") or []
            for voice in voices:
                blob = " ".join(
                    str(x).lower()
                    for x in (
                        getattr(voice, "id", ""),
                        getattr(voice, "name", ""),
                        getattr(voice, "languages", ""),
                    )
                )
                if "en" in blob or "english" in blob or "us" in blob:
                    engine.setProperty("voice", voice.id)
                    return
        except Exception:
            return

    @staticmethod
    def list_output_devices() -> list[tuple[int, str, int, float, str]]:
        import sounddevice as sd

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        out: list[tuple[int, str, int, float, str]] = []
        for i, dev in enumerate(devices):
            max_outputs = int(dev.get("max_output_channels", 0) or 0)
            if max_outputs <= 0:
                continue
            hostapi_name = ""
            try:
                hostapi_name = str(hostapis[int(dev.get("hostapi", -1))].get("name", ""))
            except Exception:
                pass
            out.append(
                (
                    i,
                    str(dev.get("name", "")),
                    max_outputs,
                    float(dev.get("default_samplerate", 0.0) or 0.0),
                    hostapi_name,
                )
            )
        return out

    def _resolve_output_device(self):
        if self._resolved_output_device is not None:
            return self._resolved_output_device
        if self.output_device is None:
            return None
        if isinstance(self.output_device, int):
            try:
                import sounddevice as sd

                sd.query_devices(self.output_device, "output")
            except Exception:
                self.last_tts_error = f"output device {self.output_device} unavailable"
                return None
            self._resolved_output_device = self.output_device
            return self._resolved_output_device
        needle = EnglishListener._norm_device_name(str(self.output_device))
        matches: list[tuple[int, float, str]] = []
        for idx, name, _channels, rate, _hostapi in self.list_output_devices():
            hay = EnglishListener._norm_device_name(name)
            if needle in hay:
                rate_penalty = abs(rate - 48000.0)
                matches.append((idx, rate_penalty, name))
        if not matches:
            self._resolved_output_device = self.output_device
            return self._resolved_output_device
        matches.sort(key=lambda x: (x[1], x[0]))
        self._resolved_output_device = matches[0][0]
        return self._resolved_output_device

    def _resolve_duplex_input_device(self):
        if self._resolved_duplex_input_device is not None:
            return self._resolved_duplex_input_device
        if self.duplex_input_device is None:
            return None
        if isinstance(self.duplex_input_device, int):
            try:
                import sounddevice as sd

                sd.query_devices(self.duplex_input_device, "input")
            except Exception:
                self.last_tts_error = f"input device {self.duplex_input_device} unavailable"
                return None
            self._resolved_duplex_input_device = self.duplex_input_device
            return self._resolved_duplex_input_device
        needle = EnglishListener._norm_device_name(str(self.duplex_input_device))
        matches: list[tuple[int, float, str]] = []
        for idx, name, _channels, rate, _hostapi in EnglishListener.list_input_devices():
            hay = EnglishListener._norm_device_name(name)
            if needle in hay:
                rate_penalty = abs(rate - 16000.0)
                matches.append((idx, rate_penalty, name))
        if not matches:
            self._resolved_duplex_input_device = self.duplex_input_device
            return self._resolved_duplex_input_device
        matches.sort(key=lambda x: (x[1], x[0]))
        self._resolved_duplex_input_device = matches[0][0]
        return self._resolved_duplex_input_device


class EnglishListener:
    def __init__(
        self,
        model_path: str = "models/whisper-large-v3-turbo",
        device: str = "cuda",
        sample_rate: int = 16000,
        input_device: int | str | None = None,
        min_rms: float = 0.0035,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.min_rms = min_rms
        self._model = None
        self._resolved_model = ""
        self._resolved_input_device = None
        self.last_rms = 0.0
        self.last_duration_s = 0.0
        self.last_audio = np.zeros(0, dtype=np.float32)
        self.last_record_mode = ""

    def available(self) -> bool:
        return self._resolve_model_path() is not None

    def model_description(self) -> str:
        return self._resolved_model or str(self._resolve_model_path() or self.model_path)

    def input_description(self) -> str:
        import sounddevice as sd

        device = self._resolve_input_device()
        try:
            info = sd.query_devices(device, "input")
            idx = device if device is not None else sd.default.device[0]
            return (
                f"{idx}: {info.get('name', '')} "
                f"inputs={info.get('max_input_channels', '?')} "
                f"default_rate={float(info.get('default_samplerate', 0.0) or 0.0):.0f}"
            )
        except Exception as e:
            return f"{device!r} ({e})"

    def warmup(self, audio_seconds: float = 0.15) -> None:
        if self._model is None:
            self._load()
        if audio_seconds > 0:
            self._record(audio_seconds)

    def listen_once(self, seconds: float = 5.0, *, whisper_vad: bool = False) -> str:
        if self._model is None:
            self._load()
        with _AUDIO_IO_LOCK:
            audio = self._record(seconds)
        return self._transcribe_audio(audio, whisper_vad=whisper_vad)

    def listen_utterance(
        self,
        *,
        start_timeout_s: float = 10.0,
        max_s: float = 12.0,
        silence_s: float = 0.8,
        pre_roll_s: float = 1.5,
        whisper_vad: bool = False,
        cue=None,
        input_warmup_s: float = 0.8,
        cue_tone_device=None,
    ) -> str:
        if self._model is None:
            self._load()
        with _AUDIO_IO_LOCK:
            audio = self._record_utterance(
                start_timeout_s=start_timeout_s,
                max_s=max_s,
                silence_s=silence_s,
                pre_roll_s=pre_roll_s,
                cue=cue,
                input_warmup_s=input_warmup_s,
                cue_tone_device=cue_tone_device,
            )
        return self._transcribe_audio(audio, whisper_vad=whisper_vad)

    def _transcribe_audio(self, audio: np.ndarray, *, whisper_vad: bool = False) -> str:
        self.last_audio = audio.astype(np.float32, copy=True)
        self.last_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        self.last_duration_s = float(audio.size) / float(self.sample_rate) if audio.size else 0.0
        if self.last_rms < self.min_rms:
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=whisper_vad,
            beam_size=1,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        kept: list[str] = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) > 0.75:
                continue
            if getattr(seg, "avg_logprob", 0.0) < -1.2:
                continue
            text = seg.text.strip()
            if text:
                kept.append(text)
        return " ".join(kept).strip()

    @staticmethod
    def list_input_devices() -> list[tuple[int, str, int, float, str]]:
        import sounddevice as sd

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        out: list[tuple[int, str, int, float, str]] = []
        for i, dev in enumerate(devices):
            max_inputs = int(dev.get("max_input_channels", 0) or 0)
            if max_inputs <= 0:
                continue
            hostapi_name = ""
            try:
                hostapi_name = str(hostapis[int(dev.get("hostapi", -1))].get("name", ""))
            except Exception:
                pass
            out.append(
                (
                    i,
                    str(dev.get("name", "")),
                    max_inputs,
                    float(dev.get("default_samplerate", 0.0) or 0.0),
                    hostapi_name,
                )
            )
        return out

    def record_level(self, seconds: float = 2.0) -> float:
        audio = self._record(seconds)
        self.last_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return self.last_rms

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        model_path = self._resolve_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"local Whisper model not found: {self.model_path}. "
                "Run scripts/download_whisper.py or use typed goal."
            )
        compute_type = "float16" if self.device == "cuda" else "int8"
        self._resolved_model = str(model_path)
        self._model = WhisperModel(
            str(model_path),
            device=self.device,
            compute_type=compute_type,
        )

    def _resolve_model_path(self) -> Path | None:
        direct = Path(self.model_path)
        if direct.exists():
            return direct

        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        candidates = [
            cache_root / "models--h2oai--faster-whisper-large-v3-turbo",
            cache_root / "models--Systran--faster-whisper-small",
            cache_root / "models--Systran--faster-whisper-base",
        ]
        for root in candidates:
            ref = root / "refs" / "main"
            if ref.exists():
                try:
                    commit = ref.read_text(encoding="utf-8").strip()
                except Exception:
                    commit = ""
                snap = root / "snapshots" / commit
                if commit and snap.exists():
                    return snap
            snaps = root / "snapshots"
            if snaps.exists():
                found = sorted((p for p in snaps.iterdir() if p.is_dir()), reverse=True)
                if found:
                    return found[0]
        return None

    def _record(self, seconds: float) -> np.ndarray:
        import sounddevice as sd

        q: queue.Queue[np.ndarray] = queue.Queue()

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        chunks: list[np.ndarray] = []
        with sd.InputStream(
            device=self._resolve_input_device(),
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            callback=cb,
        ):
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                try:
                    chunks.append(q.get(timeout=0.2))
                except queue.Empty:
                    pass
        if not chunks:
            return np.zeros(int(self.sample_rate * seconds), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def _record_utterance(
        self,
        *,
        start_timeout_s: float,
        max_s: float,
        silence_s: float,
        pre_roll_s: float,
        cue=None,
        input_warmup_s: float = 0.8,
        cue_tone_device=None,
    ) -> np.ndarray:
        import sounddevice as sd

        if cue_tone_device is not None:
            try:
                audio = self._record_utterance_duplex_tone(
                    start_timeout_s=start_timeout_s,
                    max_s=max_s,
                    silence_s=silence_s,
                    pre_roll_s=pre_roll_s,
                    input_warmup_s=input_warmup_s,
                    cue_tone_device=cue_tone_device,
                )
                self.last_record_mode = "duplex_tone"
                return audio
            except Exception:
                self.last_record_mode = "input_fallback"
                pass
        else:
            self.last_record_mode = "input"

        q: queue.Queue[np.ndarray] = queue.Queue()
        blocksize = max(160, int(self.sample_rate * 0.04))
        pre_roll_chunks = max(1, int(pre_roll_s * self.sample_rate / blocksize))
        silence_chunks = max(1, int(silence_s * self.sample_rate / blocksize))
        noise_rms: deque[float] = deque(maxlen=30)
        release = max(self.min_rms * 0.7, 0.0005)

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        pre: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        chunks: list[np.ndarray] = []
        speaking = False
        quiet_seen = 0
        speech_peak = 0.0
        t0 = time.perf_counter()
        speech_t0 = 0.0
        with sd.InputStream(
            device=self._resolve_input_device(),
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=blocksize,
            callback=cb,
        ):
            warm_deadline = time.perf_counter() + max(0.0, input_warmup_s)
            while time.perf_counter() < warm_deadline:
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            if cue is not None:
                try:
                    cue()
                except Exception:
                    pass
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            t0 = time.perf_counter()
            while True:
                now = time.perf_counter()
                if not speaking and now - t0 > start_timeout_s:
                    break
                if speaking and now - speech_t0 > max_s:
                    break
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                if not speaking:
                    pre.append(chunk)
                    noise_rms.append(rms)
                    baseline = float(np.median(noise_rms)) if len(noise_rms) >= 3 else 0.0
                    trigger = max(self.min_rms, baseline * 1.25 + 0.001)
                    if rms >= trigger:
                        speaking = True
                        speech_t0 = now
                        speech_peak = rms
                        release = max(
                            self.min_rms,
                            baseline * 1.35 + 0.0003,
                            speech_peak * 0.55,
                        )
                        chunks.extend(pre)
                        pre.clear()
                        quiet_seen = 0
                    continue
                chunks.append(chunk)
                if rms > speech_peak:
                    speech_peak = rms
                    release = max(release, speech_peak * 0.55)
                if rms < release:
                    quiet_seen += 1
                    if quiet_seen >= silence_chunks:
                        break
                else:
                    quiet_seen = 0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def _record_utterance_duplex_tone(
        self,
        *,
        start_timeout_s: float,
        max_s: float,
        silence_s: float,
        pre_roll_s: float,
        input_warmup_s: float,
        cue_tone_device,
    ) -> np.ndarray:
        import sounddevice as sd

        q: queue.Queue[np.ndarray] = queue.Queue()
        blocksize = max(160, int(self.sample_rate * 0.04))
        pre_roll_chunks = max(1, int(pre_roll_s * self.sample_rate / blocksize))
        silence_chunks = max(1, int(silence_s * self.sample_rate / blocksize))
        noise_rms: deque[float] = deque(maxlen=30)
        release = max(self.min_rms * 0.7, 0.0005)
        tone_left = 0
        tone_pos = 0

        def cb(indata, outdata, frames, time_info, status):
            nonlocal tone_left, tone_pos
            q.put(indata[:, 0].copy())
            outdata.fill(0.0)
            if tone_left <= 0:
                return
            n = min(frames, tone_left)
            t = (np.arange(n, dtype=np.float32) + float(tone_pos)) / float(self.sample_rate)
            outdata[:n, 0] = 0.20 * np.sin(2.0 * np.pi * 880.0 * t)
            tone_pos += n
            tone_left -= n

        def drain_for(seconds: float) -> None:
            deadline = time.perf_counter() + max(0.0, seconds)
            while time.perf_counter() < deadline:
                try:
                    q.get(timeout=0.03)
                except queue.Empty:
                    pass

        pre: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        chunks: list[np.ndarray] = []
        speaking = False
        quiet_seen = 0
        speech_peak = 0.0
        speech_t0 = 0.0
        with sd.Stream(
            device=(self._resolve_input_device(), cue_tone_device),
            channels=(1, 1),
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=blocksize,
            callback=cb,
        ):
            drain_for(input_warmup_s)
            tone_left = max(1, int(self.sample_rate * 0.12))
            drain_for(0.18)
            t0 = time.perf_counter()
            while True:
                now = time.perf_counter()
                if not speaking and now - t0 > start_timeout_s:
                    break
                if speaking and now - speech_t0 > max_s:
                    break
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                if not speaking:
                    pre.append(chunk)
                    noise_rms.append(rms)
                    baseline = float(np.median(noise_rms)) if len(noise_rms) >= 3 else 0.0
                    trigger = max(self.min_rms, baseline * 1.25 + 0.001)
                    if rms >= trigger:
                        speaking = True
                        speech_t0 = now
                        speech_peak = rms
                        release = max(
                            self.min_rms,
                            baseline * 1.35 + 0.0003,
                            speech_peak * 0.55,
                        )
                        chunks.extend(pre)
                        pre.clear()
                        quiet_seen = 0
                    continue
                chunks.append(chunk)
                if rms > speech_peak:
                    speech_peak = rms
                    release = max(release, speech_peak * 0.55)
                if rms < release:
                    quiet_seen += 1
                    if quiet_seen >= silence_chunks:
                        break
                else:
                    quiet_seen = 0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def _resolve_input_device(self):
        if self._resolved_input_device is not None:
            return self._resolved_input_device
        if self.input_device is None:
            return None
        if isinstance(self.input_device, int):
            self._resolved_input_device = self.input_device
            return self._resolved_input_device

        needle = self._norm_device_name(str(self.input_device))
        matches: list[tuple[int, float, str]] = []
        for idx, name, _channels, rate, _hostapi in self.list_input_devices():
            hay = self._norm_device_name(name)
            if needle in hay:
                rate_penalty = abs(rate - float(self.sample_rate))
                matches.append((idx, rate_penalty, name))
        if not matches:
            self._resolved_input_device = self.input_device
            return self._resolved_input_device
        matches.sort(key=lambda x: (x[1], x[0]))
        self._resolved_input_device = matches[0][0]
        return self._resolved_input_device

    @staticmethod
    def _norm_device_name(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()
