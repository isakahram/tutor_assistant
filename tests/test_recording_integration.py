"""Интеграция: «запись → WAV → расшифровка → txt на рабочем столе».

Звуковая карта не нужна: контроллер получает два искусственных источника
(микрофон и системный звук), которые выдают сигнал в реальном времени.
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tutor_transcript import transcriber as tr
from tutor_transcript.audio_sink import read_wav_mono
from tutor_transcript.capture import AudioSource, RecordingController
from tutor_transcript.devices import LOOPBACK, MIC


class ToneSource(AudioSource):
    """Источник, выдающий синусоиду в реальном темпе."""

    def __init__(self, key: str, freq: float, rate: int = 16000,
                 amplitude: float = 0.3) -> None:
        self.key = key
        self.freq = freq
        self.rate = rate
        self.amplitude = amplitude
        self.description = f"тест {key}"
        self._position = 0
        self._next_time = 0.0
        self.closed = False

    def open(self) -> None:
        self._next_time = time.monotonic()

    def read(self, numframes: int) -> np.ndarray:
        index = np.arange(self._position, self._position + numframes)
        block = self.amplitude * np.sin(2 * np.pi * self.freq * index / self.rate)
        self._position += numframes
        self._next_time += numframes / self.rate
        delay = self._next_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return block.astype(np.float32)

    def close(self) -> None:
        self.closed = True


class _BrokenSource(AudioSource):
    """Источник, который не открывается (как «занятый» микрофон)."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.description = "сломанный источник"

    def open(self) -> None:
        raise RuntimeError("устройство занято другой программой")

    def read(self, numframes: int) -> np.ndarray:  # pragma: no cover
        raise AssertionError("не должно вызываться")

    def close(self) -> None:
        return None


class _FakeSegment:
    def __init__(self, text: str, end: float) -> None:
        self.text = text
        self.end = end


class _FakeInfo:
    def __init__(self, duration: float) -> None:
        self.duration = duration


class _FakeWhisperModel:
    def __init__(self, reference: dict) -> None:
        self.ref = reference

    def transcribe(self, audio, **kwargs):
        self.ref["size"] = len(audio)   # в модель пришли сэмплы, а не путь
        total = len(audio) / 16000
        return iter([_FakeSegment("Проверка интеграции прошла успешно.", total)]), \
            _FakeInfo(total)


class _FakePipeline:
    def __init__(self, model) -> None:
        self.model = model

    def transcribe(self, audio, **kwargs):
        return self.model.transcribe(audio, **kwargs)


class BaseRecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        tr.free_model()

    def tearDown(self) -> None:
        tr.free_model()
        sys.modules.pop("faster_whisper", None)
        self.tmp.cleanup()

    def _controller(self, sources) -> RecordingController:
        controller = RecordingController()
        controller._build_sources = lambda: (  # type: ignore[assignment]
            list(sources), [], {s.key: s.description for s in sources}
        )
        return controller


class TestRecording(BaseRecordingTest):
    def test_records_both_sources_and_mixes(self) -> None:
        mic, system = ToneSource(MIC, 440), ToneSource(LOOPBACK, 880)
        controller = self._controller([mic, system])
        wav = self.dir / "rec.wav"
        controller.start(wav)
        time.sleep(1.4)
        result = controller.stop()

        self.assertTrue(wav.exists())
        self.assertGreaterEqual(result.duration, 1.0)
        self.assertLessEqual(result.duration, 1.7)
        data, rate = read_wav_mono(wav)
        self.assertEqual(rate, 16000)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)
        self.assertTrue(mic.closed and system.closed)

    def test_pause_skips_audio(self) -> None:
        controller = self._controller([ToneSource(MIC, 440),
                                       ToneSource(LOOPBACK, 880)])
        controller.start(self.dir / "paused.wav")
        time.sleep(0.8)
        controller.pause()
        self.assertTrue(controller.is_paused)
        time.sleep(0.8)
        controller.resume()
        self.assertFalse(controller.is_paused)
        time.sleep(0.8)
        result = controller.stop()
        self.assertLess(result.duration, 2.1)
        self.assertGreater(result.duration, 1.2)

    def test_double_start_and_stop_raise(self) -> None:
        controller = self._controller([ToneSource(MIC, 440)])
        controller.start(self.dir / "x.wav")
        with self.assertRaises(RuntimeError):
            controller.start(self.dir / "y.wav")
        controller.stop()
        with self.assertRaises(RuntimeError):
            controller.stop()

    def test_timer_grows(self) -> None:
        controller = self._controller([ToneSource(MIC, 440)])
        controller.start(self.dir / "t.wav")
        first = controller.elapsed
        time.sleep(0.6)
        second = controller.elapsed
        controller.stop()
        self.assertGreater(second, first)

    def test_silence_is_reported_as_warning(self) -> None:
        controller = self._controller([ToneSource(MIC, 440, amplitude=0.0)])
        controller.start(self.dir / "silent.wav")
        time.sleep(0.4)
        result = controller.stop()
        self.assertTrue(any("низкий" in w for w in result.warnings))


class TestDegradedSources(BaseRecordingTest):
    def test_mic_failure_keeps_system_audio(self) -> None:
        controller = self._controller([_BrokenSource(MIC),
                                       ToneSource(LOOPBACK, 880)])
        wav = self.dir / "degraded.wav"
        controller.start(wav)
        time.sleep(1.2)
        result = controller.stop()
        self.assertGreater(result.duration, 0.6)
        self.assertGreaterEqual(len(result.warnings), 1)
        data, _ = read_wav_mono(wav)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)

    def test_system_failure_keeps_mic(self) -> None:
        controller = self._controller([ToneSource(MIC, 440),
                                       _BrokenSource(LOOPBACK)])
        wav = self.dir / "degraded2.wav"
        controller.start(wav)
        time.sleep(1.0)
        result = controller.stop()
        self.assertGreater(result.duration, 0.5)
        data, _ = read_wav_mono(wav)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)

    def test_all_failed_gives_warning_not_hang(self) -> None:
        controller = self._controller([_BrokenSource(MIC),
                                       _BrokenSource(LOOPBACK)])
        controller.start(self.dir / "both_broken.wav")
        time.sleep(0.4)
        result = controller.stop()
        self.assertLessEqual(result.duration, 0.5)
        self.assertGreaterEqual(len(result.warnings), 2)


class TestEndToEnd(BaseRecordingTest):
    def test_record_then_transcribe_to_desktop_dir(self) -> None:
        controller = self._controller([ToneSource(MIC, 440),
                                       ToneSource(LOOPBACK, 880)])
        wav = self.dir / "e2e.wav"
        controller.start(wav)
        time.sleep(1.0)
        controller.stop()

        reference: dict = {}
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = lambda *a, **kw: _FakeWhisperModel(reference)
        module.BatchedInferencePipeline = _FakePipeline
        sys.modules["faster_whisper"] = module

        path = tr.Transcriber().transcribe_and_save(wav, self.dir)
        self.assertTrue(path.exists())
        self.assertIn("Проверка интеграции", path.read_text(encoding="utf-8"))
        self.assertGreater(reference["size"], 16000 * 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
