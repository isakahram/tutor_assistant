"""Интеграционный тест: «запись → WAV → расшифровка → txt».

Звуковая карта в тестах не нужна: контроллер получает два искусственных
источника (микрофон и системный звук), которые выдают сигнал в реальном
времени. Проверяем смешивание, синхронизацию, паузу, длительность итогового
файла и полный проход через расшифровку.
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
from tutor_transcript.config import Config
from tutor_transcript.devices import LOOPBACK, MIC


class ToneSource(AudioSource):
    """Источник, выдающий синусоиду с заданной частотой в реальном темпе."""

    def __init__(self, key: str, freq: float, rate: int = 16000,
                 amplitude: float = 0.3) -> None:
        self.key = key
        self.freq = freq
        self.rate = rate
        self.amplitude = amplitude
        self.description = f"тест {key}"
        self._position = 0
        self._next_time = 0.0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True
        self._next_time = time.monotonic()

    def read(self, numframes: int) -> np.ndarray:
        index = np.arange(self._position, self._position + numframes)
        block = (self.amplitude * np.sin(2 * np.pi * self.freq * index / self.rate))
        self._position += numframes
        self._next_time += numframes / self.rate
        delay = self._next_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return block.astype(np.float32)

    def close(self) -> None:
        self.closed = True


class _FakeSegment:
    def __init__(self, text: str, end: float) -> None:
        self.text = text
        self.end = end


class _FakeInfo:
    def __init__(self, duration: float) -> None:
        self.duration = duration


class _FakeWhisper:
    def __init__(self, duration_ref: dict) -> None:
        self.ref = duration_ref

    def transcribe(self, audio, **kwargs):
        # проверяем, что в модель пришёл массив нужной длины
        self.ref["size"] = len(audio)
        total = len(audio) / 16000
        return iter([_FakeSegment("Проверка интеграции прошла успешно.", total)]), \
            _FakeInfo(total)


class TestRecordingIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        tr._MODEL_CACHE.clear()

    def tearDown(self) -> None:
        tr._MODEL_CACHE.clear()
        sys.modules.pop("faster_whisper", None)
        self.tmp.cleanup()

    def _controller(self) -> RecordingController:
        config = Config()
        controller = RecordingController(config)
        mic = ToneSource(MIC, 440)
        system = ToneSource(LOOPBACK, 880)
        controller._build_sources = lambda devices, mode: (  # type: ignore[assignment]
            [mic, system],
            [],
            {MIC: "тестовый микрофон", LOOPBACK: "тестовые динамики"},
        )
        self.sources = (mic, system)
        return controller

    # ------------------------------------------------------------------ #
    def test_records_both_sources_and_mixes(self) -> None:
        controller = self._controller()
        wav = self.dir / "rec.wav"
        controller.start(wav)
        time.sleep(1.4)
        result = controller.stop()

        self.assertTrue(wav.exists())
        # 1,4 с записи ± один блок синхронизации (0,2 с)
        self.assertGreaterEqual(result.duration, 1.0)
        self.assertLessEqual(result.duration, 1.7)

        data, rate = read_wav_mono(wav)
        self.assertEqual(rate, 16000)
        self.assertGreater(data.size, 0)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)
        self.assertLessEqual(float(np.max(np.abs(data))), 1.0)
        self.assertEqual(result.sources[MIC], "тестовый микрофон")
        for source in self.sources:
            self.assertTrue(source.closed)

    def test_pause_skips_audio(self) -> None:
        controller = self._controller()
        wav = self.dir / "paused.wav"
        controller.start(wav)
        time.sleep(0.8)
        controller.pause()
        self.assertTrue(controller.is_paused)
        time.sleep(0.8)
        controller.resume()
        self.assertFalse(controller.is_paused)
        time.sleep(0.8)
        result = controller.stop()

        # всего прошло ~2,4 с, но записано ~1,6 с (без паузы ~0,8 с)
        self.assertLess(result.duration, 2.1)
        self.assertGreater(result.duration, 1.2)

    def test_double_start_and_stop_raise(self) -> None:
        controller = self._controller()
        controller.start(self.dir / "x.wav")
        with self.assertRaises(RuntimeError):
            controller.start(self.dir / "y.wav")
        controller.stop()
        with self.assertRaises(RuntimeError):
            controller.stop()

    def test_timer_grows(self) -> None:
        controller = self._controller()
        controller.start(self.dir / "t.wav")
        first = controller.elapsed
        time.sleep(0.6)
        second = controller.elapsed
        controller.stop()
        self.assertGreater(second, first)

    # ------------------------------------------------------------------ #
    def test_end_to_end_transcribe(self) -> None:
        """Запись → WAV → (поддельная) модель → txt в выбранной папке."""
        controller = self._controller()
        wav = self.dir / "e2e.wav"
        controller.start(wav)
        time.sleep(1.0)
        controller.stop()

        reference: dict = {}
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = lambda *a, **kw: _FakeWhisper(reference)
        sys.modules["faster_whisper"] = module

        config = Config(model_size="tiny")
        transcriber = tr.Transcriber(config)
        path = transcriber.transcribe_and_save(wav, output_dir=self.dir)

        self.assertTrue(path.exists())
        self.assertTrue(path.name.startswith("transcript_"))
        text = path.read_text(encoding="utf-8")
        self.assertIn("Проверка интеграции", text)
        # модель получила массив сэмплов, а не путь к файлу
        self.assertIsInstance(reference["size"], int)
        self.assertGreater(reference["size"], 16000 * 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)


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


class TestDegradedSources(unittest.TestCase):
    """Если один источник не открылся, второй должен продолжать писаться."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _controller(self, sources, warnings=None):
        controller = RecordingController(Config())
        controller._build_sources = lambda devices, mode: (  # type: ignore[assignment]
            list(sources), list(warnings or []), {s.key: s.description for s in sources}
        )
        return controller

    def test_mic_failure_keeps_system_audio(self) -> None:
        system = ToneSource(LOOPBACK, 880)
        controller = self._controller([_BrokenSource(MIC), system])
        wav = self.dir / "degraded.wav"
        controller.start(wav)
        time.sleep(1.2)
        result = controller.stop()

        self.assertGreater(result.duration, 0.6)
        self.assertGreaterEqual(len(result.warnings), 1)
        data, _rate = read_wav_mono(wav)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)

    def test_system_failure_keeps_mic(self) -> None:
        mic = ToneSource(MIC, 440)
        controller = self._controller([mic, _BrokenSource(LOOPBACK)])
        wav = self.dir / "degraded2.wav"
        controller.start(wav)
        time.sleep(1.0)
        result = controller.stop()
        self.assertGreater(result.duration, 0.5)
        data, _rate = read_wav_mono(wav)
        self.assertGreater(float(np.max(np.abs(data))), 0.05)

    def test_all_failed_gives_warning_not_hang(self) -> None:
        controller = self._controller([_BrokenSource(MIC), _BrokenSource(LOOPBACK)])
        wav = self.dir / "both_broken.wav"
        controller.start(wav)
        time.sleep(0.4)
        result = controller.stop()
        self.assertLessEqual(result.duration, 0.5)
        self.assertGreaterEqual(len(result.warnings), 2)
