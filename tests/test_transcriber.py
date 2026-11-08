"""Тесты расшифровки с поддельной моделью faster-whisper.

Настоящая модель весит больше гигабайта, поэтому в тестах подменяем
``WhisperModel`` маленькой заглушкой: проверяем подготовку данных, расчёт
прогресса, склейку текста, сохранение файла и обработку ошибок.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tutor_transcript import transcriber as tr
from tutor_transcript.audio_sink import WavWriter
from tutor_transcript.config import Config


class _Segment:
    def __init__(self, text: str, end: float) -> None:
        self.text = text
        self.end = end


class _Info:
    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.language = "ru"


class _FakeModel:
    """Модель-заглушка: возвращает заранее заданные сегменты."""

    def __init__(self, text_parts, duration: float, record: dict) -> None:
        self.text_parts = text_parts
        self.duration = duration
        self.record = record

    def transcribe(self, audio, **kwargs):
        self.record["audio"] = audio
        self.record["kwargs"] = kwargs
        step = self.duration / max(1, len(self.text_parts))
        segments = [
            _Segment(part, step * (index + 1))
            for index, part in enumerate(self.text_parts)
        ]
        return iter(segments), _Info(self.duration)


def _install_fake_faster_whisper(model: _FakeModel) -> None:
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = lambda *a, **kw: model
    module.__version__ = "test"
    sys.modules["faster_whisper"] = module


def _write_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> None:
    writer = WavWriter(path, rate)
    rng = np.random.default_rng(0)
    writer.write(0.05 * rng.standard_normal(int(rate * seconds)).astype(np.float32))
    writer.close()


class TestTranscribeWithFakeModel(unittest.TestCase):
    def setUp(self) -> None:
        tr._MODEL_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = self.dir / "rec.wav"
        _write_wav(self.wav)

    def tearDown(self) -> None:
        tr._MODEL_CACHE.clear()
        sys.modules.pop("faster_whisper", None)
        self.tmp.cleanup()

    def test_text_is_assembled_and_file_saved(self) -> None:
        record: dict = {}
        _install_fake_faster_whisper(
            _FakeModel(["Здравствуйте!", "сегодня мы разберём дроби",
                        "и решим пару задач."], 2.0, record)
        )
        config = Config(model_size="tiny")
        progress_events = []
        transcriber = tr.Transcriber(config, on_progress=progress_events.append)
        path = transcriber.transcribe_and_save(self.wav, output_dir=self.dir)

        self.assertTrue(path.exists())
        self.assertTrue(path.name.startswith("transcript_"))
        self.assertTrue(path.name.endswith(".txt"))
        content = path.read_text(encoding="utf-8")
        self.assertIn("Здравствуйте!", content)
        self.assertIn("разберём дроби", content)

        kwargs = record["kwargs"]
        self.assertEqual(kwargs["language"], "ru")
        self.assertTrue(kwargs["without_timestamps"])
        self.assertFalse(kwargs["word_timestamps"])
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertIsInstance(record["audio"], np.ndarray)

        self.assertTrue(progress_events)
        self.assertEqual(progress_events[-1].stage, "done")
        self.assertEqual(progress_events[-1].percent, 100.0)

    def test_medium_is_default_model(self) -> None:
        record: dict = {}
        model = _FakeModel(["тест"], 1.0, record)
        _install_fake_faster_whisper(model)
        config = Config()
        self.assertEqual(config.model_size, "medium")
        tr.load_model(config.model_size)
        loaded = tr._MODEL_CACHE[(config.model_size, "auto", "")]
        self.assertIs(loaded, model)

    def test_empty_speech_raises(self) -> None:
        _install_fake_faster_whisper(_FakeModel([" ", "."], 1.0, {}))
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber(Config()).transcribe_file(self.wav)

    def test_too_short_audio_raises(self) -> None:
        short = self.dir / "short.wav"
        _write_wav(short, seconds=0.1)
        _install_fake_faster_whisper(_FakeModel(["тест"], 0.1, {}))
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber(Config()).transcribe_file(short)

    def test_missing_file_raises(self) -> None:
        _install_fake_faster_whisper(_FakeModel(["тест"], 1.0, {}))
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber(Config()).transcribe_file(self.dir / "нет.wav")

    def test_progress_percent_grows(self) -> None:
        _install_fake_faster_whisper(
            _FakeModel(["раз", "два", "три", "четыре"], 4.0, {})
        )
        events = []
        tr.Transcriber(Config(), on_progress=events.append).transcribe_file(self.wav)
        transcribing = [e for e in events if e.stage == "transcribing" and e.percent]
        self.assertGreaterEqual(len(transcribing), 4)
        percents = [e.percent for e in transcribing]
        self.assertEqual(percents, sorted(percents))

    def test_cancel_stops_early(self) -> None:
        _install_fake_faster_whisper(_FakeModel(["раз", "два", "три"], 3.0, {}))
        transcriber = tr.Transcriber(Config())
        transcriber.cancel()
        with self.assertRaises(tr.TranscriptionError):
            transcriber.transcribe_file(self.wav)

    def test_missing_library_message(self) -> None:
        sys.modules["faster_whisper"] = None  # имитируем отсутствие пакета
        tr._MODEL_CACHE.clear()
        try:
            with self.assertRaises(tr.TranscriptionError) as ctx:
                tr.load_model("medium")
            self.assertIn("faster-whisper", str(ctx.exception))
        finally:
            sys.modules.pop("faster_whisper", None)


class TestOutputHelpers(unittest.TestCase):
    def test_filename_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            moment = datetime(2026, 11, 9, 18, 30, 0)
            first = tr.save_transcript("текст", directory, moment=moment)
            second = tr.save_transcript("текст", directory, moment=moment)
            self.assertEqual(first.name, "transcript_2026-11-09_18-30-00.txt")
            self.assertEqual(second.name, "transcript_2026-11-09_18-30-00_2.txt")

    def test_header_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = tr.save_transcript("текст", Path(tmp), header="Занятие 5")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("Занятие 5"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
