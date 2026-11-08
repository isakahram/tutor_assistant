"""Тесты распознавания: быстрое окружение вызова и работа с файлами.

Настоящая модель весит мегабайты, поэтому в тестах подменяем
``faster_whisper`` маленькой заглушкой и проверяем главное:

* что используются именно «быстрые» параметры (батчи, beam_size=1,
  без меток времени), — от них зависит цель «час за 3 минуты»;
* что текст склеивается в сплошной и сохраняется в .txt;
* что прогресс и отмена работают;
* что ошибки объясняются по-русски.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tutor_transcript import config, transcriber as tr
from tutor_transcript.audio_sink import WavWriter


class _Segment:
    def __init__(self, text: str, end: float) -> None:
        self.text = text
        self.end = end


class _Info:
    def __init__(self, duration: float) -> None:
        self.duration = duration


class _FakeModel:
    """Модель-заглушка: запоминает параметры вызова и отдаёт сегменты."""

    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, "kwargs": kwargs})
        duration = 4.0
        texts = ["Здравствуйте!", "сегодня разберём дроби", "и решим пару задач."]
        step = duration / len(texts)
        segments = [_Segment(t, step * (i + 1)) for i, t in enumerate(texts)]
        return iter(segments), _Info(duration)


class _FakePipeline:
    """Замена BatchedInferencePipeline."""

    def __init__(self, model) -> None:
        self.model = model

    def transcribe(self, audio, **kwargs):
        return self.model.transcribe(audio, **kwargs)


def _install_fake_faster_whisper(model: _FakeModel) -> dict:
    """Подставляет фальшивый модуль faster_whisper; возвращает настройки model."""
    captured = {}

    def _whisper_model(*args, **kwargs):
        captured["init"] = {"args": args, "kwargs": kwargs}
        return model

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _whisper_model
    module.BatchedInferencePipeline = _FakePipeline
    sys.modules["faster_whisper"] = module
    return captured


def _write_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> None:
    writer = WavWriter(path, rate)
    rng = np.random.default_rng(0)
    writer.write(0.05 * rng.standard_normal(int(rate * seconds)).astype(np.float32))
    writer.close()


class TestFastSettings(unittest.TestCase):
    def setUp(self) -> None:
        tr.free_model()
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = Path(self.tmp.name) / "rec.wav"
        _write_wav(self.wav)

    def tearDown(self) -> None:
        tr.free_model()
        sys.modules.pop("faster_whisper", None)
        self.tmp.cleanup()

    def test_batched_pipeline_and_fast_options(self) -> None:
        model = _FakeModel()
        captured = _install_fake_faster_whisper(model)

        pipeline = tr.load_model()
        self.assertIsInstance(pipeline, _FakePipeline)

        # --- настройки самой модели ---
        init = captured["init"]
        self.assertEqual(init["args"][0], "base")              # модель base
        self.assertEqual(init["kwargs"]["device"], "cpu")      # без попытки CUDA
        self.assertEqual(init["kwargs"]["compute_type"], "int8")
        self.assertEqual(init["kwargs"]["download_root"], str(config.models_dir()))

        text = tr.Transcriber().transcribe_file(self.wav)
        self.assertIn("дроби", text)

        # --- настройки распознавания (от них зависит скорость) ---
        kwargs = model.calls[0]["kwargs"]
        self.assertEqual(kwargs["language"], "ru")
        self.assertEqual(kwargs["batch_size"], config.BATCH_SIZE)
        self.assertGreaterEqual(kwargs["batch_size"], 8)      # батчи, а не по 1
        self.assertEqual(kwargs["chunk_length"], config.CHUNK_SECONDS)
        self.assertEqual(kwargs["beam_size"], 1)              # жадный поиск
        self.assertEqual(kwargs["best_of"], 1)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertTrue(kwargs["without_timestamps"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertNotIn("word_timestamps", kwargs)
        self.assertIsInstance(model.calls[0]["audio"], np.ndarray)

    def test_model_is_loaded_once(self) -> None:
        model = _FakeModel()
        captured = _install_fake_faster_whisper(model)
        tr.load_model()
        first = sys.modules["faster_whisper"].WhisperModel
        tr.load_model()
        self.assertIs(sys.modules["faster_whisper"].WhisperModel, first)
        self.assertIn("init", captured)

    def test_free_model_drops_cache(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        tr.load_model()
        tr.free_model()
        self.assertIsNone(tr._PIPELINE)

    def test_text_is_assembled_and_saved(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        out_dir = Path(self.tmp.name) / "out"
        progress_events = []
        path = tr.Transcriber(on_progress=progress_events.append).transcribe_and_save(
            self.wav, out_dir, header="Занятие"
        )
        self.assertTrue(path.exists())
        self.assertTrue(path.name.startswith("transcript_"))
        content = path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("Занятие"))
        self.assertIn("разберём дроби", content)
        self.assertTrue(progress_events)
        self.assertEqual(progress_events[-1].percent, 100.0)

    def test_progress_grows(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        events = []
        tr.Transcriber(on_progress=events.append).transcribe_file(self.wav)
        done = [e.done_seconds for e in events if e.done_seconds]
        self.assertGreaterEqual(len(done), 3)
        self.assertEqual(done, sorted(done))

    def test_cancel_stops_early(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        transcriber = tr.Transcriber()
        transcriber.cancel()
        with self.assertRaises(tr.TranscriptionError):
            transcriber.transcribe_file(self.wav)

    def test_too_short_audio_raises(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        short = Path(self.tmp.name) / "short.wav"
        _write_wav(short, seconds=0.1)
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber().transcribe_file(short)

    def test_missing_file_raises(self) -> None:
        _install_fake_faster_whisper(_FakeModel())
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber().transcribe_file(Path(self.tmp.name) / "нет.wav")

    def test_empty_text_raises(self) -> None:
        class _Silent(_FakeModel):
            def transcribe(self, audio, **kwargs):
                return iter([_Segment(" ", 1.0)]), _Info(1.0)

        _install_fake_faster_whisper(_Silent())
        with self.assertRaises(tr.TranscriptionError):
            tr.Transcriber().transcribe_file(self.wav)

    def test_missing_library_message(self) -> None:
        tr.free_model()
        sys.modules["faster_whisper"] = None  # имитируем отсутствие пакета
        try:
            with self.assertRaises(tr.TranscriptionError) as ctx:
                tr.load_model()
            self.assertIn("faster-whisper", str(ctx.exception))
        finally:
            sys.modules.pop("faster_whisper", None)


class TestSpeedBudget(unittest.TestCase):
    """Час записи на модели base должен укладываться в 3 минуты.

    Реальную скорость задаёт процессор, поэтому здесь проверяется наш вклад:
    за одно обращение к модели уходит 8 × 30 = 240 секунд звука, значит для
    часа записи нужно всего 15 вызовов (и ни одного лишнего прохода VAD).
    """

    def test_hour_of_audio_needs_only_15_model_calls(self) -> None:
        chunk = config.CHUNK_SECONDS
        batch = config.BATCH_SIZE
        self.assertEqual((chunk, batch), (30, 8))
        hour = 3600
        calls = -(-hour // (chunk * batch))  # округление вверх
        self.assertEqual(calls, 15)

    def test_chunk_and_batch_are_not_tiny(self) -> None:
        self.assertGreaterEqual(config.BATCH_SIZE, 8)
        self.assertGreaterEqual(config.CHUNK_SECONDS, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSequentialFallback(unittest.TestCase):
    """Если батчевый режим недоступен, программа обязана всё равно работать."""

    def setUp(self) -> None:
        tr.free_model()
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = Path(self.tmp.name) / "rec.wav"
        _write_wav(self.wav)

    def tearDown(self) -> None:
        tr.free_model()
        sys.modules.pop("faster_whisper", None)
        self.tmp.cleanup()

    def test_works_without_batched_pipeline(self) -> None:
        model = _FakeModel()
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = lambda *a, **kw: model
        # BatchedInferencePipeline отсутствует — как в faster-whisper < 1.1
        sys.modules["faster_whisper"] = module

        pipeline = tr.load_model()
        self.assertIsInstance(pipeline, tr.SequentialPipeline)
        text = tr.Transcriber().transcribe_file(self.wav)
        self.assertIn("дроби", text)
        kwargs = model.calls[0]["kwargs"]
        # «батчевые» параметры убраны, чтобы старый API не ругался
        self.assertNotIn("batch_size", kwargs)
        self.assertNotIn("chunk_length", kwargs)
        self.assertEqual(kwargs["beam_size"], 1)
