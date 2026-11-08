"""Тесты текстовой и файловой логики (не требуют звуковой карты и GUI).

Запуск:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tutor_transcript.audio_sink import WavWriter, read_wav_mono, resample_linear
from tutor_transcript.capture import BlockBuffer, Levels, mix_channels
from tutor_transcript.config import Config, load_config, save_config
from tutor_transcript.devices import DeviceInfo, LOOPBACK, MIC, pick_default
from tutor_transcript.transcriber import (
    join_segments,
    normalize_text,
    save_transcript,
    transcript_filename,
    unique_path,
    wrap_text,
)


class TestConfig(unittest.TestCase):
    def test_defaults_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            cfg = load_config(path)
            self.assertEqual(cfg.model_size, "medium")
            self.assertEqual(cfg.language, "ru")
            cfg.model_size = "small"
            cfg.output_dir = tmp
            save_config(cfg, path)
            again = load_config(path)
            self.assertEqual(again.model_size, "small")
            self.assertEqual(again.output_dir, tmp)

    def test_broken_json_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{ это не json", encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.model_size, "medium")

    def test_invalid_values_normalized(self) -> None:
        cfg = Config.from_dict({"model_size": "huge", "beam_size": 99,
                                "capture_mode": "nonsense", "appearance": "neon"})
        self.assertEqual(cfg.model_size, "medium")
        self.assertEqual(cfg.beam_size, 10)
        self.assertEqual(cfg.capture_mode, "both")
        self.assertEqual(cfg.appearance, "dark")


class TestDevices(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = [
            DeviceInfo(LOOPBACK, "Speakers (Realtek)", 2, True),
            DeviceInfo(LOOPBACK, "CABLE Input (VB-Audio)", 2, False),
            DeviceInfo(MIC, "Headset Microphone", 1, False),
            DeviceInfo(MIC, "Микрофон (Realtek)", 2, True),
        ]

    def test_prefers_default_loopback(self) -> None:
        chosen = pick_default(self.devices, LOOPBACK)
        self.assertEqual(chosen.name, "Speakers (Realtek)")
        self.assertFalse(chosen.is_virtual_cable)

    def test_remembers_saved_name(self) -> None:
        chosen = pick_default(self.devices, LOOPBACK, "CABLE Input (VB-Audio)")
        self.assertTrue(chosen.is_virtual_cable)

    def test_mic_prefers_real_microphone(self) -> None:
        chosen = pick_default(self.devices, MIC)
        self.assertEqual(chosen.key, MIC)
        self.assertIn("Realtek", chosen.name)

    def test_missing_kind_returns_none(self) -> None:
        only_mics = [d for d in self.devices if d.key == MIC]
        self.assertIsNone(pick_default(only_mics, LOOPBACK))

    def test_display_is_unique(self) -> None:
        labels = [d.display() for d in self.devices]
        self.assertEqual(len(labels), len(set(labels)))


class TestTextAssembly(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_text("  привет   мир "), "привет мир")
        self.assertEqual(normalize_text(" . "), "")

    def test_join_segments_single_stream(self) -> None:
        text = join_segments([
            "Сегодня разберём квадратные уравнения.",
            "начнём с простого примера",
            "и найдём дискриминант",
            "Готово.",
        ])
        # после точки — новый абзац, внутри фразы — пробел
        self.assertIn("уравнения.\nначнём", text)
        # заглавная буква в новом сегменте тоже означает новое предложение
        self.assertIn("дискриминант\nГотово.", text)
        self.assertNotIn("  ", text)
        self.assertNotIn(" ,", text)

    def test_join_keeps_enumeration_on_one_line(self) -> None:
        text = join_segments(["Возьмём, например,", "число пять", "и прибавим два."])
        self.assertEqual(text, "Возьмём, например, число пять и прибавим два.")

    def test_wrap_keeps_paragraphs(self) -> None:
        text = "слово " * 40
        wrapped = wrap_text("Абзац один.\n\n" + text, width=30)
        lines = wrapped.split("\n")
        self.assertTrue(all(len(line) <= 30 for line in lines if line))
        self.assertIn("", lines)  # пустая строка между абзацами

    def test_transcript_filename_format(self) -> None:
        name = transcript_filename(datetime(2026, 11, 9, 18, 30, 0))
        self.assertEqual(name, "transcript_2026-11-09_18-30-00.txt")

    def test_unique_path_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "transcript_x.txt").write_text("old", encoding="utf-8")
            path = unique_path(Path(tmp), "transcript_x.txt")
            self.assertEqual(path.name, "transcript_x_2.txt")

    def test_save_transcript_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_transcript("Привет, мир.", Path(tmp), header="Занятие")
            content = path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("Занятие"))
            self.assertIn("Привет, мир.", content)


class TestAudioPlumbing(unittest.TestCase):
    def test_wav_writer_and_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            writer = WavWriter(path, 16000)
            signal = (0.5 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000))
            writer.write(signal.astype(np.float32))
            writer.write(np.zeros(1600, dtype=np.float32))
            self.assertAlmostEqual(writer.duration, 0.2, places=3)
            writer.close()

            data, rate = read_wav_mono(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(data.size, 3200)
            self.assertGreater(float(np.max(np.abs(data))), 0.4)
            self.assertLessEqual(float(np.max(np.abs(data))), 0.51)

    def test_block_buffer_pairs_sources(self) -> None:
        buffer = BlockBuffer([MIC, LOOPBACK], blocksize=10)
        mic_block = np.full(10, 0.5, dtype=np.float32)
        sys_block = np.full(10, 0.25, dtype=np.float32)

        self.assertIsNone(buffer.push(MIC, mic_block))
        ready = buffer.push(LOOPBACK, sys_block)
        self.assertIsNotNone(ready)
        self.assertEqual(set(ready), {MIC, LOOPBACK})
        self.assertEqual(buffer.pending_counts(), {MIC: 0, LOOPBACK: 0})

    def test_block_buffer_fits_short_blocks(self) -> None:
        buffer = BlockBuffer([MIC], blocksize=10)
        ready = buffer.push(MIC, np.ones(4, dtype=np.float32))
        self.assertEqual(ready[MIC].size, 10)
        self.assertEqual(ready[MIC][4:].tolist(), [0.0] * 6)

    def test_block_buffer_drops_old_blocks_when_flooded(self) -> None:
        buffer = BlockBuffer([MIC, LOOPBACK], blocksize=4, max_pending=3)
        for _ in range(10):
            buffer.push(MIC, np.ones(4, dtype=np.float32))
        self.assertEqual(buffer.pending_counts()[MIC], 3)
        self.assertEqual(buffer.dropped_blocks, 7)

    def test_block_buffer_pop_all(self) -> None:
        buffer = BlockBuffer([MIC, LOOPBACK], blocksize=4)
        buffer.push(MIC, np.ones(4, dtype=np.float32))
        buffer.push(MIC, np.ones(4, dtype=np.float32))
        buffer.push(LOOPBACK, np.ones(4, dtype=np.float32))
        groups = buffer.pop_all()
        # одна полная пара (mic+loopback) уже была отдана push,
        # остался «лишний» блок микрофона — он должен дойти до записи
        self.assertEqual(len(groups), 1)
        self.assertIsNotNone(groups[0][MIC])
        self.assertIsNone(groups[0][LOOPBACK])

    def test_mix_two_channels_does_not_clip(self) -> None:
        parts = {
            MIC: np.full(8, 0.9, dtype=np.float32),
            LOOPBACK: np.full(8, 0.9, dtype=np.float32),
        }
        mixed = mix_channels(parts, 8)
        self.assertEqual(mixed.size, 8)
        self.assertTrue(bool(np.all(mixed <= 1.0)))
        self.assertAlmostEqual(float(mixed[0]), 1.0, places=3)

    def test_mix_single_channel_passthrough(self) -> None:
        mixed = mix_channels({MIC: np.full(4, 0.3, dtype=np.float32)}, 4)
        self.assertAlmostEqual(float(mixed[0]), 0.3, places=5)

    def test_resample_changes_length(self) -> None:
        data = np.sin(np.linspace(0, 10, 48000)).astype(np.float32)
        out = resample_linear(data, 48000, 16000)
        self.assertAlmostEqual(out.size / 16000, data.size / 48000, places=3)

    def test_levels_decay(self) -> None:
        levels = Levels()
        levels.update(MIC, np.array([0.8], dtype=np.float32))
        self.assertAlmostEqual(levels.get(MIC), 0.8, places=3)
        levels.update(MIC, np.array([0.0], dtype=np.float32))
        self.assertLess(levels.get(MIC), 0.8)


class TestFallbackSource(unittest.TestCase):
    """Проверяем, что при сбое основного пути включается резервный."""

    def test_switches_to_fallback(self) -> None:
        from tutor_transcript.capture import AudioSource, FallbackSource

        class Broken(AudioSource):
            key = LOOPBACK

            def open(self) -> None:
                raise RuntimeError("нет устройства")

            def read(self, numframes: int) -> np.ndarray:  # pragma: no cover
                raise AssertionError

            def close(self) -> None:
                pass

        class Working(AudioSource):
            key = LOOPBACK
            description = "резерв"

            def open(self) -> None:
                self.opened = True

            def read(self, numframes: int) -> np.ndarray:
                return np.full(numframes, 0.1, dtype=np.float32)

            def close(self) -> None:
                pass

        messages = []
        source = FallbackSource(
            LOOPBACK,
            primary_factory=Broken,
            fallback_factory=Working,
            on_fallback=messages.append,
        )
        source.open()
        self.assertTrue(source.using_fallback)
        np.testing.assert_allclose(source.read(3), np.full(3, 0.1, dtype=np.float32))
        self.assertTrue(messages)

    def test_raises_when_no_fallback(self) -> None:
        from tutor_transcript.capture import AudioSource, CaptureError, FallbackSource

        class Broken(AudioSource):
            key = MIC

            def open(self) -> None:
                raise RuntimeError("нет устройства")

            def read(self, numframes: int) -> np.ndarray:  # pragma: no cover
                raise AssertionError

            def close(self) -> None:
                pass

        source = FallbackSource(MIC, primary_factory=Broken, fallback_factory=None)
        with self.assertRaises(CaptureError):
            source.open()


if __name__ == "__main__":
    unittest.main(verbosity=2)


class _FakeRecorder:
    """Поддельная запись soundcard: синусоида 1000 Гц в частоте устройства."""

    def __init__(self, samplerate: int, tone: float = 1000.0) -> None:
        self.samplerate = samplerate
        self.tone = tone
        self._position = 0
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        self.exited = True

    def record(self, numframes=None):
        index = np.arange(self._position, self._position + int(numframes))
        block = np.sin(2 * np.pi * self.tone * index / self.samplerate)
        self._position += int(numframes)
        return block.astype(np.float32).reshape(-1, 1)


class _FakeMic:
    def __init__(self, allowed_rates) -> None:
        self.allowed_rates = allowed_rates
        self.requests = []

    def recorder(self, samplerate, channels=1):
        self.requests.append(samplerate)
        if samplerate not in self.allowed_rates:
            raise OSError(f"формат {samplerate} Гц не поддерживается")
        return _FakeRecorder(samplerate)


def _dominant_frequency(signal: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return float(np.fft.rfftfreq(signal.size, 1.0 / rate)[np.argmax(spectrum)])


class TestSoundCardSourceRates(unittest.TestCase):
    """Устройства, которые не соглашаются на 16 кГц, должны работать."""

    def test_native_16k_is_used_directly(self) -> None:
        from tutor_transcript.capture import SoundCardSource

        mic = _FakeMic({16000})
        source = SoundCardSource(MIC, lambda: mic, 16000, "тестовый микрофон")
        source.open()
        block = source.read(3200)
        self.assertEqual(mic.requests, [16000])
        self.assertEqual(source.device_rate, 16000)
        self.assertEqual(block.size, 3200)
        self.assertAlmostEqual(_dominant_frequency(block, 16000), 1000.0, delta=20)
        source.close()

    def test_falls_back_to_48k_and_resamples(self) -> None:
        from tutor_transcript.capture import SoundCardSource

        mic = _FakeMic({48000})
        source = SoundCardSource(MIC, lambda: mic, 16000, "bluetooth гарнитура")
        source.open()
        self.assertEqual(source.device_rate, 48000)
        # читаем три блока подряд: частота должна сохраняться, размер — точным
        blocks = [source.read(3200) for _ in range(3)]
        for block in blocks:
            self.assertEqual(block.size, 3200)
            self.assertAlmostEqual(_dominant_frequency(block, 16000), 1000.0, delta=25)
        source.close()

    def test_raises_when_no_rate_supported(self) -> None:
        from tutor_transcript.capture import CaptureError, SoundCardSource

        mic = _FakeMic(set())
        source = SoundCardSource(MIC, lambda: mic, 16000, "сломанный микрофон")
        with self.assertRaises(CaptureError):
            source.open()
