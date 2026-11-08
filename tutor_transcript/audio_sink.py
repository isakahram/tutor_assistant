"""Запись WAV и вспомогательные операции со звуком.

``soundcard`` умеет отдавать только «сырые» float-массивы, поэтому WAV-файл
формируем сами: так мы сразу пишем 16 кГц моно — ровно тот формат, который
ждёт Whisper, и не тратим время на пересчёт после записи.

Заголовок файла обновляется на каждом блоке, поэтому WAV остаётся читаемым
даже если приложение закрыли или компьютер выключили во время занятия.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np


class WavWriter:
    """Потоковая запись моно-WAV (16 бит, заданная частота)."""

    def __init__(self, path: str | Path, samplerate: int) -> None:
        self.path = Path(path)
        self.samplerate = int(samplerate)
        self._frames = 0
        self._file: BinaryIO | None = None
        self._wav: wave.Wave_write | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._wav = wave.open(self._file, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.samplerate)

    # ------------------------------------------------------------------ #
    def write(self, samples: np.ndarray) -> None:
        """Дописывает блок сэмплов float32 в диапазоне [-1, 1]."""
        if self._wav is None:
            raise RuntimeError("WavWriter уже закрыт")
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._wav.writeframes(pcm.tobytes())
        self._frames += int(pcm.size)

    @property
    def duration(self) -> float:
        """Длительность записанного звука в секундах."""
        return self._frames / float(self.samplerate)

    def close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()  # дописывает реальные размеры в заголовок
            finally:
                self._wav = None
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Линейная передискретизация моно-сигнала.

    Нужна только если аудиоустройство не отдаёт 16 кГц (бывает у Bluetooth-
    гарнитур); обычно частота запрашивается сразу нужная и пересчёт не нужен.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or data.size == 0:
        return data
    duration = data.size / float(src_rate)
    new_size = max(1, int(round(duration * dst_rate)))
    src_t = np.linspace(0.0, duration, num=data.size, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=new_size, endpoint=False, dtype=np.float64)
    return np.interp(dst_t, src_t, data).astype(np.float32)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Читает WAV и возвращает (моно float32, частота)."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"ожидался 16-битный WAV, получено {width * 8} бит")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate
