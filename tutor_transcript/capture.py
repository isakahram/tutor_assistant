"""Захват звука: источники (loopback + микрофон), смешивание и управление записью.

Схема работы
------------

1. Для каждого источника запускается отдельный поток.
2. Поток читает звук блоками фиксированной длины (по умолчанию 0,2 с) и кладёт
   блок в общий :class:`BlockBuffer`.
3. Как только все источники отдали блок, буфер отдаёт «полный» блок, который
   сразу пишется в WAV-файл как сумма каналов (микрофон + звук компьютера).

Устройства по умолчанию имеют независимые часы, поэтому блоки синхронизируются
по счётчику: если один источник отстал, его данные дополняются тишиной —
рассинхронизация не накапливается (это критично для занятий длиной в час и
больше).

Если ``soundcard`` не может открыть loopback-устройство, системный звук
автоматически переключается на запасной путь — WASAPI loopback через
``sounddevice`` (:class:`SoundDeviceLoopbackSource`).
"""

from __future__ import annotations

import difflib
import logging
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .audio_sink import WavWriter, read_wav_mono, resample_linear
from .config import SAMPLE_RATE, Config, app_data_dir
from .devices import DeviceInfo, LOOPBACK, MIC, pick_default

log = logging.getLogger(__name__)

#: Длительность одного блока в секундах (определяет «зерно» синхронизации).
BLOCK_SECONDS = 0.2

#: Сколько блоков максимум держим в запасе (страховка от утечки памяти).
MAX_PENDING_BLOCKS = 50


class CaptureError(RuntimeError):
    """Ошибка открытия или чтения аудиоустройства."""


# --------------------------------------------------------------------------- #
#  COM для Windows: soundcard работает через MediaFoundation
# --------------------------------------------------------------------------- #
def com_init() -> None:
    """Инициализирует COM в текущем потоке (нужно перед работой со soundcard).

    Библиотека ``soundcard`` инициализирует COM только в том потоке, где её
    импортировали. Мы читаем звук в отдельных потоках, поэтому там нужно
    вызвать ``CoInitializeEx`` ещё раз. Повторный вызов безопасен.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        COINIT_MULTITHREADED = 0x0
        RPC_E_CHANGED_MODE = -2147417850  # 0x80010106 — COM уже инициализирован
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        if hr not in (0, 1, RPC_E_CHANGED_MODE):  # S_OK, S_FALSE, смена модели
            log.debug("CoInitializeEx вернул 0x%08X", hr & 0xFFFFFFFF)
    except Exception as exc:  # noqa: BLE001 - не критично
        log.debug("Не удалось инициализировать COM: %s", exc)


# --------------------------------------------------------------------------- #
#  Буфер блоков и смешивание
# --------------------------------------------------------------------------- #
class BlockBuffer:
    """Собирает блоки от нескольких источников и отдаёт их «полными».

    Потокобезопасен: ``push`` вызывают потоки-источники, ``take`` — поток записи.
    """

    def __init__(self, keys: Iterable[str], blocksize: int,
                 max_pending: int = MAX_PENDING_BLOCKS) -> None:
        self.keys = tuple(keys)
        self.blocksize = int(blocksize)
        self.max_pending = int(max_pending)
        self._pending: Dict[str, List[np.ndarray]] = {k: [] for k in self.keys}
        self._lock = threading.Lock()
        self.dropped_blocks = 0

    def push(self, key: str, samples: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Добавляет блок источника; возвращает готовый набор блоков или None."""
        block = self._fit(samples)
        with self._lock:
            if key not in self._pending or not self.keys:
                return None  # источник отключился/уже не нужен
            per_key = self._pending[key]
            per_key.append(block)
            if len(per_key) > self.max_pending:
                # Потребитель не успевает: выбрасываем самый старый блок,
                # чтобы не съесть память на многочасовой лекции.
                per_key.pop(0)
                self.dropped_blocks += 1
            if not all(self._pending[k] for k in self.keys):
                return None
            return {k: self._pending[k].pop(0) for k in self.keys}

    def _fit(self, samples: np.ndarray) -> np.ndarray:
        """Приводит блок к ровно ``blocksize`` моно-отсчётам float32."""
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == self.blocksize:
            return data
        if data.size > self.blocksize:
            return data[: self.blocksize]
        return np.concatenate(
            [data, np.zeros(self.blocksize - data.size, dtype=np.float32)]
        )

    def pop_all(self) -> List[Dict[str, Optional[np.ndarray]]]:
        """Забирает все накопленные блоки (для финального «дописывания»)."""
        with self._lock:
            sets: List[Dict[str, Optional[np.ndarray]]] = []
            while any(self._pending[k] for k in self.keys):
                sets.append({
                    k: (self._pending[k].pop(0) if self._pending[k] else None)
                    for k in self.keys
                })
            return sets

    def set_keys(self, keys: Iterable[str]) -> None:
        """Меняет набор активных источников (например, если один отвалился).

        Данные отключённого источника выбрасываются, а оставшиеся перестают
        ждать «пару» — иначе запись встала бы и в файл попало бы 0,2 секунды
        звука вместо всего занятия.
        """
        new_keys = tuple(keys)
        with self._lock:
            self._pending = {k: self._pending.get(k, []) for k in new_keys}
            self.keys = new_keys

    def reset(self) -> None:
        with self._lock:
            for key in self.keys:
                self._pending[key].clear()

    def pending_counts(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._pending.items()}


def mix_channels(parts: Dict[str, Optional[np.ndarray]],
                 blocksize: int) -> np.ndarray:
    """Складывает каналы в одну дорожку без клиппинга.

    Если каналов два, каждый предварительно ослабляется, чтобы перегрузка
    случалась реже; итог ограничивается диапазоном [-1, 1].
    """
    present = [p for p in parts.values() if p is not None]
    mixed = np.zeros(blocksize, dtype=np.float32)
    for data in present:
        chunk = np.asarray(data, dtype=np.float32).reshape(-1)
        if chunk.size < blocksize:
            chunk = np.concatenate(
                [chunk, np.zeros(blocksize - chunk.size, dtype=np.float32)]
            )
        mixed += chunk[:blocksize]
    if len(present) > 1:
        mixed *= 0.75
    return np.clip(mixed, -1.0, 1.0)


class Levels:
    """Индикаторы уровня сигнала (потокобезопасно)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {"mic": 0.0, "system": 0.0, "mix": 0.0}

    def update(self, key: str, samples: Optional[np.ndarray]) -> None:
        if samples is None or samples.size == 0:
            return
        peak = float(np.max(np.abs(samples)))
        with self._lock:
            previous = self._values.get(key, 0.0)
            # сглаживание, чтобы полоска не «дёргалась»
            self._values[key] = peak if peak > previous else previous * 0.6

    def get(self, key: str) -> float:
        with self._lock:
            return min(1.0, self._values.get(key, 0.0))

    def reset(self) -> None:
        with self._lock:
            for key in self._values:
                self._values[key] = 0.0


# --------------------------------------------------------------------------- #
#  Источники звука
# --------------------------------------------------------------------------- #
class AudioSource:
    """Базовый класс источника: открыть устройство, читать блоки, закрыть."""

    key: str = MIC
    description: str = ""

    def open(self) -> None:  # pragma: no cover - переопределяется
        raise NotImplementedError

    def read(self, numframes: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SoundCardSource(AudioSource):
    """Источник на библиотеке soundcard (MediaFoundation / WASAPI).

    Работает и для микрофона, и для loopback-копии динамиков.

    Частота: WASAPI обычно соглашается на 16 кГц (драйвер сам пересчитывает),
    но некоторые устройства (Bluetooth-гарнитуры, старые USB-микрофоны)
    принимают только «родную» частоту. Поэтому открываем первую подходящую
    из списка и, если частота отличается, пересчитываем сами.
    """

    #: Частоты, которые пробуем по очереди.
    FALLBACK_RATES = (48_000, 44_100)

    def __init__(self, key: str, mic_factory: Callable[[], object],
                 samplerate: int, description: str = "") -> None:
        self.key = key
        self.description = description
        self._mic_factory = mic_factory
        self.samplerate = int(samplerate)   # частота записи (16 кГц)
        self.device_rate = self.samplerate  # частота устройства
        self._recorder = None
        self._mic = None
        self._leftover = np.zeros(0, dtype=np.float32)

    def open(self) -> None:
        com_init()
        errors = []
        for rate in (self.samplerate,) + self.FALLBACK_RATES:
            try:
                mic = self._mic_factory()
                recorder = mic.recorder(samplerate=rate, channels=1)
                recorder.__enter__()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rate} Гц: {exc}")
                continue
            self._mic = mic
            self._recorder = recorder
            self.device_rate = rate
            if rate != self.samplerate:
                log.info(
                    "Устройство «%s» не отдаёт %s Гц — пишем в %s Гц и "
                    "пересчитываем сами", self.description, self.samplerate, rate,
                )
            log.info("Открыт источник %s: %s (%s Гц)",
                     self.key, self.description, rate)
            return
        raise CaptureError(
            f"не удалось открыть «{self.description or self.key}»: "
            + "; ".join(errors[-3:])
        )

    def read(self, numframes: int) -> np.ndarray:
        if self._recorder is None:
            raise CaptureError(f"источник {self.key} не открыт")
        device_frames = numframes
        if self.device_rate != self.samplerate:
            # просим у устройства столько отсчётов, сколько нужно для
            # numframes на частоте записи
            device_frames = max(
                1, int(round(numframes * self.device_rate / self.samplerate))
            )
        try:
            data = self._recorder.record(numframes=device_frames)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(
                f"сбой чтения «{self.description or self.key}»: {exc}"
            ) from exc
        block = np.asarray(data, dtype=np.float32)
        if block.ndim > 1:
            block = block.mean(axis=1)
        block = block.reshape(-1)

        if self.device_rate == self.samplerate:
            return block

        # пересчёт частоты: склеиваем с остатком и берём ровно numframes
        buffer = np.concatenate([self._leftover, block]) if self._leftover.size else block
        converted = resample_linear(buffer, self.device_rate, self.samplerate)
        if converted.size >= numframes:
            self._leftover = converted[numframes:]
            return converted[:numframes]
        self._leftover = np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [converted, np.zeros(numframes - converted.size, dtype=np.float32)]
        )

    def close(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            try:
                recorder.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.debug("Ошибка закрытия источника %s: %s", self.key, exc)


class SilentSource(AudioSource):
    """Заглушка: отдаёт тишину в реальном времени (если устройство не найдено)."""

    def __init__(self, key: str, samplerate: int,
                 description: str = "тишина (устройство не найдено)") -> None:
        self.key = key
        self.samplerate = int(samplerate)
        self.description = description
        self._next_time = 0.0

    def open(self) -> None:
        self._next_time = time.monotonic()
        log.warning("Источник %s заменён заглушкой (%s)", self.key, self.description)

    def read(self, numframes: int) -> np.ndarray:
        # выдерживаем реальный темп, чтобы второй источник не «убежал» вперёд
        self._next_time += numframes / float(self.samplerate)
        delay = self._next_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return np.zeros(numframes, dtype=np.float32)

    def close(self) -> None:
        return None


class FallbackSource(AudioSource):
    """Источник с автоматическим переходом на запасной путь.

    Сначала пробуем основной способ (soundcard). Если он не открылся или
    сломался во время записи — переключаемся на запасной (sounddevice/WASAPI),
    не прерывая занятие.
    """

    def __init__(self, key: str, primary_factory: Callable[[], AudioSource],
                 fallback_factory: Optional[Callable[[], AudioSource]],
                 on_fallback: Optional[Callable[[str], None]] = None) -> None:
        self.key = key
        self.description = ""
        self._primary_factory = primary_factory
        self._fallback_factory = fallback_factory
        self._on_fallback = on_fallback
        self._active: Optional[AudioSource] = None
        self._using_fallback = False

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _activate_fallback(self, reason: str) -> bool:
        if self._fallback_factory is None:
            return False
        try:
            source = self._fallback_factory()
            source.open()
        except Exception as exc:  # noqa: BLE001
            log.error("Запасной источник для %s не открылся: %s", self.key, exc)
            return False
        if self._active is not None:
            try:
                self._active.close()
            except Exception:  # noqa: BLE001
                pass
        self._active = source
        self._using_fallback = True
        self.description = source.description
        message = (
            f"Системный звук пишется резервным способом ({source.description}). "
            f"Причина: {reason}"
            if self.key == LOOPBACK
            else f"Источник переключён на резервный: {source.description} ({reason})"
        )
        log.warning(message)
        if self._on_fallback is not None:
            try:
                self._on_fallback(message)
            except Exception:  # noqa: BLE001
                pass
        return True

    def open(self) -> None:
        try:
            source = self._primary_factory()
            source.open()
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            if not self._activate_fallback(reason):
                raise CaptureError(reason) from exc
            return
        self._active = source
        self.description = source.description

    def read(self, numframes: int) -> np.ndarray:
        if self._active is None:
            raise CaptureError("источник не открыт")
        try:
            return self._active.read(numframes)
        except Exception as exc:  # noqa: BLE001
            if not self._using_fallback and self._activate_fallback(str(exc)):
                return self._active.read(numframes)  # type: ignore[union-attr]
            raise

    def close(self) -> None:
        source, self._active = self._active, None
        if source is not None:
            try:
                source.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Ошибка закрытия источника %s: %s", self.key, exc)


class SoundDeviceLoopbackSource(AudioSource):
    """Запасной путь для системного звука: sounddevice + WASAPI loopback.

    Используется, если soundcard не смог открыть loopback-устройство.
    Требует, чтобы PortAudio в установленной сборке поддерживал loopback
    (в официальных колёсах sounddevice для Windows это так).
    """

    key = LOOPBACK
    description = "WASAPI loopback (sounddevice)"

    def __init__(self, target_name: str, samplerate: int) -> None:
        self.samplerate = int(samplerate)
        self.target_name = target_name or ""
        self._stream = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._device_rate = self.samplerate
        self._leftover = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[\s()\[\]\-_,.]", "", name.lower())

    def _find_portaudio_device(self) -> Tuple[int, dict]:
        """Ищет WASAPI-устройство, из которого можно читать системный звук."""
        import sounddevice as sd

        devices = list(sd.query_devices())
        hostapis = list(sd.query_hostapis())
        wasapi = [i for i, api in enumerate(hostapis)
                  if "wasapi" in str(api["name"]).lower()]

        candidates = []
        for index, info in enumerate(devices):
            info = dict(info)
            if wasapi and info["hostapi"] not in wasapi:
                continue
            name = str(info["name"])
            # Порто-аудио-сборка PortAudio с loopback: у динамиков появляется
            # одноимённое входное устройство с пометкой [Loopback].
            if info["max_input_channels"] > 0 and "loopback" in name.lower():
                candidates.append((index, info))

        if not candidates:
            raise CaptureError(
                "запасной способ недоступен: sounddevice не нашёл устройств "
                "WASAPI loopback (нужна сборка sounddevice с поддержкой loopback)"
            )

        if self.target_name:
            wanted = self._normalize(self.target_name)
            scored = []
            for index, info in candidates:
                candidate = self._normalize(str(info["name"]))
                ratio = difflib.SequenceMatcher(None, wanted, candidate).ratio()
                scored.append((ratio, index, info))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > 0.3:
                return scored[0][1], scored[0][2]
        return candidates[0][0], candidates[0][1]

    def open(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"sounddevice недоступен: {exc}") from exc

        index, info = self._find_portaudio_device()
        self._device_rate = int(info["default_samplerate"]) or 48000
        channels = 2 if int(info["max_input_channels"]) >= 2 else 1

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("sounddevice status: %s", status)
            block = np.asarray(indata, dtype=np.float32)
            if block.ndim > 1:
                block = block.mean(axis=1)
            try:
                self._queue.put_nowait(block.copy())
            except queue.Full:
                log.debug("Очередь loopback переполнена, блок пропущен")

        self._stream = sd.InputStream(
            device=index,
            channels=channels,
            samplerate=self._device_rate,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        self.description = (
            f"WASAPI loopback: {info['name']} ({self._device_rate} Гц)"
        )
        log.info("Открыт запасной источник системного звука: %s", self.description)

    def read(self, numframes: int) -> np.ndarray:
        deadline = time.monotonic() + 5.0
        while self._buffer.size < numframes:
            if self._stream is not None and not self._stream.active:
                raise CaptureError("поток WASAPI loopback остановился")
            timeout = min(0.2, max(0.01, deadline - time.monotonic()))
            try:
                chunk = self._queue.get(timeout=timeout)
            except queue.Empty:
                if time.monotonic() > deadline:
                    # тишина в системе: отдаём нули, запись продолжается
                    timeout_zero = np.zeros(numframes, dtype=np.float32)
                    if self._device_rate != self.samplerate:
                        zero_count = int(round(
                            numframes * self._device_rate / self.samplerate
                        ))
                        timeout_zero = np.zeros(zero_count, dtype=np.float32)
                        return resample_linear(
                            timeout_zero, self._device_rate, self.samplerate
                        )[:numframes]
                    return timeout_zero
                continue
            self._buffer = np.concatenate([self._buffer, chunk])

        take, self._buffer = self._buffer, np.zeros(0, dtype=np.float32)
        if self._device_rate != self.samplerate:
            take = resample_linear(take, self._device_rate, self.samplerate)
        if take.size < numframes:
            take = np.concatenate(
                [take, np.zeros(numframes - take.size, dtype=np.float32)]
            )
        return take[:numframes]

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Ошибка закрытия WASAPI loopback: %s", exc)


# --------------------------------------------------------------------------- #
#  Поток-источник
# --------------------------------------------------------------------------- #
class SourceThread(threading.Thread):
    """Читает звук из источника блоками и отдаёт их в буфер."""

    def __init__(self, source: AudioSource, blocksize: int,
                 on_block: Callable[[str, np.ndarray], None],
                 on_error: Callable[[str, str], None],
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        super().__init__(name=f"capture-{source.key}", daemon=True)
        self.source = source
        self.blocksize = blocksize
        self._on_block = on_block
        self._on_error = on_error
        self._on_log = on_log
        self._stop_event = threading.Event()
        self.blocks_read = 0
        self.error: Optional[str] = None
        self.open_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        com_init()
        try:
            self.source.open()
        except Exception as exc:  # noqa: BLE001
            self.open_error = str(exc)
            self._on_error(self.source.key, str(exc))
            log.error("Источник %s не открылся: %s", self.source.key, exc)
            return

        if self._on_log is not None and self.source.description:
            self._on_log(f"{self.source.key}: {self.source.description}")

        try:
            while not self._stop_event.is_set():
                try:
                    block = self.source.read(self.blocksize)
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
                    self._on_error(self.source.key, str(exc))
                    log.error("Ошибка чтения %s: %s", self.source.key, exc)
                    break
                self.blocks_read += 1
                self._on_block(self.source.key, block)
        finally:
            try:
                self.source.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Ошибка закрытия %s: %s", self.source.key, exc)


# --------------------------------------------------------------------------- #
#  Результат записи
# --------------------------------------------------------------------------- #
@dataclass
class RecordingResult:
    """Итог записи: файл, длительность и предупреждения."""

    wav_path: Path
    duration: float
    started_at: datetime
    stopped_at: datetime
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: True — файл выбран пользователем (его нельзя удалять/перезаписывать).
    external: bool = False

    @property
    def empty(self) -> bool:
        return self.duration < 1.0


# --------------------------------------------------------------------------- #
#  Контроллер записи
# --------------------------------------------------------------------------- #
class RecordingController:
    """Управляет записью: старт → пауза/продолжение → стоп.

    Пример::

        controller = RecordingController(config)
        controller.start()
        ...
        result = controller.stop()
    """

    def __init__(self, config: Config, levels: Optional[Levels] = None,
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        self.levels = levels or Levels()
        self._on_log = on_log

        self.samplerate = SAMPLE_RATE
        self.blocksize = int(round(BLOCK_SECONDS * SAMPLE_RATE))
        self.buffer = BlockBuffer([], self.blocksize)

        self._writer: Optional[WavWriter] = None
        self._threads: List[SourceThread] = []
        self._paused = threading.Event()
        self._lock = threading.RLock()
        self._warnings: List[str] = []
        self._sources_info: Dict[str, str] = {}
        self._started_at: Optional[datetime] = None
        self._recording = False
        self._sink_error: Optional[str] = None
        self._source_errors: List[str] = []
        self._dropped_blocks = 0
        self._overall_peak = 0.0
        self.wav_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    #  Свойства состояния
    # ------------------------------------------------------------------ #
    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def elapsed(self) -> float:
        """Сколько звука уже записано (секунды)."""
        writer = self._writer
        return writer.duration if writer is not None else 0.0

    @property
    def source_errors(self) -> List[str]:
        return list(self._source_errors)

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    @property
    def source_keys(self) -> Tuple[str, ...]:
        """Ключи активных источников (mic / loopback)."""
        return tuple(self.buffer.keys)

    # ------------------------------------------------------------------ #
    #  Сборка источников
    # ------------------------------------------------------------------ #
    def _build_sources(
        self, devices: List[DeviceInfo], mode: str
    ) -> Tuple[List[AudioSource], List[str], Dict[str, str]]:
        """Создаёт источники согласно режиму записи."""
        warnings: List[str] = []
        info: Dict[str, str] = {}
        sources: List[AudioSource] = []

        need_system = mode in ("both", "system")
        need_mic = mode in ("both", "mic")

        system_device = pick_default(devices, LOOPBACK, self.config.system_device)
        mic_device = pick_default(devices, MIC, self.config.mic_device)

        # --- системный звук ------------------------------------------------
        if need_system:
            speaker_name = system_device.name if system_device else ""

            def _primary() -> AudioSource:
                return SoundCardSource(
                    LOOPBACK,
                    mic_factory=lambda: _soundcard_loopback(speaker_name),
                    samplerate=self.samplerate,
                    description=speaker_name or "динамики по умолчанию",
                )

            fallback_factory: Optional[Callable[[], AudioSource]] = None
            if self.config.allow_sounddevice_fallback:
                try:
                    import sounddevice  # noqa: F401

                    fallback_factory = lambda: SoundDeviceLoopbackSource(  # noqa: E731
                        speaker_name, self.samplerate
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        "sounddevice не установлен — резервный захват системного "
                        f"звука недоступен ({exc})."
                    )

            sources.append(
                FallbackSource(
                    LOOPBACK,
                    primary_factory=_primary,
                    fallback_factory=fallback_factory,
                    on_fallback=self._on_runtime_warning,
                )
            )
            info[LOOPBACK] = speaker_name or "динамики по умолчанию"
            if system_device is None:
                warnings.append(
                    "Loopback-устройство не найдено: попробуем резервный способ. "
                    "Если звука компьютера нет, запустите воспроизведение "
                    "(видео/звонок) и нажмите «Обновить устройства»."
                )

        # --- микрофон ------------------------------------------------------
        if need_mic:
            if mic_device is not None:
                mic_name = mic_device.name
                sources.append(
                    SoundCardSource(
                        MIC,
                        mic_factory=lambda: _soundcard_microphone(mic_name),
                        samplerate=self.samplerate,
                        description=mic_name,
                    )
                )
                info[MIC] = mic_name
            else:
                warnings.append(
                    "Микрофон не найден. Запись будет идти без вашего голоса — "
                    "проверьте, что микрофон подключён, и обновите устройства."
                )
                sources.append(
                    SilentSource(MIC, self.samplerate, "микрофон не найден")
                )
                info[MIC] = "—"

        if not sources:
            warnings.append("Не выбрано ни одного источника звука.")
            sources.append(SilentSource(MIC, self.samplerate, "нет источников"))

        return sources, warnings, info

    # ------------------------------------------------------------------ #
    #  Старт / пауза / стоп
    # ------------------------------------------------------------------ #
    def start(self, output_wav: Optional[Path] = None) -> Path:
        """Начинает запись. Возвращает путь к временному WAV-файлу."""
        with self._lock:
            if self._recording:
                raise RuntimeError("Запись уже идёт")

            from .devices import list_devices

            devices = list_devices()
            mode = self.config.capture_mode
            sources, warnings, info = self._build_sources(devices, mode)
            self._warnings = list(warnings)
            self._sources_info = info
            self._source_errors = []

            if output_wav is None:
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_wav = app_data_dir() / "temp" / f"recording_{stamp}.wav"
            self.wav_path = Path(output_wav)
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)

            self.buffer = BlockBuffer([s.key for s in sources], self.blocksize)
            self.levels.reset()
            self._paused.clear()
            self._sink_error = None
            self._dropped_blocks = 0
            self._overall_peak = 0.0
            self._writer = WavWriter(self.wav_path, self.samplerate)
            self._threads = [
                SourceThread(
                    source,
                    self.blocksize,
                    on_block=self._on_block,
                    on_error=self._on_source_error,
                    on_log=self._log,
                )
                for source in sources
            ]
            self._started_at = datetime.now()
            self._recording = True
            for thread in self._threads:
                thread.start()

            self._log(
                "Запись началась. Источники: "
                + "; ".join(f"{k}: {v}" for k, v in info.items())
            )
            for warning in warnings:
                self._log("Внимание: " + warning)
            return self.wav_path

    def pause(self) -> None:
        """Приостанавливает запись (звук читается, но в файл не пишется)."""
        if not self._recording:
            return
        self._paused.set()
        self._log("Запись на паузе.")

    def resume(self) -> None:
        """Продолжает запись после паузы."""
        if not self._recording:
            return
        self.buffer.reset()
        self._paused.clear()
        self._log("Запись продолжена.")

    def toggle_pause(self) -> bool:
        """Переключает паузу. Возвращает True, если запись сейчас на паузе."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()
        return self.is_paused

    def stop(self) -> RecordingResult:
        """Останавливает запись и возвращает информацию о файле."""
        with self._lock:
            if not self._recording:
                raise RuntimeError("Запись не идёт")
            self._recording = False
            self._paused.clear()

            for thread in self._threads:
                thread.stop()
            for thread in self._threads:
                thread.join(timeout=5.0)

            # забираем остатки из буфера, чтобы не потерять последние 0,2 с
            self._flush_pending()

            writer, self._writer = self._writer, None
            if writer is not None:
                writer.close()

            warnings = list(self._warnings) + list(self._source_errors)
            if self._sink_error:
                warnings.append(self._sink_error)
            if self.buffer.dropped_blocks:
                warnings.append(
                    "Часть звука не успела записаться (компьютер был перегружен). "
                    "Попробуйте закрыть лишние программы."
                )
            if self.is_too_quiet():
                warnings.append(
                    "Похоже, звук почти не поступал (уровень очень низкий). "
                    "Проверьте, что микрофон и звук компьютера слышны."
                )

            duration = 0.0
            if self.wav_path is not None and self.wav_path.exists():
                try:
                    data, rate = read_wav_mono(self.wav_path)
                    duration = data.size / float(rate) if rate else 0.0
                except Exception as exc:  # noqa: BLE001
                    log.warning("Не удалось прочитать запись: %s", exc)

            result = RecordingResult(
                wav_path=self.wav_path or Path(),
                duration=duration,
                started_at=self._started_at or datetime.now(),
                stopped_at=datetime.now(),
                sources=dict(self._sources_info),
                warnings=warnings,
            )
            self._threads = []
            self._log(f"Запись остановлена. Длительность: {format_hms(duration)}")
            return result

    # ------------------------------------------------------------------ #
    def is_too_quiet(self) -> bool:
        """Проверяет, был ли записанный звук подозрительно тихим."""
        return self._overall_peak < 0.002

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _on_block(self, key: str, block: np.ndarray) -> None:
        """Колбэк потока-источника: складываем блок в буфер."""
        self.levels.update(key, block)
        if self._paused.is_set():
            return  # на паузе данные читаются, но не пишутся
        ready = self.buffer.push(key, block)
        if ready is not None:
            self._write_ready(ready)

    def _write_ready(self, ready: Dict[str, np.ndarray]) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            if len(ready) == 1:
                mixed = next(iter(ready.values()))
                self.levels.update("mix", mixed)
            else:
                mixed = mix_channels(ready, self.blocksize)
                self.levels.update("mix", mixed)
            peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
            if peak > self._overall_peak:
                self._overall_peak = peak
            writer.write(mixed)
        except Exception as exc:  # noqa: BLE001
            if self._sink_error is None:  # предупреждаем только один раз
                self._sink_error = f"ошибка записи в файл: {exc}"
                log.error(self._sink_error)

    def _flush_pending(self) -> None:
        """Дописывает блоки, для которых не нашлось пары (источник отстал)."""
        writer = self._writer
        if writer is None:
            return
        try:
            for group in self.buffer.pop_all():
                writer.write(mix_channels(group, self.blocksize))
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось дописать остатки: %s", exc)

    def _on_source_error(self, key: str, message: str) -> None:
        """Источник не открылся или сломался — продолжаем без него."""
        if message not in self._source_errors:
            self._source_errors.append(message)
        self._log("Ошибка: " + message)

        remaining = [k for k in self.buffer.keys if k != key]
        if remaining and key in self.buffer.keys:
            self.buffer.set_keys(remaining)
            labels = {MIC: "микрофон", LOOPBACK: "системный звук"}
            self._on_runtime_warning(
                f"Источник «{labels.get(key, key)}» отключился — "
                "запись продолжается с остальными."
            )
        elif not remaining:
            self._log(
                "Больше не осталось работающих источников: запись остановится "
                "пустой. Проверьте устройства и попробуйте снова."
            )

    def _on_runtime_warning(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)
        self._log("Внимание: " + message)

    def _log(self, message: str) -> None:
        log.info(message)
        if self._on_log is not None:
            try:
                self._on_log(message)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
#  Помощники
# --------------------------------------------------------------------------- #
def _soundcard_loopback(name: str):
    """Возвращает loopback-«микрофон» нужных динамиков (по имени)."""
    import soundcard as sc

    if name:
        try:
            return sc.get_microphone(id=name, include_loopback=True)
        except Exception as exc:  # noqa: BLE001
            log.info("Не удалось взять loopback «%s»: %s", name, exc)
    # имена могли измениться — берём динамики по умолчанию
    return sc.get_microphone(id=str(sc.default_speaker().name),
                             include_loopback=True)


def _soundcard_microphone(name: str):
    """Возвращает микрофон по имени (иначе — микрофон по умолчанию)."""
    import soundcard as sc

    if name:
        try:
            return sc.get_microphone(id=name, include_loopback=False)
        except Exception as exc:  # noqa: BLE001
            log.info("Не удалось взять микрофон «%s»: %s", name, exc)
    return sc.default_microphone()


def format_hms(seconds: float) -> str:
    """Форматирует секунды как ЧЧ:ММ:СС."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
