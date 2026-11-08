"""Максимально быстрое распознавание речи: faster-whisper, модель ``base``.

Что сделано для скорости (порядок — по величине выигрыша):

* **BatchedInferencePipeline, batch_size=8** — 8 фрагментов по 30 с уходят в
  модель одним вызовом и считаются параллельно на всех ядрах (×2–4);
* **beam_size=1, best_of=1, temperature=0** — жадный поиск вместо перебора
  пяти вариантов (×2–3);
* **int8 и все ядра** (``compute_type="int8"``, ``cpu_threads=0``) — самый
  быстрый режим вычислений на процессоре (×1,5–2);
* **without_timestamps=True** — не тратим время на токены времени (~×1,1);
* **condition_on_previous_text=False** — не тянем контекст между фрагментами
  (~×1,1);
* **device="cpu" напрямую** — без попытки инициализировать CUDA (−3–5 с).

Дополнительно запись ведётся сразу в 16 кГц моно, поэтому перед распознаванием
не нужно ни пересчитывать частоту, ни смешивать каналы.

``vad_filter`` включён: в faster-whisper это один быстрый проход Silero VAD,
который вырезает паузы — модели достаётся меньше звука, и итог получается
быстрее, а не медленнее.

Итог на 4-ядерном процессоре — примерно ×20–25 к реальному времени: час
занятия расшифровывается за 2,5–3 минуты. На 8 и более ядрах — быстрее.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .audio_sink import read_wav_mono
from .config import (
    BATCH_SIZE,
    CHUNK_SECONDS,
    CPU_THREADS,
    LANGUAGE,
    MODEL_SIZE,
    SAMPLE_RATE,
    models_dir,
)

log = logging.getLogger(__name__)

#: Ширина строки в итоговом тексте (чтобы файл было удобно читать).
WRAP_WIDTH = 110

#: Короткая подсказка модели: стиль текста и пунктуация. Длинные промпты
#: замедляют декодирование, поэтому здесь всего одна фраза.
INITIAL_PROMPT = "Расшифровка урока с репетитором на русском языке."


class TranscriptionError(RuntimeError):
    """Ошибка распознавания (нет библиотеки, модели, пустая запись)."""


# --------------------------------------------------------------------------- #
#  Склейка текста
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"[.!?…]+$")


def normalize_text(text: str) -> str:
    """Чистит один сегмент: лишние пробелы, пустые реплики, «♪»."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if cleaned in {".", ",", "-", "—", "♪", "*"}:
        return ""
    return cleaned.replace("♪", "").strip()


def join_segments(segments: Iterable[str]) -> str:
    """Склеивает сегменты в сплошной текст (без меток времени).

    Новый абзац начинается, когда предыдущий фрагмент закончился точкой или
    новый фрагмент начинается с заглавной буквы. Запятая/двоеточие/тире —
    признак того, что фраза продолжается с новой строки Whisper.
    """
    parts: List[str] = []
    continuation = (",", ";", ":", "—", "-")
    for raw in segments:
        segment = normalize_text(raw)
        if not segment:
            continue
        if not parts:
            parts.append(segment)
            continue
        previous = parts[-1]
        previous_continues = previous.endswith(continuation)
        starts_new_sentence = segment[:1].isupper()
        if not previous_continues and (
            _SENTENCE_END.search(previous) or starts_new_sentence
        ):
            parts.append("\n" + segment)
        else:
            parts.append(" " + segment)

    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wrap_text(text: str, width: int = WRAP_WIDTH) -> str:
    """Переносит длинные строки по словам, сохраняя абзацы."""
    if width <= 0:
        return text
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
#  Сохранение файла
# --------------------------------------------------------------------------- #
def transcript_filename(moment: Optional[datetime] = None) -> str:
    """Имя файла: ``transcript_2026-11-09_18-30-00.txt``."""
    moment = moment or datetime.now()
    return f"transcript_{moment.strftime('%Y-%m-%d_%H-%M-%S')}.txt"


def unique_path(directory: Path, filename: str) -> Path:
    """Не перезаписывает существующие файлы: добавляет ``_2``, ``_3`` и т. д."""
    path = Path(directory) / filename
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = Path(directory) / f"{path.stem}_{counter}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_transcript(text: str, directory: Path,
                    moment: Optional[datetime] = None,
                    header: Optional[str] = None) -> Path:
    """Сохраняет текст в .txt и возвращает путь."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_path(directory, transcript_filename(moment))
    body = text if text.endswith("\n") else text + "\n"
    if header:
        body = header.rstrip() + "\n\n" + body
    path.write_text(body, encoding="utf-8")
    log.info("Расшифровка сохранена: %s", path)
    return path


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    """Состояние распознавания (для строки статуса в окне)."""

    done_seconds: float = 0.0      # сколько секунд звука уже обработано
    total_seconds: float = 0.0     # длительность записи
    started_at: float = field(default_factory=time.time)
    message: str = ""

    @property
    def percent(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return min(100.0, self.done_seconds / self.total_seconds * 100.0)

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def speed(self) -> Optional[float]:
        """Во сколько раз быстрее реального времени идёт обработка."""
        if self.done_seconds <= 0 or self.elapsed <= 0:
            return None
        return self.done_seconds / self.elapsed


_MODEL_LOCK = threading.Lock()
_MODEL: Optional[object] = None          # WhisperModel
_PIPELINE: Optional[object] = None       # BatchedInferencePipeline / Sequential


class SequentialPipeline:
    """Запасной путь для старых версий faster-whisper (без батчей).

    Работает так же, только медленнее: по одному 30-секундному фрагменту.
    Нужен, чтобы программа не падала, если установлена версия < 1.1.0.
    """

    def __init__(self, model) -> None:
        self.model = model

    def transcribe(self, audio, **kwargs):
        kwargs.pop("batch_size", None)
        kwargs.pop("chunk_length", None)
        return self.model.transcribe(audio, **kwargs)


def load_model(on_status: Optional[Callable[[str], None]] = None):
    """Загружает модель и обёртку для батчевой обработки (кэшируется в памяти).

    Первая загрузка скачивает модель ``base`` (~145 МБ) — об этом сообщаем.
    """
    global _MODEL, _PIPELINE
    with _MODEL_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(
            "Не установлена библиотека faster-whisper. Установите её командой:\n"
            "    pip install faster-whisper\n"
            f"Подробности: {exc}"
        ) from exc

    if on_status is not None:
        on_status(
            f"Загрузка модели «{MODEL_SIZE}» (первый запуск скачивает ~145 МБ)…"
        )

    try:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",          # CUDA не пробуем: неудачный старт стоит секунд
            compute_type="int8",   # самый быстрый режим для процессора
            cpu_threads=int(CPU_THREADS),   # 0 = занять все ядра
            download_root=str(models_dir()),
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(
            f"Не удалось загрузить модель «{MODEL_SIZE}»: {exc}\n\n"
            "Проверьте подключение к интернету (модель скачивается один раз) "
            "и свободное место на диске."
        ) from exc

    try:
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model=model)
    except Exception as exc:  # noqa: BLE001 - старая версия библиотеки
        log.warning("Батчевый режим недоступен (%s), работаем по одному чанку", exc)
        pipeline = SequentialPipeline(model)

    with _MODEL_LOCK:
        _MODEL, _PIPELINE = model, pipeline
    log.info("Модель %s загружена (int8, батч %s, чанк %s с)",
             MODEL_SIZE, BATCH_SIZE, CHUNK_SECONDS)
    if on_status is not None:
        on_status("Модель готова.")
    return pipeline


def free_model() -> None:
    """Выгружает модель из памяти (~150 МБ)."""
    global _MODEL, _PIPELINE
    with _MODEL_LOCK:
        _MODEL = None
        _PIPELINE = None
    gc.collect()


def preload(on_status: Optional[Callable[[str], None]] = None) -> None:
    """Заранее загружает модель (вызывается при старте записи)."""
    try:
        load_model(on_status)
    except TranscriptionError as exc:
        log.warning("Модель пока не загружена: %s", exc)


# --------------------------------------------------------------------------- #
#  Распознавание
# --------------------------------------------------------------------------- #
class Transcriber:
    """Распознаёт WAV и сохраняет сплошной текст в .txt."""

    def __init__(self, on_progress: Optional[Callable[[Progress], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self._on_progress = on_progress
        self._on_status = on_status
        self.progress = Progress()
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    def _emit(self, **changes) -> None:
        """Публикует «снимок» состояния (его отрисует поток интерфейса)."""
        for key, value in changes.items():
            setattr(self.progress, key, value)
        if self._on_progress is not None:
            try:
                self._on_progress(replace(self.progress))
            except Exception:  # noqa: BLE001
                pass

    def _status(self, message: str) -> None:
        self._emit(message=message)
        if self._on_status is not None:
            try:
                self._on_status(message)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    def transcribe_file(self, wav_path: str | Path) -> str:
        """Распознаёт файл и возвращает текст, ничего не сохраняя."""
        path = Path(wav_path)
        if not path.exists():
            raise TranscriptionError(f"Файл записи не найден: {path}")

        self._emit(started_at=time.time(), message="Чтение записи…")
        try:
            audio, rate = read_wav_mono(path)
        except Exception as exc:  # noqa: BLE001
            # не наш формат — пусть Whisper декодирует сам (через PyAV/FFmpeg)
            log.info("Не удалось прочитать WAV (%s), отдаю файл модели", exc)
            audio, rate = None, SAMPLE_RATE  # type: ignore[assignment]

        total = (audio.size / float(rate)) if audio is not None and rate else 0.0
        self._emit(total_seconds=total)
        if audio is not None and total < 0.3:
            raise TranscriptionError(
                "Запись слишком короткая (меньше секунды) — распознавать нечего."
            )

        pipeline = load_model(on_status=self._status)
        self._status("Распознаю…")

        source = path if audio is None else audio
        try:
            segments, info = pipeline.transcribe(
                source,
                language=LANGUAGE,
                task="transcribe",
                batch_size=BATCH_SIZE,
                chunk_length=CHUNK_SECONDS,
                beam_size=1,                 # жадный поиск — быстрее всего
                best_of=1,
                temperature=0.0,             # без перебора вариантов
                without_timestamps=True,     # не нужны метки времени
                condition_on_previous_text=False,  # не тянем контекст между чанками
                initial_prompt=INITIAL_PROMPT,
                vad_filter=True,             # вырезаем паузы: меньше чанков — быстрее
                vad_parameters={"min_silence_duration_ms": 500},
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Ошибка распознавания: {exc}") from exc

        total = float(getattr(info, "duration", 0.0) or total or 0.0)
        self._emit(total_seconds=total)

        pieces: List[str] = []
        done = 0.0
        for segment in segments:
            if self._cancel.is_set():
                raise TranscriptionError("Распознавание отменено.")
            pieces.append(segment.text)
            done = float(getattr(segment, "end", done) or done)
            self._emit(done_seconds=done)
            speed = self.progress.speed
            self._status(
                f"Распознано {format_hms(done)} из {format_hms(total)}"
                + (f" (×{speed:.0f} к реальному времени)" if speed else "")
            )

        text = join_segments(pieces)
        if not text.strip():
            raise TranscriptionError(
                "Речь не распознана: в записи нет разборчивого голоса. "
                "Проверьте микрофон и громкость системного звука."
            )
        self._emit(message="Готово.")
        return text

    # ------------------------------------------------------------------ #
    def transcribe_and_save(self, wav_path: str | Path, directory: Path,
                            header: Optional[str] = None) -> Path:
        """Распознаёт запись и сохраняет результат в .txt."""
        text = self.transcribe_file(wav_path)
        self._status("Сохраняю файл…")
        return save_transcript(wrap_text(text), directory, header=header)


def format_hms(seconds: float) -> str:
    """ЧЧ:ММ:СС без долей секунды."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
