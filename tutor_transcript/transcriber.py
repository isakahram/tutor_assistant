"""Распознавание речи через faster-whisper и сохранение расшифровки.

Ключевые решения
----------------

* Модель по умолчанию — ``medium`` (хороший баланс качества и скорости для
  русского языка). Если на вашем компьютере она работает медленно, в
  настройках можно переключиться на ``small`` — результат будет готов
  в 2-3 раза быстрее.
* ``language="ru"`` передаётся явно, чтобы модель не тратила время на
  определение языка и не «прыгала» на английский в технических терминах.
* Метки времени не нужны: склеиваем сегменты в сплошной текст с нормальными
  пробелами и абзацами.
* Звук в модель подаётся массивом NumPy, а не через FFmpeg. Так приложению
  не нужен FFmpeg при расшифровке (он остаётся нужен для декодирования
  «чужих» форматов — mp3/m4a, если вы подсовываете их вручную).
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
from .config import SAMPLE_RATE

log = logging.getLogger(__name__)

#: Сколько символов примерно в одной строке итогового текста.
WRAP_WIDTH = 110

#: Стандартный приветственный промпт: подсказывает модели стиль и лексику,
#: а также помогает с пунктуацией в русском тексте.
DEFAULT_PROMPT = (
    "Это расшифровка онлайн-занятия с репетитором на русском языке. "
    "Обсуждаются темы урока, домашнее задание, примеры и объяснения."
)


class TranscriptionError(RuntimeError):
    """Ошибка распознавания (нет библиотеки, нет модели, битый файл)."""


# --------------------------------------------------------------------------- #
#  Сборка текста
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"[.!?…]+$")


def normalize_text(text: str) -> str:
    """Чистит один сегмент: убирает лишние пробелы и пустые реплики."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    # Whisper иногда выдаёт одинокие знаки препинания или «музыкальные» метки.
    if cleaned in {".", ",", "-", "—", "♪", "*"}:
        return ""
    cleaned = cleaned.replace("♪", "").strip()
    return cleaned


def join_segments(segments: Iterable[str]) -> str:
    """Склеивает сегменты Whisper в сплошной текст.

    Whisper отдаёт куски фраз. Разделяем их пробелом, но начинаем новый абзац,
    если предыдущий фрагмент закончился точкой/вопросом или если новый
    фрагмент начинается с заглавной буквы (значит, это новое предложение).
    Запятая, двоеточие и тире — наоборот, признак того, что фраза продолжается.
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
    # Приводим кавычки-ёлочки и убираем двойные пробелы после склейки.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wrap_text(text: str, width: int = WRAP_WIDTH) -> str:
    """Переносит длинные строки по словам (абзацы сохраняются)."""
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
#  Имя и сохранение файла
# --------------------------------------------------------------------------- #
def transcript_filename(moment: Optional[datetime] = None) -> str:
    """Имя файла расшифровки: ``transcript_2026-11-09_18-30-00.txt``."""
    moment = moment or datetime.now()
    return f"transcript_{moment.strftime('%Y-%m-%d_%H-%M-%S')}.txt"


def unique_path(directory: Path, filename: str) -> Path:
    """Не перезаписывает существующие файлы: добавляет ``_2``, ``_3`` и т. д."""
    path = Path(directory) / filename
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = Path(directory) / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_transcript(text: str, directory: Path,
                    moment: Optional[datetime] = None,
                    header: Optional[str] = None) -> Path:
    """Сохраняет текст в файл .txt, возвращает итоговый путь."""
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
#  Движок
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptionProgress:
    """Текущее состояние распознавания (для индикатора в интерфейсе)."""

    stage: str = "idle"          # idle | loading | transcribing | saving | done | error
    percent: Optional[float] = None  # 0..100, если известно
    audio_seconds: float = 0.0
    processed_seconds: float = 0.0
    message: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def speed(self) -> Optional[float]:
        """Во сколько раз быстрее реального времени идёт обработка."""
        if self.processed_seconds <= 0 or self.elapsed <= 0:
            return None
        return self.processed_seconds / self.elapsed

    def eta_seconds(self) -> Optional[float]:
        if not self.percent or self.percent <= 0:
            return None
        remaining = max(0.0, self.audio_seconds - self.processed_seconds)
        speed = self.speed()
        if not speed:
            return None
        return remaining / speed


#: Кэш загруженных моделей: ключ → WhisperModel.
_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


def model_compute_type(model_size: str, device: str = "cpu") -> str:
    """Подбирает тип вычислений: int8 на CPU (быстро), float16 на GPU."""
    return "float16" if device == "cuda" else "int8"


def models_dir() -> Path:
    """Папка, куда скачиваются модели (внутри данных приложения)."""
    from .config import app_data_dir

    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(model_size: str, device: str = "auto",
               compute_type: Optional[str] = None,
               on_status: Optional[Callable[[str], None]] = None):
    """Загружает (или берёт из кэша) модель faster-whisper.

    Первая загрузка модели ``medium`` скачивает ~1,5 ГБ с Hugging Face,
    поэтому занимает время — об этом сообщаем через ``on_status``.
    """
    size = (model_size or "medium").strip()
    requested_device = (device or "auto").strip()
    key = (size, requested_device, compute_type or "")

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    try:
        from faster_whisper import WhisperModel  # ленивый импорт
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(
            "Не установлена библиотека faster-whisper. "
            "Установите её командой:  pip install faster-whisper\n"
            f"Подробности: {exc}"
        ) from exc

    attempts = []
    if requested_device == "auto":
        # Сразу выбираем «правильное» устройство, чтобы не тратить время на
        # неудачную попытку и не пугать пользователя лишними сообщениями.
        if _cuda_available():
            attempts = [("cuda", "float16"), ("cpu", "int8")]
        else:
            attempts = [("cpu", "int8")]
    elif requested_device == "cuda":
        attempts = [("cuda", compute_type or "float16")]
    else:
        attempts = [("cpu", compute_type or "int8")]

    last_error: Optional[Exception] = None
    for dev, ctype in attempts:
        if compute_type:
            ctype = compute_type
        if on_status is not None:
            on_status(
                f"Загрузка модели «{size}» ({dev}, {ctype}). "
                "Первый запуск может занять несколько минут — модель скачивается."
            )
        try:
            model = WhisperModel(
                size,
                device=dev,
                compute_type=ctype,
                download_root=str(models_dir()),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.warning("Не удалось загрузить модель на %s/%s: %s", dev, ctype, exc)
            continue
        with _MODEL_LOCK:
            _MODEL_CACHE[(size, requested_device, compute_type or "")] = model
        log.info("Модель %s загружена (%s, %s)", size, dev, ctype)
        return model

    raise TranscriptionError(
        "Не удалось загрузить модель распознавания "
        f"«{size}»: {last_error}\n\nПроверьте подключение к интернету "
        "(модель скачивается один раз) и свободное место на диске."
    )


def _cuda_available() -> bool:
    """Есть ли на компьютере видеокарта NVIDIA, пригодная для CTranslate2."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # noqa: BLE001 - нет библиотеки/нет драйвера
        return False


def free_models() -> None:
    """Выгружает модели из памяти (освобождает ~1,5 ГБ ОЗУ)."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
    gc.collect()


class Transcriber:
    """Распознаёт WAV-файл и сохраняет .txt.

    Класс используется из фонового потока интерфейса, поэтому все «живые»
    обновления передаются через колбэки (их вызывает рабочий поток, а
    отрисовку делает GUI-поток).
    """

    def __init__(self, config, on_progress: Optional[Callable[[TranscriptionProgress], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
        self.config = config
        self._on_progress = on_progress
        self._on_status = on_status
        self.progress = TranscriptionProgress()
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        """Просит остановить распознавание (проверяется между сегментами)."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ #
    def _emit(self, **changes) -> None:
        """Обновляет состояние и отдаёт подписчику «снимок» (не живой объект).

        Снимок важен: интерфейс получает его через очередь и отрисовывает
        позже, поэтому изменяемый объект «уехал» бы вперёд по прогрессу.
        """
        for key, value in changes.items():
            setattr(self.progress, key, value)
        if self._on_progress is not None:
            snapshot = replace(self.progress)
            try:
                self._on_progress(snapshot)
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
        """Распознаёт файл и возвращает текст (без сохранения)."""
        path = Path(wav_path)
        if not path.exists():
            raise TranscriptionError(f"Файл записи не найден: {path}")

        self._emit(stage="loading", percent=0.0,
                   message="Чтение записи…", started_at=time.time())
        try:
            audio, rate = read_wav_mono(path)
        except Exception as exc:  # noqa: BLE001
            # Не наш формат (например, m4a) — пробуем отдать файл Whisper:
            # он сам декодирует через FFmpeg/PyAV.
            log.info("Не удалось прочитать WAV (%s), пробуем как файл", exc)
            audio, rate = None, SAMPLE_RATE  # type: ignore[assignment]

        audio_seconds = (audio.size / float(rate)) if audio is not None and rate else 0.0
        self._emit(audio_seconds=audio_seconds)

        if audio is not None and audio_seconds < 0.3:
            raise TranscriptionError(
                "Запись слишком короткая (меньше секунды) — распознавать нечего."
            )

        model = load_model(
            self.config.model_size,
            on_status=self._status,
        )

        self._emit(stage="transcribing", percent=0.0,
                   message="Распознавание речи…")

        source: object = path if audio is None else audio
        try:
            segments, info = model.transcribe(
                source,  # type: ignore[arg-type]
                language=self.config.language or "ru",
                task="transcribe",
                beam_size=max(1, int(self.config.beam_size)),
                vad_filter=bool(self.config.vad_filter),
                vad_parameters={"min_silence_duration_ms": 500} if self.config.vad_filter else None,
                condition_on_previous_text=True,
                initial_prompt=DEFAULT_PROMPT,
                without_timestamps=True,
                word_timestamps=False,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Ошибка распознавания: {exc}") from exc

        total = float(getattr(info, "duration", 0.0) or audio_seconds or 0.0)
        self._emit(audio_seconds=total)

        pieces: List[str] = []
        processed = 0.0
        for segment in segments:
            if self._cancel.is_set():
                raise TranscriptionError("Распознавание отменено пользователем.")
            pieces.append(segment.text)
            processed = float(getattr(segment, "end", processed) or processed)
            percent = (processed / total * 100.0) if total > 0 else None
            self._emit(processed_seconds=processed, percent=percent)
            speed = self.progress.speed()
            speed_text = f", ×{speed:.1f} к реальному времени" if speed else ""
            self._emit(
                message=(
                    f"Распознано {format_hms(processed)} из {format_hms(total)}"
                    f"{speed_text}"
                )
            )

        text = join_segments(pieces)
        if not text.strip():
            raise TranscriptionError(
                "Речь не распознана: в записи не оказалось разборчивого голоса. "
                "Проверьте уровень микрофона и громкость системного звука."
            )
        self._emit(stage="done", percent=100.0, message="Готово.")
        return text

    # ------------------------------------------------------------------ #
    def transcribe_and_save(self, wav_path: str | Path,
                            output_dir: Optional[Path] = None,
                            header: Optional[str] = None) -> Path:
        """Распознаёт запись и сохраняет результат в .txt."""
        text = self.transcribe_file(wav_path)
        directory = Path(output_dir) if output_dir else self.config.resolved_output_dir()
        self._emit(stage="saving", percent=100.0, message="Сохранение файла…")
        path = save_transcript(wrap_text(text), directory, header=header)
        self._emit(stage="done", message=f"Сохранено: {path}")
        return path


def format_hms(seconds: float) -> str:
    """ЧЧ:ММ:СС без долей секунды."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def transcribe_file_simple(wav_path: str | Path, config,
                           output_dir: Optional[Path] = None) -> Path:
    """Удобная обёртка для скриптов и тестов (без интерфейса)."""
    return Transcriber(config).transcribe_and_save(wav_path, output_dir)


def warm_up(config, on_status: Optional[Callable[[str], None]] = None) -> None:
    """Заранее загружает модель (чтобы «Стоп» не ждал скачивания)."""
    load_model(config.model_size, on_status=on_status)
