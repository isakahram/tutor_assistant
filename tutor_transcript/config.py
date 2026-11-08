"""Настройки приложения, служебные пути и журнал работы.

Все пользовательские настройки хранятся в файле ``settings.json`` рядом
с данными приложения:

* Windows: ``%APPDATA%\\TutorTranscript\\settings.json``
* Linux/macOS: ``~/.config/TutorTranscript/settings.json``

Файл можно править вручную — при следующем запуске настройки подхватятся.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

APP_NAME = "TutorTranscript"
APP_TITLE = "Расшифровка занятий"
APP_SUBTITLE = "Запись системного звука и микрофона → текст (локально, без облака)"

#: Доступные размеры модели faster-whisper.
MODEL_CHOICES = ("tiny", "base", "small", "medium", "large-v3")

#: Режимы записи: ключ → подпись в интерфейсе.
CAPTURE_MODES: Dict[str, str] = {
    "both": "Оба источника",
    "system": "Только звук компьютера",
    "mic": "Только микрофон",
}

#: Частота дискретизации, в которой мы пишем WAV (её же ждёт Whisper).
SAMPLE_RATE = 16_000

SETTINGS_FILENAME = "settings.json"
LOG_FILENAME = "app.log"


# --------------------------------------------------------------------------- #
#  Пути
# --------------------------------------------------------------------------- #
def app_data_dir() -> Path:
    """Папка для служебных файлов приложения (создаётся при необходимости)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    """Папка «Документы» текущего пользователя — сюда сохраняем расшифровки."""
    candidates = []

    if sys.platform == "win32":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / "Documents")
    # Домашняя папка / Documents на остальных системах
    candidates.append(Path.home() / "Documents")
    candidates.append(Path.home() / "Документы")
    candidates.append(Path.home())

    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:  # pragma: no cover - защита от необычных ФС
            continue
    return Path.home()


def settings_path() -> Path:
    return app_data_dir() / SETTINGS_FILENAME


def log_path() -> Path:
    return app_data_dir() / LOG_FILENAME


# --------------------------------------------------------------------------- #
#  Настройки
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Пользовательские настройки (сериализуются в settings.json)."""

    # --- распознавание ---
    model_size: str = "medium"       # medium по умолчанию, small — если медленно
    language: str = "ru"             # язык распознавания
    compute_type: str = ""           # "" = подобрать автоматически (int8 для CPU)
    beam_size: int = 5               # качество/скорость: 1 — быстрее, 5 — точнее
    vad_filter: bool = True          # вырезать паузы перед распознаванием

    # --- запись ---
    capture_mode: str = "both"       # both | system | mic
    system_device: str = ""          # имя loopback-устройства ("" = по умолчанию)
    mic_device: str = ""             # имя микрофона ("" = по умолчанию)
    allow_sounddevice_fallback: bool = True  # запасной путь для системного звука

    # --- файлы ---
    output_dir: str = ""             # "" = папка «Документы»
    keep_audio: bool = False         # сохранять рядом с расшифровкой и WAV

    # --- интерфейс ---
    appearance: str = "dark"         # dark | light | system

    # ------------------------------------------------------------------ #
    def resolved_output_dir(self) -> Path:
        """Папка для сохранения расшифровок (существующая, если возможно)."""
        if self.output_dir:
            path = Path(self.output_dir).expanduser()
            if path.is_dir():
                return path
        return default_output_dir()

    def normalized(self) -> "Config":
        """Приводит значения к допустимым (защита от ручной правки json)."""
        if self.model_size not in MODEL_CHOICES:
            self.model_size = "medium"
        if self.capture_mode not in CAPTURE_MODES:
            self.capture_mode = "both"
        if self.appearance not in ("dark", "light", "system"):
            self.appearance = "dark"
        try:
            self.beam_size = int(self.beam_size)
        except (TypeError, ValueError):
            self.beam_size = 5
        self.beam_size = max(1, min(10, self.beam_size))
        if not isinstance(self.language, str) or not self.language.strip():
            self.language = "ru"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**clean).normalized()


def load_config(path: Optional[Path] = None) -> Config:
    """Читает настройки; при ошибке возвращает значения по умолчанию."""
    path = Path(path) if path is not None else settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("в settings.json ожидался объект JSON")
        return Config.from_dict(data)
    except FileNotFoundError:
        return Config().normalized()
    except Exception as exc:  # повреждённый json — не падаем
        logging.getLogger(__name__).warning("Не удалось прочитать настройки: %s", exc)
        return Config().normalized()


def save_config(cfg: Config, path: Optional[Path] = None) -> None:
    """Сохраняет настройки (ошибки записи не должны ронять приложение)."""
    path = Path(path) if path is not None else settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cfg.normalized().to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("Не удалось сохранить настройки: %s", exc)


# --------------------------------------------------------------------------- #
#  Журнал
# --------------------------------------------------------------------------- #
def setup_logging(debug: bool = False) -> logging.Logger:
    """Настраивает запись в файл (%APPDATA%\\TutorTranscript\\app.log) и консоль."""
    root = logging.getLogger()
    if getattr(root, "_tutor_configured", False):
        return root

    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    try:
        handler = logging.handlers.RotatingFileHandler(
            log_path(), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)
    except Exception:  # noqa: BLE001 - журнал не критичен
        pass

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    root._tutor_configured = True  # type: ignore[attr-defined]
    return root
