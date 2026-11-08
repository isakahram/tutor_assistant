"""Пути, константы и журнал работы.

Настроек больше нет: программа делает ровно одну вещь — пишет системный звук
вместе с микрофоном и распознаёт запись моделью ``base`` настолько быстро,
насколько позволяет компьютер. Все параметры — константы в этом файле,
их можно поправить прямо в коде.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

APP_NAME = "TutorTranscript"
APP_TITLE = "Расшифровка занятий"
APP_SUBTITLE = "Запись звука занятия → текст"

#: Папка сохранения расшифровок — рабочий стол (Windows).
DESKTOP_WINDIR = Path(r"C:\Users\TREIDCOMPUTERS\Desktop")

#: Модель распознавания. ``base`` — требование скорости; менять не нужно.
MODEL_SIZE = "base"

#: Язык распознавания (передаётся в модель явно).
LANGUAGE = "ru"

#: Частота дискретизации WAV (её же ждёт Whisper — без пересчёта).
SAMPLE_RATE = 16_000

#: Сколько секунд звука модель обрабатывает за один проход (батч).
#: Чанки по 30 с — это максимальный контекст Whisper; 8 штук в батче дают
#: максимальную загрузку всех ядер процессора.
CHUNK_SECONDS = 30
BATCH_SIZE = 8

#: Сколько потоков отдавать CTranslate2 (0 — определить автоматически).
CPU_THREADS = 0

#: Сколько секунд будем ждать закрытия потока при остановке записи.
STOP_TIMEOUT = 5.0

#: Раз в сколько секунд обновлять состояние в журнале/интерфейсе.
UI_TICK_MS = 250

LOG_FILENAME = "app.log"


# --------------------------------------------------------------------------- #
#  Пути
# --------------------------------------------------------------------------- #
def app_data_dir() -> Path:
    """Папка для служебных файлов приложения (журнал, модели, временный WAV)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    """Куда сохранять расшифровки: рабочий стол, а если его нет — домой.

    На Windows это всегда ``C:\\Users\\TREIDCOMPUTERS\\Desktop`` (создаётся,
    если его почему-то нет). На других системах берётся «Рабочий стол»
    текущего пользователя, чтобы программу можно было запустить и проверить
    не только на вашем компьютере.
    """
    if sys.platform == "win32":
        desktop = DESKTOP_WINDIR
    else:
        home = Path.home()
        desktop = home / "Desktop"
        if not desktop.is_dir():
            desktop = home / "Рабочий стол"
        if not desktop.is_dir():
            desktop = home
    try:
        desktop.mkdir(parents=True, exist_ok=True)
        return desktop
    except OSError:
        fallback = Path.home()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def temp_dir() -> Path:
    """Папка для временного WAV (удаляется сразу после расшифровки)."""
    path = app_data_dir() / "temp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Папка, куда один раз скачивается модель распознавания."""
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return app_data_dir() / LOG_FILENAME


# --------------------------------------------------------------------------- #
#  Журнал
# --------------------------------------------------------------------------- #
def setup_logging(debug: bool = False) -> logging.Logger:
    """Пишет журнал в файл (``%APPDATA%\\TutorTranscript\\app.log``) и в консоль."""
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
