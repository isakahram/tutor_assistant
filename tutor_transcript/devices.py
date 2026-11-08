"""Выбор двух устройств записи: системный звук (loopback) и микрофон.

Пользователь всегда пишет оба источника, поэтому выбирать ничего не нужно:
берём динамики по умолчанию (их loopback-копию) и микрофон по умолчанию.
Модуль ничего не записывает и не держит интерфейс — только отвечает, что
именно открывать.
"""

from __future__ import annotations

import difflib
import logging
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

#: Ключи источников: системный звук (loopback) и микрофон.
LOOPBACK = "loopback"
MIC = "mic"

#: Слова-маркеры, по которым узнаём «микрофон», а не «динамики».
_MIC_KEYWORDS = (
    "microphone", "mic", "micro", "микрофон", "headset", "гарнитура",
    "webcam", "веб-камера", "line in", "line-in",
)

#: Слова, по которым узнаём виртуальные кабели (их в списке помечаем).
_VIRTUAL_KEYWORDS = ("cable", "vb-audio", "virtual", "voicemeeter", "кабель")


@dataclass(frozen=True)
class DeviceInfo:
    """Одно аудиоустройство, которое мы умеем открывать."""

    key: str            # LOOPBACK или MIC
    name: str           # имя устройства
    channels: int       # число каналов
    is_default: bool    # устройство по умолчанию в системе

    @property
    def is_loopback(self) -> bool:
        return self.key == LOOPBACK

    @property
    def kind_label(self) -> str:
        return "звук ПК" if self.is_loopback else "микрофон"

    @property
    def is_virtual_cable(self) -> bool:
        low = self.name.lower()
        return self.is_loopback and any(word in low for word in _VIRTUAL_KEYWORDS)

    def display(self) -> str:
        """Короткое описание для журнала: «звук ПК: Динамики (Realtek)»."""
        return f"{self.kind_label}: {self.name}"


# --------------------------------------------------------------------------- #
#  Звуковая карта (soundcard) — основной путь
# --------------------------------------------------------------------------- #
_REQUIRED_SOUNDCARD_API = (
    "all_microphones", "default_microphone", "default_speaker", "get_microphone",
)


def _soundcard_module():
    """Импортирует soundcard. Возвращает (модуль, текст ошибки).

    Тонкость: если импорт падает (например, нет системной библиотеки),
    Python оставляет в ``sys.modules`` полуинициализированный модуль. Тогда
    повторный ``import`` уже «успешен», но функций внутри нет. Поэтому
    убираем такой модуль из кеша и проверяем, что нужный API на месте.
    """
    try:
        import soundcard  # noqa: WPS433 - ленивый импорт оправдан
    except Exception as exc:  # noqa: BLE001 - OSError, ImportError, ...
        sys.modules.pop("soundcard", None)
        return None, f"{type(exc).__name__}: {exc}"

    missing = [name for name in _REQUIRED_SOUNDCARD_API
               if not hasattr(soundcard, name)]
    if missing:
        sys.modules.pop("soundcard", None)
        return None, "soundcard загрузился не полностью (нет: " + ", ".join(missing) + ")"
    return soundcard, None


def _default_names(sc) -> tuple:
    """Имена устройств по умолчанию (динамики, микрофон)."""
    speaker = mic = None
    try:
        spk = sc.default_speaker()
        speaker = getattr(spk, "name", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        mc = sc.default_microphone()
        mic = getattr(mc, "name", None)
    except Exception:  # noqa: BLE001
        pass
    return speaker, mic


def list_devices() -> List[DeviceInfo]:
    """Все доступные устройства: сначала loopback-источники, затем микрофоны.

    Никогда не выбрасывает исключение — при отсутствии soundcard вернёт
    пустой список (причина пишется в журнал).
    """
    sc, error = _soundcard_module()
    if sc is None:
        log.warning("soundcard недоступен: %s", error)
        return []

    devices: List[DeviceInfo] = []
    default_speaker, default_mic = _default_names(sc)

    # 1. Системный звук: «микрофонные» устройства, которые на самом деле
    #    являются loopback-копией динамиков.
    try:
        loopbacks = list(sc.all_microphones(include_loopback=True))
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось получить список loopback-устройств: %s", exc)
        loopbacks = []

    for mic in loopbacks:
        if not getattr(mic, "isloopback", False):
            continue
        devices.append(
            DeviceInfo(
                key=LOOPBACK,
                name=str(getattr(mic, "name", "Без имени")),
                channels=int(len(getattr(mic, "channels", []) or [])),
                is_default=str(getattr(mic, "name", "")) == str(default_speaker),
            )
        )

    # 2. Настоящие микрофоны.
    try:
        microphones = list(sc.all_microphones(include_loopback=False))
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось получить список микрофонов: %s", exc)
        microphones = []

    for mic in microphones:
        if getattr(mic, "isloopback", False):
            continue
        name = str(getattr(mic, "name", "Без имени"))
        devices.append(
            DeviceInfo(
                key=MIC,
                name=name,
                channels=int(len(getattr(mic, "channels", []) or [])),
                is_default=name == str(default_mic),
            )
        )

    log.info("Найдено устройств: %s", ", ".join(d.display() for d in devices) or "нет")
    return devices


def pick_default(devices: Iterable[DeviceInfo], key: str,
                 prefer: Optional[str] = None) -> Optional[DeviceInfo]:
    """Выбирает устройство: сначала сохранённое, затем системное по умолчанию.

    Для микрофона предпочитаем настоящий микрофон, а не гарнитуру-динамики и не
    виртуальный кабель; для loopback — устройство, совпадающее с динамиками
    по умолчанию.
    """
    candidates = [d for d in devices if d.key == key]
    if not candidates:
        return None

    if prefer:
        exact = [d for d in candidates if d.name == prefer]
        if exact:
            return exact[0]
        similar = difflib.get_close_matches(prefer, [d.name for d in candidates], n=1)
        if similar:
            return next(d for d in candidates if d.name == similar[0])

    # 1) помеченное «по умолчанию»
    preferred = [d for d in candidates if d.is_default]
    if preferred:
        # среди микрофонов — предпочтём не виртуальный кабель
        preferred.sort(key=lambda d: d.is_virtual_cable)
        return preferred[0]

    # 2) эвристика по имени
    if key == MIC:
        real = [d for d in candidates if any(w in d.name.lower() for w in _MIC_KEYWORDS)]
        if real:
            return real[0]
        non_virtual = [d for d in candidates if not d.is_virtual_cable]
        return (non_virtual or candidates)[0]

    return candidates[0]


def describe_runtime() -> str:
    """Одна строка о платформе и установленных библиотеках — для журнала."""
    parts = [f"Python {sys.version.split()[0]}", sys.platform]
    sc, error = _soundcard_module()
    parts.append("soundcard OK" if sc else f"soundcard НЕТ ({error})")
    try:
        import sounddevice  # noqa: F401

        parts.append("sounddevice OK")
    except Exception:  # noqa: BLE001
        parts.append("sounddevice не установлен")
    try:
        import faster_whisper  # noqa: F401

        parts.append("faster-whisper OK")
    except Exception:  # noqa: BLE001
        parts.append("faster-whisper НЕТ")
    return " · ".join(parts)


def choose_sources() -> Tuple[Optional[DeviceInfo], Optional[DeviceInfo]]:
    """Возвращает пару (системный звук, микрофон) — по устройству по умолчанию.

    ``None`` означает «такого устройства сейчас нет»: тогда запись пойдёт без
    него (второй источник продолжит писаться).
    """
    devices = list_devices()
    system = pick_default(devices, LOOPBACK)
    mic = pick_default(devices, MIC)
    return system, mic
