"""Локальная расшифровка звука онлайн-занятий в текст.

Пакет с логикой приложения:

* :mod:`tutor_transcript.config`   — настройки, пути, журнал;
* :mod:`tutor_transcript.devices`  — поиск аудиоустройств (loopback + микрофон);
* :mod:`tutor_transcript.audio_sink` — запись смеси каналов в WAV;
* :mod:`tutor_transcript.capture`  — потоки захвата звука и контроллер записи;
* :mod:`tutor_transcript.transcriber` — распознавание речи (faster-whisper);
* :mod:`tutor_transcript.gui`      — графический интерфейс (customtkinter).
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
