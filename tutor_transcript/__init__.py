"""Локальная расшифровка звука онлайн-занятий в текст.

Пакет с логикой приложения:

* :mod:`tutor_transcript.config`      — пути, константы, журнал;
* :mod:`tutor_transcript.devices`     — выбор микрофона и динамиков (loopback);
* :mod:`tutor_transcript.audio_sink`  — запись WAV;
* :mod:`tutor_transcript.capture`     — запись системного звука + микрофона;
* :mod:`tutor_transcript.transcriber` — быстрое распознавание (faster-whisper);
* :mod:`tutor_transcript.gui`         — окно с тремя кнопками.
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
