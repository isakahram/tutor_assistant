"""Проверка окружения: ``python diagnostics.py``.

Скрипт печатает всё, что нужно для поиска проблем:

* версию Python и операционной системы;
* список установленных библиотек;
* все аудиоустройства, которые видит soundcard (включая loopback);
* устройства, которые видит PortAudio/sounddevice (включая WASAPI loopback);
* доступные модели faster-whisper и папку с ними;
* есть ли FFmpeg.

Если что-то не работает — пришлите вывод этого скрипта.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

LINE = "─" * 70


def header(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def main() -> int:
    header("Система")
    print(f"Python:      {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"ОС:          {platform.platform()}")
    print(f"Исполняемый: {sys.executable}")

    header("Библиотеки")
    packages = [
        ("numpy", "numpy"),
        ("soundcard", "SoundCard"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("customtkinter", "customtkinter"),
        ("faster_whisper", "faster-whisper"),
        ("ctranslate2", "ctranslate2"),
        ("av", "av (PyAV)"),
        ("PIL", "Pillow (для customtkinter)"),
    ]
    for module, label in packages:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            print(f"  ✔ {label:28} {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✖ {label:28} НЕ УСТАНОВЛЕН ({type(exc).__name__})")

    header("FFmpeg")
    ffmpeg = shutil.which("ffmpeg")
    print(f"  {'✔ ' + ffmpeg if ffmpeg else '✖ не найден в PATH'}")
    print("  (нужен для декодирования mp3/m4a при ручной расшифровке файлов;")
    print("   звук, записанный программой, расшифровывается без FFmpeg)")

    header("Устройства soundcard (MediaFoundation / WASAPI)")
    try:
        import soundcard as sc

        try:
            speaker = sc.default_speaker()
            print(f"  динамики по умолчанию: {speaker.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  динамики по умолчанию: не определены ({exc})")
        try:
            mic = sc.default_microphone()
            print(f"  микрофон по умолчанию: {mic.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  микрофон по умолчанию: не определён ({exc})")

        print("\n  Все устройства (loopback помечены):")
        microphones = list(sc.all_microphones(include_loopback=True))
        if not microphones:
            print("    — список пуст —")
        for index, device in enumerate(microphones):
            mark = "LOOPBACK" if getattr(device, "isloopback", False) else "микрофон"
            channels = len(getattr(device, "channels", []) or [])
            print(f"    {index:2}. [{mark:8}] {device.name} ({channels} кан.)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ОШИБКА soundcard: {type(exc).__name__}: {exc}")
        print("  Проверьте установку: pip install soundcard")

    header("Устройства PortAudio (sounddevice) — резервный путь")
    try:
        import sounddevice as sd

        print(f"  Версия PortAudio: {sd.get_portaudio_version()[1]}")
        wasapi_loopbacks = 0
        for index, info in enumerate(sd.query_devices()):
            api = sd.query_hostapis(info["hostapi"])["name"]
            if "wasapi" not in str(api).lower():
                continue
            mark = ""
            if info["max_input_channels"] > 0:
                mark = "  ← можно писать"
                if "loopback" in str(info["name"]).lower():
                    mark += " (LOOPBACK)"
                    wasapi_loopbacks += 1
            print(f"    {index:3}. {info['name']} | in={info['max_input_channels']} "
                  f"out={info['max_output_channels']}{mark}")
        if wasapi_loopbacks == 0:
            print("\n  Loopback-устройств не найдено: резервный захват системного "
                  "звука недоступен.\n  Основной путь (soundcard) при этом работает.")
    except Exception as exc:  # noqa: BLE001
        print(f"  ОШИБКА sounddevice: {type(exc).__name__}: {exc}")

    header("Модели faster-whisper")
    try:
        from tutor_transcript.config import app_data_dir

        models_dir = Path(app_data_dir()) / "models"
        print(f"  Папка моделей: {models_dir}")
        if models_dir.exists():
            found = [p.name for p in models_dir.rglob("model.bin")]
            print(f"  Скачанные модели: {found or 'пока нет'}")
        else:
            print("  Модели ещё не скачивались (это произойдёт при первом запуске).")
    except Exception as exc:  # noqa: BLE001
        print(f"  Не удалось проверить модели: {exc}")

    try:
        from tutor_transcript.config import log_path

        print(f"\nЖурнал приложения: {log_path()}")
    except Exception:  # noqa: BLE001
        pass

    print(f"\n{LINE}\nГотово. Если запись не работает — пришлите этот вывод.\n{LINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
