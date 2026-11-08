"""Запуск приложения: ``python main.py``.

Скрипт делает несколько вещей, важных для пользователя без опыта:

* проверяет, что запущен Python 3.9+ и установлены нужные библиотеки,
  и выдаёт понятное сообщение вместо «красной простыни» ошибок;
* пишет журнал рядом с приложением (см. ``app.log``);
* при аргументе ``--self-test`` проверяет микрофон и системный звук на 5 секунд
  и сохраняет пробную запись (удобно, чтобы убедиться, что всё работает).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

MIN_PYTHON = (3, 9)

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "soundcard": "SoundCard",
    "sounddevice": "sounddevice",
    "customtkinter": "customtkinter",
    "faster_whisper": "faster-whisper",
}

INSTALL_HINT = (
    "Установите зависимости командой (в папке программы):\n\n"
    "    pip install -r requirements.txt\n\n"
    "Если pip не найден, попробуйте:  python -m pip install -r requirements.txt"
)


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        current = ".".join(map(str, sys.version_info[:3]))
        sys.exit(
            f"Нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} или новее, "
            f"а запущен {current}.\nСкачайте актуальную версию с python.org."
        )


def check_packages() -> list:
    """Возвращает список отсутствующих пакетов (не прерывая запуск).

    Используется ``find_spec``: он только ищет модуль, но не импортирует его.
    Это важно для скорости — импорт faster-whisper занимает несколько секунд.
    """
    import importlib.util

    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001 - битый пакет/нет DLL
            found = False
        if not found:
            missing.append(package)
    return missing


def _tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401  (проверяем наличие модуля)

        return True
    except Exception:  # noqa: BLE001
        return False


def pause_and_exit(message: str, code: int = 1) -> None:
    """Печатает сообщение и ждёт Enter, чтобы окно консоли не закрылось."""
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)
    try:
        input("\nНажмите Enter, чтобы закрыть окно…")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(code)


# --------------------------------------------------------------------------- #
#  Самопроверка (--self-test)
# --------------------------------------------------------------------------- #
def run_self_test(seconds: int = 5) -> int:
    """Пишет 5 секунд звука с микрофона и системного звука, сохраняет WAV."""
    from tutor_transcript.config import Config, app_data_dir
    from tutor_transcript.capture import RecordingController, format_hms
    from tutor_transcript.devices import list_devices

    print("Ищу аудиоустройства…")
    devices = list_devices()
    if not devices:
        print("Устройства не найдены. Проверьте, что микрофон подключён.")
        return 2
    for device in devices:
        print(f"  • {device.display()}")

    print(f"\nЗаписываю {seconds} секунд. Говорите в микрофон и включите "
          "какой-нибудь звук на компьютере…")
    config = Config()
    controller = RecordingController(config, on_log=lambda m: print("  " + m))
    controller.start()
    import time

    time.sleep(seconds)
    result = controller.stop()

    print(f"\nГотово. Длительность: {format_hms(result.duration)}")
    for warning in result.warnings:
        print(f"  ⚠  {warning}")
    print(f"Файл: {result.wav_path}")

    try:
        import numpy as np

        from tutor_transcript.audio_sink import read_wav_mono

        data, rate = read_wav_mono(result.wav_path)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        print(f"Пиковый уровень: {peak:.3f} "
              f"({'звук есть' if peak > 0.01 else 'почти тишина!'})")

        target = Path(app_data_dir()) / "self_test.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(result.wav_path, target)
        print(f"Копия записи: {target}")
        print("Если в записи слышны и ваш голос, и звук компьютера — всё работает.")
        return 0 if peak > 0.01 else 3
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось проверить запись: {exc}")
        return 4


# --------------------------------------------------------------------------- #
def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Расшифровка звука онлайн-занятий в текст (локально)."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="проверить запись звука (5 секунд) и выйти")
    parser.add_argument("--seconds", type=int, default=5,
                        help="длительность самопроверки, сек (по умолчанию 5)")
    parser.add_argument("--debug", action="store_true", help="подробный журнал")
    parser.add_argument("--model", default=None,
                        help="размер модели: tiny/base/small/medium/large-v3")
    parser.add_argument("--console", action="store_true",
                        help="без интерфейса: запись до Enter и расшифровка")
    args = parser.parse_args(argv)

    check_python_version()

    missing = check_packages()
    if missing or not _tkinter_available():
        # Без GUI программа работать не может, но объясним это по-человечески.
        if not _tkinter_available():
            pause_and_exit(
                "В вашей установке Python нет модуля tkinter.\n\n"
                "На Windows он ставится вместе с Python: запустите установщик "
                "с python.org и включите галочку «tcl/tk and IDLE».\n"
                "После установки запустите программу снова."
            )
        pause_and_exit(
            "Не хватает библиотек: " + ", ".join(missing) + "\n\n" + INSTALL_HINT
        )

    from tutor_transcript.config import load_config, setup_logging

    setup_logging(debug=args.debug)
    config = load_config()
    if args.model:
        config.model_size = args.model

    if args.self_test:
        return run_self_test(max(2, args.seconds))

    if args.console:
        return run_console(config)

    from tutor_transcript.gui import TutorTranscriptApp

    app = TutorTranscriptApp(config)
    app.mainloop()
    return 0


def run_console(config) -> int:
    """Режим без интерфейса: запись до Enter, затем расшифровка."""
    from tutor_transcript.capture import RecordingController, format_hms
    from tutor_transcript.transcriber import Transcriber, save_transcript, wrap_text

    controller = RecordingController(config, on_log=lambda m: print("• " + m))
    print("Нажмите Enter, чтобы начать запись…")
    input()
    controller.start()
    print("Идёт запись. Нажмите Enter, чтобы остановить и расшифровать…")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    result = controller.stop()
    print(f"Записано {format_hms(result.duration)}")
    for warning in result.warnings:
        print("⚠ " + warning)
    if result.duration < 1.0:
        print("Запись слишком короткая.")
        return 2

    print("Распознаю (первый запуск может занять время — скачивается модель)…")
    transcriber = Transcriber(config, on_status=lambda m: print("• " + m))
    text = transcriber.transcribe_file(result.wav_path)
    path = save_transcript(wrap_text(text), config.resolved_output_dir())
    print(f"\nГотово: {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 - последняя линия обороны для новичка
        traceback_text = traceback.format_exc()
        print(traceback_text)
        pause_and_exit(
            "Программа неожиданно завершилась. Скопируйте текст выше — "
            "по нему можно понять причину.",
        )
