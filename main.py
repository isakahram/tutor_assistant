"""Запуск приложения: ``python main.py``.

Что делает скрипт:

* проверяет версию Python и наличие библиотек — и объясняет по-русски, чего
  не хватает, вместо «красной простыни» трассировки;
* при ``--self-test`` записывает 10 секунд звука и расшифровывает их, чтобы
  сразу увидеть, работает ли захват (никакой лишней диагностики не нужно);
* ведёт журнал ``%APPDATA%\\TutorTranscript\\app.log``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

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
    "Если pip не найден:  python -m pip install -r requirements.txt"
)


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        current = ".".join(map(str, sys.version_info[:3]))
        sys.exit(
            f"Нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} или новее, "
            f"а запущен {current}.\nСкачайте актуальную версию с python.org."
        )


def check_packages() -> list:
    """Возвращает список отсутствующих пакетов, не импортируя их (это быстрее)."""
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
#  Самопроверка: запись 10 секунд и расшифровка
# --------------------------------------------------------------------------- #
def run_self_test(seconds: int = 10) -> int:
    """Пишет несколько секунд звука, расшифровывает и печатает результат."""
    from tutor_transcript.capture import RecordingController, format_hms
    from tutor_transcript.config import output_dir
    from tutor_transcript.devices import describe_runtime
    from tutor_transcript.transcriber import Transcriber

    print(describe_runtime())
    print(f"\nЗаписываю {seconds} секунд. Говорите в микрофон, а на компьютере "
          "пусть играет любой звук…")
    controller = RecordingController(on_log=lambda m: print("  " + m))
    started = time.time()
    controller.start()
    time.sleep(seconds)
    result = controller.stop()
    print(f"\nЗаписано {format_hms(result.duration)} за "
          f"{time.time() - started:.1f} с реального времени.")
    for warning in result.warnings:
        print("  ⚠  " + warning)

    print("\nРаспознаю…")
    transcriber = Transcriber(on_status=lambda m: print("  " + m))
    try:
        text = transcriber.transcribe_file(result.wav_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось распознать: {exc}")
        print(f"Запись осталась здесь: {result.wav_path}")
        return 3
    finally:
        pass

    elapsed = result.duration / max(1e-6, transcriber.progress.elapsed)
    print(f"\nСкорость: ×{elapsed:.0f} к реальному времени "
          f"(1 час звука ≈ {60 / max(elapsed, 0.01):.1f} мин).")
    print(f"\nТекст:\n{'-' * 70}\n{text}\n{'-' * 70}")
    print(f"\nФайлы расшифровок сохраняются в: {output_dir()}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Запись звука занятия и быстрая расшифровка в текст."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="записать и расшифровать 10 секунд звука и выйти")
    parser.add_argument("--seconds", type=int, default=10,
                        help="длительность самопроверки, сек (по умолчанию 10)")
    parser.add_argument("--debug", action="store_true", help="подробный журнал")
    args = parser.parse_args(argv)

    check_python_version()

    if not _tkinter_available():
        pause_and_exit(
            "В вашей установке Python нет модуля tkinter.\n\n"
            "На Windows он ставится вместе с Python: запустите установщик "
            "с python.org и включите галочку «tcl/tk and IDLE».\n"
            "После установки запустите программу снова."
        )

    missing = check_packages()
    if missing:
        pause_and_exit(
            "Не хватает библиотек: " + ", ".join(missing) + "\n\n" + INSTALL_HINT
        )

    from tutor_transcript.config import setup_logging

    setup_logging(debug=args.debug)

    if args.self_test:
        return run_self_test(max(3, args.seconds))

    from tutor_transcript.gui import App

    App().mainloop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 - последняя линия обороны для новичка
        print(traceback.format_exc())
        pause_and_exit(
            "Программа неожиданно завершилась. Скопируйте текст выше — "
            "по нему можно понять причину."
        )
