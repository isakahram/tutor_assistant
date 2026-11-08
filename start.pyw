#!/usr/bin/env python3
"""Автозапуск для Windows без консольного окна.

Двойной щелчок по этому файлу (``start.pyw``) запускает программу так же, как
``main.py``, но без чёрного окна консоли. Ошибки записываются в
``%APPDATA%\\TutorTranscript\\app.log`` и показываются в окне сообщения.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def _show_error(text: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Расшифровка занятий", text)
        root.destroy()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    try:
        import main as app_main

        return app_main.main([])
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        details = traceback.format_exc()
        try:
            from tutor_transcript.config import log_path

            path = log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n=== Ошибка запуска ===\n" + details)
            hint = f"\n\nПодробности записаны в файл:\n{path}"
        except Exception:  # noqa: BLE001
            hint = ""
        _show_error("Не удалось запустить программу.\n\n" + details[-2000:] + hint)
        return 1


if __name__ == "__main__":
    sys.exit(main())
