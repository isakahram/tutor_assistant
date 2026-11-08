"""Окно программы: три кнопки и строка состояния.

Задача интерфейса — не мешать. Поэтому здесь нет ни выбора устройств, ни
индикаторов уровня, ни настроек: программа всегда пишет системный звук и
микрофон, всегда распознаёт моделью ``base`` и всегда сохраняет текст на
рабочий стол.

Кнопки:

* **Старт** — начинает запись;
* **Пауза** — приостанавливает (повторное нажатие продолжает);
* **Расшифровка** — останавливает запись и сразу сохраняет текст в .txt.

Вся тяжёлая работа (чтение звука, распознавание) идёт в отдельных потоках,
окно только опрашивает очередь сообщений, поэтому оно не «зависает».
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import customtkinter as ctk

from . import __version__
from .capture import RecordingController, RecordingResult, format_hms
from .config import APP_TITLE, MODEL_SIZE, output_dir
from .devices import describe_runtime
from .transcriber import Progress, Transcriber, TranscriptionError, free_model

log = logging.getLogger(__name__)

#: Цвета интерфейса.
SUCCESS = "#2E9E5B"
SUCCESS_HOVER = "#247C47"
PAUSE_COLOR = "#4A4A55"
PAUSE_HOVER = "#3B3B44"
ACCENT = "#2F80ED"
DANGER = "#D9534F"
MUTED = ("#5A5A5A", "#A9B4C0")


class App(ctk.CTk):
    """Главное (и единственное) окно."""

    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title(f"{APP_TITLE} {__version__}")
        self.geometry("600x440")
        self.resizable(False, False)
        apply_window_icon(self)

        # --- состояние ---
        self.controller: Optional[RecordingController] = None
        self.recording_result: Optional[RecordingResult] = None
        self.transcriber: Optional[Transcriber] = None
        self.worker: Optional[threading.Thread] = None
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.last_transcript: Optional[Path] = None
        self._failed = False      # последняя расшифровка закончилась ошибкой
        self._busy = False        # идёт расшифровка/сохранение
        self._closing = False

        self._build_ui()
        self._set_state("idle")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._poll_events)
        self.after(250, self._tick)

        self._log(f"Текст сохраняется в: {output_dir()}")
        # проверка окружения и загрузка модели — в фоне, чтобы окно
        # появлялось мгновенно; к концу занятия модель уже в памяти
        threading.Thread(target=self._prepare, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Интерфейс
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        ctk.CTkLabel(
            header, text="🎓  " + APP_TITLE,
            font=ctk.CTkFont(size=20, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text=f"Модель {MODEL_SIZE} · системный звук + микрофон",
            font=ctk.CTkFont(size=12), text_color=MUTED, anchor="w",
        ).grid(row=1, column=0, sticky="w")

        self.timer_label = ctk.CTkLabel(
            self, text="00:00:00", font=ctk.CTkFont(size=56, weight="bold"),
        )
        self.timer_label.grid(row=1, column=0, pady=(18, 0))

        self.status_label = ctk.CTkLabel(
            self, text="Готово к записи", font=ctk.CTkFont(size=13),
            text_color=MUTED,
        )
        self.status_label.grid(row=2, column=0, pady=(0, 18))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=3, column=0, pady=(0, 18))
        for column in range(3):
            buttons.grid_columnconfigure(column, weight=1, uniform="btn")

        self.start_button = ctk.CTkButton(
            buttons, text="●  Старт", width=150, height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=SUCCESS, hover_color=SUCCESS_HOVER, command=self.on_start,
        )
        self.start_button.grid(row=0, column=0, padx=6)

        self.pause_button = ctk.CTkButton(
            buttons, text="⏸  Пауза", width=150, height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=PAUSE_COLOR, hover_color=PAUSE_HOVER, command=self.on_pause,
        )
        self.pause_button.grid(row=0, column=1, padx=6)

        self.transcribe_button = ctk.CTkButton(
            buttons, text="■  Расшифровка", width=170, height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=DANGER, hover_color="#B2403D", command=self.on_transcribe,
        )
        self.transcribe_button.grid(row=0, column=2, padx=6)

        self.open_button = ctk.CTkButton(
            self, text="Открыть последний текст", width=220, height=30,
            fg_color="transparent", border_width=1,
            state="disabled", command=self.open_last_transcript,
        )
        self.open_button.grid(row=4, column=0, pady=(0, 12))

        self.log_box = ctk.CTkTextbox(
            self, height=110, wrap="word", font=ctk.CTkFont(size=12),
        )
        self.log_box.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 16))
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------ #
    def _set_state(self, state: str) -> None:
        """idle / recording / paused / stopping / transcribing."""
        if state == "idle":
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled", text="⏸  Пауза")
            self.transcribe_button.configure(state="disabled")
        elif state == "recording":
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal", text="⏸  Пауза")
            self.transcribe_button.configure(state="normal")
        elif state == "paused":
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal", text="▶  Продолжить")
            self.transcribe_button.configure(state="normal")
        else:  # stopping / transcribing
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
            self.transcribe_button.configure(state="disabled")
        self._busy = state in ("stopping", "transcribing")

    def _set_status(self, text: str, color=MUTED) -> None:
        try:
            self.status_label.configure(text=text, text_color=color)
        except Exception:  # noqa: BLE001
            pass

    def _log(self, message: str) -> None:
        """Добавляет строку в журнал (можно вызывать из любого потока)."""
        self.events.put(("log", message))

    def _log_now(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:  # noqa: BLE001 - журнал не должен ломать окно
            pass
        log.info("%s", message)

    # ------------------------------------------------------------------ #
    #  Подготовка: сообщить о библиотеках и загрузить модель
    # ------------------------------------------------------------------ #
    def _prepare(self) -> None:
        try:
            self.events.put(("log", describe_runtime()))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", f"не удалось проверить окружение: {exc}"))
        try:
            from .transcriber import preload

            preload(lambda message: self.events.put(("log", message)))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", f"модель ещё не готова: {exc}"))

    # ------------------------------------------------------------------ #
    #  Кнопки
    # ------------------------------------------------------------------ #
    def on_start(self) -> None:
        if self._busy or (self.controller is not None and self.controller.is_recording):
            return

        controller = RecordingController(on_log=lambda message: self._log(message))
        try:
            wav_path = controller.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Не удалось начать запись")
            self._log(f"Не удалось начать запись: {exc}")
            messagebox.showerror(
                "Не удалось начать запись",
                f"{exc}\n\nЧто проверить:\n"
                "• микрофон не занят другой программой;\n"
                "• в системе выбраны нужные микрофон и динамики;\n"
                "• подробности — в журнале окна и в app.log.",
                parent=self,
            )
            self._set_state("idle")
            return

        self.controller = controller
        self.recording_result = None
        self._failed = False
        self._set_state("recording")
        self._set_status("● Идёт запись", SUCCESS)
        self._log(f"Пишу во временный файл: {wav_path}")
        threading.Thread(
            target=self._watch_quiet_start,
            args=(controller, controller.source_keys),
            daemon=True,
        ).start()

    def _watch_quiet_start(self, controller: RecordingController,
                           keys: tuple) -> None:
        """Через 10 секунд предупреждаем, если не слышно ни одного источника."""
        import time

        time.sleep(10.0)
        if not controller.is_recording:
            return
        if controller.levels.get("mix") < 0.002:
            labels = {"mic": "микрофон", "loopback": "звук компьютера"}
            silent = [labels.get(key, key) for key in keys
                      if controller.levels.get(key) < 0.002]
            self._log(
                "Похоже, сейчас не слышно: " + (", ".join(silent) or "звук")
                + ". Проверьте устройства и громкость."
            )

    def on_pause(self) -> None:
        controller = self.controller
        if controller is None or not controller.is_recording:
            return
        if controller.toggle_pause():
            self._set_state("paused")
            self._set_status("⏸ Пауза — звук не записывается", "#E0A100")
        else:
            self._set_state("recording")
            self._set_status("● Идёт запись", SUCCESS)

    def on_transcribe(self) -> None:
        """«Расшифровка»: останавливает запись и запускает распознавание."""
        controller = self.controller
        if controller is None or not controller.is_recording:
            return
        self._set_state("stopping")
        self._set_status("Останавливаю запись…")
        threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    def _stop_and_transcribe(self) -> None:
        controller = self.controller
        if controller is None:
            self.events.put(("idle", None))
            return
        try:
            result = controller.stop()
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка остановки записи")
            self.events.put(("log", f"Ошибка остановки записи: {exc}"))
            self.events.put(("idle", None))
            return

        self.recording_result = result
        for warning in result.warnings:
            self.events.put(("log", "Внимание: " + warning))

        if result.duration < 1.0:
            self.events.put(("log", "Запись почти пустая — расшифровывать нечего."))
            self._cleanup_audio()
            self.events.put(("idle", None))
            return

        self.events.put(("transcribing", result.duration))
        self._run_transcription(result)

    def _run_transcription(self, result: RecordingResult) -> None:
        header = (
            f"Расшифровка занятия — {result.stopped_at.strftime('%d.%m.%Y %H:%M')}\n"
            f"Длительность: {format_hms(result.duration)}"
        )
        target_dir = output_dir()
        transcriber = Transcriber(
            on_progress=lambda progress: self.events.put(("progress", progress)),
            on_status=lambda message: self.events.put(("status", message)),
        )
        self.transcriber = transcriber

        def worker() -> None:
            saved = False
            try:
                path = transcriber.transcribe_and_save(
                    result.wav_path, target_dir, header=header
                )
                saved = True
                self.events.put(("transcript", path))
            except TranscriptionError as exc:
                self.events.put(("error", str(exc)))
            except Exception as exc:  # noqa: BLE001
                log.error("Ошибка расшифровки:\n%s", traceback.format_exc())
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                # при ошибке запись не удаляем: её можно расшифровать позже,
                # когда причина будет устранена
                if saved:
                    self._cleanup_audio()
                else:
                    self.events.put((
                        "log",
                        f"Запись сохранена: {result.wav_path}",
                    ))
                self.events.put(("idle", None))

        self.worker = threading.Thread(target=worker, name="transcribe", daemon=True)
        self.worker.start()

    def _cleanup_audio(self) -> None:
        """Удаляет временный WAV: он нужен только до расшифровки."""
        result = self.recording_result
        if result is None:
            return
        try:
            if result.wav_path.exists():
                result.wav_path.unlink()
        except Exception as exc:  # noqa: BLE001
            log.debug("Не удалось удалить временный файл: %s", exc)

    def open_last_transcript(self) -> None:
        if self.last_transcript and self.last_transcript.exists():
            open_in_explorer(self.last_transcript)

    # ------------------------------------------------------------------ #
    #  Циклы обновления
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self._closing:
            return
        controller = self.controller
        try:
            if controller is not None and controller.is_recording:
                self.timer_label.configure(text=format_hms(controller.elapsed))
        except Exception:  # noqa: BLE001
            pass
        self.after(250, self._tick)

    def _poll_events(self) -> None:
        if self._closing:
            return
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_event(kind, payload)
            except Exception:  # noqa: BLE001 - одно событие не должно рушить окно
                log.error("Ошибка обработки события «%s»:\n%s", kind,
                          traceback.format_exc())
        self.after(150, self._poll_events)

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "log":
            self._log_now(str(payload))
        elif kind == "transcribing":
            self._set_state("transcribing")
            self._set_status("Распознаю запись…", ACCENT)
        elif kind == "status":
            self._set_status(str(payload), ACCENT)
        elif kind == "progress":
            progress: Progress = payload
            self._set_status(progress.message or "Распознаю…", ACCENT)
        elif kind == "transcript":
            path = Path(payload)
            self._failed = False
            self.last_transcript = path
            self.open_button.configure(state="normal")
            self._set_status("Готово — текст сохранён", SUCCESS)
            self.timer_label.configure(text="00:00:00")
            self._log_now(f"Расшифровка сохранена: {path}")
            try:
                self.bell()
            except Exception:  # noqa: BLE001
                pass
        elif kind == "error":
            self._failed = True
            self._set_status("Не удалось расшифровать", DANGER)
            self._log_now(f"Ошибка: {payload}")
            wav_hint = ""
            if self.recording_result is not None:
                wav_hint = (
                    "\n\nАудиозапись сохранена, её можно расшифровать позже:\n"
                    f"{self.recording_result.wav_path}"
                )
            messagebox.showerror("Расшифровка не удалась",
                                 f"{payload}{wav_hint}", parent=self)
        elif kind == "idle":
            self._set_state("idle")
            if not self._failed and self.last_transcript is None:
                self._set_status("Готово к записи")
            self.timer_label.configure(text="00:00:00")

    # ------------------------------------------------------------------ #
    def _on_close(self) -> None:
        controller = self.controller
        if controller is not None and controller.is_recording:
            if not messagebox.askyesno(
                "Идёт запись",
                "Запись ещё идёт. Остановить её и закрыть программу?\n"
                "Текст не будет сохранён.",
                parent=self,
            ):
                return
            try:
                controller.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._busy and not messagebox.askyesno(
            "Идёт расшифровка",
            "Расшифровка ещё не закончена. Закрыть программу?",
            parent=self,
        ):
            return
        if self.transcriber is not None:
            self.transcriber.cancel()
        self._closing = True
        free_model()
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
#  Вспомогательные функции
# --------------------------------------------------------------------------- #
def apply_window_icon(window) -> None:
    """Ставит иконку окна из папки assets (если файл на месте)."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    ico, png = assets / "icon.ico", assets / "icon.png"
    try:
        if sys.platform == "win32" and ico.exists():
            window.iconbitmap(str(ico))
            return
        if png.exists():
            from PIL import Image, ImageTk

            with Image.open(png) as opened:
                image = ImageTk.PhotoImage(opened)
            window.iconphoto(True, image)
            window._tutor_icon = image  # держим ссылку, иначе иконка исчезнет
    except Exception as exc:  # noqa: BLE001 - иконка не критична
        log.debug("Не удалось поставить иконку окна: %s", exc)


def open_in_explorer(path: Path) -> None:
    """Открывает папку и выделяет файл в проводнике."""
    try:
        if sys.platform == "win32":
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось открыть %s: %s", path, exc)


def run() -> None:
    """Запускает окно (используется из main.py и тестов)."""
    App().mainloop()
