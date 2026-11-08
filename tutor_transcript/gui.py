"""Графический интерфейс приложения (customtkinter).

Окно состоит из трёх блоков:

1. **Источники звука** — какие устройства пишем, режим записи и индикаторы
   уровня (чтобы сразу видеть, «слышит» ли программа голос и звук компьютера).
2. **Запись** — большая кнопка времени и три кнопки: «Старт», «Пауза», «Стоп».
3. **Расшифровка** — модель, папка сохранения и журнал работы.

Интерфейс не выполняет тяжёлую работу сам: запись живёт в своих потоках,
распознавание — в отдельном рабочем потоке, а обмен сообщениями идёт через
очередь, которую окно опрашивает каждые 150 мс. Поэтому окно не «зависает».
"""

from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk

from . import __version__
from .capture import Levels, RecordingController, RecordingResult, format_hms
from .config import (
    APP_SUBTITLE,
    APP_TITLE,
    CAPTURE_MODES,
    MODEL_CHOICES,
    Config,
    load_config,
    save_config,
)
from .devices import DeviceInfo, LOOPBACK, MIC, cached_devices, describe_runtime, pick_default
from .transcriber import Transcriber, TranscriptionError, TranscriptionProgress, free_models

log = logging.getLogger(__name__)

#: Цветовая схема.
ACCENT = "#2F80ED"
ACCENT_HOVER = "#1F6BD0"
SUCCESS = "#2E9E5B"
WARNING = "#E0A100"
DANGER = "#D9534F"
MUTED = ("#5A5A5A", "#A9B4C0")

#: Заголовок файла расшифровки.
HEADER_TEMPLATE = "Расшифровка занятия — {date}"


class TutorTranscriptApp(ctk.CTk):
    """Главное окно приложения."""

    # ------------------------------------------------------------------ #
    #  Создание окна
    # ------------------------------------------------------------------ #
    def __init__(self, config: Optional[Config] = None,
                 autoload_devices: bool = True) -> None:
        super().__init__()

        self.config_data: Config = config or load_config()
        ctk.set_appearance_mode(self.config_data.appearance)
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_TITLE} {__version__}")
        self.geometry("980x830")
        self.minsize(880, 720)
        apply_window_icon(self)

        # --- состояние ---
        self.devices: List[DeviceInfo] = []
        self._device_by_display: Dict[str, DeviceInfo] = {}
        self.controller: Optional[RecordingController] = None
        self.levels = Levels()
        self.recording_result: Optional[RecordingResult] = None
        self.transcriber: Optional[Transcriber] = None
        self.worker: Optional[threading.Thread] = None
        self._external_copy: Optional[Path] = None
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.last_transcript: Optional[Path] = None
        self._busy = False           # идёт распознавание/сохранение
        self._stopping = False
        self._closing = False
        self._recording_started_at: Optional[datetime] = None

        self._build_ui()
        self._apply_state("idle")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._poll_events)
        self.after(300, self._tick)

        if autoload_devices:
            self.after(200, lambda: self.refresh_devices(silent=True))
        self._log(f"Версия {__version__}")
        # проверка библиотек импортирует тяжёлые пакеты — делаем это в фоне,
        # чтобы окно появлялось мгновенно
        threading.Thread(target=self._log_runtime, daemon=True).start()

    def _log_runtime(self) -> None:
        try:
            description = describe_runtime()
        except Exception as exc:  # noqa: BLE001
            description = f"не удалось проверить окружение: {exc}"
        self.events.put(("log", (description, "info")))

    # ------------------------------------------------------------------ #
    #  Построение интерфейса
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        # Тянуть вниз должен только журнал: иначе таймер записи «съедается».
        self.grid_rowconfigure(3, weight=1)

        # ---------------- шапка ----------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="🎓  " + APP_TITLE,
            font=ctk.CTkFont(size=22, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text=APP_SUBTITLE, font=ctk.CTkFont(size=12),
            text_color=MUTED, anchor="w",
        ).grid(row=1, column=0, sticky="w")

        theme_box = ctk.CTkFrame(header, fg_color="transparent")
        theme_box.grid(row=0, column=1, rowspan=2, sticky="e")
        ctk.CTkLabel(theme_box, text="Тема:", font=ctk.CTkFont(size=11),
                     text_color=MUTED).grid(row=0, column=0, padx=(0, 6))
        self.theme_menu = ctk.CTkOptionMenu(
            theme_box, values=["Тёмная", "Светлая", "Системная"],
            width=110, command=self._on_theme_change,
        )
        self.theme_menu.set({"dark": "Тёмная", "light": "Светлая",
                             "system": "Системная"}.get(self.config_data.appearance, "Тёмная"))
        self.theme_menu.grid(row=0, column=1)

        # ---------------- источники звука ----------------
        self.sources_card = self._make_card(self, "1. Источники звука")
        self.sources_card.grid(row=1, column=0, sticky="ew", padx=18, pady=6)
        self.sources_card.grid_columnconfigure(1, weight=1)
        self.sources_card.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(self.sources_card, text="Звук компьютера",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, sticky="w", padx=(14, 8), pady=(0, 4))
        self.system_device_combo = ctk.CTkComboBox(
            self.sources_card, values=["—  нет данных —"],
            command=lambda _v: self._on_device_change(LOOPBACK),
            width=310, height=32,
        )
        self.system_device_combo.grid(row=1, column=1, columnspan=3, sticky="ew",
                                      padx=(0, 14), pady=(0, 4))

        ctk.CTkLabel(self.sources_card, text="Микрофон",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=(14, 8), pady=(0, 4))
        self.mic_device_combo = ctk.CTkComboBox(
            self.sources_card, values=["—  нет данных —"],
            command=lambda _v: self._on_device_change(MIC),
            width=310, height=32,
        )
        self.mic_device_combo.grid(row=2, column=1, columnspan=3, sticky="ew",
                                   padx=(0, 14), pady=(0, 4))

        # режим и обновление
        mode_row = ctk.CTkFrame(self.sources_card, fg_color="transparent")
        mode_row.grid(row=3, column=0, columnspan=4, sticky="ew", padx=14, pady=(6, 2))
        mode_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(mode_row, text="Записывать:",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 8))
        mode_values = list(CAPTURE_MODES.values())
        self.mode_switch = ctk.CTkSegmentedButton(
            mode_row, values=mode_values, command=self._on_mode_change,
        )
        self.mode_switch.set(CAPTURE_MODES.get(self.config_data.capture_mode, mode_values[0]))
        self.mode_switch.grid(row=0, column=1, sticky="w")

        self.refresh_button = ctk.CTkButton(
            mode_row, text="⟳  Обновить устройства", width=180, height=30,
            fg_color="transparent", border_width=1, command=self.refresh_devices,
        )
        self.refresh_button.grid(row=0, column=2, sticky="e")

        # индикаторы уровня
        levels_row = ctk.CTkFrame(self.sources_card, fg_color="transparent")
        levels_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=14, pady=(6, 12))
        levels_row.grid_columnconfigure(1, weight=1)
        levels_row.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(levels_row, text="Уровень микрофона", width=140, anchor="w",
                     font=ctk.CTkFont(size=11), text_color=MUTED).grid(
            row=0, column=0, padx=(0, 6))
        self.mic_level = ctk.CTkProgressBar(levels_row, height=10, progress_color=SUCCESS)
        self.mic_level.set(0)
        self.mic_level.grid(row=0, column=1, sticky="ew", padx=(0, 16))

        ctk.CTkLabel(levels_row, text="Уровень звука ПК", width=130, anchor="w",
                     font=ctk.CTkFont(size=11), text_color=MUTED).grid(
            row=0, column=2, padx=(0, 6))
        self.system_level = ctk.CTkProgressBar(levels_row, height=10, progress_color=ACCENT)
        self.system_level.set(0)
        self.system_level.grid(row=0, column=3, sticky="ew")

        # ---------------- запись ----------------
        self.record_card = self._make_card(self, "2. Запись занятия")
        self.record_card.grid(row=2, column=0, sticky="ew", padx=18, pady=6)
        self.record_card.grid_columnconfigure(0, weight=1)

        self.timer_label = ctk.CTkLabel(
            self.record_card, text="00:00:00",
            font=ctk.CTkFont(size=52, weight="bold"),
        )
        self.timer_label.grid(row=1, column=0, pady=(2, 0))

        self.status_label = ctk.CTkLabel(
            self.record_card, text="Готово к записи",
            font=ctk.CTkFont(size=13), text_color=MUTED,
        )
        self.status_label.grid(row=2, column=0, pady=(0, 10), sticky="n")

        buttons = ctk.CTkFrame(self.record_card, fg_color="transparent")
        buttons.grid(row=3, column=0, pady=(0, 10))
        for column in range(3):
            buttons.grid_columnconfigure(column, weight=1, uniform="btn")

        self.start_button = ctk.CTkButton(
            buttons, text="●  Старт", width=180, height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=SUCCESS, hover_color="#247C47", command=self.on_start,
        )
        self.start_button.grid(row=0, column=0, padx=8)

        self.pause_button = ctk.CTkButton(
            buttons, text="⏸  Пауза", width=180, height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#4A4A55", hover_color="#3B3B44", command=self.on_pause,
        )
        self.pause_button.grid(row=0, column=1, padx=8)

        self.stop_button = ctk.CTkButton(
            buttons, text="■  Стоп и расшифровать", width=230, height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=DANGER, hover_color="#B2403D", command=self.on_stop,
        )
        self.stop_button.grid(row=0, column=2, padx=8)

        # ---------------- расшифровка / журнал ----------------
        self.log_card = self._make_card(self, "3. Расшифровка и сохранение")
        self.log_card.grid(row=3, column=0, sticky="nsew", padx=18, pady=(6, 16))
        self.log_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.log_card, text="Модель",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=1, column=0, sticky="w", padx=(14, 8))
        self.model_menu = ctk.CTkOptionMenu(
            self.log_card, values=list(MODEL_CHOICES), width=140,
            command=self._on_model_change,
        )
        self.model_menu.set(self.config_data.model_size)
        self.model_menu.grid(row=1, column=1, sticky="w")

        ctk.CTkLabel(
            self.log_card,
            text="medium — точнее, small — быстрее на слабом ПК",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).grid(row=1, column=2, columnspan=3, sticky="w", padx=(10, 14))

        ctk.CTkLabel(self.log_card, text="Папка",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=(14, 8), pady=(8, 4))
        self.path_entry = ctk.CTkEntry(self.log_card, height=32)
        self.path_entry.insert(0, str(self.config_data.resolved_output_dir()))
        self.path_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 4))

        path_buttons = ctk.CTkFrame(self.log_card, fg_color="transparent")
        path_buttons.grid(row=2, column=3, columnspan=2, sticky="e", padx=(8, 14), pady=(8, 4))
        self.choose_dir_button = ctk.CTkButton(
            path_buttons, text="Выбрать…", width=100, height=30,
            fg_color="transparent", border_width=1, command=self.choose_output_dir,
        )
        self.choose_dir_button.grid(row=0, column=0, padx=(0, 6))
        self.open_dir_button = ctk.CTkButton(
            path_buttons, text="Открыть папку", width=130, height=30,
            fg_color="transparent", border_width=1, command=self.open_output_dir,
        )
        self.open_dir_button.grid(row=0, column=1)

        # прогресс
        progress_row = ctk.CTkFrame(self.log_card, fg_color="transparent")
        progress_row.grid(row=4, column=0, columnspan=5, sticky="new",
                          padx=14, pady=(4, 0))
        progress_row.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(progress_row, height=12)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_label = ctk.CTkLabel(
            progress_row, text="", width=70, anchor="e",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        )
        self.progress_label.grid(row=0, column=1, padx=(10, 0))

        # журнал
        self.log_box = ctk.CTkTextbox(
            self.log_card, height=170, wrap="word",
            font=ctk.CTkFont(size=12),
        )
        self.log_box.grid(row=5, column=0, columnspan=5, sticky="nsew",
                          padx=14, pady=(8, 6))
        self.log_card.grid_rowconfigure(5, weight=1)

        # нижняя строка действий
        actions = ctk.CTkFrame(self.log_card, fg_color="transparent")
        actions.grid(row=6, column=0, columnspan=5, sticky="ew",
                     padx=14, pady=(0, 12))
        actions.grid_columnconfigure(1, weight=1)

        self.keep_audio_check = ctk.CTkCheckBox(
            actions, text="Сохранять и аудиозапись (WAV) рядом с текстом",
            font=ctk.CTkFont(size=12), command=self._on_keep_audio_change,
        )
        if self.config_data.keep_audio:
            self.keep_audio_check.select()
        self.keep_audio_check.grid(row=0, column=0, sticky="w")

        self.open_file_button = ctk.CTkButton(
            actions, text="Открыть расшифровку", width=170, height=30,
            fg_color="transparent", border_width=1,
            state="disabled", command=self.open_last_transcript,
        )
        self.open_file_button.grid(row=0, column=2, padx=(8, 0))

        self.transcribe_file_button = ctk.CTkButton(
            actions, text="Расшифровать файл…", width=170, height=30,
            fg_color="transparent", border_width=1, command=self.transcribe_external_file,
        )
        self.transcribe_file_button.grid(row=0, column=3, padx=(8, 0))

        self._log(f"Папка для расшифровок: {self.config_data.resolved_output_dir()}")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------ #
    def _make_card(self, parent, title: str) -> ctk.CTkFrame:
        """Рамка-«карточка» с заголовком."""
        card = ctk.CTkFrame(parent, corner_radius=12)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).grid(row=0, column=0, columnspan=5, sticky="ew", padx=14, pady=(12, 6))
        return card

    # ------------------------------------------------------------------ #
    #  Журнал и статус
    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: str = "info") -> None:
        """Добавляет строку в журнал окна (можно вызывать из любого потока)."""
        self.events.put(("log", (message, level)))

    def _log_now(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "", "warn": "⚠ ", "error": "✖ ", "ok": "✔ "}.get(level, "")
        line = f"[{timestamp}] {prefix}{message}\n"
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:  # noqa: BLE001 - журнал не должен ломать приложение
            pass
        log.info("%s", message)

    def _set_status(self, text: str, color=MUTED) -> None:
        try:
            self.status_label.configure(text=text, text_color=color)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    #  Состояния интерфейса
    # ------------------------------------------------------------------ #
    def _apply_state(self, state: str) -> None:
        """Включает/выключает кнопки под текущее состояние.

        Состояния: idle, recording, paused, stopping, transcribing, busy.
        """
        block_sources = state in ("recording", "paused", "stopping", "transcribing")
        if state == "idle":
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled", text="⏸  Пауза")
            self.stop_button.configure(state="disabled")
            self.theme_menu.configure(state="normal")
            self.transcribe_file_button.configure(state="normal")
        elif state == "recording":
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal", text="⏸  Пауза")
            self.stop_button.configure(state="normal")
            self.transcribe_file_button.configure(state="disabled")
        elif state == "paused":
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal", text="▶  Продолжить")
            self.stop_button.configure(state="normal")
        elif state == "stopping":
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
        elif state in ("transcribing", "busy"):
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.transcribe_file_button.configure(state="disabled")
        for widget in (self.system_device_combo, self.mic_device_combo,
                       self.mode_switch, self.refresh_button,
                       self.choose_dir_button, self.keep_audio_check):
            try:
                widget.configure(state="disabled" if block_sources else "normal")
            except Exception:  # noqa: BLE001 - не все виджеты знают state
                pass
        self._busy = state in ("transcribing", "busy", "stopping")

    def _set_progress(self, value: Optional[float], text: str = "") -> None:
        """value: 0..100 или None (неизвестно — «бегущая» полоска)."""
        try:
            if value is None:
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start()
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_bar.set(max(0.0, min(100.0, value)) / 100.0)
            self.progress_label.configure(
                text=text or (f"{value:.0f}%" if value is not None else "")
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    #  Устройства
    # ------------------------------------------------------------------ #
    def refresh_devices(self, silent: bool = False) -> None:
        """Перечитывает список устройств в фоне (WASAPI бывает медленным)."""
        if self.controller is not None and self.controller.is_recording:
            return
        if not silent:
            self._log("Ищу аудиоустройства…")
        self.refresh_button.configure(state="disabled")
        threading.Thread(target=self._refresh_devices_worker, daemon=True).start()

    def _refresh_devices_worker(self) -> None:
        try:
            devices = cached_devices(refresh=True)
            self.events.put(("devices", devices))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", (f"Не удалось получить список устройств: {exc}", "error")))
            self.events.put(("devices", []))

    def _apply_devices(self, devices: List[DeviceInfo]) -> None:
        self.devices = devices
        self._device_by_display = {d.display(): d for d in devices}

        system_devices = [d for d in devices if d.key == LOOPBACK]
        mic_devices = [d for d in devices if d.key == MIC]

        def fill(combo: ctk.CTkComboBox, items: List[DeviceInfo], key: str,
                 prefer: str, empty_text: str) -> None:
            values = [d.display() for d in items]
            if not values:
                values = [empty_text]
            combo.configure(values=values)
            chosen = pick_default(items, key, prefer)
            combo.set(chosen.display() if chosen else empty_text)

        fill(self.system_device_combo, system_devices, LOOPBACK,
             self.config_data.system_device,
             "—  системный звук не найден —")
        fill(self.mic_device_combo, mic_devices, MIC,
             self.config_data.mic_device,
             "—  микрофон не найден —")

        if system_devices:
            chosen = pick_default(system_devices, LOOPBACK, self.config_data.system_device)
            if chosen is not None:
                self.config_data.system_device = chosen.name
            self._log(f"Системный звук: {chosen.name if chosen else '—'}",
                      "ok" if chosen else "warn")
        else:
            self._log(
                "Не вижу loopback-устройство для звука компьютера. Обычно это "
                "значит, что сейчас нет активного воспроизведения: включите "
                "любое видео/звонок и нажмите «Обновить устройства».",
                "warn",
            )

        if mic_devices:
            chosen_mic = pick_default(mic_devices, MIC, self.config_data.mic_device)
            if chosen_mic is not None:
                self.config_data.mic_device = chosen_mic.name
            self._log(f"Микрофон: {chosen_mic.name if chosen_mic else '—'}",
                      "ok" if chosen_mic else "warn")
        else:
            self._log("Микрофон не найден. Подключите его и обновите список.", "warn")

        self.refresh_button.configure(state="normal")

    def _on_device_change(self, key: str) -> None:
        combo = self.system_device_combo if key == LOOPBACK else self.mic_device_combo
        info = self._device_by_display.get(combo.get())
        if info is None:
            return
        if key == LOOPBACK:
            self.config_data.system_device = info.name
        else:
            self.config_data.mic_device = info.name
        self._log(f"Выбрано устройство ({info.kind_label}): {info.name}")

    def _on_mode_change(self, label: str) -> None:
        for key, text in CAPTURE_MODES.items():
            if text == label:
                self.config_data.capture_mode = key
                break

    def _on_model_change(self, value: str) -> None:
        self.config_data.model_size = value
        self._log(f"Модель распознавания: {value}")

    def _on_theme_change(self, label: str) -> None:
        mapping = {"Тёмная": "dark", "Светлая": "light", "Системная": "system"}
        mode = mapping.get(label, "dark")
        self.config_data.appearance = mode
        ctk.set_appearance_mode(mode)

    def _on_keep_audio_change(self) -> None:
        self.config_data.keep_audio = bool(self.keep_audio_check.get())

    # ------------------------------------------------------------------ #
    #  Папка сохранения
    # ------------------------------------------------------------------ #
    def choose_output_dir(self) -> None:
        current = self.path_entry.get().strip() or str(self.config_data.resolved_output_dir())
        chosen = filedialog.askdirectory(
            parent=self, title="Куда сохранять расшифровки?", initialdir=current
        )
        if chosen:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, chosen)
            self.config_data.output_dir = chosen
            self._log(f"Папка для расшифровок: {chosen}")

    def open_output_dir(self) -> None:
        path = self.path_entry.get().strip() or str(self.config_data.resolved_output_dir())
        open_path_in_explorer(Path(path).expanduser())

    def open_last_transcript(self) -> None:
        if self.last_transcript and self.last_transcript.exists():
            open_path_in_explorer(self.last_transcript)

    def _current_output_dir(self) -> Path:
        raw = self.path_entry.get().strip()
        path = Path(raw).expanduser() if raw else self.config_data.resolved_output_dir()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Папка недоступна ({exc}), сохраняю в «Документы».", "warn")
            path = self.config_data.resolved_output_dir()
        return path

    # ------------------------------------------------------------------ #
    #  Запись
    # ------------------------------------------------------------------ #
    def on_start(self) -> None:
        if self._busy or (self.controller is not None and self.controller.is_recording):
            return
        self._sync_config_from_ui()
        save_config(self.config_data)

        # если модель ещё не в памяти — начинаем загружать её заранее,
        # чтобы после «Стоп» расшифровка стартовала сразу
        threading.Thread(target=self._preload_model, daemon=True).start()

        controller = RecordingController(
            self.config_data,
            levels=self.levels,
            on_log=lambda message: self._log(message),
        )
        try:
            wav_path = controller.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Не удалось начать запись")
            self._log(f"Не удалось начать запись: {exc}", "error")
            messagebox.showerror(
                "Не удалось начать запись",
                f"{exc}\n\nЧто можно сделать:\n"
                "• проверьте, что микрофон не занят другой программой;\n"
                "• нажмите «Обновить устройства» и выберите другое устройство;\n"
                "• подробности — в журнале ниже и в файле app.log.",
                parent=self,
            )
            self._apply_state("idle")
            return

        self.controller = controller
        self.recording_result = None
        self._recording_started_at = datetime.now()
        self._apply_state("recording")
        self._set_status("● Идёт запись", SUCCESS)
        self._set_progress(0, "")
        self._log(f"Запись: {wav_path}", "ok")
        threading.Thread(
            target=self._watch_quiet_start,
            args=(controller, controller.source_keys),
            daemon=True,
        ).start()

    def _watch_quiet_start(self, controller: RecordingController,
                           keys: tuple) -> None:
        """Через 8 секунд после старта предупреждаем, если звук не поступает."""
        import time

        time.sleep(8.0)
        if not controller.is_recording:
            return
        for key in keys:
            if controller.levels.get(key) >= 0.002:
                continue
            label = "микрофон" if key == MIC else "звук компьютера"
            self.events.put((
                "log",
                (f"Похоже, {label} сейчас молчит. Проверьте выбранное устройство "
                 "и уровень громкости в системе.", "warn"),
            ))

    def on_pause(self) -> None:
        controller = self.controller
        if controller is None or not controller.is_recording:
            return
        paused = controller.toggle_pause()
        if paused:
            self._apply_state("paused")
            self._set_status("⏸ Пауза — звук не записывается", WARNING)
        else:
            self._apply_state("recording")
            self._set_status("● Идёт запись", SUCCESS)

    def on_stop(self) -> None:
        controller = self.controller
        if controller is None or not controller.is_recording:
            return
        self._apply_state("stopping")
        self._set_status("Останавливаю запись…", MUTED)
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        controller = self.controller
        if controller is None:
            self.events.put(("idle", None))
            return
        try:
            result = controller.stop()
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка остановки записи")
            self.events.put(("log", (f"Ошибка остановки записи: {exc}", "error")))
            self.events.put(("idle", None))
            return

        self.events.put(("recorded", result))

        if result.duration < 1.0:
            self.events.put((
                "log",
                ("Запись почти пустая — расшифровывать нечего. "
                 "Проверьте источники звука и попробуйте снова.", "warn"),
            ))
            self.events.put(("idle", None))
            return

        self.events.put(("start_transcription", result))

    # ------------------------------------------------------------------ #
    #  Расшифровка
    # ------------------------------------------------------------------ #
    def _preload_model(self) -> None:
        try:
            from .transcriber import load_model

            load_model(self.config_data.model_size,
                       on_status=lambda message: self.events.put(("model_status", message)))
            self.events.put(("log", ("Модель распознавания загружена и готова.", "ok")))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", (f"Модель ещё не готова: {exc}", "warn")))

    def _start_transcription(self, result: RecordingResult) -> None:
        if self.worker is not None and self.worker.is_alive():
            self._log("Расшифровка уже идёт — дождитесь её окончания.", "warn")
            return

        header = HEADER_TEMPLATE.format(
            date=result.stopped_at.strftime("%d.%m.%Y %H:%M")
        )
        sources_text = ", ".join(
            f"{'звук ПК' if k == LOOPBACK else 'микрофон'}: {v}"
            for k, v in result.sources.items()
        )
        header += f"\nДлительность: {format_hms(result.duration)}"
        if sources_text:
            header += f"\nИсточники: {sources_text}"

        transcriber = Transcriber(
            self.config_data,
            on_progress=lambda progress: self.events.put(("progress", progress)),
            on_status=lambda message: self.events.put(("status", message)),
        )
        self.transcriber = transcriber

        output_dir = self._current_output_dir()
        keep_audio = bool(self.config_data.keep_audio)
        self.last_transcript = None
        self.open_file_button.configure(state="disabled")
        self._apply_state("transcribing")
        self._set_status("Расшифровываю запись…", ACCENT)
        self._set_progress(None, "")
        self._log(
            "Начинаю расшифровку. Это может занять время, "
            "если компьютер не очень быстрый.", "info"
        )

        def worker() -> None:
            source_wav = result.wav_path
            external_copy: Optional[Path] = None
            try:
                # Файл, выбранный пользователем, не трогаем: работаем с копией
                # во временной папке (её удалит уборщик после расшифровки).
                if result.external:
                    external_copy = _copy_to_temp(source_wav)
                    source_wav = external_copy
                text = transcriber.transcribe_file(source_wav)
                self.events.put(("status", "Сохраняю файл…"))
                from .transcriber import save_transcript, wrap_text

                path = save_transcript(wrap_text(text), output_dir, header=header)
                self.events.put(("transcript", path))

                if keep_audio and source_wav.exists() and not result.external:
                    target = path.with_suffix(".wav")
                    shutil.copy2(source_wav, target)
                    self.events.put(("log", (f"Аудиозапись сохранена: {target}", "ok")))
                self.events.put(("transcription_finished", None))
            except TranscriptionError as exc:
                self.events.put(("transcription_error", str(exc)))
            except Exception as exc:  # noqa: BLE001
                log.error("Ошибка расшифровки:\n%s", traceback.format_exc())
                self.events.put(("transcription_error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("cleanup", external_copy))

        self.worker = threading.Thread(target=worker, name="transcribe", daemon=True)
        self.worker.start()

    def _on_transcription_done(self, error: Optional[str]) -> None:
        self._apply_state("idle")
        if error:
            self._set_status("Не удалось расшифровать", DANGER)
            self._set_progress(0, "")
            self._log(error, "error")
            if "отменено" in error.lower():
                return
            messagebox.showerror(
                "Расшифровка не удалась",
                error + "\n\nЕсли запись сохранилась, её можно расшифровать "
                "позже кнопкой «Расшифровать файл…» (включите «Сохранять "
                "аудиозапись» перед записью).",
                parent=self,
            )
        else:
            self._set_status("Готово — расшифровка сохранена", SUCCESS)
            self.timer_label.configure(text="00:00:00")

    def transcribe_external_file(self) -> None:
        """Расшифровать готовый аудиофайл (WAV/MP3/M4A и т. п.)."""
        if self._busy:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Выберите аудиофайл для расшифровки",
            filetypes=[
                ("Аудио", "*.wav *.mp3 *.m4a *.ogg *.flac *.aac *.opus"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        result = RecordingResult(
            wav_path=Path(path),
            duration=0.0,
            started_at=datetime.now(),
            stopped_at=datetime.now(),
            sources={},
            warnings=[],
            external=True,
        )
        self._log(f"Выбран файл: {path}")
        self._start_transcription(result)

    # ------------------------------------------------------------------ #
    #  Циклы обновления
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """Обновляет таймер и индикаторы уровня 4 раза в секунду."""
        if self._closing:
            return
        controller = self.controller
        try:
            if controller is not None and controller.is_recording:
                self.timer_label.configure(text=format_hms(controller.elapsed))
            self.mic_level.set(level_percent(self.levels.get(MIC)))
            self.system_level.set(level_percent(self.levels.get(LOOPBACK)))
        except Exception:  # noqa: BLE001
            pass
        self.after(250, self._tick)

    def _poll_events(self) -> None:
        """Забирает сообщения из очереди (вызывается в потоке интерфейса)."""
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
            message, level = payload
            self._log_now(message, level)
        elif kind == "devices":
            self._apply_devices(payload)
        elif kind == "status":
            self._set_status(str(payload), ACCENT)
        elif kind == "model_status":
            self._log(str(payload))
        elif kind == "progress":
            progress: TranscriptionProgress = payload
            percent = progress.percent
            self._set_progress(percent, f"{percent:.0f}%" if percent is not None else "")
            if progress.message:
                self._set_status(progress.message, ACCENT)
        elif kind == "recorded":
            self.recording_result = payload
            for warning in payload.warnings:
                self._log_now(warning, "warn")
            self._log_now(
                f"Записано {format_hms(payload.duration)} звука.", "ok"
            )
            self._set_progress(0, "")
        elif kind == "start_transcription":
            self._start_transcription(payload)
        elif kind == "transcript":
            path = Path(payload)
            self.last_transcript = path
            self.open_file_button.configure(state="normal")
            self._log_now(f"Расшифровка сохранена: {path}", "ok")
            self._set_progress(100, "100%")
            try:
                self.bell()
            except Exception:  # noqa: BLE001
                pass
        elif kind == "transcription_error":
            self._on_transcription_done(str(payload))
        elif kind == "transcription_finished":
            if self.last_transcript is not None:
                self._apply_state("idle")
        elif kind == "cleanup":
            self._cleanup_temp_audio(extra=payload)
        elif kind == "transcribe_file":
            self.transcribe_external_file()
        elif kind == "idle":
            self._apply_state("idle")
            self._set_status("Готово к записи", MUTED)
            self.timer_label.configure(text="00:00:00")

    def _cleanup_temp_audio(self, extra: Optional[Path] = None) -> None:
        """Удаляет временные WAV-файлы, если пользователь не просил их сохранять.

        Копия файла, выбранного пользователем (``extra``), удаляется всегда —
        оригинал при этом не трогаем.
        """
        paths: List[Path] = []
        result = self.recording_result
        if result is not None and not self.config_data.keep_audio:
            paths.append(result.wav_path)
        if extra is not None:
            paths.append(Path(extra))
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    log.info("Временный файл удалён: %s", path)
            except Exception as exc:  # noqa: BLE001
                log.debug("Не удалось удалить временный файл: %s", exc)
        self._external_copy = None

    # ------------------------------------------------------------------ #
    def _sync_config_from_ui(self) -> None:
        """Переносит значения полей в настройки перед сохранением."""
        self.config_data.output_dir = self.path_entry.get().strip()
        try:
            self.config_data.keep_audio = bool(self.keep_audio_check.get())
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    #  Закрытие
    # ------------------------------------------------------------------ #
    def _on_close(self) -> None:
        controller = self.controller
        if controller is not None and controller.is_recording:
            keep = messagebox.askyesno(
                "Идёт запись",
                "Запись ещё идёт. Остановить её и закрыть программу?\n"
                "Аудиофайл сохранится, но расшифровка не запустится.",
                parent=self,
            )
            if not keep:
                return
            try:
                controller.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._busy:
            if not messagebox.askyesno(
                "Идёт расшифровка",
                "Расшифровка ещё не закончена. Закрыть программу? "
                "Результат не сохранится.",
                parent=self,
            ):
                return
            if self.transcriber is not None:
                self.transcriber.cancel()
        self._closing = True
        self._sync_config_from_ui()
        save_config(self.config_data)
        free_models()
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
#  Вспомогательные функции
# --------------------------------------------------------------------------- #
def _copy_to_temp(source: Path) -> Path:
    """Копирует выбранный пользователем файл во временную папку программы."""
    from .config import app_data_dir

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = app_data_dir() / "temp" / f"external_{stamp}{source.suffix or '.wav'}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def level_percent(peak: float) -> float:
    """Переводит пиковый уровень (0..1) в положение полоски индикатора (0..1).

    Линейная шкала «не видит» тихую речь, поэтому используем логарифмическую:
    -60 дБ → 0, 0 дБ → 1. Так полоска реагирует даже на негромкий голос.
    """
    if peak <= 0:
        return 0.0
    decibels = 20.0 * math.log10(max(peak, 1e-6))
    return max(0.0, min(1.0, (decibels + 60.0) / 60.0))


def apply_window_icon(window) -> None:
    """Ставит иконку окна из папки assets (если файл на месте)."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    ico, png = assets / "icon.ico", assets / "icon.png"
    try:
        if sys.platform == "win32" and ico.exists():
            window.iconbitmap(str(ico))
            return
        if png.exists():
            from PIL import Image, ImageTk  # Pillow нужен customtkinter

            with Image.open(png) as opened:
                image = ImageTk.PhotoImage(opened)
            window.iconphoto(True, image)
            window._tutor_icon = image  # держим ссылку, иначе иконка исчезнет
    except Exception as exc:  # noqa: BLE001 - иконка не критична
        log.debug("Не удалось поставить иконку окна: %s", exc)


def open_path_in_explorer(path: Path) -> None:
    """Открывает папку (или выделяет файл) в системном проводнике."""
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
        log.warning("Не удалось открыть папку %s: %s", path, exc)


def run(config: Optional[Config] = None) -> None:
    """Запускает приложение (удобно для тестов и для main.py)."""
    app = TutorTranscriptApp(config)
    app.mainloop()
