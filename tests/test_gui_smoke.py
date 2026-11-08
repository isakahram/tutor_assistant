"""Дымовой тест интерфейса без реального Tk.

На машине без дисплея (например, на сервере) нельзя открыть окно, но можно
подменить ``customtkinter`` и ``tkinter`` простыми заглушками и проверить, что
вся логика окна работает: переключение состояний, заполнение списков
устройств, обработка событий и сохранение результата.

Запуск:  python -m unittest tests.test_gui_smoke -v
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
#  Заглушки вместо Tk
# --------------------------------------------------------------------------- #
class _Widget:
    """Минимальная «пустышка» вместо виджета."""

    def __init__(self, master=None, **kwargs) -> None:
        self.master = master
        self.options = dict(kwargs)
        self._state = kwargs.get("state", "normal")
        self._value = kwargs.get("values", [""])[0] if kwargs.get("values") else ""

    # геометрия
    def grid(self, *args, **kwargs):
        return None

    def pack(self, *args, **kwargs):
        return None

    def grid_columnconfigure(self, *args, **kwargs):
        return None

    def grid_rowconfigure(self, *args, **kwargs):
        return None

    # свойства
    def configure(self, **kwargs):
        if "state" in kwargs:
            self._state = kwargs["state"]
        self.options.update(kwargs)

    config = configure

    def cget(self, key):
        if key == "state":
            return self._state
        return self.options.get(key, "")

    def set(self, value):
        self._value = value

    def get(self):
        return self._value

    def insert(self, *args, **kwargs):
        # как у настоящего Entry: insert(index, text) — текст последний
        if args and isinstance(args[-1], str):
            self._value = args[-1]
        return None

    def delete(self, *args, **kwargs):
        self._value = ""
        return None

    def see(self, *args, **kwargs):
        return None

    def select(self):
        self._checked = True

    def deselect(self):
        self._checked = False

    def start(self):
        return None

    def stop(self):
        return None

    def destroy(self):
        return None


class _Root(_Widget):
    """Замена CTk: окно с минимальным набором методов."""

    def title(self, text):
        self._title = text

    def geometry(self, spec):
        self._geometry = spec

    def minsize(self, width, height):
        return None

    def protocol(self, *args, **kwargs):
        return None

    def after(self, delay, callback=None, *args):
        self._last_after = (delay, callback)
        return "after-id"

    def after_cancel(self, *args):
        return None

    def mainloop(self, *args, **kwargs):
        return None

    def bell(self):
        return None

    def update(self):
        return None

    def grid_columnconfigure(self, *args, **kwargs):
        return None

    def grid_rowconfigure(self, *args, **kwargs):
        return None


class _Font:
    def __init__(self, *args, **kwargs):
        self.options = kwargs


def _install_stubs() -> None:
    """Подставляет фальшивые customtkinter/tkinter в sys.modules."""
    ctk = types.ModuleType("customtkinter")
    for name in (
        "CTkFrame", "CTkLabel", "CTkButton", "CTkComboBox", "CTkSegmentedButton",
        "CTkProgressBar", "CTkTextbox", "CTkEntry", "CTkOptionMenu",
        "CTkScrollableFrame", "CTkSlider", "CTkSwitch",
    ):
        setattr(ctk, name, type(name, (_Widget,), {}))

    class _CheckBox(_Widget):
        """Как настоящий CTkCheckBox: get() возвращает 1 или 0."""

        def __init__(self, master=None, **kwargs) -> None:
            super().__init__(master, **kwargs)
            self._checked = False

        def get(self):
            return 1 if self._checked else 0

    ctk.CTkCheckBox = _CheckBox
    ctk.CTk = _Root
    ctk.CTkFont = _Font
    ctk.set_appearance_mode = lambda mode: None
    ctk.set_default_color_theme = lambda theme: None
    ctk.get_appearance_mode = lambda: "Dark"
    sys.modules["customtkinter"] = ctk

    tk = types.ModuleType("tkinter")

    class _Tk(_Root):
        def withdraw(self):
            return None

    tk.Tk = _Tk
    tk.NORMAL = "normal"
    tk.DISABLED = "disabled"
    tk.END = "end"
    tk.TclError = RuntimeError

    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.askdirectory = lambda **kwargs: ""
    filedialog.askopenfilename = lambda **kwargs: ""
    tk.filedialog = filedialog

    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showerror = lambda *a, **k: None
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None
    messagebox.askyesno = lambda *a, **k: True
    tk.messagebox = messagebox

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox


_install_stubs()

from tutor_transcript.capture import RecordingResult  # noqa: E402
from tutor_transcript.config import Config  # noqa: E402
from tutor_transcript.devices import DeviceInfo, LOOPBACK, MIC  # noqa: E402
from tutor_transcript.gui import TutorTranscriptApp  # noqa: E402
from tutor_transcript.transcriber import TranscriptionProgress  # noqa: E402


class TestGuiSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.app = TutorTranscriptApp(Config(), autoload_devices=False)

    def tearDown(self) -> None:
        try:
            self.app._closing = True
            self.app.destroy()
        except Exception:
            pass
        self.tmp.cleanup()

    # ------------------------------------------------------------------ #
    def test_all_states_toggle_without_error(self) -> None:
        for state in ("idle", "recording", "paused", "stopping",
                      "transcribing", "busy", "idle"):
            self.app._apply_state(state)
        self.assertEqual(self.app.start_button.cget("state"), "normal")

    def test_devices_are_listed(self) -> None:
        devices = [
            DeviceInfo(LOOPBACK, "Динамики (Realtek)", 2, True),
            DeviceInfo(MIC, "Микрофон (USB)", 1, True),
        ]
        self.app._apply_devices(devices)
        self.assertEqual(len(self.app.devices), 2)
        self.assertIn("Динамики", self.app.system_device_combo.get())
        self.assertIn("Микрофон", self.app.mic_device_combo.get())

    def test_no_devices_gives_warning(self) -> None:
        self.app._apply_devices([])
        self.assertEqual(self.app.devices, [])
        self.assertIn("не найден", self.app.mic_device_combo.get())

    def test_events_are_handled(self) -> None:
        result = RecordingResult(
            wav_path=Path(self.tmp.name) / "rec.wav",
            duration=12.5,
            started_at=datetime.now(),
            stopped_at=datetime.now(),
            sources={LOOPBACK: "Динамики", MIC: "Микрофон"},
            warnings=["тестовое предупреждение"],
        )
        self.app._handle_event("log", ("проверка журнала", "info"))
        self.app._handle_event("progress", TranscriptionProgress(percent=42.0,
                                                                message="идёт"))
        self.app._handle_event("status", "статус")
        self.app._handle_event("model_status", "модель")
        self.app._handle_event("recorded", result)
        self.app._handle_event("idle", None)

        self.assertTrue(True)  # главное — отсутствие исключений

    def test_transcript_saved_reaction(self) -> None:
        path = Path(self.tmp.name) / "transcript_x.txt"
        path.write_text("текст", encoding="utf-8")
        self.app._handle_event("transcript", path)
        self.assertEqual(self.app.last_transcript, path)
        self.assertEqual(self.app.open_file_button.cget("state"), "normal")

    def test_error_reported(self) -> None:
        self.app._on_transcription_done("нет модели")
        self.assertEqual(self.app.start_button.cget("state"), "normal")

    def test_config_roundtrip_via_ui(self) -> None:
        self.app.path_entry.delete(0, "end")
        self.app.path_entry.insert(0, self.tmp.name)
        self.app._on_mode_change("Только микрофон")
        self.app._on_model_change("small")
        self.app._sync_config_from_ui()
        self.assertEqual(self.app.config_data.capture_mode, "mic")
        self.assertEqual(self.app.config_data.model_size, "small")
        self.assertEqual(self.app.config_data.output_dir, self.tmp.name)
        self.assertEqual(self.app._current_output_dir(), Path(self.tmp.name))

    def test_start_recording_without_devices_still_starts(self) -> None:
        """Даже без устройств запись стартует (источник-заглушка) и корректно
        останавливается — это важно, чтобы приложение не зависало."""
        self.app._preload_model = lambda: None
        self.app.on_start()
        self.assertIsNotNone(self.app.controller)
        self.assertTrue(self.app.controller.is_recording)
        self.app._apply_state("recording")
        self.app.on_pause()
        self.assertTrue(self.app.controller.is_paused)
        self.app.on_pause()
        self.assertFalse(self.app.controller.is_paused)
        result = self.app.controller.stop()
        self.assertIsInstance(result, RecordingResult)
        self.assertGreaterEqual(result.duration, 0.0)

    def test_close_stops_and_saves(self) -> None:
        self.app._on_close()
        self.assertTrue(self.app._closing)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class _FakeController:
    """Подмена RecordingController для проверки полного сценария в окне."""

    def __init__(self, config, levels=None, on_log=None) -> None:
        self.config = config
        self.levels = levels
        self._on_log = on_log
        self.is_recording = False
        self.is_paused = False
        self.source_errors = []
        self.warnings = []
        self.source_keys = (MIC, LOOPBACK)
        self.elapsed = 0.0
        self.wav_path = Path(tempfile.mkdtemp()) / "fake.wav"

    def start(self):
        self.is_recording = True
        self.wav_path.write_bytes(b"RIFF....")  # имитация файла записи
        return self.wav_path

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def stop(self):
        self.is_recording = False
        from tutor_transcript.capture import RecordingResult

        return RecordingResult(
            wav_path=self.wav_path,
            duration=125.0,
            started_at=datetime.now(),
            stopped_at=datetime.now(),
            sources={MIC: "Микрофон", LOOPBACK: "Динамики"},
            warnings=[],
        )


class _FakeTranscriber:
    """Подмена Transcriber: «распознаёт» заготовленный текст."""

    calls = []

    def __init__(self, config, on_progress=None, on_status=None) -> None:
        self.config = config
        self._on_progress = on_progress
        self._on_status = on_status

    def transcribe_file(self, path):
        _FakeTranscriber.calls.append(Path(path))
        if self._on_progress is not None:
            self._on_progress(TranscriptionProgress(percent=50.0, message="половина"))
        return "Сегодня разбирали задачи на проценты. Домашнее задание: № 12, 14."


class TestFullScenario(unittest.TestCase):
    """Старт → пауза → стоп → расшифровка → файл на диске."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        self.out.mkdir()
        import tutor_transcript.gui as gui

        self.gui = gui
        self._orig_controller = gui.RecordingController
        self._orig_transcriber = gui.Transcriber
        gui.RecordingController = _FakeController
        gui.Transcriber = _FakeTranscriber
        _FakeTranscriber.calls = []

        self.app = TutorTranscriptApp(Config(), autoload_devices=False)
        self.app.path_entry.delete(0, "end")
        self.app.path_entry.insert(0, str(self.out))
        self.app._preload_model = lambda: None

    def tearDown(self) -> None:
        self.gui.RecordingController = self._orig_controller
        self.gui.Transcriber = self._orig_transcriber
        try:
            self.app._closing = True
            self.app.destroy()
        except Exception:
            pass
        self.tmp.cleanup()

    def _drain(self) -> None:
        """Разбирает очередь событий, как это делает _poll_events."""
        import queue as queue_module

        while True:
            try:
                kind, payload = self.app.events.get_nowait()
            except queue_module.Empty:
                break
            self.app._handle_event(kind, payload)

    def test_recording_transcription_and_cleanup(self) -> None:
        self.app.on_start()
        self.assertTrue(self.app.controller.is_recording)
        self.assertEqual(self.app.stop_button.cget("state"), "normal")

        self.app.on_pause()
        self.assertEqual(self.app.status_label.options.get("text"),
                         "⏸ Пауза — звук не записывается")
        self.app.on_pause()
        self.assertEqual(self.app.status_label.options.get("text"), "● Идёт запись")

        self.app.on_stop()
        # дождаться рабочего потока остановки и расшифровки
        for _ in range(100):
            if self.app.last_transcript is not None:
                break
            time.sleep(0.05)
            self._drain()

        self.assertIsNotNone(self.app.last_transcript, "расшифровка не сохранилась")
        saved = self.app.last_transcript
        self.assertTrue(saved.exists())
        self.assertTrue(saved.name.startswith("transcript_"))
        self.assertIn("проценты", saved.read_text(encoding="utf-8"))
        self.assertIn("Длительность: 00:02:05", saved.read_text(encoding="utf-8"))
        self.assertEqual(self.app.start_button.cget("state"), "normal")
        self.assertEqual(self.app.open_file_button.cget("state"), "normal")
        self.assertEqual(_FakeTranscriber.calls[0], self.app.controller.wav_path)

    def test_temp_audio_removed_when_not_kept(self) -> None:
        self.app.on_start()
        wav = self.app.controller.wav_path
        self.app.on_stop()
        for _ in range(100):
            if self.app.last_transcript is not None and not wav.exists():
                break
            time.sleep(0.05)
            self._drain()
        self.assertFalse(wav.exists(), "временный файл должен удаляться")
        # рядом с текстом не должно появиться WAV
        self.assertEqual(list(self.out.glob("*.wav")), [])

    def test_audio_kept_when_requested(self) -> None:
        self.app.keep_audio_check.select()
        self.app.config_data.keep_audio = True
        self.app.on_start()
        self.app.on_stop()
        for _ in range(100):
            if self.app.last_transcript is not None and list(self.out.glob("*.wav")):
                break
            time.sleep(0.05)
            self._drain()
        self.assertTrue(list(self.out.glob("*.wav")), "WAV должен остаться рядом с текстом")
