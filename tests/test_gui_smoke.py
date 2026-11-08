"""Дымовые тесты окна без реального Tk.

На машине без дисплея (сервер, CI) окно не открыть, но можно подменить
``customtkinter``/``tkinter`` заглушками и проверить всю логику интерфейса:
состояния кнопок, полный сценарий «Старт → Пауза → Расшифровка», сохранение
файла, удаление временного WAV и показ ошибок.

Запуск:  python -m unittest tests.test_gui_smoke -v
"""

from __future__ import annotations

import queue as queue_module
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
        self._value = ""

    def grid(self, *args, **kwargs):
        return None

    def pack(self, *args, **kwargs):
        return None

    def grid_columnconfigure(self, *args, **kwargs):
        return None

    def grid_rowconfigure(self, *args, **kwargs):
        return None

    def configure(self, **kwargs):
        if "state" in kwargs:
            self._state = kwargs["state"]
        self.options.update(kwargs)

    config = configure

    def cget(self, key):
        return self._state if key == "state" else self.options.get(key, "")

    def set(self, value):
        self._value = value

    def get(self):
        return self._value

    def insert(self, *args, **kwargs):
        if args and isinstance(args[-1], str):
            self._value += args[-1]

    def delete(self, *args, **kwargs):
        self._value = ""

    def see(self, *args, **kwargs):
        return None

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

    def resizable(self, *args):
        return None

    def protocol(self, *args, **kwargs):
        return None

    def after(self, delay, callback=None, *args):
        return "after-id"

    def after_cancel(self, *args):
        return None

    def mainloop(self, *args, **kwargs):
        return None

    def bell(self):
        return None

    def destroy(self):
        return None


class _Font:
    def __init__(self, *args, **kwargs):
        self.options = kwargs


def _install_stubs() -> None:
    ctk = types.ModuleType("customtkinter")
    for name in (
        "CTkFrame", "CTkLabel", "CTkButton", "CTkTextbox", "CTkEntry",
        "CTkProgressBar", "CTkComboBox", "CTkOptionMenu", "CTkCheckBox",
        "CTkSegmentedButton", "CTkScrollableFrame", "CTkSlider", "CTkSwitch",
    ):
        setattr(ctk, name, type(name, (_Widget,), {}))
    ctk.CTk = _Root
    ctk.CTkFont = _Font
    ctk.set_appearance_mode = lambda mode: None
    ctk.set_default_color_theme = lambda theme: None

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
    tk.filedialog = filedialog

    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showerror = lambda *a, **k: None
    messagebox.showinfo = lambda *a, **k: None
    messagebox.askyesno = lambda *a, **k: True
    tk.messagebox = messagebox

    sys.modules["customtkinter"] = ctk
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox


_install_stubs()

import tutor_transcript.gui as gui  # noqa: E402
from tutor_transcript.capture import RecordingResult  # noqa: E402
from tutor_transcript.devices import LOOPBACK, MIC  # noqa: E402
from tutor_transcript.transcriber import Progress  # noqa: E402


class _FakeController:
    """Подмена RecordingController."""

    workdir = Path(tempfile.mkdtemp())

    def __init__(self, on_log=None) -> None:
        self._on_log = on_log
        self.levels = types.SimpleNamespace(get=lambda key: 0.5)
        self.is_recording = False
        self.is_paused = False
        self.source_keys = (MIC, LOOPBACK)
        self.elapsed = 0.0
        self.wav_path = _FakeController.workdir / f"rec_{id(self)}.wav"

    def start(self, output_wav=None):
        self.is_recording = True
        self.wav_path.write_bytes(b"RIFF0000WAVE")  # имитация записи
        return self.wav_path

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def stop(self):
        self.is_recording = False
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
    fail_with = None

    def __init__(self, on_progress=None, on_status=None) -> None:
        self._on_progress = on_progress
        self._on_status = on_status

    def cancel(self) -> None:
        return None

    def transcribe_and_save(self, wav_path, directory, header=None):
        _FakeTranscriber.calls.append({"wav": Path(wav_path),
                                       "dir": Path(directory), "header": header})
        if _FakeTranscriber.fail_with:
            raise gui.TranscriptionError(_FakeTranscriber.fail_with)
        if self._on_progress is not None:
            self._on_progress(Progress(done_seconds=60.0, total_seconds=120.0,
                                       message="Распознано 00:01:00 из 00:02:00"))
        text = "Сегодня разбирали задачи на проценты. Домашнее задание: № 12, 14."
        return _save(directory, header, text)


def _save(directory, header, text):
    """Сохраняет «распознанный» текст так же, как это делает программа."""
    from tutor_transcript.transcriber import save_transcript

    return save_transcript(text, directory, header=header)


class BaseGuiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "desktop"
        self.out.mkdir()

        self._orig = (gui.RecordingController, gui.Transcriber, gui.output_dir)
        gui.RecordingController = _FakeController
        gui.Transcriber = _FakeTranscriber
        gui.output_dir = lambda: self.out
        _FakeTranscriber.calls = []
        _FakeTranscriber.fail_with = None

        self.app = gui.App()
        # фоновую подготовку (описание окружения и загрузку модели) не ждём
        self.app._prepare = lambda: None

    def tearDown(self) -> None:
        gui.RecordingController, gui.Transcriber, gui.output_dir = self._orig
        try:
            self.app._closing = True
            self.app.destroy()
        except Exception:
            pass
        self.tmp.cleanup()

    def drain(self) -> None:
        while True:
            try:
                kind, payload = self.app.events.get_nowait()
            except queue_module.Empty:
                break
            self.app._handle_event(kind, payload)

    def wait_done(self, predicate, timeout: float = 8.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.drain()
            if predicate():
                return
            time.sleep(0.05)
        self.fail("сценарий не завершился за отведённое время")


class TestUiStructure(BaseGuiTest):
    """Интерфейс должен содержать только то, что просил пользователь."""

    def test_only_three_buttons_and_no_extra_widgets(self) -> None:
        self.assertTrue(hasattr(self.app, "start_button"))
        self.assertTrue(hasattr(self.app, "pause_button"))
        self.assertTrue(hasattr(self.app, "transcribe_button"))
        for unwanted in ("mode_switch", "mic_level", "system_level",
                         "model_menu", "path_entry", "refresh_button",
                         "choose_dir_button", "keep_audio_check",
                         "theme_menu", "progress_bar", "system_device_combo"):
            self.assertFalse(hasattr(self.app, unwanted),
                             f"лишний элемент интерфейса: {unwanted}")

    def test_buttons_names(self) -> None:
        self.assertIn("Старт", self.app.start_button.options["text"])
        self.assertIn("Пауза", self.app.pause_button.options["text"])
        self.assertIn("Расшифровка", self.app.transcribe_button.options["text"])

    def test_initial_state(self) -> None:
        self.assertEqual(self.app.start_button.cget("state"), "normal")
        self.assertEqual(self.app.pause_button.cget("state"), "disabled")
        self.assertEqual(self.app.transcribe_button.cget("state"), "disabled")

    def test_states_toggle(self) -> None:
        for state in ("recording", "paused", "stopping", "transcribing", "idle"):
            self.app._set_state(state)
        self.assertEqual(self.app.start_button.cget("state"), "normal")


class TestScenario(BaseGuiTest):
    def test_start_pause_resume(self) -> None:
        self.app.on_start()
        self.assertTrue(self.app.controller.is_recording)
        self.assertEqual(self.app.transcribe_button.cget("state"), "normal")
        self.assertEqual(self.app.status_label.options["text"], "● Идёт запись")

        self.app.on_pause()
        self.assertTrue(self.app.controller.is_paused)
        self.assertIn("Пауза", self.app.status_label.options["text"])
        self.assertIn("Продолжить", self.app.pause_button.options["text"])

        self.app.on_pause()
        self.assertFalse(self.app.controller.is_paused)
        self.assertEqual(self.app.status_label.options["text"], "● Идёт запись")

    def test_transcribe_saves_text_and_removes_wav(self) -> None:
        self.app.on_start()
        wav = self.app.controller.wav_path
        self.assertTrue(wav.exists())

        self.app.on_transcribe()
        self.wait_done(lambda: self.app.last_transcript is not None)

        saved = self.app.last_transcript
        self.assertTrue(saved.exists())
        self.assertTrue(saved.parent == self.out, "текст должен лечь на рабочий стол")
        content = saved.read_text(encoding="utf-8")
        self.assertIn("проценты", content)
        self.assertIn("Длительность: 00:02:05", content)

        # временный WAV удалён, рядом с текстом ничего лишнего
        self.wait_done(lambda: not wav.exists())
        self.assertEqual(list(self.out.glob("*.wav")), [])

        # интерфейс вернулся в исходное состояние
        self.wait_done(lambda: self.app.start_button.cget("state") == "normal")
        self.assertEqual(self.app.open_button.cget("state"), "normal")
        self.assertIn("Готово", self.app.status_label.options["text"])

        # расшифровке передан именно тот файл, который записали
        self.assertEqual(_FakeTranscriber.calls[0]["wav"], wav)
        self.assertEqual(_FakeTranscriber.calls[0]["dir"], self.out)

    def test_transcription_error_keeps_ui_alive(self) -> None:
        _FakeTranscriber.fail_with = "модель не загрузилась"
        self.app.on_start()
        self.app.on_transcribe()
        self.wait_done(lambda: self.app.start_button.cget("state") == "normal")
        self.assertIn("Не удалось", self.app.status_label.options["text"])
        self.assertIn("модель не загрузилась", self.app.log_box._value)

    def test_error_keeps_audio_for_retry(self) -> None:
        _FakeTranscriber.fail_with = "нет интернета"
        self.app.on_start()
        wav = self.app.controller.wav_path
        self.app.on_transcribe()
        self.wait_done(lambda: self.app.start_button.cget("state") == "normal")
        self.assertTrue(wav.exists(), "при ошибке запись нельзя удалять")
        self.assertIsNone(self.app.last_transcript)

    def test_empty_recording_is_not_transcribed(self) -> None:
        class _Empty(_FakeController):
            def stop(self):
                result = super().stop()
                result.duration = 0.2
                return result

        gui.RecordingController = _Empty
        self.app.on_start()
        self.app.on_transcribe()
        self.wait_done(lambda: self.app.start_button.cget("state") == "normal")
        self.assertEqual(_FakeTranscriber.calls, [])
        self.assertIsNone(self.app.last_transcript)

    def test_close_stops_recording(self) -> None:
        self.app.on_start()
        self.app._on_close()
        self.assertTrue(self.app._closing)
        self.assertFalse(self.app.controller.is_recording)


if __name__ == "__main__":
    unittest.main(verbosity=2)
