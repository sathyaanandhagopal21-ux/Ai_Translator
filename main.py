"""
main.py
-------
Entry point for the AI Voice Translator.

Wires the GUI to the processing pipeline:
    * Builds the shared SubtitleManager and TranslationPipeline.
    * Connects the pipeline's callbacks to the GUI's thread-safe update methods.
    * Handles Start / Stop / Clear, keeping the GUI responsive (start/stop work
      happens off the main thread).

Usage:
    python main.py              # launch the app
    python main.py --selftest   # build the GUI off-screen, then exit (CI check)
"""

import sys
import threading
import tkinter as tk

import config
from gui import TranslatorGUI
from subtitle_manager import SubtitleManager


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.subtitles = SubtitleManager()
        self.pipeline = None            # created lazily (needs audio libs)
        self._lock = threading.Lock()   # serialize start/stop

        # Load optional Groq/Gemini provider keys from .env. Harmless if absent.
        config.load_env_file()

        self.gui = TranslatorGUI(
            root,
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_clear=self._on_clear,
            list_input_devices=self._list_input_devices,
        )

    # ------------------------------------------------------------- device list
    def _list_input_devices(self):
        """Real device list, with a safe fallback so the GUI always launches."""
        try:
            import audio_input
            return audio_input.list_input_devices()
        except Exception as exc:
            return [{"label": f"(audio unavailable: {exc})", "id": None}]

    # ------------------------------------------------------------- build pipeline
    def _ensure_pipeline(self):
        """Create the pipeline on first use (imports the audio/AI modules)."""
        if self.pipeline is not None:
            return self.pipeline
        from pipeline import TranslationPipeline
        self.pipeline = TranslationPipeline(
            self.subtitles,
            on_original=self.gui.set_original,
            on_subtitle=self.gui.set_subtitle,
            on_detected=self.gui.set_detected_language,
            on_status=self.gui.set_translation_status,
            on_tts_status=self.gui.set_tts_status,
            on_level=self._on_level,
            on_error=self._on_error,
        )
        return self.pipeline

    # ------------------------------------------------------------------ handlers
    def _on_start(self, settings):
        """Start translating. Runs setup off the main thread to avoid any UI lag."""
        def worker():
            with self._lock:
                try:
                    self._ensure_pipeline().start(settings)
                except Exception as exc:
                    # Revert the UI and show what went wrong.
                    self.gui.set_translation_status(f"start failed: {exc}",
                                                    config.COLORS["stop"])
                    self.gui.notify_stopped()
        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self):
        """Stop translating. Tear-down happens off the main thread (can take ~1s)."""
        def worker():
            with self._lock:
                if self.pipeline is not None:
                    self.pipeline.stop()
        threading.Thread(target=worker, daemon=True).start()
        # UI is reset immediately by the GUI's own Stop handler.

    def _on_clear(self):
        self.subtitles.clear()

    def _on_level(self, rms, is_speech, label):
        # Keep the indicator lively and drive the live audio level meter.
        self.gui.set_input_indicator(True, label)
        self.gui.set_level(rms, is_speech)

    def _on_error(self, message):
        """Called from a worker thread when something fails fatally."""
        self.gui.set_translation_status(message, config.COLORS["stop"])
        self.gui.notify_stopped()
        # Stop the pipeline from a SEPARATE thread (never join our own thread).
        threading.Thread(target=self._safe_stop, daemon=True).start()

    def _safe_stop(self):
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.stop()


def main():
    selftest = "--selftest" in sys.argv

    root = tk.Tk()
    if selftest:
        root.withdraw()

    app = TranslatorApp(root)

    if selftest:
        root.update_idletasks()
        root.update()
        root.destroy()
        print("SELFTEST OK: full app constructed and rendered without errors.")
        return

    # Make sure the pipeline is stopped if the window is closed while running.
    def on_close():
        try:
            if app.pipeline is not None:
                app.pipeline.stop()
        finally:
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


if __name__ == "__main__":
    main()
