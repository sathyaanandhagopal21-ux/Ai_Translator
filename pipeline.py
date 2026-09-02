"""
pipeline.py
-----------
The orchestrator. Connects every module using background threads + queues so
that audio processing and translation NEVER run on the Tkinter main thread
(the GUI stays responsive).

Data flow:

    AudioCapture ──(utterance PCM)──▶ audio_queue
                                          │
                        [STT + Translate worker thread]
                                          │
                 updates subtitle (via callbacks) ──▶ GUI
                                          │
                                     tts_queue
                                          │
                            [TTS worker thread] ──▶ speaker
                            (pauses capture while speaking to
                             avoid re-capturing our own voice)

All GUI updates go through callbacks that are themselves thread-safe (the GUI
enqueues them for the main thread), so the pipeline can call them from any
thread.
"""

import queue
import threading

import config
from audio_input import AudioCapture
from speech_to_text import get_stt_engine, stt_language_code, STTError
from translator import get_translation_service, detect_language, TranslationError
from text_to_speech import get_tts_service


def _noop(*_args, **_kwargs):
    pass


class TranslationPipeline:
    def __init__(self, subtitle_manager,
                 on_original=None, on_subtitle=None, on_detected=None,
                 on_status=None, on_tts_status=None, on_level=None,
                 on_error=None):
        # Engines / services (each is swappable behind its interface).
        self.stt = get_stt_engine()
        self.translation = get_translation_service()
        self.tts = get_tts_service()
        self.subtitles = subtitle_manager

        # Callbacks to the GUI (all optional / thread-safe).
        self.on_original = on_original or _noop
        self.on_subtitle = on_subtitle or _noop
        self.on_detected = on_detected or _noop
        self.on_status = on_status or _noop
        self.on_tts_status = on_tts_status or _noop
        self.on_level = on_level or _noop
        self.on_error = on_error or _noop

        # Runtime state
        self._audio_queue = queue.Queue()
        self._interim_queue = queue.Queue(maxsize=2)
        self._tts_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._capture = None
        self._worker = None
        self._interim_worker = None
        self._tts_worker = None
        self._sequence_lock = threading.Lock()
        self._finalized_uids = set()

        # Per-run settings (set in start()).
        self._source_name = config.DEFAULT_SOURCE
        self._target_name = config.DEFAULT_TARGET
        self._speak = True
        self._capture_error_reported = False

    # ------------------------------------------------------------------ start
    def start(self, settings):
        """Begin translating. `settings` comes from the GUI (get_settings())."""
        self._source_name = settings.get("source_name", config.DEFAULT_SOURCE)
        self._target_name = settings.get("target_name", config.DEFAULT_TARGET)
        self._speak = bool(settings.get("speak", True))
        device_id = settings.get("device_id")

        self._stop_event.clear()
        self._capture_error_reported = False
        # Drain any leftovers from a previous run.
        self._drain(self._audio_queue)
        self._drain(self._interim_queue)
        self._drain(self._tts_queue)
        with self._sequence_lock:
            self._finalized_uids.clear()

        # Finished utterances and bounded provisional chunks use separate queues
        # so a slow interim request can never delay an authoritative final.
        self._capture = AudioCapture(
            on_utterance=lambda pcm, rate: self._audio_queue.put((pcm, rate, 0)),
            on_utterance_with_id=self._enqueue_final,
            on_interim=self._enqueue_interim,
            on_level=self._handle_level,
        )

        # Start worker threads BEFORE capture so nothing is missed.
        self._worker = threading.Thread(target=self._process_loop,
                                        daemon=True, name="STT-Translate")
        self._interim_worker = threading.Thread(target=self._interim_loop,
                                                daemon=True, name="STT-Interim")
        self._tts_worker = threading.Thread(target=self._tts_loop,
                                            daemon=True, name="TTS")
        self._worker.start()
        self._interim_worker.start()
        self._tts_worker.start()

        try:
            self._capture.start(device_id)
        except Exception as exc:
            self.on_error(f"Could not start audio: {exc}")
            self.stop()
            raise

        self.on_status("listening", config.COLORS["start"])

    # ------------------------------------------------------------------- stop
    def stop(self):
        """Stop everything and tear down threads. Safe to call more than once."""
        self._stop_event.set()
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
        # Unblock the worker queues with sentinels.
        self._audio_queue.put(None)
        try:
            self._interim_queue.put_nowait(None)
        except queue.Full:
            self._drain(self._interim_queue)
            self._interim_queue.put_nowait(None)
        self._tts_queue.put(None)
        for t in (self._worker, self._interim_worker, self._tts_worker):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._worker = self._interim_worker = self._tts_worker = None
        self._capture = None

    # --------------------------------------------------------------- internals
    def _enqueue_final(self, pcm, rate, utterance_id):
        """Queue a final and invalidate any late provisional result for it."""
        with self._sequence_lock:
            self._finalized_uids.add(utterance_id)
        self._audio_queue.put((pcm, rate, utterance_id))

    def _enqueue_interim(self, pcm, rate, utterance_id):
        """Keep only the newest provisional chunk when the service is busy."""
        with self._sequence_lock:
            if utterance_id in self._finalized_uids:
                return
        item = (pcm, rate, utterance_id)
        try:
            self._interim_queue.put_nowait(item)
        except queue.Full:
            try:
                self._interim_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._interim_queue.put_nowait(item)
            except queue.Full:
                pass

    def _is_finalized(self, utterance_id):
        with self._sequence_lock:
            return utterance_id in self._finalized_uids

    # --------------------------------------------------------------- internals
    @staticmethod
    def _drain(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _handle_level(self, rms, is_speech):
        # Reflect live input on the GUI indicator.
        label = "Hearing speech…" if is_speech else "Listening…"
        self.on_level(rms, is_speech, label)

    def _interim_loop(self):
        """Process provisional chunks independently of the final queue."""
        while not self._stop_event.is_set():
            try:
                item = self._interim_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            pcm, rate, utterance_id = item
            if self._is_finalized(utterance_id):
                continue
            self._handle_utterance(pcm, rate, interim=True,
                                   utterance_id=utterance_id)

    def _process_loop(self):
        """STT + language detection + translation (background thread)."""
        while not self._stop_event.is_set():
            # If the capture stream died (e.g. device unplugged), report once.
            if self._capture and self._capture.error and \
                    not self._capture_error_reported:
                self._capture_error_reported = True
                self.on_error(self._capture.error)
                break

            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:                    # stop sentinel
                break

            pcm, rate, utterance_id = item
            self._handle_utterance(pcm, rate, utterance_id=utterance_id)

        # let the GUI know we're idle again if we exited on our own
        self.on_status("stopped", config.COLORS["muted"])

    def _handle_utterance(self, pcm, rate, interim=False, utterance_id=0):
        """Run a final or provisional chunk through STT -> translate."""
        if interim and self._is_finalized(utterance_id):
            return

        # ---- Speech to text ----
        self.on_status("transcribing…" if not interim else "listening…",
                       config.COLORS["accent"])
        try:
            min_confidence = getattr(config, "STT_MIN_CONFIDENCE", 0.0)
            if self._source_name == config.SOURCE_AUTO and not interim and \
                    hasattr(self.stt, "transcribe_candidates"):
                names = [config.AUTO_PROBE_LANGUAGE] + list(
                    getattr(config, "AUTO_FALLBACK_LANGUAGES", ()))
                codes = [stt_language_code(name) for name in names]
                text, confidence = self.stt.transcribe_candidates(
                    pcm, rate, codes, min_confidence=min_confidence)
            else:
                text, confidence = self.stt.transcribe_detailed(
                    pcm, rate, stt_language_code(self._source_name),
                    min_confidence=min_confidence)
        except STTError as exc:
            if not interim:
                self.on_status("speech service error", config.COLORS["stop"])
                self.on_error(f"Speech-to-text: {exc}")
            return

        if not text or (interim and self._is_finalized(utterance_id)):
            if not interim:
                self.on_status("listening", config.COLORS["start"])
            return

        self.on_original(text)

        # ---- Language detection (for the "Detected:" label) ----
        _code, detected_name = detect_language(text)
        if not interim:
            self.on_detected(detected_name)

        # ---- Translation ----
        self.on_status("translating…" if not interim else "live translating…",
                       config.COLORS["accent"])
        try:
            translated = self.translation.translate(
                text, self._source_name, self._target_name)
        except TranslationError as exc:
            if not interim:
                self.on_status("translation error", config.COLORS["stop"])
                self.on_error(f"Translation: {exc}")
            return

        if not translated or (interim and self._is_finalized(utterance_id)):
            if not interim:
                self.on_status("no translation", config.COLORS["stop"])
            return

        if interim:
            # Provisional text is display-only: it is never saved or spoken.
            self.on_subtitle(translated)
            confidence_text = "" if confidence is None else \
                f" · {confidence:.0%}"
            self.on_status(f"live · interim{confidence_text}",
                           config.COLORS["accent"])
            return

        # ---- Update subtitle + queue speech ----
        self.subtitles.add(text, translated, detected_name)
        self.on_subtitle(translated)
        self.on_status("translated ✓", config.COLORS["start"])

        if self._speak:
            self._queue_tts(translated)

    def _queue_tts(self, text):
        """Keep speech current instead of letting old captions form a backlog."""
        try:
            while True:
                self._tts_queue.get_nowait()
        except queue.Empty:
            pass
        self._tts_queue.put_nowait(text)

    def _tts_loop(self):
        """Speak queued translations (background thread)."""
        target = config.get_language(self._target_name)
        tts_code = target["tts"] if target else "en"

        while not self._stop_event.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:                    # stop sentinel
                break

            # Pause capturing so loopback/mic doesn't record our own voice.
            paused = False
            if config.PAUSE_CAPTURE_WHILE_SPEAKING and self._capture:
                self._capture.pause()
                paused = True

            self.on_tts_status("speaking…", config.COLORS["accent"])
            ok, info = self.tts.speak(text, tts_code)
            self.on_tts_status(info, config.COLORS["start"] if ok
                               else config.COLORS["muted"])

            if paused and self._capture:
                self._capture.resume()
