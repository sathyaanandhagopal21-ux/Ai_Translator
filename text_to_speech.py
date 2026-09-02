"""
text_to_speech.py
-----------------
Turn translated text into spoken audio.

Engines (behind a common TTSEngine interface, so they're swappable):
    * GTTSEngine    - Google Text-to-Speech. Online, multilingual (Tamil,
                      Hindi, etc.). Produces an mp3 which we play through the
                      Windows MCI system via ctypes - so NO extra audio-playback
                      dependency is needed.
    * Pyttsx3Engine - Offline Windows SAPI voices. Instant, but most systems
                      only have English-ish voices installed, so it's a
                      fallback for when gTTS/network is unavailable.

TTSService tries engines in order and, like the translator, puts a failing
engine on a short cooldown so we don't repeatedly wait on a dead network.

If nothing can speak the chosen language, we fail quietly (subtitles still
show) - speech is a bonus, not something that should crash the app.
"""

import ctypes
import io
import os
import tempfile
import threading
import time

import config

# Windows Multimedia API, used only for DESKTOP mp3 playback. Loaded lazily/
# guarded so this module also imports on non-Windows hosts (the Flask web app
# synthesizes audio and sends it to the browser instead of playing it here).
try:
    _winmm = ctypes.windll.winmm
except (AttributeError, OSError):
    _winmm = None

TTS_COOLDOWN_SECONDS = 30


class TTSError(Exception):
    """Raised by an engine when it cannot speak the given text."""


# ---------------------------------------------------------------------------
# Synthesis-only helper (used by the web app: audio is played in the browser)
# ---------------------------------------------------------------------------
def synthesize_mp3(text, lang_code):
    """
    Return the spoken `text` as mp3 bytes using gTTS. Does NOT play anything.

    The web server calls this and streams the bytes to the browser, which plays
    them client-side. Raises TTSError on failure so the caller can decide what
    to do (speech is a bonus and should never crash a request).
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("nothing to speak")
    try:
        from gtts import gTTS
    except Exception as exc:
        raise TTSError(f"gTTS not available ({exc})") from exc
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=(lang_code or "en")).write_to_fp(buf)
        return buf.getvalue()
    except Exception as exc:
        # e.g. no internet, or gTTS doesn't support this language code
        raise TTSError(f"gtts: {exc}") from exc


# ---------------------------------------------------------------------------
# Low-level Windows mp3 playback (blocking) via MCI
# ---------------------------------------------------------------------------
def _mci(command):
    """Send one MCI command string; return the error code (0 == success)."""
    if _winmm is None:
        raise TTSError("mp3 playback is only available on Windows")
    return int(_winmm.mciSendStringW(ctypes.c_wchar_p(command), None, 0, 0))


def _play_mp3_blocking(path, alias="aivt_tts"):
    """Play an mp3 and block until it finishes. Raises TTSError on failure."""
    _mci(f"close {alias}")  # clear any stale handle
    # 'mpegvideo' is the MCI device that plays mp3; fall back to type-less open.
    if _mci(f'open "{path}" type mpegvideo alias {alias}') != 0:
        if _mci(f'open "{path}" alias {alias}') != 0:
            raise TTSError("could not open audio for playback")
    try:
        if _mci(f"play {alias} wait") != 0:
            raise TTSError("audio playback failed")
    finally:
        _mci(f"close {alias}")


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
class TTSEngine:
    name = "base"

    def speak(self, text, tts_code):
        """Speak text in the given language code. Raise TTSError on failure."""
        raise NotImplementedError


class GTTSEngine(TTSEngine):
    name = "gtts"

    def speak(self, text, tts_code):
        try:
            from gtts import gTTS
        except Exception as exc:
            raise TTSError(f"gTTS not available ({exc})") from exc

        path = None
        try:
            tts = gTTS(text=text, lang=(tts_code or "en"))
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="aivt_")
            os.close(fd)
            tts.save(path)             # network call (synthesis) happens here
            _play_mp3_blocking(path)
        except TTSError:
            raise
        except Exception as exc:
            # e.g. no internet, or gTTS doesn't support this language code
            raise TTSError(f"gtts: {exc}") from exc
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


class Pyttsx3Engine(TTSEngine):
    name = "pyttsx3"

    def speak(self, text, tts_code):
        try:
            import pyttsx3
            # A fresh engine per utterance avoids a known Windows SAPI hang when
            # runAndWait() is reused across calls.
            engine = pyttsx3.init()
            # Best effort: pick a voice whose language matches, if any exists.
            self._select_voice(engine, tts_code)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            raise TTSError(f"pyttsx3: {exc}") from exc

    def _select_voice(self, engine, tts_code):
        if not tts_code:
            return
        base = tts_code.split("-")[0].lower()
        try:
            for voice in engine.getProperty("voices"):
                langs = " ".join(str(x).lower() for x in (voice.languages or []))
                if base in langs or base in (voice.id or "").lower() \
                        or base in (voice.name or "").lower():
                    engine.setProperty("voice", voice.id)
                    return
        except Exception:
            pass  # keep default voice


# ---------------------------------------------------------------------------
# Service used by the pipeline
# ---------------------------------------------------------------------------
class TTSService:
    def __init__(self, primary=None):
        primary = (primary or config.TTS_ENGINE or "gtts").lower()
        gtts, offline = GTTSEngine(), Pyttsx3Engine()
        self.engines = [gtts, offline] if primary == "gtts" else [offline, gtts]
        self._cooldown_until = {}
        self._lock = threading.Lock()

    def speak(self, text, tts_code):
        """
        Speak `text`. Returns (ok: bool, info: str). Never raises - speech is a
        bonus feature and must not crash the pipeline.
        """
        text = (text or "").strip()
        if not text:
            return (False, "nothing to speak")

        now = time.monotonic()
        errors = []
        with self._lock:  # serialize playback so clips don't overlap
            for engine in self.engines:
                if self._cooldown_until.get(engine.name, 0) > now:
                    continue
                try:
                    engine.speak(text, tts_code)
                    return (True, f"spoke ({engine.name})")
                except TTSError as exc:
                    errors.append(str(exc))
                    self._cooldown_until[engine.name] = now + TTS_COOLDOWN_SECONDS
        return (False, "; ".join(errors) or "no voice available")


def get_tts_service():
    return TTSService()
