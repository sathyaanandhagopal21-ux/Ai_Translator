"""
config.py
---------
Central configuration for the AI Voice Translator.

Everything that the rest of the app might want to tweak lives here so we never
hard-code magic numbers or language codes deep inside other modules.

IMPORTANT about language codes:
    The three services we use each expect a DIFFERENT style of language code:
        - Speech-to-Text (Google) wants a locale code   ->  "en-US", "ta-IN"
        - Translation (deep-translator/Google) wants ISO ->  "en",    "ta"
        - Text-to-Speech (gTTS) wants a short code       ->  "en",    "ta"
    So each language below stores all three. Other modules just ask this file
    for the right code and never guess.

API keys for optional hosted AI backends are read from environment variables /
a local .env file and are NEVER hard-coded here.
"""

import os
import re


# ---------------------------------------------------------------------------
# Language table
# ---------------------------------------------------------------------------
# Each entry maps a friendly display name to the codes each service needs.
#   name  : shown in the GUI dropdowns
#   stt   : Google Speech-to-Text locale code
#   trans : translation code (ISO 639-1 style used by deep-translator)
#   tts   : gTTS voice code
LANGUAGES = [
    {"name": "English",             "stt": "en-US", "trans": "en",    "tts": "en"},
    {"name": "Tamil",               "stt": "ta-IN", "trans": "ta",    "tts": "ta"},
    {"name": "Hindi",               "stt": "hi-IN", "trans": "hi",    "tts": "hi"},
    {"name": "Telugu",              "stt": "te-IN", "trans": "te",    "tts": "te"},
    {"name": "Kannada",             "stt": "kn-IN", "trans": "kn",    "tts": "kn"},
    {"name": "Malayalam",           "stt": "ml-IN", "trans": "ml",    "tts": "ml"},
    {"name": "French",              "stt": "fr-FR", "trans": "fr",    "tts": "fr"},
    {"name": "Spanish",             "stt": "es-ES", "trans": "es",    "tts": "es"},
    {"name": "German",              "stt": "de-DE", "trans": "de",    "tts": "de"},
    {"name": "Japanese",            "stt": "ja-JP", "trans": "ja",    "tts": "ja"},
    {"name": "Chinese (Simplified)","stt": "zh-CN", "trans": "zh-CN", "tts": "zh-CN"},
    {"name": "Arabic",              "stt": "ar-SA", "trans": "ar",    "tts": "ar"},
    {"name": "Russian",             "stt": "ru-RU", "trans": "ru",    "tts": "ru"},
    {"name": "Portuguese",          "stt": "pt-BR", "trans": "pt",    "tts": "pt"},
]

# Special label for "let the app figure out the spoken language".
# NOTE: The free Google STT needs a language hint, so "Auto" is BEST-EFFORT:
# we transcribe using AUTO_PROBE_LANGUAGE, then double-check the text with a
# language detector. If you already know the source language, selecting it
# explicitly is always more accurate.
SOURCE_AUTO = "Auto (best-effort)"
AUTO_PROBE_LANGUAGE = "English"  # language STT assumes while in Auto mode
# Free Google STT has no true language auto-detect. These optional fallbacks are
# only used when the probe is empty/low-confidence; keep the list short because
# each candidate is an additional public-endpoint request. Explicit source
# selection remains the fastest and most accurate mode.
AUTO_FALLBACK_LANGUAGES = ("Tamil", "Hindi")

# Sensible defaults for the dropdowns.
DEFAULT_SOURCE = "English"
DEFAULT_TARGET = "Tamil"


# ---------------------------------------------------------------------------
# Optional faster/paid-provider backends
# ---------------------------------------------------------------------------
# The default ``auto`` mode uses Groq/Gemini when their keys are present and
# keeps the existing Google/MyMemory services as fallbacks. Explicit provider
# selection is useful for comparing latency and translation quality.
STT_ENGINE = "auto"
TRANSLATION_ENGINE = "auto"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
# Older Gemini Flash model IDs may be unavailable to newer API keys; use the
# current Flash model recommended by the Generative Language API by default.
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_MODEL_ALIASES = {
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-2.5-flash": "gemini-3.6-flash",
}


# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000        # 16 kHz mono is what Google STT likes best
CHANNELS = 1               # mono
BLOCK_DURATION = 0.05      # seconds of audio per block (smaller = snappier VAD)

# Voice Activity Detection (VAD) - simple energy based.
# We start capturing when audio is louder than the threshold and end the
# "utterance" after a short silence, so we translate sentence-by-sentence.
#
# LATENCY BUDGET (why these numbers):
#   Google STT round-trip ~0.5-0.7s + translation ~0.45s = ~1.0-1.2s of
#   unavoidable service time. Everything else is ours to tune. The old
#   8-second force-cut meant continuous speech (a video, a lecture) could play
#   for 8s before a chunk was even sent - ~9s behind. A 4s cut plus interim
#   captions below brings the visible lag to roughly 1-2s, close to YouTube.
VAD_ENERGY_THRESHOLD = 0.010   # 0..1 RMS energy; raise if it triggers on noise
VAD_SILENCE_SECONDS = 0.32     # short gap ends a natural phrase
VAD_MIN_SPEECH_SECONDS = 0.25  # ignore clicks and very short blips
VAD_MAX_UTTERANCE_SECONDS = 4.5  # cap continuous speech for live captions

# --- Live ("interim") captions ---------------------------------------------
# During a long unbroken stretch of speech, translate what we have so far
# every VAD_INTERIM_SECONDS and show it as a provisional caption, which the
# final result then replaces. This is how the app keeps up with a video
# instead of showing nothing for a whole block.
#
# Because interim captions carry the responsiveness, the FINAL chunk is allowed
# to be longer (VAD_MAX_UTTERANCE_SECONDS above). Longer chunks give the speech
# recognizer more context, which it needs to get words right.
#
# Cost: interim chunks are extra STT+translate calls. They only fire when
# someone talks for longer than VAD_INTERIM_SECONDS without pausing, so
# ordinary conversation (which has gaps) pays nothing. Set to 0 to disable.
VAD_INTERIM_SECONDS = 1.8  # one provisional update per live phrase

# When speech must be force-cut, don't slice at an arbitrary instant - that
# lands mid-word and the recognizer mangles both halves. Instead look back over
# this much audio and cut at its quietest point (the gap between two words).
# Whatever follows the cut is carried into the next chunk, so nothing is lost.
VAD_CUT_SEARCH_SECONDS = 1.2

# System/app audio (a video, a call) sits on top of music, effects and room
# tone, so a single fixed threshold either never ends an utterance or chops it
# to pieces. Track the background level and require speech to be this many
# times louder than it. VAD_ENERGY_THRESHOLD stays the absolute floor, so a
# quiet microphone in a quiet room behaves exactly as before.
VAD_NOISE_MULTIPLIER = 2.2

# Media dialogue is often mixed quiet. 16-bit samples that only use a fraction
# of their range transcribe badly, so each chunk is scaled up to this peak
# before being sent. The gain is capped so near-silence isn't blown up into
# noise that the recognizer then "hears" words in.
AUDIO_NORMALIZE_PEAK = 0.85
AUDIO_NORMALIZE_MAX_GAIN = 12.0

# Google returns a confidence score with each transcript. Music, sound effects
# and crowd noise come back as low-confidence nonsense words; dropping those is
# what stops the subtitles filling up with garbage. Lower this if real speech is
# being discarded, raise it if junk still gets through. 0 disables the check.
STT_MIN_CONFIDENCE = 0.45

# The user chose "System / app audio" as the primary source, so default to
# Windows WASAPI loopback (capture whatever is playing on the PC).
USE_LOOPBACK_BY_DEFAULT = True


# ---------------------------------------------------------------------------
# Text-to-Speech settings
# ---------------------------------------------------------------------------
# "gtts"    -> online Google TTS: supports Tamil/Hindi/etc. (needs internet)
# "pyttsx3" -> offline Windows SAPI voices: instant, but many languages
#              (incl. Tamil) usually have NO installed voice.
# The TTS module will fall back to pyttsx3 if gTTS fails, and will simply show
# subtitles (no crash) if neither can speak the chosen language.
TTS_ENGINE = "gtts"
SPEAK_ENABLED_BY_DEFAULT = True

# When capturing system/loopback audio, our own spoken translation would be
# re-captured and re-translated (a feedback loop). So pause capturing while we
# speak. Turn this off only if TTS goes to a different device / headphones.
PAUSE_CAPTURE_WHILE_SPEAKING = True


# ---------------------------------------------------------------------------
# GUI appearance
# ---------------------------------------------------------------------------
APP_TITLE = "AI Voice Translator"
WINDOW_SIZE = "1040x820"

# A refined dark palette. Existing keys (bg/panel/text/muted/accent/start/stop/
# idle) are kept because other modules reference them; the rest are new and used
# by the polished GUI.
COLORS = {
    "bg":        "#0f1220",   # window background (deep navy)
    "panel":     "#171a2b",   # large panels / text boxes
    "card":      "#1f2338",   # raised cards (controls, status)
    "border":    "#2c3150",   # subtle separators / outlines
    "text":      "#f5f7ff",   # normal text
    "muted":     "#8b91b0",   # secondary text
    "accent":    "#7ee7ff",   # subtitle highlight (cyan)
    "accent2":   "#a78bfa",   # secondary accent (violet)
    "start":     "#4ade80",   # green
    "stop":      "#fb7185",   # red / pink
    "idle":      "#5b6488",   # indicator when not listening
    "level_bg":  "#242844",   # audio meter track
}

# A font that ships with Windows and renders Indic scripts (Tamil, Hindi, ...)
# correctly. Falls back gracefully if unavailable.
SUBTITLE_FONT = ("Nirmala UI", 32, "bold")
ORIGINAL_FONT = ("Nirmala UI", 15)
UI_FONT = ("Segoe UI", 10)
TITLE_FONT = ("Segoe UI Semibold", 17)
LABEL_FONT = ("Segoe UI", 9)
STATUS_FONT = ("Segoe UI Semibold", 10)
BADGE_FONT = ("Segoe UI", 9, "bold")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def language_names():
    """All selectable language names (without the Auto option)."""
    return [lang["name"] for lang in LANGUAGES]


def source_language_names():
    """Languages for the SOURCE dropdown (includes the Auto option first)."""
    return [SOURCE_AUTO] + language_names()


def target_language_names():
    """Languages for the TARGET dropdown (no Auto - must pick a real one)."""
    return language_names()


def get_language(name):
    """Return the language dict for a display name, or None if not found."""
    for lang in LANGUAGES:
        if lang["name"] == name:
            return lang
    return None


# ---------------------------------------------------------------------------
# .env / environment loading (no external dependency needed)
# ---------------------------------------------------------------------------
def load_env_file(path=".env"):
    """
    Minimal .env loader using only the standard library.

    Reads KEY=VALUE lines and puts them into os.environ (without overwriting
    variables that are already set). This lets us keep provider API keys out
    of source code. Safe to call even if .env is missing.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                # skip blank lines and comments
                if not line or line.startswith("#"):
                    continue
                # match KEY=VALUE (value may be quoted)
                match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
                if not match:
                    continue
                key, value = match.group(1), match.group(2).strip()
                value = value.strip('"').strip("'")
                os.environ.setdefault(key, value)
    except OSError:
        # If the file can't be read we just proceed without it.
        pass


def get_api_key(name):
    """Fetch an API key from the environment (returns None if not set)."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def get_backend_setting(name, default):
    """Read a provider setting after .env has been loaded."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default
