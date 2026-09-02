"""
webapp.py
---------
Flask web version of the AI Voice Translator (browser microphone edition).

Unlike the desktop app (main.py + gui.py + tkinter), here the BROWSER captures
the microphone with the Web Audio API, does the live level meter + voice-
activity detection (VAD) in JavaScript, and sends one finished utterance at a
time to this server. The server reuses the exact same AI modules as the desktop
app:

    POST /api/translate   raw 16-bit PCM  -> STT -> detect -> translate -> JSON
    POST /api/tts         {text, target}  -> gTTS mp3 bytes (played in browser)
    POST /api/clear       reset the server-side transcript history
    GET  /api/export.csv  download the transcript (pandas)
    GET  /                the single-page UI

Honest limitation: a browser can capture only the MICROPHONE (a security
boundary of the web platform), NOT system/app/phone audio. For that, use the
desktop version, which captures system/loopback audio via Python.

Provider keys are never hard-coded. config.load_env_file() is called at startup
so the optional Groq/Gemini backends can read keys from a local .env file.

Run:
    python webapp.py                 # then open http://127.0.0.1:5000
    HOST=0.0.0.0 PORT=8000 python webapp.py
"""

import os
import time

from flask import Flask, Response, jsonify, render_template, request

import config
from speech_to_text import get_stt_engine, stt_language_code, STTError
from translator import get_translation_service, detect_language, TranslationError
from text_to_speech import synthesize_mp3, TTSError
from subtitle_manager import SubtitleManager

# Load optional Groq/Gemini provider keys from .env. Harmless if absent.
config.load_env_file()

app = Flask(__name__)

# Shared services built once (none of these need the audio libraries, so the
# web app runs even on a machine without PyAudioWPatch installed).
_stt = get_stt_engine()
_translation = get_translation_service()
_subtitles = SubtitleManager()

# NOTE: chunks are deliberately NOT serialized. Each request does ~1s of
# network I/O (STT ~0.6s + translate ~0.45s), and the browser sends a chunk
# every ~1.5-4s. A global lock made chunks queue behind each other, so during
# continuous speech the captions fell further behind with every chunk instead
# of holding a steady lag. Running them concurrently is safe here:
#   * recognize_google() keeps no state on the recognizer between calls,
#   * requests.Session (used by the translator) is thread-safe for GETs,
#   * SubtitleManager has its own internal lock.
# Flask runs with threaded=True, so each chunk gets its own thread.


def _frontend_config():
    """The subset of config the browser needs (languages, VAD, defaults)."""
    return {
        "sourceLanguages": config.source_language_names(),
        "targetLanguages": config.target_language_names(),
        "defaultSource": config.DEFAULT_SOURCE,
        "defaultTarget": config.DEFAULT_TARGET,
        "autoLabel": config.SOURCE_AUTO,
        "speakDefault": config.SPEAK_ENABLED_BY_DEFAULT,
        "sampleRate": config.SAMPLE_RATE,
        "vad": {
            "energyThreshold": config.VAD_ENERGY_THRESHOLD,
            "silenceSeconds": config.VAD_SILENCE_SECONDS,
            "minSpeechSeconds": config.VAD_MIN_SPEECH_SECONDS,
            "maxUtteranceSeconds": config.VAD_MAX_UTTERANCE_SECONDS,
            "interimSeconds": config.VAD_INTERIM_SECONDS,
            "cutSearchSeconds": config.VAD_CUT_SEARCH_SECONDS,
            "noiseMultiplier": config.VAD_NOISE_MULTIPLIER,
        },
        "audio": {
            "normalizePeak": config.AUDIO_NORMALIZE_PEAK,
            "normalizeMaxGain": config.AUDIO_NORMALIZE_MAX_GAIN,
        },
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html",
                           app_title=config.APP_TITLE,
                           config_json=_frontend_config())


# ---------------------------------------------------------------------------
# API: one utterance -> transcript + translation
# ---------------------------------------------------------------------------
@app.post("/api/translate")
def api_translate():
    """
    Body: raw 16-bit little-endian mono PCM (one chunk of speech).
    Query:
        source, target  friendly language names
        rate            sample rate of the PCM, in Hz
        uid, part       chunk identity, echoed back so the browser can discard
                        results that arrive out of order (chunks run in parallel)
        interim         "1" for a provisional mid-speech chunk: it is displayed
                        but NOT written to the transcript, and language
                        detection is skipped to save time
    Returns JSON {original, detected, translated, empty, confidence,
    uid, part, ms} or {error}.
    """
    pcm = request.get_data(cache=False)
    started = time.monotonic()

    # Echoed back unchanged so the client can order/discard responses.
    uid = request.args.get("uid", "0")
    part = request.args.get("part", "0")
    interim = request.args.get("interim") == "1"

    def reply(payload, status=200):
        payload.update(uid=uid, part=part, interim=interim,
                       ms=int((time.monotonic() - started) * 1000))
        return jsonify(payload), status

    if not pcm:
        return reply({"error": "no audio received"}, 400)

    try:
        rate = int(request.args.get("rate", config.SAMPLE_RATE))
    except (TypeError, ValueError):
        rate = config.SAMPLE_RATE
    source = request.args.get("source", config.DEFAULT_SOURCE)
    target = request.args.get("target", config.DEFAULT_TARGET)

    if config.get_language(target) is None:
        return reply({"error": f"unsupported target language: {target}"}, 400)

    # ---- Speech to text ----
    # transcribe_detailed also reports Google's confidence, and returns no text
    # at all when that confidence is below config.STT_MIN_CONFIDENCE. That is
    # how music / sound effects / background noise from system audio are kept
    # out of the subtitles instead of appearing as invented words.
    try:
        if source == config.SOURCE_AUTO and not interim and \
                hasattr(_stt, "transcribe_candidates"):
            names = [config.AUTO_PROBE_LANGUAGE] + list(
                getattr(config, "AUTO_FALLBACK_LANGUAGES", ()))
            codes = [stt_language_code(name) for name in names]
            text, confidence = _stt.transcribe_candidates(
                pcm, rate, codes,
                min_confidence=getattr(config, "STT_MIN_CONFIDENCE", 0.0))
        else:
            text, confidence = _stt.transcribe_detailed(
                pcm, rate, stt_language_code(source),
                min_confidence=getattr(config, "STT_MIN_CONFIDENCE", 0.0))
    except STTError as exc:
        return reply({"error": f"speech-to-text: {exc}"}, 502)

    if not text:
        # Captured audio but no confident speech in it -> keep listening.
        # `confidence` is still reported so a rejected guess is visible in the
        # UI rather than looking like the app simply missed the audio.
        return reply({"original": "", "detected": "", "translated": "",
                      "empty": True, "confidence": confidence,
                      "rejected": confidence is not None})

    # ---- Language detection (for the "Detected" pill) ----
    # Only worth doing when we don't already know the language, and never on an
    # interim chunk - the final chunk will report it a moment later anyway.
    if interim or source != config.SOURCE_AUTO:
        detected = "" if interim else source
    else:
        _code, detected = detect_language(text)

    # ---- Translation ----
    try:
        translated = _translation.translate(text, source, target)
    except TranslationError as exc:
        return reply({"original": text, "detected": detected,
                      "confidence": confidence,
                      "error": f"translation: {exc}"}, 502)

    if not translated:
        return reply({"original": text, "detected": detected,
                      "confidence": confidence,
                      "error": "no translation available"}, 502)

    # Provisional chunks are display-only; only finals enter the transcript.
    # `started` is the moment this chunk was received, which IS speech order -
    # chunks are sent in order but, running in parallel, can finish out of it.
    if not interim:
        _subtitles.add(text, translated, detected, seq=started)

    return reply({"original": text, "detected": detected,
                  "translated": translated, "empty": False,
                  "confidence": confidence})


# ---------------------------------------------------------------------------
# API: synthesize speech for the browser to play
# ---------------------------------------------------------------------------
@app.post("/api/tts")
def api_tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    target = data.get("target", config.DEFAULT_TARGET)
    if not text:
        return jsonify(error="no text to speak"), 400

    lang = config.get_language(target)
    tts_code = lang["tts"] if lang else "en"
    try:
        mp3 = synthesize_mp3(text, tts_code)
    except TTSError as exc:
        # Speech is a bonus; report politely so the browser can just skip it.
        return jsonify(error=str(exc)), 502
    return Response(mp3, mimetype="audio/mpeg")


# ---------------------------------------------------------------------------
# API: transcript history (clear + CSV export)
# ---------------------------------------------------------------------------
@app.post("/api/clear")
def api_clear():
    _subtitles.clear()
    return jsonify(ok=True)


@app.get("/api/history")
def api_history():
    return jsonify(history=_subtitles.history())


@app.get("/api/export.csv")
def api_export_csv():
    """Download the session transcript as CSV (built with pandas)."""
    rows = _subtitles.to_rows()
    try:
        import pandas as pd
        df = pd.DataFrame(
            rows, columns=["time", "detected", "original", "translated"])
        csv = df.to_csv(index=False)
    except Exception:
        # Fallback if pandas is unavailable: build CSV with the stdlib.
        import csv as _csv
        import io
        buf = io.StringIO()
        writer = _csv.writer(buf)
        writer.writerow(["time", "detected", "original", "translated"])
        writer.writerows(rows)
        csv = buf.getvalue()
    return Response(
        csv, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=subtitles.csv"})


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print("=" * 60)
    print("  AI Voice Translator (web) is starting…")
    print(f"  Open this in your browser:  {url}")
    print("  Microphone access requires localhost or HTTPS.")
    print("  Press CTRL+C to stop.")
    print("=" * 60)
    # threaded=True so the level/translate requests don't block each other.
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
