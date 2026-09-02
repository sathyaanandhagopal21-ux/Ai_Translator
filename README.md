# AI Voice Translator

A near-real-time voice interpreter in Python:

> **Voice → Speech-to-Text → Language Detection → Translation → Subtitle → Translated Voice**

It listens to an audio source, transcribes speech, detects the language,
translates it, shows a large live subtitle, and speaks the translation aloud.

---

## 1. What it can (and can't) do

**Can (on a normal Windows PC, in Python):**
- Capture your **microphone**.
- Capture **system / app audio** — whatever is playing on your speakers — via
  WASAPI loopback (no virtual cable or "Stereo Mix" needed).
- Neural **speech-to-text**, **language detection**, and neural **translation**.
- Large live **subtitles** + **spoken** translation (Tamil, Hindi, etc.).

**Can't (honest limitations):**
- It **cannot reach inside another app's private audio** or your phone directly.
  The operating system doesn't expose that. Audio must arrive at an **input
  device** first (see §4).
- It is **near-real-time**, not word-by-word simultaneous. It shows a provisional
  caption while a long phrase is still being spoken, then replaces it with the
  final translation after a short pause. On a healthy connection, expect roughly
  **1-3 seconds** for a final short phrase (mostly network); public-service load
  can make this longer.
- No speech recognizer or machine translator can guarantee literal **100%
  accuracy**. For the best result with this free backend, select the spoken
  source language explicitly, use a clean audio input, and avoid overlapping
  speakers/music. `Auto (best-effort)` uses a bounded language fallback and is
  inherently less reliable.
- The default backend uses **free public endpoints** that can occasionally
  rate-limit or change. The app fails over between two of them, but for
  production you'd switch to an official paid API (the code is built for that).

---

## 2. Install

Requires **Python 3.10+** on Windows.

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
python main.py
```

Then: pick **Audio input**, choose **From / To** languages, click **Start
Translation**. Speak (or play audio into the selected device). Subtitles appear
and the translation is spoken. **Stop** ends it; **Clear Subtitles** resets.

> Self-test (builds the UI and exits, useful for a quick check):
> ```bash
> python main.py --selftest
> ```

---

## 4. Getting audio IN (mic / system / phone)

The app lists every input device the OS exposes. Choose the one that matches
your source:

| Source | How | Pick in the app |
|---|---|---|
| **Someone near the PC** | Just talk | `Microphone: ...` |
| **A video call / video on this PC** | Nothing to set up | `System audio: ...` (loopback) |
| **Phone audio** | See below | whichever device the route creates |

**Routing phone audio to the PC** (Python can't grab it directly):
1. **Cable** — phone headphone-out → PC line-in/mic-in. Pick that mic.
2. **Wi-Fi streaming app** (e.g. AudioRelay / SoundWire): stream the phone to
   the PC; it creates a playback device — then pick `System audio` (loopback).
3. **Speaker → mic** — play the phone out loud; pick the `Microphone`. (Lowest
   quality; picks up room noise.)

> **Feedback loop warning:** when capturing **System audio** and speaking the
> translation through the **same speakers**, the app would hear its own voice.
> It automatically **pauses capture while speaking**. For best results, send the
> spoken output to **headphones** or a different device, or untick **🔊 Speak
> translation** and use subtitles only.

---

## 5. Architecture

```
main.py            entry point; wires GUI <-> pipeline, handles Start/Stop/Clear
config.py          settings, language table (STT/translate/TTS codes), .env loader
gui.py             Tkinter UI; thread-safe updates via a queue + root.after()
audio_input.py     PyAudioWPatch capture (mic + loopback) + energy-based VAD
speech_to_text.py  STTEngine interface + Groq Whisper / Google fallback
translator.py      TranslationService (Gemini -> Google -> MyMemory) + detection
text_to_speech.py  TTSEngine interface + gTTS (MCI playback) + pyttsx3 fallback
subtitle_manager.py subtitle history (ready for CSV export via pandas)
pipeline.py        threads + queues connecting capture -> STT -> translate -> TTS
```

**Threads & queues** (the GUI never blocks):
- Capture thread → detects an utterance → `audio_queue`
- STT+Translate worker → transcribe → detect → translate → update subtitle →
  `tts_queue`
- TTS worker → pause capture → speak → resume

### Where the "AI" is
- **Speech recognition** (hosted Whisper via Groq when configured, otherwise
  Google's neural recognizer).
- **Language detection** (statistical classifier, `langdetect`).
- **Translation** (Gemini when configured, then Google/MyMemory neural
  translation — grammar/word-order aware, not a dictionary).

---

## 6. Swapping backends (it's modular)

Each stage sits behind a small interface, so you can add an engine without
touching the rest of the app:

- **Translation** — add a class implementing `TranslationEngine.translate(...)`
  in `translator.py` (e.g. OpenAI/DeepL/Google Cloud), then add it to
  `TranslationService`.
- **Speech-to-text** — add an `STTEngine` in `speech_to_text.py`
  (e.g. Vosk/Whisper for offline).
- **Text-to-speech** — add a `TTSEngine` in `text_to_speech.py`.

**API keys** go in a `.env` file (copy `.env.example`), read by
`config.load_env_file()`. **Never hard-code keys** in source.

---

## 7. Optional Groq + Gemini backends

The app can use faster/context-aware hosted backends while keeping the original
free services as fallbacks. The calls use the already-installed `requests`
package, so no extra SDK is required.

Copy `.env.example` to `.env` and add keys locally (never paste keys into source
control or chat):

```env
GROQ_API_KEY=your-groq-key
GROQ_STT_MODEL=whisper-large-v3-turbo
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.6-flash
STT_ENGINE=auto
TRANSLATION_ENGINE=auto
```

In `auto` mode, the desktop and web apps use this order when keys are present:

```text
Speech:      Groq Whisper -> Google Web Speech
Translation: Gemini       -> Google fast endpoint -> MyMemory
```

`GEMINI_MODEL` can be changed to `gemini-3.5-flash-lite` for a lighter/faster
comparison. Older `gemini-2.0-flash` and `gemini-2.5-flash` values are
automatically mapped to the current `gemini-3.6-flash` model. `STT_ENGINE=google` or `TRANSLATION_ENGINE=google` returns to the
original Google-only path. Explicit `groq` or `gemini` selection is also
available, but `auto` is recommended because it preserves fallback behavior.

Groq is an inference platform rather than a model; this project uses its hosted
Whisper model for speech recognition. Gemini is used only for translation. Free
API quotas, model availability, rate limits, and latency can change, and no
speech or translation model can guarantee 100% accuracy. Clear audio and an
explicit source language still produce the best results.

---

## 8. Dependencies (why each is needed)

| Library | Role | Notes |
|---|---|---|
| `PyAudioWPatch` | Audio capture (mic + **loopback**) | The only way to grab system audio in pure Python on Windows |
| `numpy` | Downmix to mono + RMS energy for VAD | You already know it |
| `SpeechRecognition` | Speech → text (Google, no key) | Bundles a FLAC encoder; no PyAudio needed |
| `deep-translator` | Neural translation (Google + MyMemory) | No key; swappable |
| `gTTS` | Multilingual voice output | Plays via Windows MCI (ctypes) — no extra playback lib |
| `pyttsx3` | Offline voice fallback | Uses Windows SAPI voices |
| `langdetect` | "Detected language" display + Auto mode | Lightweight |

Standard-library modules used throughout: `tkinter`, `threading`, `queue`,
`time`, `os`, `sys`, `ctypes`, `tempfile`, `re`, `json`.

---

## 8. Troubleshooting

- **No `System audio` device listed** → make sure something has played on the
  default speakers at least once; click **⟳ Refresh**.
- **Subtitles stay blank** → check the audio level indicator turns to *"Hearing
  speech…"*; raise the source volume, or lower `VAD_ENERGY_THRESHOLD` in
  `config.py`.
- **"translation error" / "speech service error"** → usually a network blip;
  the app retries on the next sentence and fails over between backends.
- **No Tamil/Hindi voice offline** → that's expected for `pyttsx3`; the default
  `gTTS` handles those (needs internet).
- **It translated its own voice** → keep **Speak translation** with capture-
  pause on (default), or use headphones for output.
