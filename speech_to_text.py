"""
speech_to_text.py
-----------------
Speech-to-Text (STT): turn a chunk of recorded audio into text.

This is one of the genuinely "AI" parts of the app: GroqSTT sends the audio
to hosted Whisper when configured, with GoogleSTT as the free fallback.

The engine sits behind a small interface (STTEngine) so it can be swapped later
for an offline engine (Vosk / Whisper) or a paid API WITHOUT changing the rest
of the app. The pipeline only ever calls `engine.transcribe(...)`.
"""

import io
import wave

import requests
import speech_recognition as sr
import numpy as np

from concurrent.futures import ThreadPoolExecutor

import config


STT_SAMPLE_RATE = 16000


def resample_pcm16(pcm_bytes, source_rate, target_rate=STT_SAMPLE_RATE):
    """Convert little-endian mono PCM to the rate preferred by Google STT."""
    if not pcm_bytes:
        return b""
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return bytes(pcm_bytes)

    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if samples.size == 0:
        return b""
    output_size = max(1, int(round(samples.size * target_rate / source_rate)))
    if samples.size == 1:
        output = np.repeat(samples, output_size)
    else:
        positions = np.linspace(0, samples.size - 1, output_size)
        output = np.interp(
            positions, np.arange(samples.size), samples.astype(np.float32))
    return np.clip(np.rint(output), -32768, 32767).astype("<i2").tobytes()


class STTError(Exception):
    """Raised for service/network failures (as opposed to empty speech)."""


class STTEngine:
    """Interface every STT backend implements."""

    def transcribe(self, pcm_bytes, sample_rate, language_code):
        """
        Convert 16-bit mono PCM audio to text.

        Returns the transcript string. Returns "" when the audio contained no
        recognizable speech. Raises STTError on network/service failures.
        """
        raise NotImplementedError

    def transcribe_detailed(self, pcm_bytes, sample_rate, language_code):
        """
        Same as transcribe(), but also report how sure the recognizer was.

        Returns (text, confidence) where confidence is a float 0..1 or None if
        the backend doesn't provide one. Backends that can't do better may just
        return (self.transcribe(...), None).
        """
        return self.transcribe(pcm_bytes, sample_rate, language_code), None


class GoogleSTT(STTEngine):
    """
    Free Google Web Speech backend (via the SpeechRecognition library).

    No API key required. Needs internet. Note: the free endpoint expects a
    language hint (e.g. "en-US"), so for the "Auto" source we pass a probe
    language and let the translator/langdetect figure out the real one.

    We ask for the RAW response (show_all=True) rather than just the top string,
    because the raw response carries a confidence score. That score is the only
    signal we get about whether the audio really was speech: music, sound
    effects and background chatter still come back as words, but with low
    confidence. Filtering on it is what keeps junk out of the subtitles when
    listening to system audio.
    """

    # 16-bit audio => 2 bytes per sample. Capture devices may run at 44.1/48 kHz,
    # but the request is normalized to 16 kHz in transcribe_detailed for a
    # smaller payload and more consistent recognizer behavior.
    SAMPLE_WIDTH = 2

    def __init__(self):
        self._recognizer = sr.Recognizer()

    def transcribe_candidates(self, pcm_bytes, sample_rate, language_codes,
                              min_confidence=None):
        """Try a small configured set of locales concurrently for Auto mode."""
        codes = list(dict.fromkeys(code for code in (language_codes or []) if code))
        if not codes:
            return "", None
        if len(codes) == 1:
            return self.transcribe_detailed(pcm_bytes, sample_rate, codes[0],
                                            min_confidence=min_confidence)

        # The probe is the common path. Only spend extra requests when it was
        # empty or below the confidence bar, keeping normal Auto latency low.
        try:
            first = self.transcribe_detailed(
                pcm_bytes, sample_rate, codes[0],
                min_confidence=min_confidence)
        except STTError:
            first = ("", None)
        if first[0]:
            return first

        def run(code):
            try:
                return self.transcribe_detailed(
                    pcm_bytes, sample_rate, code, min_confidence=min_confidence)
            except STTError:
                return "", None

        with ThreadPoolExecutor(max_workers=len(codes) - 1) as pool:
            results = [first] + list(pool.map(run, codes[1:]))
        valid = [(text, confidence) for text, confidence in results if text]
        if not valid:
            confidences = [confidence for _text, confidence in results
                           if confidence is not None]
            return "", max(confidences) if confidences else None
        return max(valid, key=lambda item: item[1] if item[1] is not None else 0.0)

    def transcribe(self, pcm_bytes, sample_rate, language_code):
        text, _confidence = self.transcribe_detailed(
            pcm_bytes, sample_rate, language_code)
        return text

    def transcribe_detailed(self, pcm_bytes, sample_rate, language_code,
                            min_confidence=None):
        """
        Transcribe and return (text, confidence).

        Results below `min_confidence` are treated as "not speech" and come back
        as ("", confidence) so the caller can show/log the score but not the
        text. Defaults to config.STT_MIN_CONFIDENCE; pass 0 to keep everything.
        """
        if not pcm_bytes:
            return "", None
        if min_confidence is None:
            min_confidence = getattr(config, "STT_MIN_CONFIDENCE", 0.0)

        try:
            pcm_bytes = resample_pcm16(pcm_bytes, sample_rate)
        except (TypeError, ValueError) as exc:
            raise STTError(f"invalid audio sample rate ({exc})") from exc

        audio = sr.AudioData(pcm_bytes, STT_SAMPLE_RATE, self.SAMPLE_WIDTH)
        try:
            raw = self._recognizer.recognize_google(
                audio, language=language_code or "en-US", show_all=True)
        except sr.UnknownValueError:
            # Audio was captured but no speech could be recognized -> empty.
            return "", None
        except sr.RequestError as exc:
            # Could not reach the service / quota / bad response.
            raise STTError(f"speech service unavailable ({exc})") from exc
        except Exception as exc:  # defensive: never crash the pipeline
            raise STTError(f"speech recognition failed ({exc})") from exc

        return self._pick_best(raw, min_confidence)

    @staticmethod
    def _pick_best(raw, min_confidence):
        """
        Turn Google's raw response into (text, confidence).

        With show_all=True the library hands back whatever the service sent:
            {"alternative": [{"transcript": "...", "confidence": 0.91}, ...],
             "final": true}
        Silence / unrecognizable audio gives an empty list or dict instead.
        Google only attaches `confidence` to some alternatives, so we prefer the
        best scored one and fall back to the first (which is Google's own pick).
        """
        if not raw:
            return "", None
        alternatives = raw.get("alternative") if isinstance(raw, dict) else None
        if not alternatives:
            return "", None

        scored = [a for a in alternatives if isinstance(a, dict)
                  and a.get("confidence") is not None]
        best = max(scored, key=lambda a: a["confidence"]) if scored \
            else alternatives[0]

        text = str(best.get("transcript") or "").strip()
        try:
            confidence = float(best["confidence"]) if "confidence" in best \
                else None
        except (TypeError, ValueError):
            confidence = None

        # Unscored results are kept: no score is not the same as a bad score.
        if text and confidence is not None and min_confidence \
                and confidence < min_confidence:
            return "", confidence
        return text, confidence


class GroqSTT(STTEngine):
    """Fast Whisper transcription through Groq's OpenAI-compatible API."""

    URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    TIMEOUT = 20
    SAMPLE_WIDTH = 2

    def __init__(self, api_key=None, model=None, session=None):
        self.api_key = api_key or config.get_api_key("GROQ_API_KEY")
        self.model = model or config.get_backend_setting(
            "GROQ_STT_MODEL", config.GROQ_STT_MODEL)
        self._session = session or requests.Session()

    @staticmethod
    def _locale_to_iso(language_code):
        """Groq Whisper wants a short ISO code rather than a regional locale."""
        return (language_code or "").split("-")[0].split("_")[0].lower() or None

    @staticmethod
    def _wav_bytes(pcm_bytes, sample_rate):
        """Wrap mono little-endian PCM in a WAV container for the upload API."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(GroqSTT.SAMPLE_WIDTH)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm_bytes)
        return buf.getvalue()

    def transcribe_candidates(self, pcm_bytes, sample_rate, language_codes,
                               min_confidence=None):
        """Let Whisper detect the language instead of probing several locales."""
        return self.transcribe_detailed(pcm_bytes, sample_rate, None,
                                        min_confidence=min_confidence)

    def transcribe(self, pcm_bytes, sample_rate, language_code):
        text, _confidence = self.transcribe_detailed(
            pcm_bytes, sample_rate, language_code)
        return text

    def transcribe_detailed(self, pcm_bytes, sample_rate, language_code,
                            min_confidence=None):
        """Upload a WAV conversion and return (transcript, None)."""
        if not pcm_bytes:
            return "", None
        if not self.api_key:
            raise STTError("Groq API key is not configured")
        try:
            pcm_bytes = resample_pcm16(pcm_bytes, sample_rate)
            wav_data = self._wav_bytes(pcm_bytes, STT_SAMPLE_RATE)
        except (TypeError, ValueError, wave.Error) as exc:
            raise STTError(f"invalid audio ({exc})") from exc

        data = {
            "model": self.model,
            "response_format": "json",
        }
        language = self._locale_to_iso(language_code)
        if language:
            data["language"] = language
        try:
            response = self._session.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                data=data,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            text = str(payload.get("text") or "").strip() \
                if isinstance(payload, dict) else ""
            return text, None
        except requests.RequestException as exc:
            raise STTError(f"Groq speech service unavailable ({exc})") from exc
        except (ValueError, TypeError, AttributeError) as exc:
            raise STTError(f"Groq returned an invalid response ({exc})") from exc
        except Exception as exc:
            raise STTError(f"Groq speech recognition failed ({exc})") from exc


class FallbackSTT(STTEngine):
    """Try STT engines in order, falling back only when one raises STTError."""

    def __init__(self, engines):
        self.engines = list(engines)

    def transcribe(self, pcm_bytes, sample_rate, language_code):
        text, _confidence = self.transcribe_detailed(
            pcm_bytes, sample_rate, language_code)
        return text

    @staticmethod
    def _best_empty(results):
        """Return the empty result with the strongest available confidence."""
        return max(results, key=lambda item: item[1] if item[1] is not None else 0.0) \
            if results else ("", None)

    def transcribe_detailed(self, pcm_bytes, sample_rate, language_code,
                            min_confidence=None):
        errors = []
        empty_results = []
        for engine in self.engines:
            try:
                result = engine.transcribe_detailed(
                    pcm_bytes, sample_rate, language_code,
                    min_confidence=min_confidence)
                if result[0]:
                    return result
                # An empty result means "no speech", not a provider failure;
                # still try the next backend in case this one missed the audio.
                empty_results.append(result)
            except STTError as exc:
                errors.append(str(exc))
        if empty_results:
            return self._best_empty(empty_results)
        raise STTError("; ".join(errors) or "no speech engine available")

    def transcribe_candidates(self, pcm_bytes, sample_rate, language_codes,
                              min_confidence=None):
        errors = []
        empty_results = []
        for engine in self.engines:
            try:
                method = getattr(engine, "transcribe_candidates", None)
                result = method(
                    pcm_bytes, sample_rate, language_codes,
                    min_confidence=min_confidence) if method else \
                    engine.transcribe_detailed(
                        pcm_bytes, sample_rate,
                        (language_codes or [None])[0],
                        min_confidence=min_confidence)
                if result[0]:
                    return result
                empty_results.append(result)
            except STTError as exc:
                errors.append(str(exc))
        if empty_results:
            return self._best_empty(empty_results)
        raise STTError("; ".join(errors) or "no speech engine available")


# --- factory ---------------------------------------------------------------
def get_stt_engine(name=None):
    """Return a configured STT engine, using Groq with Google fallback in auto mode."""
    name = config.get_backend_setting("STT_ENGINE", config.STT_ENGINE) \
        if name is None else name
    name = (name or "auto").lower()
    if name == "google":
        return GoogleSTT()
    if name == "groq":
        return GroqSTT()
    if name in ("auto", "default"):
        if config.get_api_key("GROQ_API_KEY"):
            return FallbackSTT([GroqSTT(), GoogleSTT()])
        return GoogleSTT()
    raise ValueError(f"Unknown STT engine: {name!r}")


def stt_language_code(source_name):
    """
    Map a GUI source-language selection to the STT locale code.

    For "Auto (best-effort)" we fall back to the configured probe language,
    because the free Google STT needs an explicit language hint.
    """
    if source_name == config.SOURCE_AUTO:
        probe = config.get_language(config.AUTO_PROBE_LANGUAGE)
        return probe["stt"] if probe else "en-US"
    lang = config.get_language(source_name)
    return lang["stt"] if lang else "en-US"
