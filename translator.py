"""
translator.py
-------------
Translation service + language detection.

This is a genuinely "AI" part of the app: the backends use neural machine
translation (grammar- and context-aware), not word-by-word dictionary swaps.

Design:
    * TranslationEngine   - interface each backend implements.
    * GeminiTranslate     - context-aware translation through Google's
                            Gemini generateContent API when configured.
    * GoogleFastTranslate - Google's free neural MT via a fast JSON endpoint
                            (~0.4-0.6s; supports source='auto').
    * MyMemoryTranslate   - a stable JSON translation API used as a fallback
                            (needs an explicit source language; region codes).
    * TranslationService  - tries configured backends in order and returns the
                            first success, with a short cooldown on a failing
                            backend so we don't waste time re-hitting it.

Free public endpoints occasionally break or rate-limit; the fallback chain is
what keeps the app working when one of them does.
"""

import time
import threading
import unicodedata
from collections import OrderedDict

import requests
from deep_translator import MyMemoryTranslator

from langdetect import detect as _ld_detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0  # make detection deterministic for short text

import config


# Skip a backend for this many seconds after it fails, so a broken/rate-limited
# endpoint doesn't add a wasted network round-trip to every single utterance.
BACKEND_COOLDOWN_SECONDS = 30
CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 512


class TranslationError(Exception):
    """Raised when every translation backend fails."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class TranslationEngine:
    """Interface every translation backend implements."""

    # Which code style this backend wants from the language table:
    #   'iso'    -> use the "trans"  code (e.g. "en", "ta")
    #   'locale' -> use the "stt"    code (e.g. "en-US", "ta-IN")
    code_kind = "iso"
    supports_auto = False          # can it auto-detect the source itself?
    name = "base"

    def translate(self, text, source_code, target_code):
        raise NotImplementedError


class GoogleFastTranslate(TranslationEngine):
    """
    Google's free neural MT via a lightweight JSON endpoint.

    We hit the `dict-chrome-ex` client endpoint directly with `requests`
    instead of scraping the web page. It answers in ~0.4-0.6s (about 4x
    faster than the fallback) and returns clean JSON, which is what makes
    the app feel closer to live. Supports source='auto'.
    """

    code_kind = "iso"
    supports_auto = True
    name = "google"

    # Primary endpoint returns a plain ["translated text"] array.
    URL = "https://clients5.google.com/translate_a/t"
    # Fallback endpoint (same data, nested array format) if the first errors.
    URL_ALT = "https://translate.googleapis.com/translate_a/single"
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    TIMEOUT = 6

    def __init__(self):
        # One pooled Session, reused for every call. Keep-alive skips the TLS
        # handshake on later requests: measured ~0.95s cold vs ~0.45s warm, so
        # this removes about half a second from all but the first translation.
        # requests.Session is safe to share across threads for plain GETs.
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    def translate(self, text, source_code, target_code):
        sl = source_code or "auto"
        try:
            resp = self._session.get(
                self.URL, timeout=self.TIMEOUT,
                params={"client": "dict-chrome-ex", "sl": sl,
                        "tl": target_code, "q": text})
            resp.raise_for_status()
            out = self._parse(resp.json())
            if out:
                return out
            # Empty parse -> try the alternate endpoint before giving up.
            return self._translate_alt(text, sl, target_code)
        except TranslationError:
            raise
        except Exception:
            # Network / JSON / HTTP error on the primary: try the alternate.
            return self._translate_alt(text, sl, target_code)

    def _translate_alt(self, text, sl, target_code):
        try:
            resp = self._session.get(
                self.URL_ALT, timeout=self.TIMEOUT,
                params={"client": "dict-chrome-ex", "sl": sl, "tl": target_code,
                        "dt": "t", "q": text})
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as exc:
            raise TranslationError(f"google: {exc}") from exc

    @staticmethod
    def _parse(data):
        """Handle both the flat (['text']) and nested ([[['text',...]]]) shapes."""
        if not isinstance(data, list) or not data:
            return ""
        # Flat form from clients5/dict-chrome-ex: ["translated"] (maybe split).
        if isinstance(data[0], str):
            return "".join(p for p in data if isinstance(p, str)).strip()
        # Nested form from translate_a/single: [[["translated","orig",...]], ...]
        if isinstance(data[0], list):
            parts = []
            for chunk in data[0]:
                if isinstance(chunk, list) and chunk and isinstance(chunk[0], str):
                    parts.append(chunk[0])
            return "".join(parts).strip()
        return ""


class MyMemoryTranslate(TranslationEngine):
    code_kind = "locale"
    supports_auto = False          # must be given an explicit source language
    name = "mymemory"

    def translate(self, text, source_code, target_code):
        try:
            result = MyMemoryTranslator(
                source=source_code, target=target_code).translate(text)
            return (result or "").strip()
        except Exception as exc:
            raise TranslationError(f"mymemory: {exc}") from exc


class GeminiTranslate(TranslationEngine):
    """Context-aware translation through Google's Gemini generateContent API."""

    code_kind = "iso"
    supports_auto = True
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    TIMEOUT = 15

    def __init__(self, api_key=None, model=None, session=None):
        self.api_key = api_key or config.get_api_key("GEMINI_API_KEY")
        configured_model = model or config.get_backend_setting(
            "GEMINI_MODEL", config.GEMINI_MODEL)
        # Keep older .env files working when their model IDs are unavailable.
        self.model = getattr(config, "GEMINI_MODEL_ALIASES", {}).get(
            configured_model, configured_model)
        self._session = session or requests.Session()

    @staticmethod
    def _prompt(text, source_code, target_code):
        source = source_code or "auto-detect"
        return (
            "Translate the text below from the source language to the target "
            "language. Return only the translation, with no explanation, labels, "
            "quotes, or transliteration. Preserve meaning, names, numbers, and "
            "punctuation.\n\n"
            f"Source language/code: {source}\n"
            f"Target language/code: {target_code}\n"
            "Text to translate:\n<<<\n"
            f"{text}\n"
            ">>>"
        )

    @staticmethod
    def _parse_response(payload):
        """Extract text from Gemini's candidate/content/parts response shape."""
        if not isinstance(payload, dict):
            return ""
        candidates = payload.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or [] if isinstance(content, dict) else []
        return "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ).strip()

    def translate(self, text, source_code, target_code):
        if not self.api_key:
            raise TranslationError("Gemini API key is not configured")
        url = self.URL.format(model=self.model)
        body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": self._prompt(
                    text, source_code, target_code)}],
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            },
        }
        try:
            response = self._session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json=body,
                timeout=self.TIMEOUT,
            )
            if not 200 <= response.status_code < 300:
                raise TranslationError(
                    f"gemini: service returned HTTP {response.status_code}")
            result = self._parse_response(response.json())
            if not result:
                raise TranslationError("gemini: empty or invalid response")
            return result
        except TranslationError:
            raise
        except requests.RequestException as exc:
            # Do not include the URL here because the API key is in its query.
            raise TranslationError(f"gemini: request failed ({exc})") from exc
        except (TypeError, ValueError, AttributeError) as exc:
            raise TranslationError(f"gemini: invalid response ({exc})") from exc
        except Exception as exc:
            raise TranslationError(f"gemini: request failed ({exc})") from exc


# ---------------------------------------------------------------------------
# Language detection (for the "Detected: <language>" display and Auto mode)
# ---------------------------------------------------------------------------
_CODE_TO_NAME = {lang["trans"].lower(): lang["name"] for lang in config.LANGUAGES}

# langdetect uses a few aliases or regional variants that are not represented
# exactly in our language table. Resolve them before choosing a translation code.
_DETECT_CODE_ALIASES = {
    "zh": "zh-cn",
    "zh-hans": "zh-cn",
    "zh-sg": "zh-cn",
    "zh-tw": "zh-cn",  # Simplified Chinese is the only Chinese option here.
}

# Distinctive Unicode ranges let short Indic/CJK utterances bypass the
# statistical detector, which is often unreliable when there are only a few
# words. Each range is deliberately limited to languages supported by the UI.
_SCRIPT_RANGES = {
    "ta": ((0x0B80, 0x0BFF),),       # Tamil
    "hi": ((0x0900, 0x097F),),       # Devanagari (Hindi)
    "te": ((0x0C00, 0x0C7F),),       # Telugu
    "kn": ((0x0C80, 0x0CFF),),       # Kannada
    "ml": ((0x0D00, 0x0D7F),),       # Malayalam
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "ru": ((0x0400, 0x04FF),),       # Cyrillic (Russian in this UI)
    "ja": ((0x3040, 0x30FF),),       # Hiragana + Katakana
    "zh-cn": ((0x4E00, 0x9FFF),),    # Han characters (when no kana exists)
}


def normalize_translation_text(text):
    """Make recognizer output stable without changing its linguistic content."""
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.split())


def _friendly_language_name(code):
    """Return the configured display name for a detector code, if known."""
    normalized = (code or "").lower().replace("_", "-")
    normalized = _DETECT_CODE_ALIASES.get(normalized, normalized)
    return _CODE_TO_NAME.get(normalized, code or "unknown")


def _script_language_code(text):
    """Return a confident supported-script code, or ``None`` for Latin text."""
    counts = {code: 0 for code in _SCRIPT_RANGES}
    letters = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        point = ord(char)
        for code, ranges in _SCRIPT_RANGES.items():
            if any(start <= point <= end for start, end in ranges):
                counts[code] += 1
                break
    if not letters:
        return None
    code, count = max(counts.items(), key=lambda item: item[1])
    # Require either a genuinely short script-only phrase or a clear majority;
    # this avoids one stray non-Latin character overriding an English sentence.
    if count and (letters <= 3 or count * 4 >= letters):
        return code
    return None


def detect_language(text):
    """
    Detect the language of `text` -> (iso_code, friendly_name).
    Falls back to ("", "unknown") when detection fails.
    """
    text = normalize_translation_text(text)
    if not text:
        return ("", "unknown")

    code = _script_language_code(text)
    if code is None:
        try:
            code = _ld_detect(text)
        except LangDetectException:
            return ("", "unknown")

    normalized = (code or "").lower().replace("_", "-")
    normalized = _DETECT_CODE_ALIASES.get(normalized, normalized)
    return (normalized, _friendly_language_name(normalized))


# ---------------------------------------------------------------------------
# The service the pipeline actually uses
# ---------------------------------------------------------------------------
class TranslationService:
    """
    Coordinates the backends. Call translate(text, source_name, target_name).

    * Resolves the source language (using detection when source is 'Auto').
    * Picks the right code style per backend from the language table.
    * Tries backends in order, skipping any on cooldown, returning the first
      non-empty translation.
    """

    def __init__(self, backends=None, cache_ttl=CACHE_TTL_SECONDS,
                 cache_max_entries=CACHE_MAX_ENTRIES):
        self.backends = backends or [GoogleFastTranslate(), MyMemoryTranslate()]
        self._cooldown_until = {}   # backend.name -> monotonic timestamp
        self._cache_ttl = max(0.0, float(cache_ttl))
        self._cache_max_entries = max(0, int(cache_max_entries))
        self._cache = OrderedDict()  # key -> (monotonic timestamp, translation)
        self._cache_lock = threading.Lock()

    def _cached(self, key, now):
        if not self._cache_max_entries:
            return None
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None:
                return None
            stored_at, result = item
            if now - stored_at > self._cache_ttl:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return result

    def _store_cached(self, key, result, now):
        if not self._cache_max_entries:
            return
        with self._cache_lock:
            self._cache[key] = (now, result)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)

    def _code_for(self, backend, lang):
        """Pick the code from a language dict in the style the backend wants."""
        if lang is None:
            return None
        return lang["trans"] if backend.code_kind == "iso" else lang["stt"]

    def translate(self, text, source_name, target_name):
        # Keep the request payload compact and consistent. This is local-only
        # cleanup: it does not add a request or change the selected backend.
        text = normalize_translation_text(text)
        if not text:
            return ""

        target = config.get_language(target_name)
        if target is None:
            raise TranslationError(f"unsupported target language: {target_name}")

        # Resolve the source language. In Auto mode we detect it from the text
        # so that even backends without their own auto-detect can be used.
        if source_name == config.SOURCE_AUTO:
            code, _name = detect_language(text)
            source_lang = config.get_language(_friendly_language_name(code)) \
                if code else None
        else:
            source_lang = config.get_language(source_name)

        now = time.monotonic()
        cache_key = (text, source_name or config.SOURCE_AUTO, target_name)
        cached = self._cached(cache_key, now)
        if cached is not None:
            return cached

        errors = []
        for backend in self.backends:
            # Respect cooldown for recently-failed backends.
            if self._cooldown_until.get(backend.name, 0) > now:
                continue

            # A backend without auto-detect can't run if we don't know the source.
            if source_lang is None and not backend.supports_auto:
                continue

            source_code = "auto" if source_lang is None else \
                self._code_for(backend, source_lang)
            target_code = self._code_for(backend, target)

            # Don't translate a language into itself.
            if source_code and target_code and \
                    source_code.split("-")[0] == target_code.split("-")[0]:
                self._store_cached(cache_key, text, now)
                return text

            try:
                result = backend.translate(text, source_code, target_code)
                if not isinstance(result, str):
                    errors.append(f"{backend.name}: invalid result")
                    continue
                result = normalize_translation_text(result)
                # A multi-word response identical to its source is usually an
                # upstream failure/HTML fallback, not a successful translation.
                # Keep short names and genuinely same-language requests valid.
                same_language = source_code and target_code and \
                    source_code.split("-")[0] == target_code.split("-")[0]
                if result and not same_language and result == text and \
                        len(text.split()) >= 2:
                    errors.append(f"{backend.name}: unchanged result")
                    continue
                if result:
                    self._store_cached(cache_key, result, now)
                    return result
                errors.append(f"{backend.name}: empty result")
            except TranslationError as exc:
                errors.append(str(exc))
                self._cooldown_until[backend.name] = now + BACKEND_COOLDOWN_SECONDS

        raise TranslationError("; ".join(errors) or "no backend could translate")


# --- factory ---------------------------------------------------------------
def get_translation_service():
    """Return the configured service with optional Gemini-first routing."""
    preference = config.get_backend_setting(
        "TRANSLATION_ENGINE", config.TRANSLATION_ENGINE).lower()
    google = GoogleFastTranslate()
    mymemory = MyMemoryTranslate()

    if preference == "google":
        backends = [google, mymemory]
    elif preference == "mymemory":
        backends = [mymemory, google]
    elif preference == "gemini":
        backends = ([GeminiTranslate(), google, mymemory]
                    if config.get_api_key("GEMINI_API_KEY")
                    else [google, mymemory])
    elif preference in ("auto", "default"):
        backends = ([GeminiTranslate(), google, mymemory]
                    if config.get_api_key("GEMINI_API_KEY")
                    else [google, mymemory])
    else:
        raise ValueError(f"Unknown translation engine: {preference!r}")
    return TranslationService(backends)
