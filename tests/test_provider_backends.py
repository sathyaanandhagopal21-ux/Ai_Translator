"""Network-free tests for the optional Groq and Gemini backends."""

import io
import os
import unittest
import wave
from unittest.mock import patch

import speech_to_text
import translator


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class ProviderBackendTests(unittest.TestCase):
    def test_groq_uploads_mono_wav_and_reads_transcript(self):
        session = RecordingSession(FakeResponse({"text": "hello there"}))
        engine = speech_to_text.GroqSTT(
            api_key="groq-test", model="whisper-test", session=session)

        text, confidence = engine.transcribe_detailed(
            b"\x00\x00" * 8, 8000, "en-US")

        self.assertEqual((text, confidence), ("hello there", None))
        _args, kwargs = session.calls[0]
        self.assertEqual(kwargs["data"], {
            "model": "whisper-test",
            "response_format": "json",
            "language": "en",
        })
        uploaded = kwargs["files"]["file"][1]
        with wave.open(io.BytesIO(uploaded), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)

    def test_groq_http_failure_becomes_stt_error(self):
        session = RecordingSession(FakeResponse({}, status_code=500))
        engine = speech_to_text.GroqSTT(
            api_key="groq-test", session=session)

        with self.assertRaises(speech_to_text.STTError):
            engine.transcribe_detailed(b"\x00\x00", 16000, "en-US")

    def test_gemini_extracts_translation_and_sends_language_prompt(self):
        session = RecordingSession(FakeResponse({
            "candidates": [{
                "content": {"parts": [{"text": "Bonjour"}]}
            }]
        }))
        engine = translator.GeminiTranslate(
            api_key="gemini-test", model="gemini-test", session=session)

        result = engine.translate("hello", "en", "fr")

        self.assertEqual(result, "Bonjour")
        args, kwargs = session.calls[0]
        self.assertIn("gemini-test", args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "gemini-test")
        self.assertNotIn("key=", args[0])
        prompt = kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("Source language/code: en", prompt)
        self.assertIn("Target language/code: fr", prompt)
        self.assertIn("hello", prompt)

    def test_gemini_empty_response_becomes_translation_error(self):
        session = RecordingSession(FakeResponse({"candidates": []}))
        engine = translator.GeminiTranslate(
            api_key="gemini-test", session=session)

        with self.assertRaises(translator.TranslationError):
            engine.translate("hello", "en", "fr")

    def test_retired_gemini_model_id_uses_current_alias(self):
        engine = translator.GeminiTranslate(
            api_key="gemini-test", model="gemini-2.0-flash")
        self.assertEqual(engine.model, "gemini-3.6-flash")

    def test_stt_falls_back_when_first_engine_returns_no_speech(self):
        class EmptySTT(speech_to_text.STTEngine):
            def transcribe_detailed(self, *_args, **_kwargs):
                return "", 0.2

        class WorkingSTT(speech_to_text.STTEngine):
            def transcribe_detailed(self, *_args, **_kwargs):
                return "hello", 0.9

        fallback = speech_to_text.FallbackSTT([EmptySTT(), WorkingSTT()])
        self.assertEqual(
            fallback.transcribe_detailed(b"audio", 16000, "en-US"),
            ("hello", 0.9),
        )

    def test_stt_candidate_fallback_when_first_engine_returns_no_speech(self):
        class EmptySTT(speech_to_text.STTEngine):
            def transcribe_candidates(self, *_args, **_kwargs):
                return "", 0.2

        class WorkingSTT(speech_to_text.STTEngine):
            def transcribe_candidates(self, *_args, **_kwargs):
                return "hello", 0.9

        fallback = speech_to_text.FallbackSTT([EmptySTT(), WorkingSTT()])
        self.assertEqual(
            fallback.transcribe_candidates(
                b"audio", 16000, ["en-US", "ta-IN"]),
            ("hello", 0.9),
        )

    def test_auto_factories_prefer_optional_backends_when_keys_exist(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "groq-test",
            "GEMINI_API_KEY": "gemini-test",
            "STT_ENGINE": "auto",
            "TRANSLATION_ENGINE": "auto",
        }, clear=False):
            stt = speech_to_text.get_stt_engine()
            service = translator.get_translation_service()

        self.assertIsInstance(stt, speech_to_text.FallbackSTT)
        self.assertIsInstance(stt.engines[0], speech_to_text.GroqSTT)
        self.assertIsInstance(service.backends[0], translator.GeminiTranslate)

    def test_auto_factories_keep_original_backends_without_keys(self):
        with patch.dict(os.environ, {
            "STT_ENGINE": "auto",
            "TRANSLATION_ENGINE": "auto",
        }, clear=False):
            with patch.object(speech_to_text.config, "get_api_key",
                              side_effect=lambda name: None):
                stt = speech_to_text.get_stt_engine()
            with patch.object(translator.config, "get_api_key",
                              side_effect=lambda name: None):
                service = translator.get_translation_service()

        self.assertIsInstance(stt, speech_to_text.GoogleSTT)
        self.assertIsInstance(service.backends[0], translator.GoogleFastTranslate)


if __name__ == "__main__":
    unittest.main()
