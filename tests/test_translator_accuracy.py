"""Network-free regression tests for translation quality helpers."""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import translator
    from speech_to_text import resample_pcm16
except (ImportError, OSError) as exc:  # optional runtime dependencies
    raise unittest.SkipTest(f"translator dependencies unavailable: {exc}")


class RecordingEngine(translator.TranslationEngine):
    code_kind = "iso"
    supports_auto = False
    name = "recording"

    def __init__(self, result="ok"):
        self.calls = []
        self.result = result

    def translate(self, text, source_code, target_code):
        self.calls.append((text, source_code, target_code))
        return self.result


class TranslationAccuracyTests(unittest.TestCase):
    def test_supported_scripts_override_short_text_detector(self):
        examples = {
            "ta": "வணக்கம்",
            "hi": "नमस्ते",
            "te": "నమస్కారం",
            "kn": "ನಮಸ್ಕಾರ",
            "ml": "നമസ്കാരം",
            "ar": "مرحبا",
            "ru": "Привет",
            "ja": "こんにちは",
            "zh-cn": "你好",
        }
        for expected, text in examples.items():
            with self.subTest(expected=expected):
                code, _name = translator.detect_language(text)
                self.assertEqual(code, expected)

    def test_detector_alias_maps_to_configured_language(self):
        with patch.object(translator, "_ld_detect", return_value="zh-TW"):
            code, name = translator.detect_language("hello")
        self.assertEqual(code, "zh-cn")
        self.assertEqual(name, "Chinese (Simplified)")

    def test_translation_text_normalization_is_local_and_stable(self):
        self.assertEqual(
            translator.normalize_translation_text("  café\n\t au lait  "),
            "café au lait",
        )

    def test_auto_source_uses_script_detection_for_backend_routing(self):
        backend = RecordingEngine()
        service = translator.TranslationService([backend])
        result = service.translate("  வணக்கம்  ", "Auto (best-effort)", "English")
        self.assertEqual(result, "ok")
        self.assertEqual(backend.calls, [("வணக்கம்", "ta", "en")])

    def test_explicit_source_and_target_codes_are_unchanged(self):
        backend = RecordingEngine()
        service = translator.TranslationService([backend])
        service.translate(" hello ", "English", "Tamil")
        self.assertEqual(backend.calls, [("hello", "en", "ta")])

    def test_successful_translation_is_cached(self):
        backend = RecordingEngine(result="bonjour")
        service = translator.TranslationService([backend], cache_ttl=60)
        self.assertEqual(service.translate("hello", "English", "French"),
                         "bonjour")
        self.assertEqual(service.translate(" hello ", "English", "French"),
                         "bonjour")
        self.assertEqual(len(backend.calls), 1)

    def test_resampling_converts_native_audio_to_16khz(self):
        source = np.array([0, 1000, -1000, 2000], dtype="<i2")
        output = resample_pcm16(source.tobytes(), 8000, 16000)
        self.assertEqual(len(output), len(source) * 2 * 2)
        self.assertEqual(np.frombuffer(output, dtype="<i2")[0], 0)

    def test_unchanged_multword_result_uses_next_backend(self):
        first = RecordingEngine(result="hello world")
        second = RecordingEngine(result="வணக்கம் உலகம்")
        service = translator.TranslationService([first, second])
        self.assertEqual(
            service.translate("hello world", "English", "Tamil"),
            "வணக்கம் உலகம்",
        )
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)


if __name__ == "__main__":
    unittest.main()
