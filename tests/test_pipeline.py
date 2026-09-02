"""Hardware-free tests for final/provisional pipeline semantics."""

import unittest

try:
    from pipeline import TranslationPipeline
    from subtitle_manager import SubtitleManager
    from speech_to_text import STTEngine
    from translator import TranslationEngine
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"pipeline dependencies unavailable: {exc}")


class FakeSTT(STTEngine):
    def transcribe_detailed(self, pcm_bytes, sample_rate, language_code,
                            min_confidence=None):
        return "hello", 0.9


class FakeTranslation(TranslationEngine):
    name = "fake"

    def translate(self, text, source_code, target_code):
        return "வணக்கம்"


class FakeTTS:
    def speak(self, text, tts_code):
        return True, "spoke"


class PipelineSemanticsTests(unittest.TestCase):
    def make_pipeline(self, subtitles, subtitles_seen):
        pipeline = TranslationPipeline(
            subtitles,
            on_subtitle=subtitles_seen.append,
        )
        pipeline.stt = FakeSTT()
        pipeline.translation = FakeTranslation()
        pipeline.tts = FakeTTS()
        pipeline._source_name = "English"
        pipeline._target_name = "Tamil"
        pipeline._speak = False
        return pipeline

    def test_interim_is_not_saved_or_spoken(self):
        subtitles = SubtitleManager()
        shown = []
        pipeline = self.make_pipeline(subtitles, shown)
        pipeline._handle_utterance(b"audio", 16000, interim=True,
                                   utterance_id=1)
        self.assertEqual(shown, ["வணக்கம்"])
        self.assertEqual(subtitles.history(), [])

    def test_final_replaces_interim_and_is_saved_once(self):
        subtitles = SubtitleManager()
        shown = []
        pipeline = self.make_pipeline(subtitles, shown)
        pipeline._handle_utterance(b"audio", 16000, interim=True,
                                   utterance_id=1)
        pipeline._enqueue_final(b"audio", 16000, 1)
        pipeline._handle_utterance(b"audio", 16000, utterance_id=1)
        self.assertEqual(len(subtitles.history()), 1)
        self.assertEqual(subtitles.current()["translated"], "வணக்கம்")


if __name__ == "__main__":
    unittest.main()
