"""Network- and hardware-free tests for desktop audio preparation."""

import unittest

try:
    import numpy as np
    from audio_input import (normalize_audio_samples, adaptive_vad_threshold,
                             quietest_cut_index, update_noise_floor)
except (ImportError, OSError) as exc:  # optional Windows audio dependency
    raise unittest.SkipTest(f"audio dependencies unavailable: {exc}")


class AudioNormalizationTests(unittest.TestCase):
    def test_quiet_audio_is_boosted_but_gain_is_bounded(self):
        source = np.array([1000, -1000, 500], dtype=np.int16)
        normalized = normalize_audio_samples(source)
        self.assertGreater(np.max(np.abs(normalized)), np.max(np.abs(source)))
        # The configured max gain is 12x, with a small rounding allowance.
        self.assertLessEqual(
            float(np.max(np.abs(normalized))), float(np.max(np.abs(source))) * 12.01
        )

    def test_near_silence_is_not_amplified(self):
        source = np.array([1, -2, 0], dtype=np.int16)
        normalized = normalize_audio_samples(source)
        np.testing.assert_array_equal(normalized, source)

    def test_loud_audio_is_not_modified(self):
        source = np.array([28000, -30000, 1000], dtype=np.int16)
        normalized = normalize_audio_samples(source)
        np.testing.assert_array_equal(normalized, source)

    def test_adaptive_threshold_tracks_noise_without_losing_floor(self):
        floor = update_noise_floor(0.0, 0.03)
        self.assertLessEqual(floor, 0.01)
        self.assertGreaterEqual(adaptive_vad_threshold(0.03, floor), 0.01)

    def test_force_cut_prefers_recent_quiet_boundary(self):
        samples = np.ones(16000, dtype=np.float32) * 0.2
        samples[15000:15500] = 0.0
        cut = quietest_cut_index(samples, 16000, search_seconds=1.0)
        self.assertGreaterEqual(cut, 15000)
        self.assertLessEqual(cut, 15500)


if __name__ == "__main__":
    unittest.main()
