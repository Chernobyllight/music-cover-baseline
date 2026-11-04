import unittest
import numpy as np
from music_cover.voice_alignment import onset_alignment
from music_cover.audio import shift_audio


class OnsetAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rate = 1000
        rng = np.random.default_rng(9)
        envelope = np.zeros(8000)
        for t in range(2300, 7800, 29):
            envelope[t:t+10] = rng.uniform(.1, .8)
        self.source = (np.repeat(envelope, 10) * np.sin(np.arange(80000) * .4))[:, None]

    def test_constant_delay_with_silent_intro_and_tail(self):
        converted = shift_audio(self.source, 120, len(self.source)) * .7
        result = onset_alignment(self.source, converted, self.rate)
        self.assertTrue(result['consistent'])
        self.assertAlmostEqual(result['estimated_delay_ms'], 120, delta=10)
        self.assertGreater(len(result['excluded_low_activity_windows']), 0)

    def test_real_time_varying_delay_is_rejected(self):
        shifted = shift_audio(self.source, 120, len(self.source))
        converted = self.source.copy()
        converted[40000:60000] = shifted[40000:60000]
        self.assertFalse(onset_alignment(self.source, converted, self.rate)['consistent'])

    def test_silence_and_unrelated_audio_cannot_pass(self):
        self.assertFalse(onset_alignment(self.source * 0, self.source * 0, self.rate)['consistent'])
        other = np.random.default_rng(32).normal(0, .1, self.source.shape)
        self.assertFalse(onset_alignment(self.source, other, self.rate)['consistent'])
