import tempfile
import unittest
from pathlib import Path
import numpy as np
from music_cover.pitch_selection import choose, load_pitch, settings, score_consistency, export_score_midi
from music_cover.pitch_validation import compare_generated, compare_conversion
from music_cover.pitch import transpose_abc, abc_tools
from test_pitch import HEADER


def fixture(path, values, rms=.1):
    values = np.asarray(values, float)
    np.savez(path, f0=440 * 2 ** ((values - 69) / 12), rms=np.full(len(values), rms), hop_seconds=.01)
    return load_pitch(path)


class PitchSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.melody = np.tile(np.repeat([57, 60, 62, 64, 67, 64, 60, 59], 50), 8)
        self.source = fixture(self.root / 'source.npz', self.melody)

    def profile(self, shift=0, values=None, rms=.1):
        return fixture(self.root / 'reference.npz', self.melody + shift if values is None else values, rms)

    def choose(self, ref, config=None, requested='auto'):
        return choose(self.source, [ref], requested, {'reference_kind': 'singing', **(config or {})})

    def test_raise_lower_and_keep_original(self):
        for shift in [-3, 0, 3]:
            with self.subTest(shift=shift):
                plan = self.choose(self.profile(shift))
                self.assertEqual(plan['recommended_semitones'], shift)
                self.assertEqual(plan['applied_semitones'], shift)
                self.assertEqual(len(plan['auto_candidates']), 13)
                self.assertEqual(plan['bootstrap']['direction_support'], 1)

    def test_unknown_and_speech_reference_do_not_claim_singing_range(self):
        for kind in ['unknown', 'speech']:
            p = self.choose(self.profile(-3), {'reference_kind': kind})
            self.assertEqual(p['recommended_semitones'], -3)
            self.assertEqual(p['applied_semitones'], 0)
            self.assertEqual(p['status'], 'reference_kind_unconfirmed')

    def test_manual_override_independent_of_candidate_and_confidence(self):
        p = self.choose(self.profile(3), {'reference_kind': 'speech'}, '-5')
        self.assertEqual(p['recommended_semitones'], 3)
        self.assertEqual(p['applied_semitones'], -5)
        self.assertEqual(p['status'], 'manual_override')

    def test_insufficient_and_ambiguous_references(self):
        self.assertEqual(self.choose(self.profile(values=[60] * 100))['status'], 'insufficient_reference')
        # Two contrasting registers make the direction unstable under whole-block resampling.
        p = self.choose(self.profile(values=np.r_[np.full(400, 50), np.full(400, 70)]))
        self.assertEqual(p['applied_semitones'], 0)
        self.assertIn(p['status'], ['ambiguous_reference', 'insignificant_improvement'])

    def test_soft_threshold_is_not_forced_nonzero(self):
        p = self.choose(self.profile(.6), {'minimum_improvement': 1})
        self.assertEqual(p['recommended_semitones'], 1)
        self.assertEqual(p['applied_semitones'], 0)

    def test_quiet_reference_and_nonfinite_empty_arrays(self):
        self.assertEqual(self.choose(self.profile(-3, rms=.0005))['applied_semitones'], -3)
        p = self.profile(values=[np.nan, np.inf, -np.inf])
        self.assertEqual(p['summary']['frames'], 0)
        self.assertEqual(self.choose(p)['status'], 'insufficient_reference')
        self.assertEqual(self.profile(values=[])['summary']['frames'], 0)

    def test_octave_spike_rejected_without_removing_sustained_pitch(self):
        v = self.melody.copy(); v[120] += 12; v[400:450] += 12
        p = self.profile(values=v)
        self.assertFalse(p['valid'][120])
        self.assertTrue(p['valid'][420])

    def test_configuration_validation_and_explicit_range(self):
        for config in [{'auto_min_semitones': 1}, {'auto_max_semitones': 13}, {'auto_min_semitones': True},
                       {'shift_penalty': float('nan')}, {'target_low_midi': 60}, {'minimum_voiced_seconds': 1},
                       {'range_references': 'bad'}, {'minimum_direction_support': 1.1}]:
            with self.subTest(config=config), self.assertRaises(ValueError): settings(config)
        p = self.choose(self.profile(), {'target_low_midi': 60, 'target_high_midi': 70, 'range_penalty': 4})
        self.assertTrue(any(row['range_excess'] > 0 for row in p['auto_candidates']))
        self.assertFalse(p['reference_sample_only'])

    def test_multiclip_equal_weight_and_reproducibility(self):
        refs = [self.profile(-3), self.profile(-3, values=np.tile(self.melody - 3, 4))]
        a = choose(self.source, refs, 'auto', {'reference_kind': 'singing'})
        b = choose(self.source, refs[::-1], 'auto', {'reference_kind': 'singing'})
        self.assertEqual(a['applied_semitones'], -3)
        self.assertEqual(a['bootstrap'], b['bootstrap'])

    def test_score_audio_octave_detection_does_not_correct(self):
        path = self.root / 'melody.lab'
        path.write_text('\n'.join(f'{i*.5} {(i+1)*.5} {self.melody[i*50]+12}' for i in range(64)))
        result = score_consistency(path, self.source)
        self.assertEqual(result['status'], 'octave_offset_detected')
        self.assertEqual(result['nearest_octave_offset'], 12)
        self.assertEqual(result['automatic_octave_correction'], 0)
        p = choose(self.source, [self.profile(-3)], 'auto', {'reference_kind': 'singing'}, result)
        self.assertEqual(p['applied_semitones'], -3)
        self.assertEqual(p['confidence'], 'medium')

    def test_audio_verification_catches_octave_error_and_vc_pitch_change(self):
        down = self.profile(-3)
        self.assertEqual(compare_generated(self.source, down, -3)['status'], 'verified')
        wrong = compare_generated(self.source, self.profile(9), -3)
        self.assertEqual(wrong['status'], 'needs_review')
        self.assertEqual(wrong['nearest_octave_error'], 12)
        self.assertEqual(compare_conversion(self.source, self.profile())['status'], 'verified')
        self.assertEqual(compare_conversion(self.source, self.profile(2))['status'], 'needs_review')

    def test_midi_matches_abc_events_with_negative_shift_and_ties(self):
        # The lightweight control environment exports MIDI without importing mido.
        score = HEADER + 'V: Vocal\n"C/E"C8D8E8F8-|F8G8z16|\nV: Ins\nc16e16|Z|\n'
        shifted, checks = transpose_abc(score, -2)
        path = self.root / 'score.mid'; export_score_midi(shifted, path)
        self.assertEqual(path.read_bytes()[:14], b'MThd\x00\x00\x00\x06\x00\x01\x00\x03\x03\xc0')
        self.assertEqual(checks['target_key'], 'Bb')
        self.assertEqual([n[1] for n in abc_tools().parse(shifted).voices['Vocal'].notes], [58, 60, 62, 63, 65])
        try:
            import pretty_midi
        except ImportError:
            return
        parsed = pretty_midi.PrettyMIDI(str(path))
        expected = abc_tools().parse(shifted)
        for instrument, name in zip(parsed.instruments, ['Vocal', 'Ins']):
            self.assertEqual(len(instrument.notes), len(expected.voices[name].notes))
            for actual, (start, pitch, duration) in zip(instrument.notes, expected.voices[name].notes):
                self.assertEqual(actual.pitch, pitch)
                self.assertAlmostEqual(actual.start, float(start) * 60 / expected.bpm, places=4)
                self.assertAlmostEqual(actual.end, float(start + duration) * 60 / expected.bpm, places=4)
