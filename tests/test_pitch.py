import unittest

from music_cover.pitch import abc_tools, choose_transposition, transpose_abc


HEADER = '''X:1
T:
M:4/4
L:1/32
Q:1/4=62
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
'''


class ScoreTests(unittest.TestCase):
    def test_tie_across_bar_and_native_accidentals_across_octaves(self):
        score = HEADER + 'V: Vocal\n^C8c8C8C8-|C8C8z16|\nV: Ins\nZ2|\n'
        shifted, checks = transpose_abc(score, 2)
        notes = abc_tools().parse(shifted).voices['Vocal'].notes
        self.assertEqual([n[1] for n in notes], [63, 75, 63, 63, 62])
        self.assertEqual([float(n[2]) for n in notes], [1, 1, 1, 2, 1])
        self.assertEqual(checks['target_key'], 'D')
        self.assertEqual(checks['notes_checked'], {'Vocal': 5, 'Ins': 0})

    def test_inline_key_chords_slash_bass_and_both_voices(self):
        score = HEADER + 'V: Vocal\n"C/E"C16[K:G]"D7/F#"F16|\nV: Ins\nc16[K:G]f16|\n'
        shifted, _ = transpose_abc(score, 3)
        parsed = abc_tools().parse(shifted)
        self.assertEqual([n[1] for n in parsed.voices['Vocal'].notes], [63, 69])
        self.assertEqual([n[1] for n in parsed.voices['Ins'].notes], [75, 81])
        self.assertEqual([c for _, c in parsed.voices['Vocal'].chords], ['Eb/G', 'F7/A'])
        self.assertEqual([k for _, k in parsed.voices['Vocal'].keys], ['Eb', 'Bb'])

    def test_octave_boundaries_minor_key_and_meter_change(self):
        score = HEADER.replace('K:C', 'K:Am') + 'V: Vocal\nB,16B16|\nV: Ins\nB16b16|\nV: Vocal\nM:1/4\nc8|\nV: Ins\nM:1/4\nz8|\n'
        shifted, checks = transpose_abc(score, 1)
        parsed = abc_tools().parse(shifted)
        self.assertEqual([n[1] for n in parsed.voices['Vocal'].notes], [60, 72, 73])
        self.assertEqual([n[1] for n in parsed.voices['Ins'].notes], [72, 84])
        self.assertEqual(checks['target_key'], 'Bbm')

    def test_out_of_range_and_unsupported_notation_rejected(self):
        score = HEADER + 'V: Vocal\nc\'\'\'\'32|\nV: Ins\nZ|\n'
        with self.assertRaisesRegex(ValueError, 'MIDI range'):
            transpose_abc(score, 12)
        with self.assertRaises(ValueError):
            transpose_abc(HEADER + 'V: Vocal\n[CEG]32|\nV: Ins\nZ|\n', 3)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def profile(low, high):
        return {'voiced_seconds': 10, 'midi': {'p10': low, 'p95': high}}

    def test_raise_low_register_and_protect_high_register(self):
        plan = choose_transposition(self.profile(50, 70), self.profile(53, 70))
        self.assertEqual(plan['selected_semitones'], 3)
        constrained = choose_transposition(self.profile(50, 75), self.profile(56, 70))
        self.assertLess(constrained['selected_semitones'], 3)

    def test_manual_override_and_insufficient_reference(self):
        self.assertEqual(choose_transposition(self.profile(50, 70), self.profile(53, 70), '-2')['selected_semitones'], -2)
        with self.assertRaisesRegex(ValueError, 'Insufficient voiced'):
            choose_transposition(self.profile(50, 70), {'voiced_seconds': 0})


if __name__ == '__main__':
    unittest.main()
