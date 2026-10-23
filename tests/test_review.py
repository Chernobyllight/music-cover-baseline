import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from music_cover.review import mix_components, stereo_aligned, write_midi, midi_stats


class ReviewAudioTests(unittest.TestCase):
    def test_post_gain_components_reconstruct_actual_pcm_master(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = np.arange(48000) / 48000
            voice = (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
            accompaniment = np.stack([0.2 * np.sin(2 * np.pi * 330 * t), 0.1 * np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32)
            metrics = {"final": {"master_gain": 0.63}, "vocal_gain": 1.7}
            sf.write(root / "voice.wav", voice, 48000, subtype="FLOAT")
            sf.write(root / "inst.wav", accompaniment, 48000, subtype="FLOAT")
            sf.write(root / "final.wav", (accompaniment + voice * 1.7) * 0.63, 48000, subtype="PCM_24")
            final, vocals, instrumental, residue = mix_components(root / "final.wav", root / "voice.wav", root / "inst.wav", metrics)
            np.testing.assert_allclose(vocals, np.repeat(voice, 2, axis=1) * 1.7 * 0.63, atol=2e-8)
            np.testing.assert_allclose(final, vocals + instrumental + residue, atol=2e-8)
            self.assertEqual(final.shape, (48000, 2))

    def test_alignment_rejects_drift_and_handles_resampling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "track.wav"
            sf.write(path, np.ones(44100, np.float32) * 0.1, 44100, subtype="FLOAT")
            self.assertEqual(stereo_aligned(path, 48000).shape, (48000, 2))
            with self.assertRaisesRegex(ValueError, "duration differs"):
                stereo_aligned(path, 96000)


class ReviewMidiTests(unittest.TestCase):
    def test_roundtrip_preserves_absolute_time_retriggers_silence_and_drum_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "score.mid"
            write_midi(path, [
                {"name": "lead", "program": 53, "notes": [[1.25, 2.0, 69, 93], [2.0, 2.5, 69, 90]]},
                {"name": "bass", "program": 33, "notes": [[1.25, 3.0, 45, 80]]},
                {"name": "drums", "drums": True, "notes": [[0.2, 0.3, 36, 100]]}], 4.0)
            result = midi_stats(path, 4)
            self.assertEqual(result["notes"], 4)
            self.assertAlmostEqual(result["midi_duration_s"], 4, places=4)
            parsed = pretty_midi.PrettyMIDI(str(path))
            self.assertEqual([n.pitch for n in parsed.instruments[0].notes], [69, 69])
            self.assertAlmostEqual(parsed.instruments[0].notes[0].start, 1.25, places=4)
            self.assertTrue(parsed.instruments[2].is_drum)
            channels = {msg.channel for msg in mido.MidiFile(path).tracks[3] if msg.type == "note_on"}
            self.assertEqual(channels, {9})

    def test_silent_track_is_valid_but_not_reported_as_successful_transcription(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "silent.mid"
            write_midi(path, [{"name": "absent", "notes": []}], 2.0)
            result = midi_stats(path, 2.0)
            self.assertEqual(result["status"], "no_notes_detected")
            self.assertEqual(result["notes"], 0)
            self.assertAlmostEqual(result["midi_duration_s"], 2)

    def test_invalid_notes_fail_instead_of_silently_moving_or_inventing_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            for note in [[-1, 1, 60, 100], [1, 5, 60, 100], [1, 2, 128, 100], [1, 2, 60, 0], [1, float("nan"), 60, 100]]:
                with self.assertRaises(ValueError):
                    write_midi(Path(temp) / "bad.mid", [{"name": "bad", "notes": [note]}], 4)


if __name__ == "__main__":
    unittest.main()
