from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from music_cover.audio import estimate_alignment, export_mp3, fit_length, mix_tracks, read_at, sha256, shift_audio
from music_cover.pipeline import Pipeline, clean_lyrics, write_json


class AudioTests(unittest.TestCase):
    def test_lrc_repeated_timestamps_preserve_repeated_lines(self):
        text = "[ar:Singer]\n[00:20.00][00:40.50]同一句\n[00:30.00]中间一句\n[00:00.00]作词：某人"
        self.assertEqual(clean_lyrics(text), "[Verse]\n同一句\n中间一句\n同一句\n")

    def test_resampling_preserves_timed_events(self):
        with tempfile.TemporaryDirectory() as directory:
            x = np.zeros((48000 * 4, 2), np.float32)
            positions = [24000, 72000, 144000]
            x[positions] = 0.5
            path = Path(directory) / "events.wav"
            sf.write(path, x, 48000, subtype="FLOAT")
            at_44 = read_at(path, 44100)
            sf.write(path, at_44, 44100, subtype="FLOAT")
            recovered = read_at(path, 48000)
            for position in positions:
                peak = np.argmax(np.abs(recovered[position - 10:position + 11, 0])) + position - 10
                self.assertLessEqual(abs(peak - position), 1)

    def test_alignment_recovers_constant_delay(self):
        rate = 8000
        t = np.arange(rate * 12) / rate
        env = sum(np.exp(-((t - center) / 0.12) ** 2) for center in [0.8, 2.1, 4.3, 6.8, 9.1, 10.6])
        x = (env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
        delayed = shift_audio(x, 640, len(x))
        result = estimate_alignment(x, delayed, rate)
        self.assertTrue(result["consistent"])
        self.assertAlmostEqual(result["estimated_delay_ms"], 80, delta=10)
        recovered = shift_audio(delayed, result["applied_shift_samples"], len(x))
        np.testing.assert_allclose(recovered[:-640], x[:-640], atol=1e-6)

    def test_mix_preserves_length_and_prevents_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rate = 48000
            t = np.arange(rate * 2) / rate
            vocal = (0.6 * np.sin(2 * np.pi * 220 * t))[:, None]
            acc = np.repeat((0.6 * np.sin(2 * np.pi * 330 * t))[:, None], 2, axis=1)
            paths = {}
            for name, values in [("vocal", np.repeat(vocal, 2, axis=1)), ("acc", acc),
                                 ("cover", acc + vocal), ("converted", vocal * 0.8)]:
                paths[name] = root / f"{name}.wav"
                sf.write(paths[name], values, rate, subtype="FLOAT")
            result = mix_tracks(paths["cover"], paths["vocal"], paths["acc"], paths["converted"], root)
            self.assertEqual(result["final"]["frames"], rate * 2)
            self.assertEqual(result["final"]["channels"], 2)
            self.assertLessEqual(result["final"]["estimated_true_peak_dbtp"], -1.0)
            self.assertEqual(result["final_mp3"]["sample_rate"], rate)
            self.assertEqual(result["final_mp3"]["channels"], 2)
            self.assertAlmostEqual(result["final_mp3"]["duration_s"], 2, delta=0.05)
            self.assertEqual(sf.info(root / "final.mp3").format, "MP3")
            self.assertGreater(result["final_mp3"]["rms"], 0.01)
            self.assertLess((root / "final.mp3").stat().st_size, (root / "final.wav").stat().st_size / 5)
            self.assertEqual(result["final_mp3"]["sha256"], sha256(root / "final.mp3"))

            # Exercise the real delivery code with the actual encoded audio.
            sf.write(root / "cover.flac", acc + vocal, rate)
            write_json(root / "mix.json", result)
            pipeline = Pipeline(root / "run", {"gpu": "0", "style": "test", "song": str(paths["cover"])})
            try:
                pipeline.deliver(
                    {"source": paths["cover"], "reference": paths["converted"]},
                    {"cover": root / "cover.flac"},
                    {"vocals": paths["vocal"], "instrumental": paths["acc"]},
                    {"converted": paths["converted"]},
                    {"final": root / "final.wav", "final_mp3": root / "final.mp3",
                     "reconstruction": root / "reconstruction.wav", "aligned_vocal": root / "converted_aligned.wav",
                     "metrics": root / "mix.json"})
                delivered = pipeline.run / "deliverables"
                for name in ["final.wav", "final.mp3"]:
                    self.assertEqual(sha256(delivered / name), sha256(root / name))
                    self.assertIn(f'src="{name}"', (delivered / "listen.html").read_text())
                    self.assertIn(f'href="{name}" download', (delivered / "listen.html").read_text())
                checks = json.loads((delivered / "checks.json").read_text())
                self.assertEqual(checks["final_mp3"]["sha256"], sha256(delivered / "final.mp3"))
                self.assertIn("final.mp3", (delivered / "report.md").read_text())
            finally:
                pipeline.lock.close()

            sf.write(paths["converted"], np.pad(vocal, ((0, rate), (0, 0))), rate)
            with self.assertRaisesRegex(ValueError, "refusing to hide drift"):
                mix_tracks(paths["cover"], paths["vocal"], paths["acc"], paths["converted"], root)

    def test_failed_mp3_export_keeps_previous_file_and_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "final.mp3"
            target.write_bytes(b"previous output")
            def fail(command, **kwargs):
                Path(command[-1]).write_bytes(b"partial encoding")
                raise subprocess.CalledProcessError(1, command)
            with patch("music_cover.audio.subprocess.run", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    export_mp3(Path(directory) / "final.wav", target)
            self.assertEqual(target.read_bytes(), b"previous output")
            self.assertEqual(list(Path(directory).iterdir()), [target])


class RecoveryTests(unittest.TestCase):
    def test_valid_cache_reused_but_corrupt_output_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("input")
            pipeline = Pipeline(root / "run", {"gpu": "0"})
            calls = []
            def action(out):
                calls.append(out)
                target = out / "output.txt"
                target.write_text("valid output")
                return {"artifact": target}
            try:
                first = pipeline.stage("01_test", {"source": source}, {}, action)
                pipeline.stage("01_test", {"source": source}, {}, action)
                self.assertEqual(len(calls), 1)
                first["artifact"].write_text("corrupt output")
                rebuilt = pipeline.stage("01_test", {"source": source}, {}, action)
                self.assertEqual(len(calls), 2)
                self.assertNotEqual(first["artifact"], rebuilt["artifact"])
            finally:
                pipeline.lock.close()

    def test_failure_is_not_committed_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("input")
            pipeline = Pipeline(root / "run", {"gpu": "0"})
            try:
                with self.assertRaises(RuntimeError):
                    pipeline.stage("01_test", {"source": source}, {}, lambda out: {"missing": out / "missing.wav"})
                self.assertEqual(pipeline.state["stages"]["01_test"]["status"], "failed")
                self.assertNotIn("outputs", pipeline.state["stages"]["01_test"])
            finally:
                pipeline.lock.close()


if __name__ == "__main__":
    unittest.main()
