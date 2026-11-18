import importlib.util
from pathlib import Path
import unittest
import tempfile
import sys
import numpy as np


from music_cover.backends import yue as runner


class GenerationLengthTests(unittest.TestCase):
    def test_six_and_half_minute_song_has_room_for_full_score_and_ending(self):
        budget = runner.cover_token_budget(397.4025974025974, 2000)
        self.assertEqual(budget["semantic_max_tokens"], 12000)
        self.assertGreater(budget["audio_ceiling_s"], 397.4025974025974 + 30)
        self.assertLessEqual(budget["prefix_tokens"] + budget["semantic_max_tokens"], 24576)

    def test_short_song_keeps_existing_limit(self):
        self.assertEqual(runner.cover_token_budget(200, 2000)["semantic_max_tokens"], 9000)

    def test_long_prompt_is_rejected_instead_of_shortening_song(self):
        with self.assertRaisesRegex(ValueError, "refusing to silently truncate"):
            runner.cover_token_budget(397.4, 13000)

    def test_invalid_score_duration_rejected(self):
        for duration in [0, -1, float("nan"), float("inf")]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                runner.cover_token_budget(duration, 2000)


class CoverCheckpointTests(unittest.TestCase):
    def test_valid_phase_survives_next_attempt_but_corruption_is_not_reused(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/YuE/src"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "attempt_001/artifacts", root / "attempt_002/artifacts"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            values = np.array([11, 22, 33], dtype=np.int32)
            runner.save_cover_checkpoint(first, "semantic", "key", values, {"timing": {"seconds": 1}})
            loaded, metadata = runner.restore_cover_checkpoint(second, "semantic", "key")
            np.testing.assert_array_equal(loaded, values)
            self.assertEqual(metadata["timing"]["seconds"], 1)
            self.assertIsNone(runner.restore_cover_checkpoint(second, "semantic", "different")[0])
            for output in (first, second):
                (output / "semantic.npy").write_bytes(b"damaged")
            self.assertIsNone(runner.restore_cover_checkpoint(second, "semantic", "key")[0])

    def test_latents_are_not_reused_for_different_semantic_tokens(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/YuE/src"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "attempt_001/artifacts", root / "attempt_002/artifacts"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            def key(tokens):
                return runner.cover_latent_key("same-request", np.asarray(tokens, dtype=np.int32))
            runner.save_cover_checkpoint(first, "latent", key([1, 2]), np.ones((2, 64), dtype=np.float32), {"seconds": 1})
            self.assertIsNone(runner.restore_cover_checkpoint(second, "latent", key([1, 3]))[0])


if __name__ == "__main__":
    unittest.main()
