import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("cover_stage_runner", Path(__file__).resolve().parents[1] / "scripts/stage_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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


if __name__ == "__main__":
    unittest.main()
