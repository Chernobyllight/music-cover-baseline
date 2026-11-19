"""Small sampler regression against the pinned upstream implementation.

Runs on CPU by default, or on Metal with MUSIC_TEST_DEVICE=mps.
No release checkpoint or audio generation is required.
"""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from yue_test_support import DEVICE, ROOT, require_assembled_yue


@unittest.skipUnless(DEVICE == "mps" or importlib.util.find_spec("yue2"), "Requires the isolated yue2 environment")
class SamplingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
        import yue2.sampling as sampling
        if DEVICE == "mps":
            require_assembled_yue(sampling)
        cls.torch, cls.current = torch, sampling
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD:third_party/YuE"], text=True).strip()
        source = subprocess.check_output(["git", "-C", str(ROOT / "third_party/YuE"),
            "show", f"{revision}:src/yue2/sampling.py"], text=True)
        if DEVICE == "mps" and Path(sampling.__file__).read_text() == source:
            raise AssertionError("The MPS workspace is missing the YuE sampling patch")
        cls.original = {"__package__": "yue2", "__name__": "baseline_sampling"}
        exec(compile(source, "pinned_sampling.py", "exec"), cls.original)
        torch.manual_seed(41)
        config = YuE2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, intermediate_size=64,
            max_position_embeddings=64, max_latent_frames=64)
        cls.model = YuE2ForCausalLM(config).eval().to(DEVICE)

    def compare(self, temperature=1.0, cfg_scale=1.0, eos=False):
        from yue2.protocol import Sampling, MUSIC_END
        torch = self.torch
        sampling = Sampling(temperature=temperature, max_tokens=4, min_tokens=0 if eos else 4)
        original_distribution = self.current.distribution
        def distribution(*args, **kwargs):
            if not eos:
                return original_distribution(*args, **kwargs)
            scores = torch.full_like(args[0], -torch.inf)
            scores[..., MUSIC_END] = 0
            return scores
        records = []
        with patch.dict(self.original, {"distribution": distribution}), patch.object(self.current, "distribution", distribution):
            for fn in (self.original["generate_tokens"], self.current.generate_tokens):
                callbacks = []
                tokens, timing, truncated = fn(self.model, [1, 4, 7], sampling, 42, "semantic",
                    negative=[1, 2] if cfg_scale != 1 else None, cfg_scale=cfg_scale,
                    on_token=lambda phase, token: callbacks.append((phase, token)), use_cuda_graph=False)
                records.append((tokens, truncated, callbacks, timing["output_tokens"]))
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[1][3], 1 if eos else 4)
        self.assertEqual(records[1][1], not eos)

    def test_stochastic_rng_and_length_limit(self):
        self.compare()

    def test_greedy(self):
        self.compare(temperature=0)

    def test_cfg_branches(self):
        self.compare(cfg_scale=1.3)

    def test_eos_callback_and_stop(self):
        self.compare(eos=True)


if __name__ == "__main__":
    unittest.main()
