"""CPU/MPS reference checks for the experimental native AR GQA callsite."""
import os
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
    from yue2 import modeling_yue2 as modeling
except ImportError:
    modeling = None


@unittest.skipIf(modeling is None, "Requires the isolated yue2 environment")
class AttentionReferenceTests(unittest.TestCase):
    def check_case(self, length, keys, causal=False, masked=False):
        device = os.environ.get("MUSIC_TEST_DEVICE", "cpu")
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        rng = torch.Generator().manual_seed(117)
        # Transposition deliberately leaves a noncontiguous token/head layout.
        q = torch.randn(length, 16, 128, generator=rng).to(dtype).transpose(0, 1)[None]
        k = torch.randn(keys, 8, 128, generator=rng).to(dtype).transpose(0, 1)[None]
        v = torch.randn(keys, 8, 128, generator=rng).to(dtype).transpose(0, 1)[None]
        mask = torch.arange(keys)[None] <= torch.arange(keys-length, keys)[:, None] if masked else None
        expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
            enable_gqa=True, is_causal=causal, attn_mask=mask)
        with patch.object(modeling.F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            actual = modeling.sdpa(q.to(device), k.to(device), v.to(device),
                is_causal=causal, attn_mask=mask.to(device) if mask is not None else None)
            expected_gqa = device != "mps" or getattr(modeling, "_MPS_NATIVE_GQA", False)
            self.assertEqual(call.call_args.kwargs["enable_gqa"], expected_gqa)
        self.assertTrue(torch.isfinite(actual).all().cpu().item())
        torch.testing.assert_close(actual.float().cpu(), expected,
            atol=0.001 if device == "mps" else 1e-6, rtol=0.02 if device == "mps" else 1e-5)

    def test_short_queries_and_long_keys(self):
        for query in range(1, 9):
            with self.subTest(query=query):
                self.check_case(query, 7864)

    def test_causal_prefill(self):
        self.check_case(33, 33, causal=True)

    def test_rectangular_visible_keys(self):
        self.check_case(3, 7864, masked=True)
