"""The local BigVGAN loader accepts both HF Hub 0.x and 1.x callers."""
import sys
import types
import unittest
from unittest.mock import patch

from music_cover.backends.seed_vc_runtime import configure_bigvgan_loader


class BigVGANCompatibility(unittest.TestCase):
    def test_old_and_new_hub_calls_preserve_checkpoint_options(self):
        class BigVGAN:
            @classmethod
            def _from_pretrained(cls, *, model_id, proxies, resume_download, **kwargs):
                return cls, model_id, proxies, resume_download, kwargs

        module = types.ModuleType('modules.bigvgan.bigvgan')
        module.BigVGAN = BigVGAN
        with patch.dict(sys.modules, {'modules.bigvgan.bigvgan': module}):
            configure_bigvgan_loader()
            result = BigVGAN._from_pretrained(model_id='/models/BigVGAN', use_cuda_kernel=False)
            self.assertEqual(result, (BigVGAN, '/models/BigVGAN', None, None, {'use_cuda_kernel': False}))
            result = BigVGAN._from_pretrained(model_id='/models/BigVGAN', proxies={'https': 'proxy'},
                                              resume_download=True, strict=True)
            self.assertEqual(result[2:], ({'https': 'proxy'}, True, {'strict': True}))
            original = BigVGAN._from_pretrained.__func__
            configure_bigvgan_loader()
            self.assertIs(BigVGAN._from_pretrained.__func__, original)

    def test_loader_with_existing_defaults_is_not_changed(self):
        class BigVGAN:
            @classmethod
            def _from_pretrained(cls, *, model_id, proxies=None, resume_download=None):
                return model_id

        module = types.ModuleType('modules.bigvgan.bigvgan')
        module.BigVGAN = BigVGAN
        original = BigVGAN._from_pretrained.__func__
        with patch.dict(sys.modules, {'modules.bigvgan.bigvgan': module}):
            configure_bigvgan_loader()
            self.assertIs(BigVGAN._from_pretrained.__func__, original)


if __name__ == '__main__':
    unittest.main()
