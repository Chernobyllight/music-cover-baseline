"""Process-local compatibility for the pinned Seed-VC on the existing base stack."""
import functools
import inspect


def configure_bigvgan_loader():
    """HF Hub 1.x no longer supplies two legacy BigVGAN download arguments.

    Keep the upstream local checkpoint loader and network unchanged; supply only
    missing legacy defaults. This also works with the older Hub 0.x caller.
    """
    from modules.bigvgan.bigvgan import BigVGAN

    original = BigVGAN._from_pretrained.__func__
    if getattr(original, '_seed_vc_base_compatible', False):
        return
    parameters = inspect.signature(original).parameters
    defaults = {name: None for name in ('proxies', 'resume_download')
                if name in parameters and parameters[name].default is inspect.Parameter.empty}
    if not defaults:
        return

    @functools.wraps(original)
    def compatible(cls, **kwargs):
        return original(cls, **{**defaults, **kwargs})

    compatible._seed_vc_base_compatible = True
    BigVGAN._from_pretrained = classmethod(compatible)
