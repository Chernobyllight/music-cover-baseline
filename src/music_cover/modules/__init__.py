"""Ordered stage registry. Dependencies and artifact roles are declared in each module."""
import importlib

NAMES = ('prepare', 'source_score', 'pitch_adaptation', 'style_generation', 'vocal_separation',
         'voice_conversion', 'mix', 'stem_separation', 'final_score', 'stem_midi', 'evaluation', 'delivery')
IDS = tuple(f'{i:02d}_{name}' for i, name in enumerate(NAMES))


def load_modules():
    return [importlib.import_module(f'{__name__}.{name}') for name in NAMES]


def active_ids(request):
    return [stage for stage in IDS if request.get('review', True) or stage[:2] not in {'07', '08', '09'}]


def normalize(value):
    matches = [stage for stage in IDS if value in {stage, stage[:2], stage[3:]}]
    if not matches:
        raise ValueError(f'Unknown stage {value}; choose from {", ".join(IDS)}')
    return matches[0]
