"""Compatibility exports. Implementations live in common/ and backends/."""
from .common.io import artifact_paths, write_json as save_json
from .common.review_audio import *
from .backends.review_stems import separate_review
from .backends.midi import write_midi, notes_csv, midi_stats, drum_onset_notes, transcribe_stems
from .backends.analysis import loudness, describe_audio
from .backends.package import build_review_package
