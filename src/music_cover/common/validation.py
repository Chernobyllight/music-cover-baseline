"""Model-free validation shared by evaluation, delivery and the independent verifier."""
import hashlib
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
import soundfile as sf

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()

class Links(HTMLParser):

    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attributes):
        self.targets += [v for (k, v) in attributes if k in {'href', 'src'} and v]

def verify_components(final_path, root, midi_root):
    import mido
    import pretty_midi
    (final_path, root, midi_root) = map(Path, (final_path, root, midi_root))
    stems = json.loads((root / 'stems.json').read_text())
    if not digest(final_path) == stems['input_final_sha256']:
        raise ValueError("Artifact validation failed: digest(final_path) == stems['input_final_sha256']")
    expected = {'lead_vocals', 'backing_vocals', 'drums', 'bass', 'guitar', 'piano', 'other', 'instrumental_vocal_leak', 'instrumental_residual', 'mix_residual'}
    if not set(stems['leaf_sum_roles']) == expected:
        raise ValueError("Artifact validation failed: set(stems['leaf_sum_roles']) == expected")
    for (role, spec) in stems['stems'].items():
        path = root / spec['path']
        info = sf.info(path)
        if not (info.samplerate, info.channels, info.frames) == (48000, 2, stems['frames']):
            raise ValueError(role)
        if not info.subtype == 'FLOAT':
            raise ValueError(role)
        if not digest(path) == spec['sha256']:
            raise ValueError(role)
    readers = [sf.SoundFile(root / stems['stems'][role]['path']) for role in stems['leaf_sum_roles']]
    (maximum, count, squared_error) = (0.0, 0, 0.0)
    try:
        with sf.SoundFile(final_path) as final:
            while True:
                reference = final.read(65536, dtype='float64', always_2d=True)
                if not len(reference):
                    break
                parts = [stream.read(len(reference), dtype='float64', always_2d=True) for stream in readers]
                if not all((x.shape == reference.shape and np.isfinite(x).all() for x in parts)):
                    raise ValueError('Artifact validation failed: all((x.shape == reference.shape and np.isfinite(x).all() for x in parts))')
                error = np.sum(parts, axis=0) - reference
                maximum = max(maximum, float(np.max(np.abs(error))))
                squared_error += float(np.sum(error ** 2))
                count += error.size
        if not maximum < 2e-06:
            raise ValueError(maximum)
    finally:
        for stream in readers:
            stream.close()
    midi_summary = json.loads((midi_root / 'midi.json').read_text())
    midi_counts = {}
    for (role, spec) in midi_summary['tracks'].items():
        path = midi_root / spec['file']
        parsed = pretty_midi.PrettyMIDI(str(path))
        notes = [n for instrument in parsed.instruments for n in instrument.notes]
        if not len(notes) == spec['notes']:
            raise ValueError(role)
        if not abs(mido.MidiFile(path).length - stems['duration_s']) < 0.002:
            raise ValueError(role)
        if not all((0 <= n.start < n.end <= stems['duration_s'] + 0.002 for n in notes)):
            raise ValueError(role)
        if role == 'drums_onsets':
            if not all((i.is_drum for i in parsed.instruments)):
                raise ValueError(role)
        midi_counts[role] = len(notes)
    if not midi_counts['lead_vocals'] > 0:
        raise ValueError('Lead MIDI unexpectedly empty')
    if not midi_counts['arrangement_estimate'] > 0:
        raise ValueError("Artifact validation failed: midi_counts['arrangement_estimate'] > 0")
    return {'technical_validation': 'passed', 'duration_s': stems['duration_s'], 'audio_stems': len(stems['stems']), 'leaf_stems': len(expected), 'reconstruction_max_abs_error': maximum, 'reconstruction_rms_error': (squared_error / count) ** 0.5, 'midi_note_counts': midi_counts, 'empty_midi_tracks': [role for (role, count) in midi_counts.items() if not count], 'subjective_quality': 'awaiting_human_review'}

def verify_delivery(deliver, review=True):
    deliver = Path(deliver).resolve()
    checks = json.loads((deliver / 'checks.json').read_text())
    from ..audio import inspect_audio
    for (name, key) in [('final.wav', 'final'), ('final.mp3', 'final_mp3')]:
        metadata = inspect_audio(deliver / name)
        if not metadata['sha256'] == checks[key]['sha256']:
            raise ValueError(name)
        if not (metadata['sample_rate'] == 48000 and metadata['channels'] == 2):
            raise ValueError(name)
        if not abs(metadata['duration_s'] - checks['final']['duration_s']) < 0.1:
            raise ValueError(name)
    pages = [deliver / 'listen.html']
    result = {'technical_validation': 'passed', 'subjective_quality': 'awaiting_human_review'}
    if review:
        root = deliver / 'review'
        manifest = json.loads((root / 'manifest.json').read_text())
        for (rel, item) in manifest['files'].items():
            path = (root / rel).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(rel)
            if not (path.is_file() and path.stat().st_size == item['bytes']):
                raise ValueError(rel)
            if not digest(path) == item['sha256']:
                raise ValueError(f'Hash mismatch: {rel}')
        result.update(verify_components(deliver / 'final.wav', root, root))
        result['checked_payload_files'] = len(manifest['files'])
        pages.append(root / 'index.html')
        with zipfile.ZipFile(root / 'midi_bundle.zip') as archive:
            if not archive.testzip() is None:
                raise ValueError('Artifact validation failed: archive.testzip() is None')
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root.resolve()):
                    raise ValueError('Artifact validation failed: (root / name).resolve().is_relative_to(root.resolve())')
                if not hashlib.sha256(archive.read(name)).hexdigest() == digest(root / name):
                    raise ValueError(name)
    pages = sorted(deliver.rglob('*.html'))
    for page in pages:
        parser = Links()
        parser.feed(page.read_text())
        for target in parser.targets:
            url = urlsplit(target)
            if not url.scheme and url.path:
                path = (page.parent / unquote(url.path)).resolve()
                if not (path.is_relative_to(deliver) and path.is_file()):
                    raise ValueError(str(path))
    return result
