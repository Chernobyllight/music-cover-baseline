from ..common.io import write_json, artifact_paths
ID = '09_stem_midi'
DEFAULTS = {'timeout_seconds': 10800, 'onset_threshold': 0.5, 'frame_threshold': 0.3, 'minimum_note_ms': 90}
CODE = ['audio.py', 'common/review_audio.py', 'backends/midi.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['midi']
def requires(r): return ['07_stem_separation', '08_final_score']
def inputs(ctx):
    roles = ('lead_vocals', 'backing_vocals', 'vocals', 'bass', 'guitar', 'piano', 'other', 'drums')
    return {'stems/stems.json': ctx.output('07_stem_separation', 'stems.json'),
            **{'stems/audio/' + role + '.wav': ctx.output('07_stem_separation', 'audio/' + role + '.wav') for role in roles},
            'chords': ctx.output('08_final_score', 'chords.mid')}
def params(r, c):
    if not 0 < c['onset_threshold'] < 1 or not 0 < c['frame_threshold'] < 1 or not 0 < c['minimum_note_ms'] <= 1000:
        raise ValueError('Invalid MIDI detection thresholds')
    return c


def run(ctx, out, i, p):
    write_json(out / 'midi_request.json', {'stems': str(i['stems/stems.json'].parent), 'transcription': str(i['chords'].parent), 'parameters': p})
    ctx.execute('separator', 'review_midi', out, '--request', out / 'midi_request.json')
    return artifact_paths(out)
