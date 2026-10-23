from pathlib import Path
from ..audio import decode
from ..common.io import write_json
from ..common.lyrics import clean_lyrics

ID = '00_prepare'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'common/lyrics.py']
TIMELINE = {'source': 'original', 'reference': 'reference_excerpt'}


def backends(r, c): return []
def requires(r): return []
def inputs(ctx): return {k: Path(ctx.request[k]) for k in ('song', 'lyrics', 'reference')}
def params(r, c): return {**c, 'start': r['reference_start'], 'duration': r['reference_duration']}


def run(ctx, out, i, p):
    source, reference = out / 'source.wav', out / 'reference.wav'
    source_meta = decode(i['song'], source)
    ref_meta = decode(i['reference'], reference, rate=44100, channels=1, start=p['start'], duration=p['duration'])
    if ref_meta['duration_s'] < 1 or ref_meta['rms'] < 1e-5:
        raise ValueError('Reference must contain at least one second of audible speech or singing')
    (out / 'lyrics.txt').write_text(clean_lyrics(i['lyrics'].read_text(encoding='utf-8-sig')))
    write_json(out / 'metadata.json', {'source': source_meta, 'reference': ref_meta,
               'reference_note': 'Cropped, not automatically source-separated'})
    return {'source': source, 'reference': reference, 'lyrics': out / 'lyrics.txt'}
