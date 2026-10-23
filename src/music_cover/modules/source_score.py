ID = '01_source_score'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['backends/sheetsage.py']
TIMELINE = {'*': 'original'}
def backends(r, c): return ['sheetsage']
def requires(r): return ['00_prepare']
def inputs(ctx): return {'source': ctx.output('00_prepare', 'source')}
def params(r, c): return {**c, 'melody_only': True}


def run(ctx, out, i, p):
    ctx.execute('sheetsage2', 'transcribe', out, '--input', i['source'])
    if 'V:' not in (out / 'score.abc').read_text():
        raise ValueError('Transcription did not produce a voiced ABC score')
    return {'score': out / 'score.abc', 'events': out / 'events.json'}
