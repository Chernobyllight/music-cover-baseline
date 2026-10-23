from ..common.io import artifact_paths
ID = '08_final_score'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['backends/sheetsage.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['sheetsage']
def requires(r): return ['06_mix']
def inputs(ctx): return {'final': ctx.output('06_mix', 'final')}
def params(r, c): return {**c, 'melody_only': False}


def run(ctx, out, i, p):
    ctx.execute('sheetsage2', 'transcribe_final', out, '--input', i['final'])
    return artifact_paths(out)
