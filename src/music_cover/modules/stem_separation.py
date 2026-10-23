from ..common.io import write_json, artifact_paths
ID = '07_stem_separation'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'common/review_audio.py', 'backends/review_stems.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['review_stems']
def requires(r): return ['04_vocal_separation', '06_mix']
def inputs(ctx): return {'final': ctx.output('06_mix', 'final'), 'vocal': ctx.output('06_mix', 'aligned_vocal'),
                         'instrumental': ctx.output('04_vocal_separation', 'instrumental'), 'mix_metrics': ctx.output('06_mix', 'metrics')}
def params(r, c): return {**c, 'seed': r['seed']}


def run(ctx, out, i, p):
    write_json(out / 'stems_request.json', {**{k: str(v) for k, v in i.items()}, 'seed': p['seed']})
    ctx.execute('separator', 'review_stems', out, '--request', out / 'stems_request.json')
    return artifact_paths(out)
