from ..common.io import write_json
ID = '03_style_generation'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['backends/yue.py', 'pitch.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['yue']
def requires(r): return ['00_prepare', '02_pitch_adaptation']
def inputs(ctx): return {'score': ctx.output('02_pitch_adaptation', 'score'), 'lyrics': ctx.output('00_prepare', 'lyrics')}
def params(r, c): return {**c, 'style': r.get('style', ''), 'seed': r['seed']}


def run(ctx, out, i, p):
    write_json(out / 'generation_request.json', {'style': p['style'], 'lyrics': i['lyrics'].read_text(), 'seed': p['seed'], 'cot': 'melody'})
    ctx.execute('yue2', 'cover', out, '--request', out / 'generation_request.json', '--score', i['score'])
    return {'cover': out / 'generated/audio.flac'}
