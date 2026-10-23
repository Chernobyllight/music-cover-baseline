from ..audio import decode
ID = '04_vocal_separation'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'backends/separator.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['separator']
def requires(r): return ['03_style_generation']
def inputs(ctx): return {'cover': ctx.output('03_style_generation', 'cover')}
def params(r, c): return c


def run(ctx, out, i, p):
    decode(i['cover'], out / 'separation_input.wav')
    ctx.execute('separator', 'separate', out, '--input', out / 'separation_input.wav')
    return {'vocals': out / 'vocals.wav', 'instrumental': out / 'instrumental.wav'}
