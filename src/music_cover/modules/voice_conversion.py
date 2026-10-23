from ..audio import decode
ID = '05_voice_conversion'
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['audio.py', 'backends/seed_vc.py']
TIMELINE = {'*': 'generated_before_alignment'}
def backends(r, c): return ['seed_vc']
def requires(r): return ['00_prepare', '04_vocal_separation']
def inputs(ctx): return {'vocals': ctx.output('04_vocal_separation', 'vocals'), 'reference': ctx.output('00_prepare', 'reference')}
def params(r, c): return {**c, 'steps': r['steps'], 'seed': r['seed'], 'semitones': 0, 'auto_f0': False}


def run(ctx, out, i, p):
    decode(i['vocals'], out / 'source_mono.wav', rate=44100, channels=1)
    ctx.execute('seed-vc', 'convert', out, '--input', out / 'source_mono.wav', '--reference', i['reference'], '--steps', p['steps'], '--seed', p['seed'])
    return {'converted': out / 'converted.wav'}
