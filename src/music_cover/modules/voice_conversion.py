from ..audio import decode
from ..common.io import write_json
from ..voice_candidates import candidate_seeds
from ..voice import resolve_voice
ID = '05_voice_conversion'
CONTRACT_VERSION = 3
DEFAULTS = {'timeout_seconds': 10800, 'backend': 'seed_vc', 'mode': 'zero_shot', 'voice_profile': None, 'candidate_count': 3, 'candidate_seeds': None}
CODE = ['audio.py', 'voice.py', 'voice_weights.py', 'voice_candidates.py', 'backends/seed_vc.py', 'backends/voice_candidates.py', 'backends/voice_metrics.py', 'pitch_selection.py', 'pitch_validation.py']
TIMELINE = {'*': 'generated_before_alignment'}
def backends(r, c): return ['seed_vc'] + (['voice_metrics'] if c.get('mode') == 'finetuned' else [])
def requires(r): return ['00_prepare', '04_vocal_separation']
def inputs(ctx):
    c = {**DEFAULTS, **ctx.config[ID]}
    voice = resolve_voice(c['mode'], c['voice_profile'], ctx.root)
    result = {'vocals': ctx.output('04_vocal_separation', 'vocals'), 'reference': ctx.output('00_prepare', 'reference'),
              'checkpoint': voice['checkpoint'], 'model_config': voice['config']}
    if 'profile' in voice: result['profile'] = voice['profile']
    return result

def params(r, c):
    c = {**DEFAULTS, **c}
    if c['backend'] != 'seed_vc': raise ValueError('Only seed_vc is implemented')
    if c['mode'] not in {'zero_shot', 'finetuned'}: raise ValueError('Invalid voice mode')
    if c['mode'] == 'finetuned' and not c['voice_profile']: raise ValueError('finetuned requires voice_profile')
    if c['mode'] == 'zero_shot':
        c.update(voice_profile=None, candidate_count=1, candidate_seeds=[r['seed']])
    else:
        c['candidate_seeds'] = candidate_seeds(r['seed'], c['candidate_count'], c['candidate_seeds'])
    return {**c, 'steps': r['steps'], 'seed': r['seed'], 'semitones': 0, 'auto_f0': False}


def run(ctx, out, i, p):
    decode(i['vocals'], out / 'source_mono.wav', rate=44100, channels=1)
    if p['mode'] == 'finetuned':
        request = {**p, 'source': str(out / 'source_mono.wav'), 'reference': str(i['reference']), 'voice_profile': str(i['profile'])}
        write_json(out / 'candidate_request.json', request)
        ctx.execute('seed-vc', 'convert_candidates', out, '--request', out / 'candidate_request.json')
        return {'converted': out / 'converted.wav', 'conversion': out / 'conversion.json', 'candidate_report': out / 'candidates.json'}
    extra = ['--voice-mode', p['mode']]
    ctx.execute('seed-vc', 'convert', out, '--input', out / 'source_mono.wav', '--reference', i['reference'], '--steps', p['steps'], '--seed', p['seed'], *extra)
    return {'converted': out / 'converted.wav', 'conversion': out / 'conversion.json'}
