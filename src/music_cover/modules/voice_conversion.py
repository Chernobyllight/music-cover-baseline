from ..audio import decode
from ..common.io import write_json
from ..voice_candidates import candidate_seeds
from ..voice import resolve_voice
ID = '05_voice_conversion'
CONTRACT_VERSION = 3
DEFAULTS = {'timeout_seconds': 10800, 'backend': 'seed_vc', 'mode': 'zero_shot', 'voice_profile': None, 'candidate_count': 3, 'candidate_seeds': None, 'svc': {}, 'rvc': {}}
CODE = ['audio.py', 'voice.py', 'voice_weights.py', 'voice_candidates.py', 'common/voice_profiles.py', 'backends/seed_vc.py', 'backends/voice_candidates.py', 'backends/voice_metrics.py', 'pitch_selection.py', 'pitch_validation.py', 'svc.py', 'backends/so_vits_svc.py', 'backends/svc_contentvec.py', 'rvc.py', 'backends/rvc.py']
TIMELINE = {'*': 'generated_before_alignment'}
def backends(r, c): return [c.get('backend', 'seed_vc')] + (['voice_metrics'] if c.get('mode') == 'finetuned' else [])
def requires(r): return ['00_prepare', '04_vocal_separation']
def inputs(ctx):
    c = {**DEFAULTS, **ctx.config[ID]}
    voice = resolve_voice(c['mode'], c['voice_profile'], ctx.root, backend=c['backend'])
    result = {'vocals': ctx.output('04_vocal_separation', 'vocals'), 'reference': ctx.output('00_prepare', 'reference'),
              'checkpoint': voice['checkpoint'], 'model_config': voice['config']}
    if 'profile' in voice: result['profile'] = voice['profile']
    for key in ('index', 'index_manifest'):
        if key in voice: result[key] = voice[key]
    return result

def params(r, c):
    c = {**DEFAULTS, **c}
    if c['backend'] not in {'seed_vc', 'so_vits_svc', 'rvc'}: raise ValueError('Unsupported voice backend')
    if c['backend'] == 'rvc':
        from ..rvc import validate_settings
        if c['mode'] != 'finetuned': raise ValueError('rvc requires finetuned mode')
        c['rvc'] = validate_settings(c['rvc'])
    elif c['rvc']:
        raise ValueError('rvc settings require the rvc backend')
    if c['backend'] == 'so_vits_svc':
        from ..svc import validate_settings
        if c['mode'] != 'finetuned': raise ValueError('so_vits_svc requires finetuned mode')
        if c['candidate_count'] != 1: raise ValueError('so_vits_svc currently requires --voice-candidates 1')
        c['svc'] = validate_settings(c['svc'])
    elif c['svc']:
        raise ValueError('svc settings require the so_vits_svc backend')
    if c['mode'] not in {'zero_shot', 'finetuned'}: raise ValueError('Invalid voice mode')
    if c['mode'] == 'finetuned' and not c['voice_profile']: raise ValueError('finetuned requires voice_profile')
    if c['mode'] == 'zero_shot':
        c.update(voice_profile=None, candidate_count=1, candidate_seeds=[r['seed']])
    else:
        c['candidate_seeds'] = candidate_seeds(r['seed'], c['candidate_count'], c['candidate_seeds'])
    return {**c, 'steps': r['steps'] if c['backend'] == 'seed_vc' else None, 'seed': r['seed'], 'semitones': 0, 'auto_f0': False}


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
