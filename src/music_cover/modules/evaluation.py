import json
from ..pitch_selection import load_pitch
from ..pitch_validation import evaluate
from ..common.io import write_json, artifact_paths
ID = '10_evaluation'
CONTRACT_VERSION = 2
DEFAULTS = {'timeout_seconds': 10800, 'pitch_check': True}
CODE = ['audio.py', 'common/validation.py', 'backends/evaluation.py', 'backends/analysis.py', 'backends/rmvpe.py', 'pitch.py', 'pitch_selection.py', 'pitch_validation.py']
TIMELINE = {'*': 'generated'}
def backends(r, c): return ['analysis'] + (['rmvpe'] if c['pitch_check'] and str(r['transpose']) != '0' else [])
def requires(r):
    return ['00_prepare', '02_pitch_adaptation', '04_vocal_separation', '06_mix'] + (['07_stem_separation', '08_final_score', '09_stem_midi'] if r['review'] else [])
def inputs(ctx):
    result = {'original': ctx.output('00_prepare', 'source'), 'final': ctx.output('06_mix', 'final'),
              'final_mp3': ctx.output('06_mix', 'final_mp3'), 'metrics': ctx.output('06_mix', 'metrics')}
    if ctx.config[ID]['pitch_check'] and str(ctx.request['transpose']) != '0':
        result.update(before=ctx.output('04_vocal_separation', 'vocals'), after=ctx.output('06_mix', 'aligned_vocal'),
                      source_f0=ctx.output('02_pitch_adaptation', 'source_f0'), plan=ctx.output('02_pitch_adaptation', 'plan'))
    if ctx.request['review']:
        result.update(ctx.all_outputs('07_stem_separation', 'stems/'))
        result.update(ctx.all_outputs('09_stem_midi', 'midi/'))
        result.update(ctx.all_outputs('08_final_score', 'final_score/'))
    return result
def params(r, c):
    if not isinstance(c['pitch_check'], bool): raise ValueError('pitch_check must be true or false')
    return {**c, 'review': r['review'], 'pitch_check': c['pitch_check'] and str(r['transpose']) != '0'}


def run(ctx, out, i, p):
    if p['pitch_check']:
        write_json(out / 'pitch_request.json', {'tracks': {'before': str(i['before']), 'after': str(i['after'])}})
        ctx.execute('seed-vc', 'pitch', out, '--request', out / 'pitch_request.json')
    request = {key: str(i[key]) for key in ('original', 'final', 'final_mp3', 'metrics')}
    request['review'] = p['review']
    if p['review']:
        request.update(stems=str(i['stems/stems.json'].parent), midi=str(i['midi/midi.json'].parent))
    write_json(out / 'evaluation_request.json', request)
    ctx.execute('separator', 'evaluate', out, '--request', out / 'evaluation_request.json')
    if p['pitch_check']:
        adherence = evaluate(load_pitch(i['source_f0']), load_pitch(out / 'before_f0.npz'),
                             load_pitch(out / 'after_f0.npz'), json.loads(i['plan'].read_text()))
        write_json(out / 'pitch_adherence.json', adherence)
        evaluation = json.loads((out / 'evaluation.json').read_text())
        evaluation['pitch_adherence'] = adherence
        write_json(out / 'evaluation.json', evaluation)
    return artifact_paths(out)
