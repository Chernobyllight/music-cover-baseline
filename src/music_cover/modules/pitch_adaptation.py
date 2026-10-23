import shutil
from pathlib import Path
from ..common.io import write_json
from ..pitch import transpose_abc
from ..pitch_selection import DEFAULTS as SELECTION_DEFAULTS, settings, load_pitch, choose, score_consistency, export_score_midi, write_report

ID = '02_pitch_adaptation'
CONTRACT_VERSION = 2
DEFAULTS = {'timeout_seconds': 10800, **SELECTION_DEFAULTS}
CODE = ['pitch.py', 'pitch_selection.py', 'backends/separator.py', 'backends/rmvpe.py']
TIMELINE = {'score': 'original', 'original_score': 'original', 'score_midi': 'original', 'source_f0': 'original', 'reference_f0': 'reference_excerpt'}

def backends(r, c): return ['separator', 'rmvpe'] if str(r['transpose']) != '0' else []
def requires(r): return ['00_prepare', '01_source_score']
def inputs(ctx):
    result = {'score': ctx.output('01_source_score', 'score')}
    if str(ctx.request['transpose']) != '0':
        result.update(source=ctx.output('00_prepare', 'source'), reference=ctx.output('00_prepare', 'reference'),
                      vocal_events=ctx.output('01_source_score', 'file:melody_vocal.lab'))
        for index, path in enumerate(settings(ctx.config[ID])['range_references']):
            result[f'range_reference_{index:03d}'] = (ctx.root / Path(path).expanduser()).resolve()
    return result

def params(r, c): return {**DEFAULTS, **c, **settings(c), 'transpose': str(r['transpose'])}


def run(ctx, out, i, p):
    shutil.copyfile(i['score'], out / 'score_original.abc')
    result = {'score': out / 'score_transposed.abc', 'original_score': out / 'score_original.abc', 'plan': out / 'transposition.json'}
    if p['transpose'] == '0':
        shutil.copyfile(i['score'], out / 'score_transposed.abc')
        plan = {'selected_semitones': 0, 'applied_semitones': 0, 'recommended_semitones': None,
                'mode': 'disabled', 'status': 'disabled', 'seed_vc_semitones': 0}
    else:
        separation = out / 'source_separation'
        ctx.execute('separator', 'separate', separation, '--input', i['source'])
        tracks = {'source': str(separation / 'vocals.wav'), 'reference': str(i['reference'])}
        # Range references should be clean singing vocals, separate from the timbre crop.
        tracks.update({key: str(path) for key, path in i.items() if key.startswith('range_reference_')})
        write_json(out / 'pitch_request.json', {'tracks': tracks})
        ctx.execute('seed-vc', 'pitch', out, '--request', out / 'pitch_request.json')
        source = load_pitch(out / 'source_f0.npz')
        reference_names = [name for name in tracks if name.startswith('range_reference_')] or ['reference']
        references = [load_pitch(out / f'{name}_f0.npz') for name in reference_names]
        consistency = score_consistency(i['vocal_events'], source)
        write_json(out / 'score_audio_consistency.json', consistency)
        plan = choose(source, references, p['transpose'], p, consistency)
        plan['range_reference_inputs'] = {name: tracks[name] for name in reference_names}
        revised, invariants = transpose_abc(i['score'].read_text(), plan['applied_semitones'])
        plan.update(score_checks=invariants, seed_vc_semitones=0)
        (out / 'score_transposed.abc').write_text(revised)
        export_score_midi(revised, out / 'score_transposed.mid')
        write_report(out, plan)
        result.update(profiles=out / 'pitch_profiles.json', source_f0=out / 'source_f0.npz', reference_f0=out / 'reference_f0.npz',
                      score_midi=out / 'score_transposed.mid', report=out / 'pitch_report.html',
                      candidates=out / 'pitch_candidates.csv', consistency=out / 'score_audio_consistency.json')
        print(f"[pitch] recommended {plan['recommended_semitones']}; applied {plan['applied_semitones']:+d} semitones; {plan['status']}; confidence={plan['confidence']}", flush=True)
    write_json(out / 'transposition.json', plan)
    return result
