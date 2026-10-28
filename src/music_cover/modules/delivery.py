import shutil
import json
from ..audio import sha256
from ..voice_candidates import contained_file, render_report
from ..common.io import write_json, artifact_paths
from ..common.delivery import build_delivery
ID = '11_delivery'
CONTRACT_VERSION = 3
DEFAULTS = {'timeout_seconds': 10800}
CODE = ['voice_candidates.py', 'audio.py', 'common/delivery.py', 'common/validation.py', 'backends/package.py', 'backends/evaluation.py', 'common/review_audio.py']
TIMELINE = {'*': 'see source artifact timeline'}
def backends(r, c): return ['package'] if r['review'] else []
def requires(r):
    return ['00_prepare', '01_source_score', '02_pitch_adaptation', '03_style_generation', '04_vocal_separation', '06_mix', '10_evaluation'] + (['07_stem_separation', '08_final_score', '09_stem_midi'] if r['review'] else [])
def inputs(ctx):
    result = {}
    for stage in requires(ctx.request):
        result.update(ctx.all_outputs(stage, stage + '/'))
    # The source lyric bytes, not a mutable external file, are packaged.
    return result
def params(r, c): return {**c, 'request': r}


def run(ctx, out, i, p):
    get = lambda stage: {key[len(stage) + 1:]: value for key, value in i.items() if key.startswith(stage + '/')}
    prepared, generated, separated, mixed = (get(s) for s in ['00_prepare', '03_style_generation', '04_vocal_separation', '06_mix'])
    evaluation = get('10_evaluation')
    adaptation = get('02_pitch_adaptation') if str(ctx.request['transpose']) != '0' else None
    pitch_check = {'profiles': evaluation['pitch_profiles.json']} if 'pitch_profiles.json' in evaluation else None
    if pitch_check and 'pitch_adherence.json' in evaluation:
        pitch_check['adherence'] = evaluation['pitch_adherence.json']
    review = None
    if ctx.request['review']:
        review_dir = out / 'review'
        request = {'stems': str(get('07_stem_separation')['stems.json'].parent),
                   'transcription': str(get('08_final_score')['transcription_summary.json'].parent),
                   'midi': str(get('09_stem_midi')['midi.json'].parent),
                   'source_transcription': str(get('01_source_score')['score'].parent),
                   'original': str(prepared['source']), 'final': str(mixed['final']),
                   'lyrics': str(prepared['lyrics']), 'run_request': ctx.request,
                   'evaluation': str(evaluation['evaluation.json'])}
        if adaptation:
            request['pitch_adaptation'] = {adaptation[k].name: str(adaptation[k]) for k in
                ('score', 'original_score', 'plan', 'report', 'candidates', 'consistency', 'score_midi') if k in adaptation}
        if pitch_check and 'adherence' in pitch_check:
            request.setdefault('pitch_adaptation', {})['pitch_adherence.json'] = str(pitch_check['adherence'])
        request_file = out / 'package_request.json'
        write_json(request_file, request)
        ctx.execute('separator', 'review_package', review_dir, '--request', request_file)
        review = artifact_paths(review_dir)
    build_delivery(out, ctx.request, ctx.state['stages'], prepared, generated, separated, mixed,
                   adaptation, pitch_check, review=None)
    if review:
        page = out / 'listen.html'
        page.write_text(page.read_text().replace('<h1>', '<p><a href="review/index.html">成品分轨与 MIDI 专业评审包</a></p><h1>', 1))
        with (out / 'report.md').open('a') as f:
            f.write('\n成品分轨及 MIDI：review/index.html。人工评分请填写运行目录 reviews 中的副本。\n')
    if 'candidate_report' in mixed:
        report = json.loads(mixed['candidate_report'].read_text())
        for row in report['candidates']:
            for key in ('final_wav', 'final_mp3'):
                if key not in row: continue
                source = contained_file(mixed['candidate_report'].parent, row[key])
                if sha256(source) != row[key + '_sha256']: raise ValueError('Mixed candidate checksum mismatch')
                target = out / row[key]; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if row['id'] == report['recommended_id'] and sha256(out / key.replace('_', '.')) != sha256(target):
                    raise ValueError('Delivered final does not match recommended candidate')
        write_json(out / 'candidates.json', report)
        render_report(report, out / 'candidates.html', mixed=True)
        page = out / 'listen.html'
        page.write_text(page.read_text().replace('<h1>', '<p><a href="candidates.html">微调候选：三个版本与定量推荐</a></p><h1>', 1))
        with (out / 'report.md').open('a') as stream:
            stream.write('\n微调候选对比：candidates.html；推荐：' + report['recommended_id'] + '。指标取自 05 转换后干声；后续分轨、MIDI 对应推荐版本。\n')
    shutil.copyfile(evaluation['evaluation.json'], out / 'evaluation.json')
    for key in ['vocal_post_gain', 'instrumental_post_gain', 'residual']:
        shutil.copyfile(mixed[key], out / mixed[key].name)
    write_json(out / 'config.resolved.json', {'request': ctx.request, 'modules': ctx.config})
    request_file = out / 'validation_request.json'
    write_json(request_file, {'directory': str(out), 'review': ctx.request['review']})
    ctx.execute('separator' if ctx.request['review'] else 'control', 'validate_delivery', out, '--request', request_file)
    return artifact_paths(out)
