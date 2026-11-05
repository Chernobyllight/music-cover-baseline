"""Generate audio, assess quality, and retain a review output even if rejected."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from ..audio import sha256, inspect_audio
from ..common.io import write_json
from ..common.paths import ROOT
from ..common.resources import python_for
from ..voice import resolve_voice, voice_record
from ..voice_candidates import candidate_seeds, assessment, finite_diagnostics, select_output, selection_fields, render_report


def generate(request, output):
    out = Path(output)
    backend = request.get('backend', 'seed_vc')
    voice = resolve_voice('finetuned', request['voice_profile'], backend=backend)
    seeds = candidate_seeds(request['seed'], request['candidate_count'], request.get('candidate_seeds'))
    source, reference = Path(request['source']), Path(request['reference'])
    source_info = inspect_audio(source)
    report = {'schema_version': 2, 'mode': 'finetuned', 'voice': voice_record(voice),
              'source_sha256': sha256(source), 'reference_sha256': sha256(reference),
              'settings': {'steps': request['steps'], 'seeds': seeds, 'semitones': 0, 'auto_f0_adjust': False},
              'candidates': [], 'recommended_id': None, 'selected_id': None,
              'selection_rule': 'eligible highest ECAPA; otherwise generated highest finite ECAPA for review, then lower seed and ID; quality never blocks delivery',
              'reference_independence': ('same_inference_reference; diagnostic_only' if backend == 'seed_vc' else ('not_inference_condition; independence_requires_dataset_check' if backend in {'rvc', 'fish_diffusion'} else 'held_out_reference; not_inference_condition')),
              'backend': backend,
              'training_performed': False, 'subjective_quality': 'awaiting_human_review', 'lyrics_accuracy': 'not_evaluated'}
    for index, seed in enumerate(seeds, 1):
        name = f'candidate_{index:03d}'; folder = out/'candidates'/name; folder.mkdir(parents=True)
        row = {'id': name, 'seed': seed, 'status': 'failed', 'generation_status':'failed', 'evaluation_status':'not_run', 'eligible': False, 'rejection_reasons': []}
        command = [sys.executable, '-m', 'music_cover.backends.cli', 'convert', '--input', str(source),
                   '--reference', str(reference), '--output', str(folder), '--steps', str(request['steps']),
                   '--seed', str(seed), '--voice-mode', 'finetuned', '--voice-profile', str(voice['profile'])]
        if backend in {'so_vits_svc', 'rvc', 'fish_diffusion'}:
            write_json(folder/'request.json', {**request, 'seed': seed})
            command = [str(python_for(ROOT, backend)), '-m', 'music_cover.backends.cli',
                       {'rvc':'convert_rvc', 'so_vits_svc':'convert_svc', 'fish_diffusion':'convert_fish'}[backend], '--request', str(folder/'request.json'), '--output', str(folder)]
        write_json(folder/'command.json', {'argv': command})
        try:
            with (folder/'inference.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=request.get('timeout_seconds', 10800))
            inspect_audio(folder/'converted.wav')
            row.update(status='generated', generation_status='succeeded', audio=str((folder/'converted.wav').relative_to(out)),
                       audio_sha256=sha256(folder/'converted.wav'), conversion=str((folder/'conversion.json').relative_to(out)))
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            row.update(error=str(error), rejection_reasons=['inference_failed'])
        report['candidates'].append(row)
        write_json(out/'candidates.json', report)
        print(f'{name}: inference {row["status"]}; seed={seed}', flush=True)
    from .voice_metrics import VoiceMetrics
    metrics, initialization_error = None, None
    if any(row['generation_status']=='succeeded' for row in report['candidates']):
        try:
            metrics = VoiceMetrics()
        except Exception as error:
            initialization_error = f'{type(error).__name__}: {error}'
    for row in report['candidates']:
        if row['generation_status'] != 'succeeded': continue
        folder = out/'candidates'/row['id']
        try:
            if initialization_error: raise RuntimeError(initialization_error)
            result = metrics.compare(source, out/row['audio'], [reference], folder/'diagnostics')
            quality = assessment(result, source_info['duration_s'])
            result = finite_diagnostics(result)
            row.update(status='succeeded', evaluation_status='succeeded', metrics=result,
                       assessment=quality, eligible=quality['status']=='passed', rejection_reasons=quality['reasons'])
            print(f'{row["id"]}: ECAPA={result["speaker_similarity"]}, eligible={row["eligible"]}', flush=True)
        except Exception as error:
            row.update(status='succeeded', evaluation_status='failed', evaluation_error=f'{type(error).__name__}: {error}',
                       assessment={'status':'error','blocks_delivery':False,'reasons':['evaluation_failed']},
                       eligible=False, rejection_reasons=['evaluation_failed'])
        write_json(out/'candidates.json', report)
    selected = select_output(report['candidates'])
    report.update(selection_fields(report['candidates']))
    write_json(out/'candidates.json', report)
    with (out/'candidates.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['id','seed','ecapa','pitch_status','eligible','recommended','default_output','reasons'])
        for row in report['candidates']:
            m=row.get('metrics', {})
            writer.writerow([row['id'],row['seed'],m.get('speaker_similarity'),m.get('pitch',{}).get('status'),row['eligible'],row['id']==report['recommended_id'],row['id']==report['selected_id'],';'.join(row['rejection_reasons'])])
    render_report(report, out/'candidates.html')
    if selected is None:
        raise RuntimeError(f'No readable generated audio; inspect {out / "candidates.html"}. Inference failed.')
    shutil.copy2(out/selected['audio'], out/'converted.wav')
    conversion=json.loads((out/selected['conversion']).read_text())
    conversion['candidate_selection']={**selection_fields(report['candidates']), 'seed':selected['seed'],
        'ecapa':selected.get('metrics',{}).get('speaker_similarity'), 'report':'candidates.json'}
    conversion['assessment']=selected.get('assessment',{'status':'unassessable','blocks_delivery':False})
    write_json(out/'conversion.json', conversion)
