"""Run identical source/prompt/seed through base and candidate weights."""
import csv
import html
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from ..audio import decode, read_at, shift_audio, estimate_alignment, sha256, export_mp3
from ..common.io import write_json
from ..common.paths import ROOT
from .voice_metrics import VoiceMetrics


def infer(source, reference, out, settings, checkpoint=None, config=None):
    out.mkdir(parents=True)
    argv = [sys.executable, '-m', 'music_cover.backends.cli', 'convert', '--input', str(source),
            '--reference', str(reference), '--output', str(out), '--steps', str(settings.get('steps', 30)),
            '--seed', str(settings.get('seed', 42))]
    if checkpoint: argv += ['--checkpoint', str(checkpoint), '--model-config', str(config)]
    print(f'Converting {source.name}: {checkpoint.name if checkpoint else "zero_shot"}; log: {out / "inference.log"}', flush=True)
    with (out / 'inference.log').open('w') as log:
        try:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=settings.get('timeout_seconds', 10800))
        except subprocess.CalledProcessError as error:
            tail = (out / 'inference.log').read_text(errors='replace')[-2000:]
            raise RuntimeError(f'Voice inference failed; see {out / "inference.log"}\n{tail}') from error
    return out / 'converted.wav'


def validate_voice(c, run, out, requested, split):
    from ..voice_training import path, dataset_manifest
    manifest = dataset_manifest(c)
    candidates = [path(x) for x in requested] if requested else sorted((run / 'checkpoints').glob('step_*.pth'))
    if not candidates: raise ValueError('No training checkpoints available')
    import torch
    training_state = json.loads((run / 'state.json').read_text())
    trained = json.loads((Path(training_state['stages']['01_train']['output']) / 'training.json').read_text())
    for checkpoint in candidates:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload.get('training_identity') != trained['training_identity']:
            raise ValueError('Candidate checkpoint does not belong to this training configuration')
        del payload
    # Test set may audit one already selected candidate, never rank checkpoints on the test set.
    if split == 'test' and len(candidates) != 1: raise ValueError('Test evaluation requires exactly one --checkpoint selected on validation')
    if split == 'test':
        reports = list((run / 'stages/02_validate').glob('attempt_*/artifacts/validation.json'))
        selected_before = [json.loads(p.read_text()) for p in reports]
        if not any(r['evaluation_split'] == 'validation' and r['selected_checkpoint'] == str(candidates[0])
                   and r['dataset_identity'] == manifest['identity'] for r in selected_before):
            raise ValueError('Run validation selection before auditing the selected checkpoint on test')
    refs = [path(c['dataset_dir']) / x['path'] for x in manifest['clips'] if x['split'] == split]
    independent = bool(refs) and not c.get('smoke_test')
    reference = path(c['reference'])
    if not refs:
        if not c.get('smoke_test'): raise ValueError(f'No {split} references')
        refs = [reference]
    source_run = path(c['validation']['source_run'])
    state = json.loads((source_run / 'stage_state.json').read_text())
    source_items = state['stages']['04_vocal_separation']['outputs']
    for role in ('vocals', 'instrumental'):
        item = source_items[role]
        if sha256(item['path']) != item['sha256']: raise ValueError(f'Corrupt source stage output: {role}')
    prompt = out / 'reference.wav'
    decode(reference, prompt, rate=44100, channels=1)
    cases = []
    for i, window in enumerate(c['validation']['windows']):
        folder = out / f'case_{i + 1:03d}'; folder.mkdir()
        source, backing = folder / 'source.wav', folder / 'instrumental.wav'
        for role, target in [('vocals', source), ('instrumental', backing)]:
            decode(source_items[role]['path'], target, rate=44100, channels=1 if role == 'vocals' else 2,
                   start=window['start'], duration=window['duration'])
        baseline = infer(source, prompt, folder / 'zero_shot', c['validation'])
        cases.append({'folder': folder, 'source': source, 'backing': backing, 'baseline': baseline, 'window': window})
    metrics = VoiceMetrics()
    for case in cases:
        case['base_metrics'] = metrics.compare(case['source'], case['baseline'], refs, case['folder'] / 'zero_shot/diagnostics')
    results, listening, key = [], [], []
    for checkpoint in candidates:
        label = checkpoint.stem
        rows = []
        for i, case in enumerate(cases):
            folder = case['folder'] / label
            converted = infer(case['source'], prompt, folder, c['validation'], checkpoint, run / 'model_config.yml')
            row = metrics.compare(case['source'], converted, refs, folder / 'diagnostics')
            row['baseline_speaker_similarity'] = case['base_metrics']['speaker_similarity']
            row['speaker_similarity_delta'] = row['speaker_similarity'] - row['baseline_speaker_similarity']
            row['case'] = i + 1
            print(f'{label} case={i+1}: similarity_delta={row["speaker_similarity_delta"]:+.4f}, pitch={row["pitch"]["status"]}', flush=True)
            rows.append(row)
            blind = out / 'listening' / f'{len(listening) + 1:03d}'; blind.mkdir(parents=True)
            files = [('zero_shot', case['baseline']), ('finetuned', converted)]
            random.Random(42 + len(listening)).shuffle(files)
            backing = read_at(case['backing'], 48000)
            source_audio = read_at(case['source'], 48000)
            mixed = []
            for mode, file in files:
                vocal = read_at(file, 48000)
                shift = estimate_alignment(source_audio, vocal, 48000)['applied_shift_samples']
                vocal = shift_audio(vocal, shift, len(backing))
                vocal *= max(float(np.sqrt(np.mean(source_audio ** 2))), 1e-5) / max(float(np.sqrt(np.mean(vocal ** 2))), 1e-5)
                mixed.append(backing + vocal)
            gain = min(1., .89 / max(float(np.max(np.abs(x))) for x in mixed))
            for letter, (mode, file), mix in zip(['A', 'B'], files, mixed):
                shutil.copy2(file, blind / f'{letter}.wav')
                sf.write(blind / f'{letter}_mix.wav', mix * gain, 48000, subtype='PCM_24')
                export_mp3(blind / f'{letter}_mix.wav', blind / f'{letter}_mix.mp3')
                key.append({'trial': blind.name, 'label': letter, 'mode': mode, 'checkpoint': str(checkpoint), 'case': i + 1})
            shutil.copy2(case['source'], blind / 'source.wav'); shutil.copy2(prompt, blind / 'reference.wav')
            rel = blind.relative_to(out).as_posix()
            listening.append(f'<article><h2>试听组 {blind.name}</h2><p>参考音色</p><audio controls src="{rel}/reference.wav"></audio><p>源人声</p><audio controls src="{rel}/source.wav"></audio>' +
                ''.join(f'<p>{letter} 干声 / 相同伴奏混音</p><audio controls src="{rel}/{letter}.wav"></audio><audio controls src="{rel}/{letter}_mix.mp3"></audio>' for letter in ['A','B']) + '</article>')
        results.append({'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'cases': rows,
                        'mean_similarity_delta': float(np.mean([x['speaker_similarity_delta'] for x in rows])),
                        'pitch_verified_cases': sum(x['pitch']['status'] == 'verified' for x in rows),
                        'eligible': all(x['pitch']['status'] == 'verified' for x in rows)})
    eligible = [x for x in results if x['pitch_verified_cases'] == len(cases)]
    selected = max(eligible or results, key=lambda x: x['mean_similarity_delta'])
    report = {'schema_version': 1, 'dataset_identity': manifest['identity'], 'evaluation_split': split,
              'reference_independent': independent, 'candidates': results, 'selected_checkpoint': selected['checkpoint'],
              'selection': 'highest speaker-similarity delta among pitch-verified candidates; fallback flagged if none',
              'pitch_eligible_candidate_exists': bool(eligible), 'source_run': str(source_run),
              'source_windows': [x['window'] for x in cases], 'reference_sha256': sha256(prompt),
              'technical_validation': 'passed', 'subjective_quality': 'awaiting_human_review',
              'quality_conclusion': 'integration_only' if not independent else 'requires_independent_test_and_listening',
              'test_references': [str(x) for x in refs], 'listening_key': 'listening_key.json'}
    write_json(out / 'validation.json', report)
    write_json(out / 'listening_key.json', key)
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['checkpoint','case','base_similarity','ft_similarity','delta','pitch_status'])
        for candidate in results:
            for row in candidate['cases']:
                writer.writerow([candidate['checkpoint'],row['case'],row['baseline_speaker_similarity'],row['speaker_similarity'],row['speaker_similarity_delta'],row['pitch']['status']])
    with (out / 'review_scores.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['reviewer','trial','preferred_timbre_A_B_tie','naturalness_A_1_5','naturalness_B_1_5','lyrics_errors','pitch_errors','notes'])
        for i in range(len(listening)): writer.writerow(['',f'{i+1:03d}','','','','','',''])
    (out / 'listening.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>音色微调对照试听</title><style>body{font:16px system-ui;max-width:950px;margin:40px auto;background:#f6f7fa}article{background:white;padding:24px;margin:20px 0;border-radius:12px}audio{width:44%;margin-right:3%}</style><h1>音色微调对照试听</h1><p>同一源人声、参考片段与推理参数。A/B 标签随机；请填写 review_scores.csv。身份解码表另存。</p><p>' + html.escape(report['quality_conclusion']) + '</p>' + ''.join(listening) + '</html>')
