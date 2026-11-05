"""Run only Fish 05 and shared 06; never regenerate or mutate source 00–04."""
import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from .audio import sha256, inspect_audio
from .common.paths import ROOT, read_run_state
from .common.voice_config import stage_artifacts
from .common.voice_paths import portable_path
from .common.io import write_json


def artifacts(run):
    state=read_run_state(run,root=ROOT)
    stems=stage_artifacts(state,'04_vocal_separation',['vocals','instrumental'])
    voice=stage_artifacts(state,'05_voice_conversion',['converted','conversion'])
    mixed=stage_artifacts(state,'06_mix',['final','final_mp3','metrics'])
    conversion=json.loads(voice['conversion'].read_text())
    return {'run':portable_path(run),'backend':conversion.get('backend'),
        'source_sha256':sha256(stems['vocals']),'instrumental_sha256':sha256(stems['instrumental']),
        'converted':voice['converted'],'final':mixed['final'],'final_mp3':mixed['final_mp3'],
        'conversion':conversion,'mix':json.loads(mixed['metrics'].read_text())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--config',required=True)
    p.add_argument('--source-run');p.add_argument('--compare-run',action='append',default=[])
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--force',action='store_true')
    a=p.parse_args();run=(ROOT/a.run).resolve();config=(ROOT/a.config).resolve()
    import yaml
    settings=yaml.safe_load(config.read_text()).get('modules',{}).get('05_voice_conversion',{})
    if settings.get('backend')!='fish_diffusion' or settings.get('mode')!='finetuned':
        raise ValueError('Validation requires an explicit Fish Diffusion finetuned config')
    if a.source_run:
        if run.exists():raise ValueError('Destination exists; omit --source-run to resume it')
        subprocess.run(['bash',str(ROOT/'scripts/05_voice_conversion/fork_run.sh'),
                        '--source-run',a.source_run,'--run',str(run)],check=True)
    before=stage_artifacts(read_run_state(run,root=ROOT),'04_vocal_separation',['vocals','instrumental'])
    hashes={k:sha256(v) for k,v in before.items()}
    for entry in ('05_voice_conversion/fish_diffusion','06_mix'):
        cmd=['bash',str(ROOT/'scripts'/entry/'run.sh'),'--run',str(run),'--config',str(config),'--gpu',str(a.gpu)]
        if a.force:cmd.append('--force')
        subprocess.run(cmd,check=True,cwd=ROOT)
    if hashes!={k:sha256(v) for k,v in before.items()}:raise RuntimeError('Source stems changed during validation')
    folder=run/'fish_validation';folder.mkdir(exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    from .backends.voice_metrics import VoiceMetrics
    try:metrics=VoiceMetrics();error=None
    except Exception as exc:metrics=None;error=f'{type(exc).__name__}: {exc}'
    ref=stage_artifacts(read_run_state(run,root=ROOT),'00_prepare',['reference'])['reference']
    rows=[]
    for index,item_run in enumerate([run,*[(ROOT/x).resolve() for x in a.compare_run]]):
        row=artifacts(item_run)
        if row['source_sha256']!=hashes['vocals'] or row['instrumental_sha256']!=hashes['instrumental']:
            raise ValueError('Comparison requires the identical 04 vocals and instrumental')
        try:
            if error:raise RuntimeError(error)
            result=metrics.compare(before['vocals'],row['converted'],[ref],folder/f'metrics_{index:02d}')
            row.update(metrics=result,evaluation_status='succeeded')
        except Exception as exc:row.update(evaluation_status='failed',evaluation_error=f'{type(exc).__name__}: {exc}')
        for key in ('converted','final','final_mp3'):row[key]=portable_path(row[key])
        rows.append(row)
    report={'backend':'fish_diffusion','same_upstream_verified':True,'source_hashes':hashes,'results':rows,
            'reference_independence':'not necessarily held out; see training validation for independent data',
            'subjective_quality':'awaiting_human_review','python':sys.executable}
    write_json(folder/'comparison.json',report)
    sections=[]
    for row in rows:
        media=os.path.relpath(ROOT/row['final_mp3'],folder)
        m=row.get('metrics',{})
        sections.append(f'<section><h2>{html.escape(str(row["backend"]))}</h2><audio controls src="{html.escape(media)}"></audio><p>ECAPA: {m.get("speaker_similarity", "unavailable")}</p><pre>{html.escape(json.dumps(m or row.get("evaluation_error"),ensure_ascii=False,indent=2))}</pre></section>')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fish 05–06 validation</title><h1>Fish Diffusion 05–06 验证</h1><p>相同源人声与伴奏；质量不合格仍保留试听结果。指标不是主观听感结论。</p>'+''.join(sections))
    print(json.dumps({'report':str(folder/'comparison.json'),'results':[{'backend':r['backend'],
        'ecapa':r.get('metrics',{}).get('speaker_similarity'),'final_mp3':r['final_mp3']} for r in rows]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
