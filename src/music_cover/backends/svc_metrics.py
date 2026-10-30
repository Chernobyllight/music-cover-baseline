"""Evaluate SVC in the existing metrics environment; never import SVC here."""
from pathlib import Path
from .voice_metrics import VoiceMetrics
from ..audio import inspect_audio
from ..voice_candidates import rejection_reasons
from ..common.io import write_json


def evaluate(request, output):
    metrics=VoiceMetrics();rows=[]
    for job in request['jobs']:
        result=metrics.compare(job['source'],job['converted'],job['references'],job['output'])
        reasons=rejection_reasons(result,inspect_audio(job['source'])['duration_s'])
        rows.append({**job,'metrics':result,'eligible':not reasons,'rejection_reasons':reasons})
        write_json(Path(output)/'metrics.json',{'cases':rows})
        print(f"case={job.get('case')} ECAPA={result['speaker_similarity']:.5f} eligible={not reasons}",flush=True)
