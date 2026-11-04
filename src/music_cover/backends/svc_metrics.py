"""Evaluate SVC in the existing metrics environment; never import SVC here."""
from pathlib import Path
from .voice_metrics import VoiceMetrics
from ..audio import inspect_audio
from ..voice_candidates import assessment
from ..common.io import write_json


def evaluate(request, output):
    metrics=VoiceMetrics();rows=[]
    for job in request['jobs']:
        result=metrics.compare(job['source'],job['converted'],job['references'],job['output'])
        quality=assessment(result,inspect_audio(job['source'])['duration_s']);reasons=quality['reasons']
        rows.append({**job,'metrics':result,'assessment':quality,'eligible':not reasons,'rejection_reasons':reasons})
        write_json(Path(output)/'metrics.json',{'cases':rows})
        print(f"case={job.get('case')} ECAPA={result['speaker_similarity']:.5f} eligible={not reasons}",flush=True)
