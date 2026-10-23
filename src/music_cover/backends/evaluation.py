import json
from pathlib import Path
from ..audio import inspect_audio, sha256
from ..common.io import write_json
from ..common.validation import verify_components, verify_delivery
from .analysis import describe_audio


def evaluate(request, output):
    output = Path(output)
    metrics = json.loads(Path(request['metrics']).read_text())
    final = inspect_audio(request['final'])
    mp3 = inspect_audio(request['final_mp3'])
    if final['sha256'] != metrics['final']['sha256'] or mp3['sha256'] != metrics['final_mp3']['sha256']:
        raise ValueError('Mixed audio checksum differs from mix.json')
    if final['sample_rate'] != 48000 or final['channels'] != 2 or abs(final['duration_s'] - mp3['duration_s']) > 0.1:
        raise ValueError('Final WAV/MP3 contract failed')
    result = {'technical_validation': 'passed', 'subjective_quality': 'awaiting_human_review',
              'audio_comparison': {k: describe_audio(request[k]) for k in ('original', 'final')},
              'mix_checks': metrics, 'review_enabled': request['review']}
    if request['review']:
        result['review_checks'] = verify_components(Path(request['final']), Path(request['stems']), Path(request['midi']))
    write_json(output / 'evaluation.json', result)


def validate_delivery(request, output):
    result = verify_delivery(Path(request['directory']), request['review'])
    write_json(Path(output) / 'delivery_validation.json', result)
