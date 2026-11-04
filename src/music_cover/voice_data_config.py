"""Create a fresh dataset-preparation YAML from verified separated recordings."""
import argparse
from collections import Counter
import json
import math
import sys
import yaml
from .audio import sha256
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path
from .common.voice_config import (project_path, check_id, check_splits, fresh_targets,
                                 checked_recordings, stage_artifacts, write_new_config)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('voice-id', 'version', 'recordings-dir', 'source-run'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--validation-ids', nargs='+', required=True)
    p.add_argument('--test-ids', nargs='+', required=True)
    p.add_argument('--segment-seconds', type=float, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--dataset-dir')
    p.add_argument('--output')
    p.add_argument('--plan', action='store_true')
    return p


def build_config(a):
    check_id(a.voice_id, 'voice-id'); check_id(a.version, 'version')
    check_splits(a.validation_ids, a.test_ids)
    if a.gpu < 0 or not math.isfinite(a.segment_seconds) or not 1 <= a.segment_seconds <= 30:
        raise ValueError('gpu >= 0 and segment-seconds in [1, 30] are required')
    path = lambda value: project_path(value, ROOT)
    portable = lambda value: portable_path(value, ROOT)
    output = path(a.output or f'configs/voice_data/{a.voice_id}_{a.version}.yaml')
    dataset = path(a.dataset_dir or f'data/voice_training/{a.voice_id}/{a.version}/dataset')
    fresh_targets({'output': output, 'dataset-dir': dataset}, [path(a.recordings_dir), path(a.source_run)])
    rows, excluded, manifest = checked_recordings(a, ROOT)
    state = read_run_state(path(a.source_run), root=ROOT)
    reference = stage_artifacts(state, '00_prepare', ['reference'])['reference']
    if sha256(reference) in {r['sha256'] for r in rows}:
        raise ValueError('Reference prompt must be held out from the dataset')
    c = {'purpose': 'dataset_preparation_only', 'voice_id': a.voice_id, 'version': a.version,
         'gpu': a.gpu, 'smoke_test': False, 'segment_seconds': a.segment_seconds,
         'dataset_dir': portable(dataset), 'reference': portable(reference),
         'sources': [{'path': r['path'], 'group': f"{a.voice_id}:{r['id']}", 'split': r['split']} for r in rows],
         'configuration_generation': {'tool': 'music_cover.voice_data_config', 'schema_version': 1,
             'separation_manifest': portable(manifest), 'separation_manifest_sha256': sha256(manifest),
             'excluded_ids': excluded, 'reference_sha256': sha256(reference)}}
    summary = {'python': sys.executable, 'output': str(output), 'dataset_dir': str(dataset),
               'recordings': dict(Counter(r['split'] for r in rows)), 'excluded_ids': excluded,
               'seconds': {s: round(sum(r['duration_s'] for r in rows if r['split'] == s), 3)
                           for s in ('train', 'validation', 'test')}}
    return c, output, summary


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    try:
        config, output, summary = build_config(a)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        if a.plan:
            print('PLAN: checks passed; no files created.')
        else:
            write_new_config(output, config)
            print(f'Created dataset config: {output}')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
