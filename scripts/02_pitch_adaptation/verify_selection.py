#!/usr/bin/env python3
"""Replay real F0 and known-shift controls without modifying any existing run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from music_cover.pitch_selection import load_pitch, choose
from music_cover.common.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-f0', required=True, type=Path)
    p.add_argument('--reference-f0', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source, reference = load_pitch(args.source_f0), load_pitch(args.reference_f0)
    config = {'reference_kind': 'singing'}
    real = choose(source, [reference], 'auto', config)
    controls = []
    for shift in [-3, 0, 3]:
        # These are artificial F0 controls, NOT new recordings or measured singer range.
        target = {**source, 'values': source['values'] + shift,
                  'blocks': [block + shift for block in source['blocks']],
                  'summary': {**source['summary'], 'quantiles_midi': {k: v + shift for k, v in source['summary']['quantiles_midi'].items()}}}
        result = choose(source, [target], 'auto', config)
        controls.append({'kind': 'synthetic_F0_control', 'known_shift': shift,
                         'recommended': result['recommended_semitones'], 'applied': result['applied_semitones'],
                         'passed': result['applied_semitones'] == shift})
    report = {'source_f0': str(args.source_f0.resolve()), 'reference_f0': str(args.reference_f0.resolve()),
              'real_data': real, 'controls': controls}
    write_json(args.output / 'selection_replay.json', report)
    print(json.dumps({'real_recommended': real['recommended_semitones'], 'real_applied': real['applied_semitones'],
                      'real_status': real['status'], 'controls': controls}, ensure_ascii=False, indent=2))
    if not all(c['passed'] for c in controls):
        raise SystemExit('Known-shift control failed')


if __name__ == '__main__': main()
