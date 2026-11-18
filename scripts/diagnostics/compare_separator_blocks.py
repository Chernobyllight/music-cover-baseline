"""Compare two completed single-block workers and their corresponding released stems."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def errors(actual, reference):
    delta = actual.astype(np.float64) - reference
    return {'max_abs': float(np.abs(delta).max()),
            'relative_rms': float(np.sqrt(np.mean(delta**2)/np.mean(reference.astype(np.float64)**2)))}


def compare(old, new, run):
    reports = [json.loads((folder/'block.json').read_text()) for folder in (old, new)]
    a,b = reports
    for key in ('input_sha256','start_sample','sample_rate','sdpa_first'):
        if a[key] != b[key]:
            raise ValueError(f'{key} differs between workers')
    state = json.loads((run/'stage_state.json').read_text())['stages']['03_separate']
    result = {'workers': reports, 'stems': {}}
    start = a['start_sample']
    for stem in ('vocals','instrumental'):
        (ref,rate), (actual,rate2) = [sf.read(folder/(stem+'.wav'), dtype='float32') for folder in (old,new)]
        assert ref.shape == actual.shape == (352800,2) and rate == rate2 == 44100
        assert np.isfinite(ref).all() and np.isfinite(actual).all()
        delivered, rate3 = sf.read(state['outputs'][stem]['path'], dtype='float32')
        assert rate3 == rate
        block = delivered[start:start+len(ref)]
        # The final block is anchored at the end; interior blocks meet at
        # multiples of the stride. Inspect local jumps at the two clip edges.
        seams = [start, start+len(ref)]
        jumps = {str(i): float(np.abs(delivered[i]-delivered[i-1]).max()) for i in seams}
        result['stems'][stem] = {'samples': len(ref), 'channels': ref.shape[1],
            'baseline_peak':float(np.abs(ref).max()), 'candidate_peak':float(np.abs(actual).max()),
            'baseline_vs_delivered': errors(ref, block), 'new_vs_old':errors(actual,ref),
            'delivered_edge_sample_jumps':jumps,
            'candidate_edge_vs_baseline_max_abs':float(np.abs(actual[[0,-1]]-ref[[0,-1]]).max())}
    if 'second_s' in a and 'second_s' in b:
        gain = a['second_s'] - b['second_s']
        load_delta = b['load_and_input_s'] - a['load_and_input_s']
        result.update(warm_time_reduction=gain/a['second_s'], load_delta_s=load_delta,
            estimated_saving_26_blocks_s=a['first_s']-b['first_s']+25*gain-load_delta)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('old', 'new', 'run'):
        parser.add_argument(arg,type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.old,args.new,args.run),indent=2))
