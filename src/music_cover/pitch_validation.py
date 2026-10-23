"""Conservative, octave-aware checks of generated and converted singing F0."""
from __future__ import annotations

import numpy as np


def bins(profile, seconds=.2):
    width = max(1, round(seconds / profile['hop']))
    result = []
    for start in range(0, len(profile['midi']), width):
        values = profile['midi'][start:start + width][profile['valid'][start:start + width]]
        result.append(float(np.median(values)) if len(values) >= max(1, width // 3) else np.nan)
    return np.asarray(result)


def summarize(differences, expected, coverage, alignment, minimum_pairs=30):
    d = np.asarray(differences)
    result = {'expected_semitones': expected, 'paired_bins': len(d), 'source_coverage': coverage,
              'alignment': alignment, 'status': 'unassessable',
              'thresholds': {'minimum_paired_bins': minimum_pairs, 'minimum_coverage': .6,
                             'maximum_median_error_semitones': .5, 'minimum_within_one_semitone_fraction': .6},
              'note': 'F0 model estimates and heuristic alignment; not a calibrated accuracy or listening verdict.'}
    if not len(d):
        return result
    shift = float(np.median(d))
    deviation = float(np.median(np.abs(d - shift)))
    in_tune = float(np.mean(np.abs(d - expected) <= 1))
    octave = round((shift - expected) / 12) * 12
    result.update(observed_semitones=shift, median_absolute_deviation=deviation,
                  within_one_semitone_fraction=in_tune,
                  nearest_octave_error=octave,
                  octave_error_support=float(np.mean(np.abs(d - expected - octave) <= 1)))
    if len(d) >= minimum_pairs and coverage >= .6 and deviation <= 1.5:
        result['status'] = 'verified' if abs(shift - expected) <= .5 and in_tune >= .6 else 'needs_review'
    return result


def compare_generated(source, generated, expected):
    """DTW relative contours; requested shift never influences the alignment."""
    a, b = bins(source), bins(generated)
    x, y = a[np.isfinite(a)], b[np.isfinite(b)]
    n, m = len(x), len(y)
    if min(n, m) < 30:
        return summarize([], expected, 0, 'relative_contour_dtw')
    xc, yc = x - np.median(x), y - np.median(y)
    # Banded monotonic alignment of voiced bins. No chroma folding: octaves remain visible.
    previous = np.full(m + 1, np.inf)
    previous[0] = 0
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    radius = max(32, round(.2 * max(n, m)))
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        center = i * m / n
        for j in range(max(1, int(center - radius)), min(m, int(center + radius)) + 1):
            costs = (previous[j - 1], previous[j] + .3, current[j - 1] + .3)
            step = min(range(3), key=costs.__getitem__)
            current[j] = costs[step] + min(abs(xc[i - 1] - yc[j - 1]), 12)
            trace[i, j] = step + 1
        previous = current
    if not np.isfinite(previous[m]):
        return summarize([], expected, 0, 'relative_contour_dtw')
    pairs = {}
    i, j = n, m
    while i > 0 and j > 0:
        pairs.setdefault(i - 1, []).append(j - 1)
        step = trace[i, j]
        if step == 1: i, j = i - 1, j - 1
        elif step == 2: i -= 1
        elif step == 3: j -= 1
        else: break
    differences = [float(np.median(y[js])) - x[k] for k, js in pairs.items()]
    result = summarize(differences, expected, len(pairs) / n, 'relative_contour_dtw')
    ratio = m / n
    result.update(voiced_length_ratio=ratio, aligned_source_bins=len(pairs), source_voiced_bins=n,
                  generated_voiced_bins=m, bin_seconds=.2,
                  alignment_warning='Voiced-bin alignment can hide missing phrases; verify lyrics and structure separately.')
    if not .5 <= ratio <= 2:
        result['status'] = 'unassessable'
    return result


def compare_conversion(before, after):
    a, b = bins(before), bins(after)
    length = min(len(a), len(b))
    valid = np.isfinite(a[:length]) & np.isfinite(b[:length])
    differences = b[:length][valid] - a[:length][valid]
    return summarize(differences, 0, float(valid.sum() / max(1, np.isfinite(a).sum())), 'same_time_after_latency_correction')


def evaluate(source, before, after, plan):
    generation = compare_generated(source, before, plan.get('applied_semitones', plan['selected_semitones']))
    conversion = compare_conversion(before, after)
    return {'schema_version': 1, 'requested_shift': plan.get('applied_semitones', plan['selected_semitones']),
            'generation': generation, 'voice_conversion': conversion,
            'status': 'verified' if generation['status'] == conversion['status'] == 'verified' else 'needs_review',
            'automatic_audio_correction': False,
            'note': 'Pipeline completion and pitch adherence are separate. Review an uncertain or mismatching result; no hidden extra pitch shift is applied.'}
