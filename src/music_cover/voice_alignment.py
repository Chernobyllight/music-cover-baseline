"""Independent onset check when singing timbre changes destabilize RMS alignment."""
import math
import numpy as np
from scipy import signal
from .audio import envelope


def onset_alignment(source, converted, rate):
    hop = round(rate * .01)
    a, b = envelope(source, hop), envelope(converted, hop)
    length = min(len(a), len(b))
    width = min(length, 2000)
    threshold = max(1e-4, float(a.max(initial=0)) * .02)
    rows, excluded = [], []
    if width:
        starts = sorted(set(list(range(0, max(1, length - width + 1), width)) + [max(0, length - width)]))
    else:
        starts = []
    for start in starts:
        x, y = a[start:start + width], b[start:start + width]
        active = float(np.mean(x > threshold))
        if len(x) < 100 or active < .2:
            excluded.append({'start_s': start * hop / rate, 'source_active_fraction': active})
            continue
        # Positive envelope changes emphasize syllable attacks instead of the
        # sustained energy shape, which can change with the target timbre.
        x, y = np.maximum(np.diff(x), 0), np.maximum(np.diff(y), 0)
        x, y = x - x.mean(), y - y.mean()
        norm = float(np.linalg.norm(x) * np.linalg.norm(y))
        confidence, lag = 0., 0
        if norm > 1e-12:
            corr = signal.correlate(y, x, mode='full', method='fft') / norm
            lags = signal.correlation_lags(len(y), len(x), mode='full')
            keep = np.abs(lags) <= 25
            best = int(np.argmax(corr[keep]))
            confidence, lag = float(corr[keep][best]), int(lags[keep][best])
        rows.append({'start_s': start * hop / rate, 'source_active_fraction': active,
                     'lag_ms': lag * hop / rate * 1000, 'correlation': confidence})
    accepted = [row['lag_ms'] for row in rows if row['correlation'] >= .3]
    minimum = max(3 if length > 4000 else 1, math.ceil(.7 * len(rows)))
    consistent = len(accepted) >= minimum and max(accepted) - min(accepted) <= 30
    delay = float(np.median(accepted)) if consistent else 0.
    return {'method': 'source_active_positive_envelope_onsets_v1', 'windows': rows,
            'excluded_low_activity_windows': excluded, 'minimum_accepted_windows': minimum,
            'consistent': consistent, 'estimated_delay_ms': delay,
            'applied_shift_samples': -round(delay * rate / 1000),
            'needs_listening_review': not consistent or abs(delay) >= 200,
            'thresholds': {'min_correlation': .3, 'max_lag_spread_ms': 30,
                           'min_active_fraction': .2, 'min_accepted_fraction': .7}}


def active_envelope_alignment(source, converted, rate):
    """Compare enough active source windows; quiet intros/outros carry no vote."""
    hop = round(rate * .01)
    a, b = envelope(source, hop), envelope(converted, hop)
    length = min(len(a), len(b)); width = min(length, 2000)
    threshold = max(1e-4, float(a.max(initial=0)) * .02)
    rows, excluded = [], []
    starts = sorted(set(list(range(0, max(1, length-width+1), width)) + [max(0,length-width)])) if width else []
    for start in starts:
        x, y = a[start:start+width], b[start:start+width]
        active = float(np.mean(x > threshold))
        if len(x) < 100 or active < .2:
            excluded.append({'start_s':start*hop/rate, 'source_active_fraction':active}); continue
        x, y = x-x.mean(), y-y.mean()
        norm = float(np.linalg.norm(x)*np.linalg.norm(y)); confidence, lag = 0., 0
        if norm > 1e-12:
            corr = signal.correlate(y,x,mode='full',method='fft')/norm
            lags = signal.correlation_lags(len(y),len(x),mode='full'); keep = np.abs(lags)<=25
            best = int(np.argmax(corr[keep])); confidence, lag = float(corr[keep][best]), int(lags[keep][best])
        rows.append({'start_s':start*hop/rate, 'source_active_fraction':active,
                     'lag_ms':lag*hop/rate*1000, 'correlation':confidence})
    accepted = [row['lag_ms'] for row in rows if row['correlation'] >= .3]
    minimum = max(3 if length > 4000 else 1, math.ceil(.7*len(rows)))
    enough = len(accepted)>=minimum
    consistent = enough and max(accepted)-min(accepted)<=30
    delay = float(np.median(accepted)) if consistent else 0.
    return {'method':'source_active_envelope_v2', 'windows':rows,
            'excluded_low_activity_windows':excluded, 'minimum_accepted_windows':minimum,
            'consistent':consistent, 'estimated_delay_ms':delay,
            'applied_shift_samples':-round(delay*rate/1000),
            'needs_listening_review':not consistent or abs(delay)>=200,
            'status':'verified' if consistent else ('needs_review' if enough else 'unassessable'),
            'thresholds':{'min_correlation':.3,'max_lag_spread_ms':30,
                          'min_active_fraction':.2,'min_accepted_fraction':.7}}


def assess_alignment(source, converted, rate):
    """Shared across models, training validation, voice evaluation and mixing."""
    primary = active_envelope_alignment(source, converted, rate)
    if primary['consistent']:
        return primary
    onset = onset_alignment(source, converted, rate)
    if onset['consistent']:
        return {**onset, 'status':'verified', 'original_envelope_check':primary}
    enough = sum(row['correlation']>=.3 for row in onset['windows']) >= onset['minimum_accepted_windows']
    return {**primary, 'onset_check':onset,
            'status':'needs_review' if primary['status']=='needs_review' or enough else 'unassessable',
            'note':'No reliable global correction; preserve timing and request listening review.'}
