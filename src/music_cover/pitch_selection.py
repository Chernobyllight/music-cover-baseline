"""Auditable global transposition. Distribution matching never warps note intervals."""
from __future__ import annotations

import csv
import html
import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .common.io import write_json
from .pitch import abc_tools

METHOD = 'bidirectional_distribution_v2'
DEFAULTS = {
    'auto_min_semitones': -6, 'auto_max_semitones': 6,
    'shift_penalty': 0.04, 'minimum_improvement': 0.15,
    'minimum_voiced_seconds': 3.0, 'minimum_direction_support': 0.60,
    'reference_kind': 'unknown', 'range_references': [],
    'target_low_midi': None, 'target_high_midi': None,
    'range_penalty': 1.0,
}


def settings(config):
    c = {**DEFAULTS, **{k: v for k, v in config.items() if k in DEFAULTS}}
    for key in ('auto_min_semitones', 'auto_max_semitones'):
        if type(c[key]) is not int or not -12 <= c[key] <= 12:
            raise ValueError(f'{key} must be an integer in -12..12')
    if not c['auto_min_semitones'] <= 0 <= c['auto_max_semitones']:
        raise ValueError('Automatic candidate range must include zero')
    for key in ('shift_penalty', 'minimum_improvement', 'minimum_voiced_seconds', 'range_penalty'):
        if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if c['minimum_voiced_seconds'] < 3:
        raise ValueError('minimum_voiced_seconds must be >= 3')
    support = c['minimum_direction_support']
    if isinstance(support, bool) or not isinstance(support, (int, float)) or not math.isfinite(support) or not .5 <= support <= 1:
        raise ValueError('minimum_direction_support must be in 0.5..1')
    if c['reference_kind'] not in ('singing', 'speech', 'unknown'):
        raise ValueError('reference_kind must be singing, speech or unknown')
    if not isinstance(c['range_references'], list) or any(not isinstance(p, str) or not p for p in c['range_references']):
        raise ValueError('range_references must be a list of singing audio file paths')
    low, high = c['target_low_midi'], c['target_high_midi']
    if (low is None) != (high is None):
        raise ValueError('Provide both target_low_midi and target_high_midi, or neither')
    if low is not None:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)) or not 0 <= low < high <= 127:
            raise ValueError('Target MIDI range must satisfy 0 <= low < high <= 127')
    return c


def load_pitch(path):
    """RMS gate relative to active signal; reject isolated spikes, never flatten vibrato."""
    with np.load(path, allow_pickle=False) as d:
        f0, rms = np.asarray(d['f0'], float), np.asarray(d['rms'], float)
        hop = float(d['hop_seconds'])
    if f0.ndim != 1 or rms.ndim != 1 or len(f0) != len(rms) or not math.isfinite(hop) or hop <= 0:
        raise ValueError(f'Invalid F0 arrays: {path}')
    finite = np.isfinite(f0) & np.isfinite(rms)
    rms_values = rms[finite & (rms > 0)]
    floor = max(1e-5, float(np.percentile(rms_values, 90)) * .03) if len(rms_values) else 1e-5
    # This covers the model's useful musical range without excluding quiet low voices.
    valid = finite & (f0 >= 50) & (f0 <= 1100) & (rms >= floor)
    midi = np.zeros(len(f0))
    midi[valid] = 69 + 12 * np.log2(f0[valid] / 440)
    neighbors = np.convolve(valid.astype(int), np.ones(5, dtype=int), mode='same') if len(f0) >= 5 else np.zeros(len(f0))
    smooth = median_filter(midi, size=5, mode='nearest') if len(midi) else midi.copy()
    spikes = valid & (neighbors == 5) & (np.abs(midi - smooth) > 7)
    valid &= ~spikes
    values = midi[valid]
    blocks = []
    # Bootstrap whole time blocks, not correlated 10-ms frames.
    width = max(1, round(4 / hop))
    for start in range(0, len(midi), width):
        v = midi[start:start + width][valid[start:start + width]]
        if len(v) * hop >= .5:
            blocks.append(v)
    summary = {'frames': int(valid.sum()), 'duration_s': len(f0) * hop,
               'voiced_seconds': float(valid.sum() * hop), 'voiced_fraction': float(valid.mean()) if len(valid) else 0,
               'rms_floor': floor, 'isolated_spikes_removed': int(spikes.sum()),
               'nonfinite_frames': int((~finite).sum()), 'usable_blocks': len(blocks),
               'quantiles_midi': {f'p{p}': float(np.percentile(values, p)) for p in (5, 10, 25, 50, 75, 90, 95)} if len(values) else {}}
    return {'midi': midi, 'valid': valid, 'values': values, 'blocks': blocks, 'hop': hop, 'summary': summary}


def choose(source, references, requested, config, consistency=None):
    c = settings(config)
    quantiles = np.linspace(.1, .9, 81)
    eligible = [r for r in references if r['summary']['voiced_seconds'] >= c['minimum_voiced_seconds']]
    source_ok = source['summary']['voiced_seconds'] >= c['minimum_voiced_seconds']
    plan = {'algorithm': METHOD, 'mode': requested, 'auto_search_semitones': [c['auto_min_semitones'], c['auto_max_semitones']],
            'settings': c, 'recommended_semitones': None, 'applied_semitones': 0, 'selected_semitones': 0,
            'seed_vc_semitones': 0, 'auto_candidates': [], 'confidence': 'low', 'status': 'insufficient_reference',
            'reasons': [], 'reference_sample_only': c['target_low_midi'] is None,
            'source_quality': source['summary'], 'reference_quality': [r['summary'] for r in references],
            'reference_kind': c['reference_kind'], 'score_audio_consistency': consistency,
            'confidence_note': 'Heuristic reliability, not a calibrated probability or a singer vocal-range measurement.'}
    if source_ok and eligible:
        sq = np.quantile(source['values'], quantiles)
        # Each reference recording gets equal weight; a long clip cannot dominate all others.
        rq = np.mean([np.quantile(r['values'], quantiles) for r in eligible], axis=0)
        def rows_for(target):
            rows = []
            for shift in range(c['auto_min_semitones'], c['auto_max_semitones'] + 1):
                distance = float(np.mean(np.abs(sq + shift - target)))
                excess = 0.0
                if c['target_low_midi'] is not None:
                    v = source['values'] + shift
                    excess = float(np.mean(np.maximum(c['target_low_midi'] - v, 0) + np.maximum(v - c['target_high_midi'], 0)))
                penalty = c['shift_penalty'] * shift ** 2
                rows.append({'semitones': shift, 'distribution_distance': distance, 'range_excess': excess,
                             'shift_penalty': penalty, 'loss': distance + c['range_penalty'] * excess + penalty})
            return rows
        rows = rows_for(rq)
        ranked = sorted(rows, key=lambda r: (r['loss'], abs(r['semitones']), r['semitones']))
        best, zero = ranked[0], next(r for r in rows if r['semitones'] == 0)
        improvement = zero['loss'] - best['loss']
        # Deterministic block bootstrap; independent of the music generation seed.
        rng = np.random.default_rng(20261023)
        votes = []
        usable = all(len(r['blocks']) >= 2 for r in eligible)
        if usable:
            for _ in range(128):
                samples = []
                for reference in eligible:
                    blocks = reference['blocks']
                    v = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
                    samples.append(np.quantile(v, quantiles))
                candidate = min(rows_for(np.mean(samples, axis=0)), key=lambda r: (r['loss'], abs(r['semitones']), r['semitones']))
                votes.append(candidate['semitones'])
        best_shift = best['semitones']
        support = float(np.mean(np.sign(votes) == np.sign(best_shift))) if votes else 0.0
        near = float(np.mean(np.abs(np.asarray(votes) - best_shift) <= 1)) if votes else 0.0
        plan.update(recommended_semitones=best_shift, auto_candidates=rows, top_candidates=ranked[:3],
                    improvement_over_zero=improvement, runner_up_gap=ranked[1]['loss'] - best['loss'] if len(ranked) > 1 else 0,
                    bootstrap={'repetitions': len(votes), 'direction_support': support, 'within_one_semitone': near,
                               'candidate_counts': {str(k): votes.count(k) for k in sorted(set(votes))}},
                    distribution={'quantiles': quantiles.tolist(), 'source': sq.tolist(), 'reference': rq.tolist()},
                    status='recommended')
        adequate = c['reference_kind'] == 'singing' and usable and support >= c['minimum_direction_support']
        if best_shift == 0:
            plan.update(status='keep_original' if adequate else 'ambiguous_reference', confidence='medium' if adequate else 'low')
            plan['reasons'].append('Zero has the lowest penalized distribution cost.' if adequate else 'Zero ranks first, but reference evidence is limited.')
        elif c['reference_kind'] != 'singing':
            plan['status'] = 'reference_kind_unconfirmed'
            plan['reasons'].append('Speech/unknown timbre reference is not an established singing-range reference; only a provisional candidate is reported.')
        elif improvement < c['minimum_improvement']:
            plan['status'] = 'insignificant_improvement'
            plan['reasons'].append('Candidate improvement over zero is below the configured margin.')
        elif not adequate:
            plan['status'] = 'ambiguous_reference'
            plan['reasons'].append('Reference time-block resampling does not support a stable direction; zero retained.')
        else:
            plan.update(applied_semitones=best_shift, selected_semitones=best_shift, confidence='medium')
            if support >= .9 and near >= .8 and sum(r['summary']['voiced_seconds'] for r in eligible) >= 20:
                plan['confidence'] = 'high'
            plan['reasons'].append('Bidirectional distribution match improves on zero with sufficient block-resampled directional support.')
        if best_shift in (c['auto_min_semitones'], c['auto_max_semitones']) and best_shift != 0:
            plan['reasons'].append('Best candidate reaches the configured search boundary; inspect before expanding the range.')
    else:
        plan['status'] = 'insufficient_source' if not source_ok else 'insufficient_reference'
        plan['reasons'].append('Too little valid voiced audio; zero is a fallback, not a successful range match.')
    if consistency and consistency['status'] != 'consistent':
        if plan['confidence'] == 'high':
            plan['confidence'] = 'medium'
        plan['reasons'].append('Score/audio octave or alignment discrepancy recorded. No automatic octave correction; generated audio must be checked.')
    if requested != 'auto':
        value = int(requested)
        if not -12 <= value <= 12:
            raise ValueError('Manual transposition must be -12..12')
        plan.update(applied_semitones=value, selected_semitones=value, status='manual_override')
        plan['reasons'].append('Explicit manual shift applied independently of statistical recommendation.')
    return plan


def score_consistency(lab, source):
    differences = []
    for line in Path(lab).read_text().splitlines():
        start, end, note = map(float, line.split())
        a = max(0, round((start + .1 * (end - start)) / source['hop']))
        b = min(len(source['midi']), round((end - .1 * (end - start)) / source['hop']))
        values = source['midi'][a:b][source['valid'][a:b]]
        if len(values) * source['hop'] >= .1:
            differences.append(note - float(np.median(values)))
    if len(differences) < 10:
        return {'status': 'insufficient_alignment', 'matched_notes': len(differences), 'automatic_octave_correction': 0}
    values = np.asarray(differences)
    middle = float(np.median(values))
    octave = round(middle / 12) * 12
    support = float(np.mean(np.abs(values - octave) <= 1))
    status = 'consistent' if octave == 0 and support >= .7 else ('octave_offset_detected' if octave != 0 and support >= .7 else 'inconsistent')
    return {'status': status, 'matched_notes': len(values), 'score_minus_audio_median_semitones': middle,
            'nearest_octave_offset': octave, 'within_one_semitone_of_offset_fraction': support,
            'automatic_octave_correction': 0,
            'note': 'Two model estimates compared on original audio timestamps; no ground-truth or generator-octave guarantee.'}


def export_score_midi(text, path):
    """Export both ABC melody voices at the score tempo without an extra environment."""
    score = abc_tools().parse(text)
    def vlq(value):
        data = [value & 127]
        while value >> 7:
            value >>= 7
            data.insert(0, (value & 127) | 128)
        return bytes(data)
    def chunk(data):
        return b'MTrk' + struct.pack('>I', len(data)) + data
    tempo = round(60_000_000 / float(score.bpm))
    tracks = [chunk(b'\x00\xff\x51\x03' + tempo.to_bytes(3, 'big') + b'\x00\xff\x2f\x00')]
    for channel, name in enumerate(('Vocal', 'Ins')):
        events = []
        for start, pitch, duration in score.voices[name].notes:
            a, b = round(float(start) * 960), round(float(start + duration) * 960)
            events.extend([(a, 1, int(pitch)), (max(a + 1, b), 0, int(pitch))])
        data = b'\x00\xff\x03' + vlq(len(name)) + name.encode() + bytes([0, 0xC0 + channel, 0])
        previous = 0
        for tick, on, pitch in sorted(events):
            data += vlq(tick - previous) + bytes([(0x90 if on else 0x80) + channel, pitch, 80 if on else 0])
            previous = tick
        end = max(previous, round(float(score.voices[name].time) * 960))
        tracks.append(chunk(data + vlq(end - previous) + b'\xff\x2f\x00'))
    Path(path).write_bytes(b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), 960) + b''.join(tracks))


def write_report(out, plan):
    out = Path(out)
    rows = plan.get('auto_candidates', [])
    with (out / 'pitch_candidates.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['semitones', 'distribution_distance', 'range_excess', 'shift_penalty', 'loss'])
        writer.writeheader()
        writer.writerows(rows)
    def esc(value): return html.escape(str(value))
    table = ''.join(f'<tr><td>{r["semitones"]:+d}</td><td>{r["distribution_distance"]:.3f}</td><td>{r["range_excess"]:.3f}</td><td>{r["shift_penalty"]:.3f}</td><td>{r["loss"]:.3f}</td></tr>' for r in rows)
    # Inline SVG makes the report offline and portable; graph axes are MIDI semitones.
    chart = ''
    distribution = plan.get('distribution')
    if distribution:
        curves = [distribution['source'], distribution['reference'], [v + plan['applied_semitones'] for v in distribution['source']]]
        low = math.floor(min(min(v) for v in curves)) - 1
        high = math.ceil(max(max(v) for v in curves)) + 1
        chart = '<svg viewBox="0 0 760 280" role="img" aria-label="音高分位数对比">'
        for note in range(low, high + 1, 2):
            y = 235 - (note - low) / (high - low) * 200
            chart += f'<path d="M50 {y}H720" stroke="#ddd"/><text x="8" y="{y+5}">{note}</text>'
        for values, color, label in zip(curves, ['#737373', '#008070', '#c44a24'], ['原曲', '参考', '按实际移调预测']):
            points = ' '.join(f'{50+i/(len(values)-1)*670:.1f},{235-(v-low)/(high-low)*200:.1f}' for i,v in enumerate(values))
            chart += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        chart += '<text x="50" y="265">P10</text><text x="360" y="265">P50</text><text x="690" y="265">P90</text></svg><p>纵轴：MIDI 半音；灰：原曲，绿：参考，橙：移调后的预测分布（不是生成音频实测）。</p>'
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双向自动选调报告</title>
<style>body{{max-width:960px;margin:32px auto;padding:0 22px;font:16px/1.65 system-ui;background:#fafaf7;color:#222}}td,th{{padding:7px 14px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap}}svg{{width:100%}}</style>
<h1>双向自动选调</h1><p>算法：{METHOD} · 状态：{esc(plan['status'])} · 可信度：{esc(plan['confidence'])}</p>
<p><strong>统计推荐：{esc(plan['recommended_semitones'])} 半音；实际采用：{plan['applied_semitones']:+d} 半音。</strong></p>
<p>负数为降调，正数为升调，0 为原调。可信度是工程判断，不是校准概率。参考片段的观测音区不等于歌手完整音域。</p>
<ul>{''.join('<li>'+esc(reason)+'</li>' for reason in plan['reasons'])}</ul>
<h2>音高分布</h2>{chart}<h2>全部候选（评分越低越好）</h2><table><tr><th>半音</th><th>分布距离</th><th>人工范围超出量</th><th>幅度惩罚</th><th>总分</th></tr>{table}</table>
<p><a href="transposition.json">决策 JSON</a> · <a href="pitch_candidates.csv">候选 CSV</a> · <a href="score_audio_consistency.json">谱面与音频检查</a> · <a href="score_original.abc">原谱</a> · <a href="score_transposed.abc">移调谱</a> · <a href="score_transposed.mid">同步 MIDI</a></p>
<details><summary>完整参数和证据</summary><pre>{esc(json.dumps(plan, ensure_ascii=False, indent=2))}</pre></details></html>'''
    (out / 'pitch_report.html').write_text(body)
