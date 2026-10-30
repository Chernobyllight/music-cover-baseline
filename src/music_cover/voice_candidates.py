"""Candidate settings, technical checks and deterministic recommendation."""
import html
import math
from pathlib import Path


def candidate_seeds(seed, count=3, explicit=None):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('candidate_count must be an integer between 1 and 8')
    values = [(seed + i) % 2**32 for i in range(count)] if explicit is None else explicit
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('candidate_seeds length must equal candidate_count')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in values) or len(set(values)) != count:
        raise ValueError('candidate_seeds must be distinct integers between 0 and 2**32 - 1')
    return values


def rejection_reasons(metrics, source_duration):
    reasons = []
    score = metrics.get('speaker_similarity')
    if not isinstance(score, (int, float)) or not math.isfinite(score): reasons.append('invalid_ecapa')
    if metrics.get('pitch', {}).get('status') != 'verified': reasons.append('pitch_not_verified')
    if not metrics.get('alignment', {}).get('consistent'): reasons.append('alignment_not_verified')
    audio = metrics.get('audio', {})
    duration, peak, rms = (audio.get(k, float('nan')) for k in ('duration_s', 'sample_peak', 'rms'))
    if not math.isfinite(duration) or abs(duration - source_duration) > .25: reasons.append('duration_mismatch')
    if not math.isfinite(peak) or peak > 1.001: reasons.append('invalid_peak')
    if not math.isfinite(rms) or rms < 1e-4: reasons.append('silent_audio')
    return reasons


def recommend(rows):
    eligible = [row for row in rows if row.get('status') == 'succeeded' and row.get('eligible')
                and math.isfinite(row['metrics']['speaker_similarity'])]
    # Stable tie break: lower seed, then candidate ID. Never fall back to a failed candidate.
    return min(eligible, key=lambda row: (-row['metrics']['speaker_similarity'], row['seed'], row['id'])) if eligible else None


def contained_file(directory, relative):
    root = Path(directory).resolve(); p = (root / relative).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError(f'Invalid candidate artifact: {relative}')
    return p


def render_report(report, destination, *, mixed=False):
    """Portable report: all media paths are relative to this report directory."""
    selected = report.get('recommended_id')
    rows = []
    for row in report['candidates']:
        metrics = row.get('metrics', {}); pitch = metrics.get('pitch', {})
        score = metrics.get('speaker_similarity')
        score = f'{score:.5f}' if isinstance(score, (float, int)) and math.isfinite(score) else '未完成'
        fraction = pitch.get('within_one_semitone_fraction')
        fraction = f'{fraction:.2%}' if fraction is not None else '—'
        media = row.get('final_mp3' if mixed else 'audio')
        player = f'<audio controls preload="none" src="{html.escape(media)}"></audio>' if media else '无音频'
        verdict = '推荐' if row['id'] == selected else ('合格' if row.get('eligible') else '未通过：' + ', '.join(row.get('rejection_reasons', [])))
        rows.append(f'<tr><td>{html.escape(row["id"])}</td><td>{row["seed"]}</td><td>{score}</td><td>{html.escape(pitch.get("status", "未完成"))}</td><td>{fraction}</td><td>{html.escape(verdict)}</td><td>{player}</td></tr>')
    title = '三个微调候选对比' if len(report['candidates']) == 3 else '微调候选对比'
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 20px;background:#f6f8fb;color:#182238}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border:1px solid #ddd}audio{width:260px}</style><h1>'+title+'</h1>'
    text += '<p>同一源人声、参考音频和微调模型，使用不同随机种子；未重新训练。先通过音高、对齐、时长与音频完整性检查，再按 ECAPA 降序推荐。ECAPA 不是身份百分比，人工听感与歌词准确性仍需确认。</p>'
    if report.get('backend') == 'so_vits_svc':
        text = text.replace('同一源人声、参考音频和微调模型，使用不同随机种子；未重新训练。', 'so-vits-svc 歌手模型转换；参考音频仅用于评估，不参与推理。当前接入提供一个候选。')
    if report.get('backend') == 'rvc':
        text = text.replace('同一源人声、参考音频和微调模型，使用不同随机种子；未重新训练。', 'RVC 歌手模型转换；同一模型、索引和源人声，显式控制随机种子。参考音频仅用于评估，不参与推理；独立测试集指标另行报告。')
    text += '<p>推荐：<strong>'+html.escape(selected or '无合格候选')+'</strong>。'+('入口 final.wav / final.mp3 对应推荐候选。' if mixed and selected else '')+'</p>'
    text += '<table><tr><th>候选</th><th>种子</th><th>ECAPA ↑</th><th>音高检查</th><th>误差≤1半音</th><th>结论</th><th>试听</th></tr>'+''.join(rows)+'</table><p>详细指标：<a href="candidates.json">candidates.json</a>。</p></html>'
    Path(destination).write_text(text, encoding='utf-8')
