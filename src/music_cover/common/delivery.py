import html
import json
import shutil
from pathlib import Path
from .io import write_json

def build_delivery(destination, request, stages, prepared, generated, separated, mixed, adaptation=None, pitch_check=None, review=None):
    destination.mkdir(exist_ok=True)
    items = {"original.wav": prepared["source"], "reference.wav": prepared["reference"],
             "restyled.flac": generated["cover"], "reconstruction.wav": mixed["reconstruction"],
             "final.wav": mixed["final"], "final.mp3": mixed["final_mp3"],
             "vocals_before.wav": separated["vocals"],
             "vocals_after.wav": mixed["aligned_vocal"], "instrumental.wav": separated["instrumental"]}
    if adaptation:
        items.update({"score_original.abc": adaptation["original_score"], "score_transposed.abc": adaptation["score"],
                      "transposition.json": adaptation["plan"], "pitch_input.json": adaptation["profiles"]})
        for role, name in [('report', 'pitch_report.html'), ('candidates', 'pitch_candidates.csv'),
                           ('consistency', 'score_audio_consistency.json'), ('score_midi', 'score_transposed.mid')]:
            if role in adaptation:
                items[name] = adaptation[role]
    if pitch_check:
        items["pitch_output.json"] = pitch_check["profiles"]
        if 'adherence' in pitch_check:
            items['pitch_adherence.json'] = pitch_check['adherence']
    if review:
        items.update({"review/" + name: path for name, path in review.items()})
    for name, source in items.items():
        temporary = destination / (name + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, temporary)
        temporary.replace(destination / name)
    metrics = json.loads(mixed["metrics"].read_text())
    write_json(destination / "checks.json", metrics)
    duration = metrics["final"]["duration_s"]
    times = "\n".join(f"| {name} | {stage['wall_seconds']:.1f} |"
                      for name, stage in stages.items() if stage["status"] == "succeeded")
    report = (
        f"完整流程已执行，输出 {duration:.2f} 秒、48 kHz 立体声。\n\n"
        "自动文件、数值、时长与峰值检查已执行；音乐风格、音色相似度和自然度仍需试听验收。\n\n"
        f"目标风格：{request['style']}\n\n"
        "依次试听 original.wav、restyled.flac、final.wav；参考人物声音在 reference.wav。"
        f"final.mp3 是同一最终混音的 {metrics['final_mp3']['bitrate_kbps']} kbps 压缩版，便于下载和分享。"
        "reconstruction.wav 用于检查分离和重混的损伤。\n\n"
        f"转换人声长度差：{metrics['alignment']['converted_duration_delta_s']:.6f} 秒。"
        f"固定延迟校正：{metrics['alignment']['applied_shift_samples']} 个 48 kHz 采样点。\n\n"
        f"最终峰值估计：{metrics['final']['estimated_true_peak_dbtp']:.2f} dBTP"
        "（4 倍过采样工程估计）。\n\n"
        f"对齐需要重点试听：{metrics['alignment']['needs_listening_review']}。\n\n"
        "| 阶段 | 耗时秒，包含进程启动和模型加载 |\n| --- | ---: |\n" + times + "\n\n"
        "完整参数、输入哈希和阶段日志保存在上一级目录。\n"
    )
    if Path(request["song"]).name == "tonight-awake.mp3":
        report += ("\n本次演示使用 YuE2 官方发布的 AI 生成中文歌曲及其配套歌词，"
                   "目标音色使用公开 LibriSpeech 语音样本；不是用户目标人物的录音。"
                   "来源与哈希见项目 data/demo_sources.json。\n")
    if adaptation:
        plan = json.loads(adaptation["plan"].read_text())
        checks = plan["score_checks"]
        report += (f"\n本次谱面整体移调 {plan['selected_semitones']:+d} 个半音："
                   f"{checks['source_key']} → {checks['target_key']}。人声及器乐音符均已移调，"
                   "音符起点、时值、节拍和速度通过逐事件检查。Seed-VC 不再额外移调。\n\n"
                   "参考观测音区不代表歌手完整音域；自动选调为启发式建议。"
                   "选择依据见 pitch_report.html、transposition.json；其中的 source_quality/reference_quality 是本算法实际使用的统计。"
                   "pitch_input.json 保留后端旧统计口径，仅供对照。"
                   "谱面检查不保证生成歌声逐音遵循；仍需检查开头是否改善以及高音是否自然。\n")
        if 'recommended_semitones' in plan:
            report += (f"\n统计推荐 {plan['recommended_semitones']}；实际采用 {plan['applied_semitones']:+d} 半音；"
                       f"状态 {plan['status']}；可信度 {plan['confidence']}。详见 pitch_report.html。\n")
    if pitch_check and 'adherence' in pitch_check:
        adherence = json.loads(pitch_check['adherence'].read_text())
        report += f"\n生成音频与换声音高核验：{adherence['status']}，详见 pitch_adherence.json。技术流程成功不等于音高遵循或听感通过。\n"
    if review:
        report += ("\n成品分轨和 MIDI 评审包已生成，入口 review/index.html。\n"
                   "所有 review/audio 轨道均同起点、同采样率、同长度；父轨不能与子轨重复相加。\n"
                   "MIDI 来自成品重新转谱，自动估计仍需专业人员校对；鼓为起音候选，和弦谱为分析指南。\n"
                   "音符 CSV、原始模型数据、校验哈希、响度/节奏描述和人工评分表均包含在 review。\n")
    (destination / "report.md").write_text(report, encoding="utf-8")
    labels = [("original.wav", "1 原曲"), ("reference.wav", "目标音色参考"),
              ("restyled.flac", "2 换风格"), ("final.wav", "3 最终混音（WAV）"),
              ("final.mp3", "最终混音（MP3）"),
              ("reconstruction.wav", "分离后直接重混对照"),
              ("vocals_before.wav", "换声前人声"), ("vocals_after.wav", "换声后人声")]
    players = "\n".join(f'<p>{html.escape(label)} · <a href="{name}" download>下载</a></p>'
                        f'<audio controls preload="none" src="{name}"></audio>'
                        for name, label in labels)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>音乐 baseline 试听</title>'
            '<style>body{max-width:850px;margin:40px auto;padding:0 20px;font:17px/1.6 sans-serif;background:#f7f7f4;color:#222}'
            'audio{width:100%}p{margin-top:28px}pre{white-space:pre-wrap;font:inherit}</style>'
            '<h1>音乐 baseline 试听</h1><p>完整流程已执行；听感质量请对照以下音频验收。</p>'
            + ('<p><a href="review/index.html">打开成品分轨与 MIDI 专业评审包</a></p>' if review else '')
            + ('<p><a href="pitch_report.html">双向选调报告与移调 MIDI</a></p>' if adaptation and 'report' in adaptation else '')
            + ('<p><a href="pitch_adherence.json">生成与换声音高核验</a></p>' if pitch_check and 'adherence' in pitch_check else '')
            + players + '<details><summary>运行报告</summary><pre>' + html.escape(report) + '</pre></details></html>')
    (destination / "listen.html").write_text(page, encoding="utf-8")
