#!/usr/bin/env python3
"""Additional, local ASR and pitch checks; these do not replace listening."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def normalized(text):
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)).lower()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def save_verification(output, result):
    output = Path(output)
    result["asr_metric_note"] = ("CER 仅去除标点和段落标签，未统一简繁体、同音字；Whisper-small 对歌唱可能漏识别或误识别。"
                                 "数值仅作复核提示，不能据此直接断定实际漏词比例。")
    result["review_flags"] = []
    if result["pitch"]["above_600_cents_fraction"] is not None and result["pitch"]["above_600_cents_fraction"] > 0.05:
        result["review_flags"].append("pitch_outliers_require_listening")
    if any(v.get("cer_against_input_lyrics", 0) > 0.3 for v in result.get("asr", {}).values() if isinstance(v, dict)):
        result["review_flags"].append("high_asr_error_requires_listening")
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pitch = result["pitch"]
    speaker = result["speaker_diagnostic"]
    def display(value, percent=False):
        return "无可比有声帧" if value is None else format(value, ".2%" if percent else ".2f")
    summary = (f"音高绝对差中位数：{display(pitch['median_absolute_cents'])} cents；"
               f"共同有声帧中，差异不超过 100 cents 的比例：{display(pitch['within_100_cents_fraction'], True)}，"
               f"超过 600 cents 的比例：{display(pitch['above_600_cents_fraction'], True)}。\n\n"
               f"CAMPPlus 与参考录音的平均余弦相似度：换声前 {speaker['mean_cosine_before']:.3f}，"
               f"换声后 {speaker['mean_cosine_after']:.3f}。该编码器同时被 Seed-VC 使用，不是独立评价。\n\n")
    if "asr" in result:
        summary += (f"Whisper-small 原始字符错误率：换声前 {result['asr']['vocals_before']['cer_against_input_lyrics']:.2%}，"
                    f"换声后 {result['asr']['vocals_after']['cer_against_input_lyrics']:.2%}。"
                    + result["asr_metric_note"] + "\n\n")
    summary += "本次自动诊断未给出听感合格结论。需要试听复核歌词清晰度、个别音高异常及目标音色自然度；逐段数据和识别文本见 verification.json。\n"
    marker = "\n<!-- automatic-verification -->\n"
    report_path = output / "report.md"
    report = report_path.read_text().split(marker)[0]
    report_path.write_text(report + marker + "\n附加自动诊断\n\n" + summary)
    page_path = output / "listen.html"
    if page_path.exists():
        page = page_path.read_text()
        page = re.sub(r'<section id="automatic-verification">.*?</section>', '', page, flags=re.S)
        section = ('<section id="automatic-verification"><h2>自动诊断与待试听项</h2><pre>'
                   + html.escape(summary) + '</pre><p><a href="verification.json">详细诊断</a> · '
                   '<a href="report.md">完整报告</a></p></section>')
        page_path.write_text(page.replace('</html>', section + '</html>'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / 'stage_state.json').exists():
        parser.error('This archived diagnostic only supports v0.1. Use scripts/verify_review.py for modular runs; no files were changed.')
    output = run / "deliverables"
    import numpy as np
    import librosa
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / "third_party/seed-vc"))
    from modules.rmvpe import RMVPE
    pitch = RMVPE(str(ROOT / "models/RMVPE/rmvpe.pt"), is_half=False, device="cuda")
    waves, pitches = {}, {}
    for name in ["vocals_before", "vocals_after"]:
        wave, _ = librosa.load(output / f"{name}.wav", sr=16000, mono=True)
        waves[name] = wave
        pitches[name] = pitch.infer_from_audio(wave, thred=0.03)
    frames = min(map(len, pitches.values()))
    before, after = [pitches[name][:frames] for name in ["vocals_before", "vocals_after"]]
    voiced = (before > 50) & (after > 50)
    cents = np.abs(1200 * np.log2(after[voiced] / before[voiced]))
    result = {
        "pitch": {"common_voiced_frames": int(voiced.sum()), "total_frames": frames,
                  "median_absolute_cents": float(np.median(cents)) if len(cents) else None,
                  "within_100_cents_fraction": float(np.mean(cents <= 100)) if len(cents) else None,
                  "above_600_cents_fraction": float(np.mean(cents > 600)) if len(cents) else None},
        "limitations": "自动 F0 与 ASR 检查，不能替代对目标风格、音色相似度和自然度的人工试听。",
    }
    del pitch
    torch.cuda.empty_cache()
    # Diagnostic only: CAMPPlus is also the conditioning encoder in Seed-VC,
    # so this is not an independent speaker-similarity benchmark.
    import torchaudio
    from modules.campplus.DTDNN import CAMPPlus
    speaker = CAMPPlus(feat_dim=80, embedding_size=192).eval().to("cuda")
    speaker.load_state_dict(torch.load(ROOT / "models/CAMPPlus/campplus_cn_common.bin", map_location="cpu", weights_only=True))
    def embedding(wave):
        features = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0),
                                                   num_mel_bins=80, dither=0, sample_frequency=16000)
        features -= features.mean(dim=0, keepdim=True)
        with torch.inference_mode():
            vector = speaker(features.unsqueeze(0).to("cuda"))[0]
        return torch.nn.functional.normalize(vector, dim=0)
    reference, _ = librosa.load(output / "reference.wav", sr=16000, mono=True)
    target_vector = embedding(reference)
    similarities = []
    width = 8 * 16000
    for fraction in [0.2, 0.5, 0.8]:
        start = max(0, min(len(waves["vocals_before"]) - width,
                           int(len(waves["vocals_before"]) * fraction) - width // 2))
        row = {"start_s": start / 16000}
        for name, wave in waves.items():
            row[name] = float(torch.dot(target_vector, embedding(wave[start:start+width])))
        similarities.append(row)
    result["speaker_diagnostic"] = {
        "encoder": "CAMPPlus; also used internally by Seed-VC, not an independent evaluation",
        "segments": similarities,
        "mean_cosine_before": float(np.mean([row["vocals_before"] for row in similarities])),
        "mean_cosine_after": float(np.mean([row["vocals_after"] for row in similarities])),
        "limitation": "跨语言、说话与歌唱差异会影响分数；不能将该值解读为身份确认或听感达标。",
    }
    del speaker, target_vector
    torch.cuda.empty_cache()
    if not args.skip_asr:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
        model_path = str(ROOT / "models/Whisper-small")
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float16,
                                                        local_files_only=True).to("cuda")
        recognizer = pipeline("automatic-speech-recognition", model=model,
                              tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                              chunk_length_s=30, stride_length_s=5, batch_size=1,
                              torch_dtype=torch.float16, device=0)
        transcripts = {}
        for name, wave in waves.items():
            print(f"ASR {name}", flush=True)
            transcripts[name] = recognizer({"array": wave, "sampling_rate": 16000},
                                          generate_kwargs={"language": "zh", "task": "transcribe"})["text"]
        request = json.loads((run / "request.json").read_text())
        lyrics = re.sub(r"\[[^\]]+\]", "", Path(request["lyrics"]).read_text())
        expected = normalized(lyrics)
        result["asr"] = {name: {"text": text, "cer_against_input_lyrics": edit_distance(expected, normalized(text)) / max(1, len(expected))}
                         for name, text in transcripts.items()}
        source_text, converted_text = map(normalized, [transcripts["vocals_before"], transcripts["vocals_after"]])
        result["asr"]["conversion_transcript_change_rate"] = edit_distance(source_text, converted_text) / max(1, len(source_text))
    save_verification(output, result)
    print(json.dumps({"pitch": result["pitch"], "asr_checked": not args.skip_asr}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
