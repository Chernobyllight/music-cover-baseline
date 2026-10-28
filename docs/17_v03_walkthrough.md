# v0.3 实操：杨坤微调＋不败的英雄＋凯尔特史诗

本例使用已经训练好的杨坤 1000 步 Seed-VC 模型，只生成一个推理候选。03 重新生成指定曲风，05 使用 finetuned 模型换声，06–11 完成混音、分轨、MIDI、评估和交付。原有 zero_shot 接口保留；歌词仍作为 03 的输入，主流程不做自动歌词检测或静音修复。

## 1. 输入与模型

项目根目录是 `/data/liuhongda/music-cover-baseline`，当前服务器上与 `/mnt/afs_f/liuhongda/music-cover-baseline` 指向同一共享目录。下面的相对路径均基于项目根目录。

| 材料 | 路径 |
| --- | --- |
| 原曲 | `data/raw/songs/bu_bai_de_ying_xiong_tang_jun_di.mp3` |
| 歌词 | `data/raw/lyrics/bu_bai_de_ying_xiong_tang_jun_di.txt` |
| 杨坤参考人声 | `data/raw/references/yang_kun/reference.wav`，取 0–20 秒 |
| 微调模型入口 | `models/voices/yang_kun/seed_vc/catalog_v001_20261027/voice.json` |
| 模型配套文件 | 同目录的 `model.pth`、`config.yml`、`validation.json` |
| 任务配置 | `configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml` |

Git 不包含录音、歌词正文、歌手权重和 Python 环境。现有服务器已有这些资源；新机器先按 [恢复说明](08-repository.md) 和 [模型清单](12_model_weights.md) 准备。模型入口加载时会校验文件与哈希，不会缺失后自动退回零样本。

风格提示是史诗凯尔特民谣与管弦乐：风笛、爱尔兰锡哨、凯尔特竖琴、小提琴、bodhran 鼓和管弦鼓组，叙事主歌与英雄感副歌。该描述是生成目标，曲风符合程度须试听。

参数：`transpose=auto`、02 的 `reference_kind=singing`、seed=42、推理 steps=30、GPU 0、review=true、voice_check=true。微调候选数为 1，种子为 `[42]`。**1000 步是已有模型的训练步数，30 步是推理步数；本次命令不会再次训练。**

## 2. 一条命令完整生成

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

# 检查各阶段环境、模型及音色 profile，不启动生成。
bash scripts/run_baseline.sh \
  --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml \
  --preflight

# 新建本例运行。已有同名目录时会检查并复用有效缓存。
bash scripts/run_baseline.sh \
  --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml \
  --run runs/lhd_celtic_epic_v03
```

首次 finetuned 推理需要 ECAPA 和 RMVPE，即使只有一个候选也会执行技术检查。只使用已训练模型不需要 OpenVoiceV2/训练向量库；准备方法见 [权重指南](12_model_weights.md)。

## 3. 自己逐模块检查

以下与完整命令等价，选一种执行方式即可。所有命令共用一个 `LHD_RUN`；首次 00 指定任务配置，后续读取运行中保存的配置。每一步都要求上游已成功且未失效。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
LHD_RUN="/data/liuhongda/music-cover-baseline/runs/lhd_celtic_epic_v03"

bash scripts/00_prepare/run.sh \
  --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml \
  --run "$LHD_RUN"
bash scripts/01_source_score/run.sh --run "$LHD_RUN"
bash scripts/02_pitch_adaptation/run.sh --run "$LHD_RUN"
bash scripts/03_style_generation/run.sh --run "$LHD_RUN"
bash scripts/04_vocal_separation/run.sh --run "$LHD_RUN"
bash scripts/05_voice_conversion/finetuned/run.sh --run "$LHD_RUN"
bash scripts/06_mix/run.sh --run "$LHD_RUN"
bash scripts/07_stem_separation/run.sh --run "$LHD_RUN"
bash scripts/08_final_score/run.sh --run "$LHD_RUN"
bash scripts/09_stem_midi/run.sh --run "$LHD_RUN"
bash scripts/10_evaluation/run.sh --run "$LHD_RUN"
bash scripts/11_delivery/run.sh --run "$LHD_RUN"
```

下表的输出均位于 `$LHD_RUN/stages/<模块>/attempt_001/artifacts/`。例如 03 的完整音频路径是 `/data/liuhongda/music-cover-baseline/runs/lhd_celtic_epic_v03/stages/03_style_generation/attempt_001/artifacts/generated/audio.flac`。重跑后可能是 attempt_002，以 `stage_state.json` 的成功 attempt 为准。

| 模块 | 使用哪个上游 | 查看本阶段产物 |
| --- | --- | --- |
| 00_prepare | 上表原曲、歌词、参考 | `source.wav`、`reference.wav`、`lyrics.txt` |
| 01_source_score | 00 的 source.wav | `score.abc`、`events.json` |
| 02_pitch_adaptation | 01 的 score.abc；00 原曲和参考 | `transposition.json`、`pitch_report.html`、`score_transposed.abc`、`score_transposed.mid` |
| 03_style_generation | 02 移调谱＋00 歌词＋风格 | `generated/audio.flac`，这是换风格后的新曲，尚未转为杨坤音色 |
| 04_vocal_separation | 03 新曲 | `vocals.wav`、`instrumental.wav` |
| 05_voice_conversion | 04 vocals.wav＋00 reference.wav＋杨坤 profile | `converted.wav`、`conversion.json`、`candidates.json`、`candidates/candidate_001/converted.wav` |
| 06_mix | 05 转换人声＋04 伴奏；03 用于重混对照 | `final.wav`、`final.mp3`、`mix.json`、`converted_aligned.wav` |
| 07_stem_separation | 06 最终混音及其增益后组成 | `stems.json`、`audio/*.wav` |
| 08_final_score | 06 final.wav | `transcription_summary.json`、`score.abc`、`chords.mid` |
| 09_stem_midi | 07 各声部＋08 和弦 MIDI | `midi.json`、`midi/*.mid`、`notes/*.csv` |
| 10_evaluation | 00 原曲/参考、02 移调计划、04/06 人声、07–09 评审数据 | `evaluation.json`、`voice_conversion_evaluation.json`、`pitch_adherence.json` |
| 11_delivery | 已通过的各模块产物 | `final.mp3`、`final.wav`、`listen.html`、`review/index.html`、`review/midi_bundle.zip`、`delivery_validation.json` |

11 校验通过后，`$LHD_RUN/deliverables/` 指向其成功产物。最终 MP3 在 `/data/liuhongda/music-cover-baseline/runs/lhd_celtic_epic_v03/deliverables/final.mp3`，分轨在同目录的 `review/audio/`，MIDI 在 `review/midi/`。02 的 MIDI 是移调后的输入旋律；09/交付包的 MIDI 是新成品重新转谱，不要混用。

## 4. 切换零样本或三个候选

保留相同歌曲/曲风，另开一个零样本运行：

```bash
bash scripts/run_baseline.sh \
  --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml \
  --voice-mode zero_shot \
  --run runs/lhd_celtic_epic_zero_shot_v03
```

此命令在 05 使用通用模型，并清除配置中的微调 profile。该任务仍启用 10 的 voice_check，因此仍需要 ECAPA；默认零样本任务不开启 voice_check 时不需要该附加评估模型。已有共同 00–04 时也可以只调用 `scripts/05_voice_conversion/zero_shot/run.sh --run ...`，随后从 06 继续。同一 run 切换模式会使 05 之后缓存失效；长期保留两套成品建议使用不同 run。

同一个杨坤模型生成三个微调候选：

```bash
bash scripts/run_baseline.sh \
  --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml \
  --voice-candidates 3 --voice-candidate-seeds 42 43 44 \
  --run runs/lhd_celtic_epic_candidates_v03
```

三个候选仅改变 05 的推理随机种子；同一 run 内共用 03/04 的生成歌曲和伴奏。技术合格后按 ECAPA 选择，`deliverables/final.mp3` 是推荐副本，其他版本在 `deliverables/candidates/candidate_00N/final.mp3`。没有合格候选时停止并保留诊断，不保证一定交付三个成功版本。详细规则见 [候选指南](15_voice_candidates.md)。

## 5. 已完成的真实验证结果

2026-10-28 已完成的目录为 `/data/liuhongda/music-cover-baseline/runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_finetuned_20261028`，与上面建议的新运行名区分。此目录的 `deliverables/final.mp3` 可直接试听，主流程 00–11、交付包检查和 78 项测试均通过。

| 指标 | 值 |
| --- | --- |
| 时长 | 128.198667 秒 |
| 02 推荐 / 采用 | -2 / -2 半音 |
| 生成实测移调中位数 | -1.982717 半音，verified |
| 10 对齐后干声 ECAPA | 0.821394563 |
| 10 独立换声 F0 检查 | ±1 半音比例 98.0545%，verified |
| 分轨 / MIDI | 12 路音频、10 项 MIDI，无空主 MIDI |

模型是用户指定的 1000 步 checkpoint，profile 状态为 `unverified_user_selected`。ECAPA 使用同一个推理参考，仅作诊断，不是“像杨坤的百分比”或独立测试集表现。技术检查与音色、曲风、发音、MIDI 准确性的人工验收分开；当前主观状态仍为 awaiting_human_review。

检查已有结果，不启动生成：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --run runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_finetuned_20261028 --status
envs/separator/bin/python scripts/verify_review.py \
  runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_finetuned_20261028 \
  --output cache/celtic_v03_delivery_check.json
```

`runs/` 与 `cache/` 均不上传 Git；保留的实测结果和历史清理范围见 [验证记录](../VALIDATION.md)。
