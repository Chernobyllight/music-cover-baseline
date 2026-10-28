# 10_evaluation · 评估音频与评审数据

检查成品音频、分轨重建和 MIDI 数据，并输出供后续评审使用的技术指标。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)、[02_pitch_adaptation](../02_pitch_adaptation/README.md)、[04_vocal_separation](../04_vocal_separation/README.md)、[06_mix](../06_mix/README.md)、[07_stem_separation](../07_stem_separation/README.md)、[08_final_score](../08_final_score/README.md)、[09_stem_midi](../09_stem_midi/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/10_evaluation/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

表中依赖按示例的 review=true、transpose=auto、pitch_check=true 展示。--no-review 时不依赖 07–09，仍做基础音频评估。技术检查通过后，主观音质状态仍标记为 awaiting_human_review。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `02_pitch_adaptation` | `02_pitch_adaptation/attempt_001/artifacts/source_f0.npz`、`transposition.json` | 启用 F0 检查时，原曲 F0 和要求的整体移调量 |
| `00_prepare` | `00_prepare/attempt_001/artifacts/source.wav` | 原曲描述性指标对照 |
| `06_mix` | `06_mix/attempt_001/artifacts/final.wav` | 成品音频指标与重建对照 |
| `06_mix` | `06_mix/attempt_001/artifacts/final.mp3` | MP3 时长及哈希检查 |
| `06_mix` | `06_mix/attempt_001/artifacts/mix.json` | 混音指标和预期哈希 |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/vocals.wav` | 启用 pitch_check 且 transpose 非 0 时，检查换声前 F0 |
| `06_mix` | `06_mix/attempt_001/artifacts/converted_aligned.wav` | 启用 pitch_check 且 transpose 非 0 时，检查换声后 F0 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/` | review=true 时纳入全部分轨产物，检查轨道重建 |
| `08_final_score` | `08_final_score/attempt_001/artifacts/` | review=true 时纳入成品转谱产物的依赖校验 |
| `09_stem_midi` | `09_stem_midi/attempt_001/artifacts/` | review=true 时纳入全部 MIDI 产物并检查合法性 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/10_evaluation/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `evaluation.json` | 音频指标、混音及可选的分轨/MIDI 检查结果；供 11 打包 |
| `pitch_adherence.json` | 原曲→生成的音高遵循检查，以及换声前→后的音高保持检查；同时写入 evaluation.json |
| `pitch_profiles.json` | 启用 F0 检查时的换声前后音高统计 |
| `before_f0.npz` | 启用 F0 检查时的换声前 F0 |
| `after_f0.npz` | 启用 F0 检查时的换声后 F0 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 10_evaluation
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/evaluation.py`。默认模块配置：`configs/modules/10_evaluation.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

音高核验使用 0.2 秒有声片段的相对轮廓 DTW（原曲→生成）和延迟校正后的同时间片段（换声前→后）。对齐不使用预期移调量，也不按八度折叠；低覆盖或偏差大时标记 unassessable / needs_review。verified 仅表示这些工程阈值通过，不等于逐音正确或听感验收。不会在评估阶段隐式移调音频。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。

## 可选音色评估

在任务配置 `modules.10_evaluation.voice_check` 设为 `true`，会增加 ECAPA 相似度和对齐后的源/转换 F0 检查，即使 transpose=0 也执行。示例：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/10_evaluation/run.sh \
  --run runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_finetuned_20261028
```

本例读取已完成的杨坤凯尔特史诗运行，其保存配置已启用 voice_check。读取04的源人声、06的已对齐转换人声、00的参考片段，以及05的转换记录；需要相同配置下的00–09已完成，创建方法见 [v0.3 实操](../../docs/17_v03_walkthrough.md)。输出本阶段 `artifacts/voice_conversion_evaluation.json` 和 `voice_diagnostics/`，并合入 `evaluation.json`。该相似度使用同一推理参考，仅为诊断，不是盲测结论。附加权重及完整流程见 [音色微调手册](../../docs/13_voice_adaptation.md)。

05 使用微调候选模式时，evaluation.json 自动包含 voice_candidates，汇总三个候选在 05 干声上的指标和推荐，不要求再开启 voice_check。后者仅额外评估推荐版本在 06 对齐后的干声。
