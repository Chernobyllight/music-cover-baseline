# 02_pitch_adaptation · 分析音区并双向移调

结合原曲乐谱与参考音色的音区，确定移调幅度并输出供生成使用的乐谱。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)、[01_source_score](../01_source_score/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/02_pitch_adaptation/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

示例继承 00 保存的 transpose=auto，当前自动范围为 -6～+6 半音，含降调、原调和升调；该任务配置已将孙燕姿参考标记为 singing。transpose=0 时只复制乐谱并记录直通计划，不进行分离和音高分析；仍需保留 02，供 03 读取统一的 score_transposed.abc。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `01_source_score` | `01_source_score/attempt_001/artifacts/score.abc` | 待移调的原曲旋律 |
| `01_source_score` | `01_source_score/attempt_001/artifacts/melody_vocal.lab` | 原谱与原曲 F0 的逐音八度一致性检查 |
| `00_prepare` | `00_prepare/attempt_001/artifacts/source.wav` | transpose=auto 或非零时，用于原曲人声分离和音高分析 |
| `00_prepare` | `00_prepare/attempt_001/artifacts/reference.wav` | transpose=auto 或非零时，用于参考音区分析 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/02_pitch_adaptation/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `score_original.abc` | 原曲乐谱副本 |
| `score_transposed.abc` | 移调后的乐谱；供 03 生成 |
| `transposition.json` | 推荐值 recommended_semitones、实际采用值 applied_semitones、可信度、全部候选、有效 F0 统计、谱面逐音检查；selected_semitones 是实际值的兼容字段 |
| `pitch_report.html` / `pitch_candidates.csv` | 可离线打开的分布图、全部候选评分及决策原因 |
| `score_audio_consistency.json` | 原谱和原曲音频的音高/八度差检查；不自动修正八度 |
| `score_transposed.mid` | 与移调 ABC 一致的人声和器乐旋律 MIDI，按谱面速度；不是成品 MIDI，也不是完整伴奏配器 |
| `source_separation/vocals.wav` | 启用移调分析时的原曲人声 |
| `pitch_profiles.json` | 后端旧口径统计，供兼容对照；新算法实际统计见 transposition.json 的 source_quality / reference_quality |
| `source_f0.npz` | 启用移调分析时的原曲 F0 数据 |
| `reference_f0.npz` | 启用移调分析时的参考 F0 数据 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 02_pitch_adaptation
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/pitch_adaptation.py`。默认模块配置：`configs/modules/02_pitch_adaptation.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

## 双向自动选调和参考质量

`auto` 比较 P10～P90 的 81 个音高分位点，评分为平均绝对半音差 + 0.04×移调半音数²；如提供人工确认的目标音域，再加超界量。用 4 秒时间块重采样 128 次，默认要求升/降方向支持率至少 0.60、相对原调评分改善至少 0.15。可信度是工程分级，不是概率。短参考只描述该片段，不代表歌手完整生理音域。

参考默认为 `unknown`。只有明确标记 `reference_kind: singing` 才自动采用非零推荐；speech/unknown 仍展示候选，但实际保留 0。有效有声不足 3 秒、方向不稳定或改善很小时也保留 0，并记录具体原因。手动 `--transpose -2` 仍会明确采用 -2。`--transpose 0` 是关闭分析，不是“auto 恰好选择原调”。

已有运行要启用双向分析并确认参考为歌声，可使用仅覆盖选调的配置；这不会改歌曲、歌词或风格：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/02_pitch_adaptation/run.sh \
  --run runs/qiu_yi_nong_modular_test \
  --config configs/tasks/pitch_singing_auto.yaml
```

这个命令仍要求 00、01 已有效完成。如果要继续到成品，改用总入口的 `--from 02`。旧 02 缓存会因算法和契约升级失效，随后 03–11 也需要按依赖重跑；历史 attempt 保留。

如有更长、跨主副歌的干净歌声，把完整文件路径放入 02 配置 `range_references` 列表（不要放伴奏混音）。非空时这些文件替代短音色片段用于音区分析；音色转换仍使用 00 的短参考。每个合格文件等权，文件内容纳入缓存哈希。多个参考都必须是目标人物的歌声。手动确认音域可设置成对的 `target_low_midi` / `target_high_midi`；不设置时不会把样本 P95 当作硬上限。搜索范围可在 -12～+12 内配置，且必须包含 0。

用已产生的 F0 做真实数据回放及已知 ±3/0 半音的人工 F0 对照（不重新加载模型、不修改旧运行）：

```bash
PYTHONPATH=src envs/control/bin/python scripts/02_pitch_adaptation/verify_selection.py \
  --source-f0 runs/qiu_yi_nong_modular_test/stages/02_pitch_adaptation/attempt_001/artifacts/source_f0.npz \
  --reference-f0 runs/qiu_yi_nong_modular_test/stages/02_pitch_adaptation/attempt_001/artifacts/reference_f0.npz \
  --output runs/pitch_selection_replay
```

输出目录必须尚不存在。人工 F0 对照只验证算法方向，不是新的真实演唱验证。完整技术说明见 [双向选调](../../docs/10_bidirectional_pitch.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
