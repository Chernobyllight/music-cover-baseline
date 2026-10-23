# 09_stem_midi · 将分轨转成 MIDI

为评审音轨估计音符，汇总分轨 MIDI、和声参考、音符表和模型激活数据。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[07_stem_separation](../07_stem_separation/README.md)、[08_final_score](../08_final_score/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/09_stem_midi/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

本模块要求 review=true。输入为 7 条有音高音轨、鼓轨及成品和弦 MIDI，不会把残差轨转成乐器音符。MIDI 不包含孙燕姿音色；固定 120 BPM 用于表达绝对秒时间，自动音符和鼓事件仍需校对。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/stems.json` | 音轨及时间轴信息 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/lead_vocals.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/backing_vocals.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/vocals.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/bass.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/guitar.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/piano.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/other.wav` | 转谱输入 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/audio/drums.wav` | 转谱输入 |
| `08_final_score` | `08_final_score/attempt_001/artifacts/chords.mid` | 成品和弦参考 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/09_stem_midi/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `midi/lead_vocals.mid` | 主唱 MIDI |
| `midi/backing_vocals.mid` | 背景人声 MIDI |
| `midi/vocals.mid` | 总人声 MIDI |
| `midi/bass.mid` | 贝斯 MIDI |
| `midi/guitar.mid` | 吉他 MIDI |
| `midi/piano.mid` | 钢琴 MIDI |
| `midi/other.mid` | 其他乐器 MIDI |
| `midi/drums_onsets.mid` | 启发式估计的鼓 MIDI |
| `midi/arrangement_estimate.mid` | 汇总编配 MIDI |
| `midi/harmony.mid` | 成品和弦参考 MIDI |
| `notes/` | 音符 CSV；供人工检查 |
| `raw_midi/` | 原始转谱 MIDI |
| `activations/` | 模型激活数据 |
| `midi.json` | 转谱参数和结果清单；供 10、11 使用 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 09_stem_midi
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/stem_midi.py`。默认模块配置：`configs/modules/09_stem_midi.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
