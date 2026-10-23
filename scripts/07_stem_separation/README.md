# 07_stem_separation · 生成供评审的成品分轨

继续拆分转换后人声与伴奏，生成同起点、同长度的主唱、背景人声及乐器评审音轨。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[04_vocal_separation](../04_vocal_separation/README.md)、[06_mix](../06_mix/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/07_stem_separation/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

本模块要求 review=true，示例任务默认已开启。12 条 WAV 含父轨和残差轨；重建时按 stems.json 指定的 10 条叶轨相加，避免重复叠加父轨。分离结果仍需人工听评。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `06_mix` | `06_mix/attempt_001/artifacts/final.wav` | 最终混音，用于时间轴及重建对照 |
| `06_mix` | `06_mix/attempt_001/artifacts/converted_aligned.wav` | 转换且已对齐的人声 |
| `06_mix` | `06_mix/attempt_001/artifacts/mix.json` | 混音增益与对齐信息 |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/instrumental.wav` | 用于乐器拆分的伴奏 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/07_stem_separation/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `stems.json` | 轨道清单与相加规则；供 09、10、11 使用 |
| `audio/lead_vocals.wav` | 主唱音轨 |
| `audio/backing_vocals.wav` | 背景人声 / 和声音轨 |
| `audio/drums.wav` | 鼓音轨 |
| `audio/bass.wav` | 贝斯音轨 |
| `audio/guitar.wav` | 吉他音轨 |
| `audio/piano.wav` | 钢琴音轨 |
| `audio/other.wav` | 其他乐器音轨 |
| `audio/vocals.wav` | 总人声父轨 |
| `audio/instrumental.wav` | 总伴奏父轨 |
| `audio/instrumental_vocal_leak.wav` | 伴奏中估计的人声泄漏 |
| `audio/instrumental_residual.wav` | 伴奏拆分残差 |
| `audio/mix_residual.wav` | 混音残差 |
| `stems_recombined.wav` | 按叶轨相加得到的重建音频 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 07_stem_separation
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/stem_separation.py`。默认模块配置：`configs/modules/07_stem_separation.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
