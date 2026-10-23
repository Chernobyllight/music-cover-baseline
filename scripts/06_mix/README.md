# 06_mix · 对齐人声并混音

对齐转换后的人声，匹配增益并与伴奏合成 WAV 和 MP3。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[03_style_generation](../03_style_generation/README.md)、[04_vocal_separation](../04_vocal_separation/README.md)、[05_voice_conversion](../05_voice_conversion/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/06_mix/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

执行到这里即可在本模块 artifacts 中试听 final.wav / final.mp3。运行目录顶层 deliverables 的发布由 11 完成。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `03_style_generation` | `03_style_generation/attempt_001/artifacts/generated/audio.flac` | 生成歌曲的长度及音频对照 |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/vocals.wav` | 换声前人声，用于对齐与增益参考 |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/instrumental.wav` | 待混入的伴奏 |
| `05_voice_conversion` | `05_voice_conversion/attempt_001/artifacts/converted.wav` | 转换后人声 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/06_mix/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `final.wav` | 48 kHz 立体声 24-bit 成品；供 07、08、10、11 使用 |
| `final.mp3` | 默认 192 kbps 的成品 MP3 |
| `converted_aligned.wav` | 对齐后人声；供 07 和可选的 10 音高检查 |
| `mix.json` | 对齐、增益及音频指标；供 07、10、11 使用 |
| `reconstruction.wav` | 换声前人声与伴奏的重混对照 |
| `vocal_post_gain.wav` | 实际混入的增益后人声 |
| `instrumental_post_gain.wav` | 实际混入的增益后伴奏 |
| `mix_residual.wav` | 最终混音与上述两项之和的数值残差 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 06_mix
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/mix.py`。默认模块配置：`configs/modules/06_mix.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
