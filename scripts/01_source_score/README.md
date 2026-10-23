# 01_source_score · 转写原曲乐谱

用 SheetSage2 从准备好的原曲提取旋律乐谱，供后续移调和曲风生成使用。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/01_source_score/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

这里提到 00，是为了说明 source.wav 的生产模块。00 已在同一运行目录成功完成时，直接运行本页的 01 命令即可；无需重复执行 00 或重复传 --config。尚未准备输入时，先执行 [00 的使用示例](../00_prepare/README.md)。本模块采用 melody_only=true，成品完整转谱在 08 进行。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `00_prepare` | `00_prepare/attempt_001/artifacts/source.wav` | 准备好的原曲音频 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/01_source_score/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `score.abc` | 原曲旋律 ABC 乐谱；供 02 移调、11 打包 |
| `events.json` | 转谱事件记录 |
| `transcription.mid` | 原曲转谱 MIDI |
| `beat.lab` | 原曲拍点记录 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 01_source_score
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/source_score.py`。默认模块配置：`configs/modules/01_source_score.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
