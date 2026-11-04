# 03_style_generation · 生成目标曲风歌曲

用 YuE2 结合移调后的旋律、歌词和目标曲风生成新编曲歌曲。

实际 YuE2 后端解释器为 `/opt/conda/bin/python`（base），预检与运行使用同一路径；
不再调用 `envs/yue2/bin/python`。主控制器仍复用已有 control 环境。
base 的最小安装与版本说明见[环境说明](../../docs/03-environment.md#yue2-直接使用-base)。
权重仍为 `models/YuE2-3B/model.safetensors` 和 `models/YuE2-Vae/model.safetensors`。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)、[02_pitch_adaptation](../02_pitch_adaptation/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

运行配置中的工程内素材保存为项目相对路径；切换机器时按当前工程根目录解析。
旧绝对路径记录也由公共路径解析器兼容，不需要创建 `/mnt/afs_f` 等挂载别名。
`--run runs/...` 的 `runs` 前不要加 `/`。详细路径契约见[跨机器续跑](../../docs/09_modular_pipeline.md#路径与跨机器续跑)。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/03_style_generation/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

曲风和 seed 从运行目录保存的请求读取；本示例为 Synthwave / Dance-pop、seed=42。孙燕姿参考录音在 05 才直接用于音色转换，03 的音频可先单独试听编曲。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `02_pitch_adaptation` | `02_pitch_adaptation/attempt_001/artifacts/score_transposed.abc` | 用于生成的旋律乐谱 |
| `00_prepare` | `00_prepare/attempt_001/artifacts/lyrics.txt` | 清理后的歌词 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/03_style_generation/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `generated/audio.flac` | 新曲风的完整歌曲；供 04 分离和 06 混音对照 |
| `generation_request.json` | 实际使用的曲风、歌词及随机种子 |
| `generation_budget.json` | 乐谱时长、上下文长度与语义 token 预算 |
| `generated/result.json` | 生成时长、采样率、截断状态与模型信息；正常完成时 truncated 均为 false |
| `generated/score.abc` | 本次生成使用的旋律乐谱 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 03_style_generation
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/style_generation.py`。默认模块配置：`configs/modules/03_style_generation.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
