# 从本次分离结果生成 Seed-VC 训练配置

这是训练 00_prepare 之前的独立入口。它根据命令参数和已完成的分离清单，从空白构建一个训练 YAML；只读取公共模型结构配置来检查类型，不复制旧实验 YAML、不分离音频、不切片、不训练、不导出模型。Shell 入口直接使用 `/opt/conda/bin/python`，无需 VC_* 变量；相对路径统一按脚本所在项目的根目录解析。

## 5．生成本次训练配置

先完成[第 4 步原始音频分离](../../../../voice_data/separate/README.md)。以下示例为单依纯音色、绿洲验证歌曲：

| 输入 | 本次路径或取值 |
|---|---|
| 本次分离清单 | `/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/manual_20261102/recordings/manifest.json`，整批须为 succeeded |
| 本次主唱音轨 | 同一 recordings 目录下 `<素材编号>/lead.wav` 及 `separation.json` |
| 歌曲上游 | `/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_modular_test/stage_state.json` 中成功的 00_prepare、04_vocal_separation |
| 验证集录音编号 | `1698244260 1842315470` |
| 测试集录音编号 | `1843079986 1864010428` |
| 训练集 | 本次清单中未排除、且不属于上述两组的全部录音 |
| 歌曲验证窗口 | 04 人声时间轴上的 `36:12 156:12`，即从 36 秒、156 秒各取 12 秒 |
| 基础主权重 | `/data/liuhongda/music-cover-baseline/models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` |
| 公共模型结构 YAML | `/data/liuhongda/music-cover-baseline/models/Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` |

执行以下命令即可生成 YAML，不要求当前终端在项目目录：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/finetuned/training/create_config/run.sh \
  --voice-id shan_yi_chun \
  --version manual_20261102 \
  --recordings-dir data/voice_training/shan_yi_chun/manual_20261102/recordings \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test \
  --validation-ids 1698244260 1842315470 \
  --test-ids 1843079986 1864010428 \
  --validation-windows 36:12 156:12 \
  --max-steps 1000
```

若只想检查和预览，在同一命令末尾加 `--plan`。它执行相同校验并在终端展示完整 YAML，但不会创建任何文件或目录，也不加载模型。缺少第 4 步结果时会提示先完成分离，不会回退到旧资料。这里只检查公共主权重存在且非空、模型结构 YAML 可解析；完整训练依赖、权重覆盖和 GPU 检查仍由后续训练入口执行。

**预期结果**：终端打印实际 Python、数据划分数量、分离录音总时长、歌曲时长、配置输出路径。按照第 4 步的 28 份成功素材，此例为 train=24、validation=2、test=2，另外 2 份继续排除；这些时长是完整主唱录音时长，尚不是筛除短段或静音后的训练切片时长。

只新增以下文件：

`/data/liuhongda/music-cover-baseline/configs/voice_training/shan_yi_chun_seedvc_manual_20261102.yaml`

查看生成内容：

```bash
cat /data/liuhongda/music-cover-baseline/configs/voice_training/shan_yi_chun_seedvc_manual_20261102.yaml
```

YAML 包含你本次指定的 sources 列表、每条录音的 group/split、00 的参考音频路径、公共主权重/模型结构路径、训练参数、验证歌曲与窗口，以及后续数据集和导出目录。`configuration_generation` 保存分离清单和选中上游产物的路径及 SHA256，记录生成当时的输入。项目内路径写为项目相对路径；明确传入项目外的路径保留为绝对路径。

| YAML 字段 | 默认写入值 |
|---|---|
| `dataset_dir` | `data/voice_training/shan_yi_chun/manual_20261102/dataset` |
| `export_dir` | `models/voices/shan_yi_chun/seed_vc/manual_20261102` |
| `smoke_test` | `false`，需要独立训练、验证、测试录音 |
| `segment_seconds` | `8` |
| `training` | `max_steps: 1000`、`save_every: 500`、`batch_size: 1`、`learning_rate: 1e-5`、`seed: 42` |
| `validation` | 命令指定的 source_run/windows；推理 `steps: 30`、`seed: 42` |
| `gpu` | `0` |

此时 dataset、训练 run、模型导出目录都不会创建；它们分别由后续步骤产生。00 和 04 的实际 attempt 从状态记录读取，不固定为 attempt_001，也不会在当前阶段失败时擅自挑选旧成功记录。参考片段用于推理提示，不作为训练 sources；04 的人声和伴奏用于验证歌曲，不进入目标音色训练集。

## 检查规则

- 要求分离清单完成，所选每份录音的处理来源、参数指纹和 lead.wav 哈希一致；续跑中已验证的 skipped 录音可用，失败批次中保留的旧文件不可用。
- 显式传入 validation/test 编号，不存在、已排除、重复或跨组重叠的编号都会报错。其余成功录音归入 train；三个集合均须非空。
- 同一原始文件哈希或同一主唱文件哈希不得跨集合。按素材编号分组；同一首歌的不同录音/编码不一定具有相同哈希，仍需用户在整理原始素材时保证歌曲独立。第 4 步示例排除了参考歌曲《续写》，不能仅凭哈希检查保证一个参考裁剪与训练歌曲无关。
- 检查状态、outputs.json 和所需 00/04 音频的大小及哈希；验证窗口必须在音频范围内，起点非负、时长为正、互不重叠。窗口是否有清晰演唱仍需试听，不会自动替用户选片段。
- 输出 YAML、dataset_dir 或 export_dir 已存在时拒绝生成，避免误用旧实验；换版本号或显式指定新路径。没有覆盖旧文件的 force 参数。

## 后续切片命令

检查 YAML 后，执行[训练 00_prepare](../00_prepare/README.md)：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/shan_yi_chun_seedvc_manual_20261102.yaml \
  --run runs/voice_training/shan_yi_chun_seedvc_manual_20261102
```

这一步才创建 `/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/manual_20261102/dataset/manifest.json` 和 train/validation/test 切片，供训练 01 使用。它与原始分离的 recordings/manifest.json 是两份用途不同的清单。

## 可选覆盖参数

`--voice-id`、`--version`、`--recordings-dir`、`--source-run`、`--validation-ids`、`--test-ids`、`--validation-windows` 必填。歌手、素材编号和歌曲窗口不写死在代码中。voice-id/version 使用英文字母、数字、下划线、连字符或点，首字符须为字母或数字。

| 参数 | 默认值或作用 |
|---|---|
| `--output` | `configs/voice_training/<voice-id>_seedvc_<version>.yaml` |
| `--dataset-dir` | `data/voice_training/<voice-id>/<version>/dataset` |
| `--export-dir` | `models/voices/<voice-id>/seed_vc/<version>` |
| `--base-checkpoint` / `--base-config` | 上表公共主权重与模型结构配置；不能传旧实验任务 YAML 充当模型结构 |
| `--max-steps` / `--save-every` | `1000` / `500` |
| `--batch-size` / `--learning-rate` | `1` / `1e-5` |
| `--segment-seconds` / `--seed` | `8` / `42` |
| `--validation-steps` / `--validation-seed` | `30` / `42` |
| `--gpu` | `0`，写入训练配置，本步骤不使用 GPU |
| `--plan` | 只读校验并打印 YAML |
