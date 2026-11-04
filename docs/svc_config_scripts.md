# SVC 手动验证：脚本生成配置

适用于单依纯音色 +《绿洲》的新实验。用户执行脚本生成每份实验配置，不复制旧歌手配置。公共权重、原始录音以及已经完成的 00–04 歌曲产物是明确的前置输入。

以下脚本直接使用 `/opt/conda/bin/python`，从脚本位置找到项目根目录。相对路径按项目根目录解析，不依赖当前目录或 `VC_*` 变量。配置生成支持 `--plan`：校验并打印配置但不写文件；去掉它才生成 YAML。同名配置、已有目标数据集或模型目录拒绝覆盖。生成配置不分离、不训练、不验证、不推理。

## 第 2 步：环境与公共权重预检

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/preflight/run.sh \
  --scope all --gpu 0
```

检查 base、CUDA、ContentVec/RMVPE、G/D 底模、ECAPA 评价依赖及权重、两个人声分离模型、FFmpeg 和 MP3 编码器。预期 `status: passed`、`python: /opt/conda/bin/python`。不安装或下载任何组件。

`--scope training` 省略分离组件；`--scope inference` 进一步省略 G/D 底模，但仍检查评价依赖。脚本不会改变父终端的 Conda 激活状态。

## 第 3–6 步：仍由用户准备新数据

依次使用现有 [fork_run.sh](../scripts/05_voice_conversion/fork_run.sh)、[录音分离脚本](../scripts/voice_data/separate/README.md)、[数据配置生成脚本](../scripts/voice_data/create_config/README.md)、[数据准备入口](../scripts/05_voice_conversion/finetuned/training/00_prepare/README.md)。最后一个入口只执行共用数据准备，不训练 Seed-VC。

本例输入：

- 原始录音：`/data/liuhongda/music-cover-baseline/data/raw/references/shan_yi_chun/sources/*/original.m4a`。
- 上游歌曲：`/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_modular_test`，fork 为 `/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104`；保留原 run。
- 新分离目录：`/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/svc_manual_20261104/recordings`。排除 `1787247852`（参考录音）与 `6779157198`（此前无合适窗口）。
- validation IDs：`1698244260 1842315470`；test IDs：`1843079986 1864010428`；其它选中录音用于 train，按录音组隔离。
- 用户新生成的切片数据集：`/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/svc_manual_20261104/dataset`，含 `manifest.json` 和三个 split 的音频。

## 第 7 步：生成 SVC 训练配置

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/training/create_config/run.sh \
  --voice-id shan_yi_chun \
  --version manual_20261104 \
  --source-dataset data/voice_training/shan_yi_chun/svc_manual_20261104/dataset \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104 \
  --validation-windows 36:12 156:12 \
  --max-steps 4000 --save-every 1000 --batch-size 6 \
  --learning-rate 0.0001 --seed 42 --gpu 0 \
  --peak-policy attenuate_if_needed --peak-limit 0.99
```

脚本检查音频哈希、歌手、集合隔离、44.1 kHz 单声道、非有限值和峰值；校验当前 00/04 成功 attempt 的产物。`reference` 自动取自当前 00，必须与数据集清单匹配。验证窗口格式为起始秒数:持续秒数，不得重叠或越界。

分离后的 FLOAT 音频可能存在绝对峰值 >= 1；直接转换 PCM16 会削顶。新配置显式保存 `audio.peak_policy: attenuate_if_needed` 和 `peak_limit: 0.99`，由后续 SVC 00 对超限片段只降低整体增益，并在新数据集 manifest 的 `audio_preparation` 中记录原峰值与增益。配置生成阶段只报告，不改音频；原数据保持不变，不改变音高或时长。可用 `--peak-policy error` 改成遇到超限就报错。旧 YAML 没有 audio 字段时，仍保持原报错行为。

默认生成 `/data/liuhongda/music-cover-baseline/configs/voice_training/so_vits_svc/shan_yi_chun_manual_20261104.yaml`。配置中指定后续数据集目录 `/data/liuhongda/music-cover-baseline/data/voice_datasets/shan_yi_chun/so_vits_svc/manual_20261104`，导出目录 `/data/liuhongda/music-cover-baseline/models/voices/shan_yi_chun/so_vits_svc/manual_20261104`。可用 `--output`、`--dataset-dir`、`--export-dir` 更改。

4000 是本轮优化步数，不是官方通用最佳值。第 8–11 步仍使用原 SVC `00_prepare`、`01_train`、`02_validate`、`03_export` 入口，均传入新 YAML 和新训练 run `/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_svc_manual_20261104`。模型 JSON 由 00 根据官方结构模板新生成。

验证显式传入本次 `checkpoints/step_00004000.pth`。查看训练 run 的 `stages/02_validate/attempt_001/artifacts/validation.json`；`selected_checkpoint` 是本次默认导出模型；`eligible` 和 `assessment_status` 单独记录质量验收。即使不合格也允许导出试听，但 profile 标记 `quality_review_required`，不得把进程成功解释为质量合格。重跑以最新 attempt 为准。四阶段说明见 [SVC 指南](18_so_vits_svc.md)。

## 第 12 步：生成推理配置

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/so_vits_svc/manual_20261104/voice.json \
  --seed 42 --f0-predictor rmvpe --noise-scale 0.4 \
  --chunk-seconds 20 --overlap-seconds 0.5 --pad-seconds 0.5
```

必须先由用户完成第 11 步模型导出。脚本检查 profile、模型/config/验证文件的哈希、编码器身份及说话人，固定一个候选。输出 `/data/liuhongda/music-cover-baseline/configs/voice_inference/so_vits_svc/shan_yi_chun_manual_20261104.yaml`，可用 `--output` 更改。

设置优先级为 SVC 默认值 < profile 的 `recommended_settings`（如果有）< 显式命令参数；最终值完整写入 YAML。上述参数与标准训练验证入口的默认值一致。后续调整推理 YAML 不会改变已有验证记录；当前训练验证入口仍使用其默认推理参数。

## 第 13–14 步：整曲转换与混音

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104 \
  --config configs/voice_inference/so_vits_svc/shan_yi_chun_manual_20261104.yaml \
  --gpu 0

bash /data/liuhongda/music-cover-baseline/scripts/06_mix/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104 --gpu 0
```

SVC 专用推理入口现已直接调用 base，不再经过旧 control 解释器。转换人声：`/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104/stages/05_voice_conversion/attempt_001/artifacts/converted.wav`。最终混音：`/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104/stages/06_mix/attempt_001/artifacts/final.mp3`。

## 范围与路径限制

新生成 YAML 内的项目路径为相对路径。SVC 原训练器的 filelist、模型 JSON 和状态仍含运行时绝对路径，本次脚本化未改动其算法或整个训练目录的迁移机制；本轮在同一台服务器完成。Seed-VC、RVC 的代码、实验配置和数据不由本次工具修改。

统一验收与持续输出策略见 [评价说明](05-evaluation.md)。
