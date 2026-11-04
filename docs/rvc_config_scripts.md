# RVC 手动验证：第 2、5、7、14 步脚本

四个入口使用现有 `/opt/conda/bin/python`，相对路径按脚本定位的项目根目录解析，可从任意工作目录执行。配置生成只读上游数据，创建新的 YAML，不分离、不裁剪、不训练、不推理；不读取以前实验的 YAML 作为模板。公共权重与静态架构配置仍是外部输入。

三个配置生成器支持 `--plan`：执行相同输入检查，打印完整配置，不创建文件或目录。正式生成只写 YAML，原子发布且拒绝覆盖已有文件；数据/模型输出目录也须使用新路径。新版本或显式 `--output`、`--dataset-dir`、`--export-dir` 可用于下一次实验。

## 第 2 步：base 与公共模型检查

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/preflight/run.sh \
  --scope all --gpu 0
```

`all` 检查 RVC 训练/推理、ECAPA 评估、两级人声分离、CUDA、FFmpeg/MP3 编码；`training` 检查 RVC 训练/推理及评估；`inference` 不要求预训练 G/D 权重。检查包含版本锁和模型哈希；评估 F0 的存在性由运行依赖检查负责。缺项时退出码非零，不安装包、不创建环境、不运行模型。终端输出 `status: passed`、实际 python/prefix、GPU 和公共资产路径。`--plan` 同样只执行检查。

脚本自身固定使用 base。要让调用它的交互终端也显示 `(base)`，在终端执行 `source /opt/conda/etc/profile.d/conda.sh`、`conda activate base`；子进程不能修改父 shell 的环境。

## 第 5 步：从本次分离结果生成数据配置

前置：第 4 步已完成新分离，且第 3 步的歌曲运行已建立。下面的 recordings 必须是你本次 `scripts/voice_data/separate/run.sh` 产生的目录。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/voice_data/create_config/run.sh \
  --voice-id shan_yi_chun \
  --version rvc_manual_20261103 \
  --recordings-dir data/voice_training/shan_yi_chun/rvc_manual_20261103/recordings \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_rvc_manual_20261103 \
  --validation-ids 1698244260 1842315470 \
  --test-ids 1843079986 1864010428 \
  --segment-seconds 8 \
  --output configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml
```

检查分离批次状态、lead.wav 哈希及来源、集合编号、重复素材跨集合泄漏和当前 00 的 reference 产物。余下有效素材为 train，不读取目录中任意残留文件。不加载任何 Seed-VC 权重。

输出 `/data/liuhongda/music-cover-baseline/configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml`，标记 `purpose: dataset_preparation_only`，不包含模型权重和训练参数。默认数据集目标为 `/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/rvc_manual_20261103/dataset`。省略 `--output` 时配置默认位于 `configs/voice_data/<voice-id>_<version>.yaml`。

第 6 步仍由你执行现有数据准备入口，它只进行解码、裁剪与划分：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml \
  --run runs/voice_data/shan_yi_chun_rvc_manual_20261103
```

输出新数据集的 `train/`、`validation/`、`test/` 和 `manifest.json`。这个数据配置不能传给 Seed-VC 训练阶段；原有 Seed-VC 训练配置格式和入口继续支持。

## 第 7 步：生成 RVC 训练配置

输入是第 6 步刚生成的 dataset，而不是 recordings 的分离 manifest。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/training/create_config/run.sh \
  --voice-id shan_yi_chun \
  --version manual_20261103 \
  --source-dataset data/voice_training/shan_yi_chun/rvc_manual_20261103/dataset \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_rvc_manual_20261103 \
  --validation-windows 36:12 156:12 \
  --max-steps 4000 --save-every 1000 --batch-size 6 --learning-rate 0.0001 \
  --seed 42 --gpu 0
```

逐一检查输入音频哈希、歌手 ID、train/validation/test、集合泄漏以及 04 人声和伴奏的当前成功产物。验证窗口不能越界或重叠。配置从参数生成，不扫描旧实验选取配置或检查点。

默认输出与目标：

| 内容 | 路径 |
|---|---|
| 本次训练配置 | `/data/liuhongda/music-cover-baseline/configs/voice_training/rvc/shan_yi_chun_manual_20261103.yaml` |
| 后续 RVC 48k 数据集 | `/data/liuhongda/music-cover-baseline/data/voice_datasets/shan_yi_chun/rvc/manual_20261103` |
| 后续模型导出位置 | `/data/liuhongda/music-cover-baseline/models/voices/shan_yi_chun/rvc/manual_20261103` |

生成器不会创建后两个目录。默认训练参数为 max_steps=4000、save_every=1000、batch_size=6、learning_rate=0.0001、seed=42、input_crop_frames=400；推理验证默认 RMVPE、index_rate=0.5、protect=0.33、chunk=8 秒、overlap=0.2 秒、pad=0.5 秒。参数完整写入 YAML，可通过相应 `--index-rate`、`--protect`、`--chunk-seconds` 等选项覆盖。4000 是本工程本次默认训练预算，不是上游通用最优值。

接着仍由你执行 RVC 训练 00–05。例如第一条：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_manual_20261103.yaml \
  --run runs/voice_training/shan_yi_chun_rvc_manual_20261103
```

完整阶段见 [RVC 指南](20_rvc.md)。

## 第 14 步：根据你导出的模型生成推理配置

前置：你已依次完成 RVC 训练 00–05，以下 `voice.json` 是第 13 步导出的产物。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/rvc/manual_20261103/voice.json \
  --candidate-count 1 --seed 42
```

输出 `/data/liuhongda/music-cover-baseline/configs/voice_inference/rvc/shan_yi_chun_manual_20261103.yaml`。先校验模型、索引、配置、验证报告的哈希和后端，再按 `RVC 默认值 < voice.json 推荐参数 < 显式命令行参数` 的优先级生成完整推理配置。训练配置与推理配置均显式写入参数，推理引擎本身行为不变。

支持 `--index-rate 0.75` 等覆盖；多个候选用 `--candidate-count 3`，默认种子从 `--seed` 连续生成，也可用 `--candidate-seeds 42 43 44` 显式指定。配置中不会产生跨后端遗留参数。

第 15 步由你执行：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_rvc_manual_20261103 \
  --config configs/voice_inference/rvc/shan_yi_chun_manual_20261103.yaml --gpu 0
```

此时才加载你训练的模型生成音频。配置生成器没有代替执行前面的任何实验节点。
