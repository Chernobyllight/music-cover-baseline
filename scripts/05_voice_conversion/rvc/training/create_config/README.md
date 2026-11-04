# RVC 训练配置生成

第 7 步；固定 base `/opt/conda/bin/python`，相对路径按项目根目录解析。配置生成器可加 `--plan` 只检查预览，已有输出不覆盖。

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

完整阶段见 [RVC 指南](../../../../../docs/20_rvc.md)。


完整四步见[配置脚本说明](../../../../../docs/rvc_config_scripts.md)。
