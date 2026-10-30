# RVC 02_train

输入：01 的训练特征和 00 的 model_config.json。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/02_train/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：训练 run 的 checkpoints/step_*.pth（续训）、model_00004000.pth（最终推理）；供 04 使用。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/02_train/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
