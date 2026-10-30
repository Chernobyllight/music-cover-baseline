# RVC 01_features

输入：00 准备的 train 音频。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/01_features/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：派生数据目录 features/*.npy、audio16/*.wav、train.txt、features.json；供 02 和 03 使用。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/01_features/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
