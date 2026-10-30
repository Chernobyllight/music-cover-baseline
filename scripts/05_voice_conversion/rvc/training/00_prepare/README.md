# RVC 00_prepare

输入：原 dataset/manifest.json，保留 train/validation/test 歌曲组划分，排除 prompt。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：派生数据目录 manifest.json 与 48 kHz FLOAT WAV；训练 run 的 model_config.json；供 01 使用。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/00_prepare/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
